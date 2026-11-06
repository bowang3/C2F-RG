"""Continue the fusion checkpoint with text-attention LoRA on two GPUs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from train_multi.data import MultiSequenceMRIDataset, MultiSequenceMRITrainingCollator
from train_multi_llm.lora import LoRALinear, lora_parameters, merge_text_attention_lora
from train_multi_llm.model_loader import load_model_and_processor
from train_utils import close_logger, create_accelerator, make_logger, save_full_model, train_epochs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", required=True)
    p.add_argument("--multi-model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--effective-global-batch-size", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-learning-rate", type=float, default=3e-5)
    p.add_argument("--fusion-learning-rate", type=float, default=1e-5)
    p.add_argument("--fusion-weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--mri-lower-percentile", type=float, default=0.5)
    p.add_argument("--mri-upper-percentile", type=float, default=99.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def set_training_mode(model) -> None:
    model.eval()
    model.model.visual.multi_sequence_fusion.train()
    for module in model.model.language_model.modules():
        if isinstance(module, LoRALinear):
            module.train()
        if module.__class__.__name__ == "Qwen3VLTextDecoderLayer" and getattr(module, "gradient_checkpointing", False):
            module.training = True


def optimizer_groups(model, args):
    fusion = model.model.visual.multi_sequence_fusion
    fusion_ids = {id(p) for p in fusion.parameters()}
    lora = list(lora_parameters(model))
    lora_ids = {id(p) for p in lora}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if not fusion_ids or not lora_ids or fusion_ids & lora_ids or trainable != fusion_ids | lora_ids:
        raise RuntimeError("Trainable parameters must be exactly fusion and text-attention LoRA.")
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise RuntimeError("Fusion and LoRA parameters must be FP32.")
    decay, no_decay = [], []
    for name, parameter in fusion.named_parameters():
        if name.endswith("bias") or parameter.ndim == 1 or "queries" in name or "sequence_embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": lora, "lr": args.lora_learning_rate, "weight_decay": 0.0},
        {"params": decay, "lr": args.fusion_learning_rate, "weight_decay": args.fusion_weight_decay},
        {"params": no_decay, "lr": args.fusion_learning_rate, "weight_decay": 0.0},
    ]


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.lora_rank, args.log_every) <= 0:
        raise ValueError("Epochs, LoRA rank and log interval must be positive.")
    if min(args.lora_alpha, args.lora_learning_rate, args.fusion_learning_rate, args.max_grad_norm) <= 0:
        raise ValueError("LoRA alpha, learning rates and gradient norm must be positive.")
    if not 0 <= args.lora_dropout < 1 or args.fusion_weight_decay < 0 or args.num_workers < 0:
        raise ValueError("Invalid dropout, weight decay or worker count.")
    if not 0 <= args.mri_lower_percentile < args.mri_upper_percentile <= 100:
        raise ValueError("MRI percentiles must satisfy 0 <= lower < upper <= 100.")
    if (Path(args.output_dir) / "final").resolve() == Path(args.multi_model_path).resolve():
        raise ValueError("Output checkpoint must differ from the source fusion checkpoint.")

    accelerator, accumulation = create_accelerator(args)
    output_dir = Path(args.output_dir)
    logger, handler = make_logger(accelerator, output_dir, args, "multi_llm")
    model, processor, preprocessing, fusion_config, lora_count = load_model_and_processor(
        args.multi_model_path, args.lora_rank, args.lora_alpha, args.lora_dropout
    )
    groups = optimizer_groups(model, args)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_training_mode(model)
    collator = MultiSequenceMRITrainingCollator(
        processor, volume_depth=preprocessing["merlin_volume_depth"],
        image_size=preprocessing["merlin_image_size"],
        sample_images=preprocessing["merlin_sample_images"], query_tokens=256,
        lower_percentile=args.mri_lower_percentile,
        upper_percentile=args.mri_upper_percentile,
    )
    dataset = MultiSequenceMRIDataset(args.train_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collator, pin_memory=True)
    optimizer = AdamW(groups, foreach=False)
    accelerator.print(f"records={len(dataset)} trainable={trainable_count} lora_modules={lora_count} accumulation={accumulation}")
    result = train_epochs(
        accelerator=accelerator, model=model, optimizer=optimizer, loader=loader,
        args=args, accumulation=accumulation, set_mode=set_training_mode, logger=logger,
    )
    metadata = {
        "source_multi": str(Path(args.multi_model_path).resolve()),
        "multi_sequence_config": fusion_config,
        "trainable_parameters": trainable_count,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                 "dropout": args.lora_dropout, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "merged": True},
        **{key: value for key, value in result.items() if key != "model"},
    }
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(result["model"])
        unwrapped.eval()
        merged = merge_text_attention_lora(unwrapped)
        if merged != lora_count:
            raise RuntimeError(f"Merged {merged} LoRA modules, expected {lora_count}.")
    save_full_model(accelerator, result["model"], processor, output_dir / "final", metadata)
    accelerator.print(f"Saved merged full checkpoint to {output_dir / 'final'}")
    close_logger(logger, handler, accelerator)


if __name__ == "__main__":
    main()
