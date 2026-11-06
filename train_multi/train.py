"""Train the patient-level axial/sagittal fusion module from the A2 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from train_multi.data import MultiSequenceMRIDataset, MultiSequenceMRITrainingCollator
from train_multi.model_loader import load_model_and_processor
from train_utils import close_logger, create_accelerator, make_logger, save_full_model, train_epochs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", required=True)
    p.add_argument("--aligned-model-path", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--dinov2-path", required=True)
    p.add_argument("--convnext-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--effective-global-batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
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
    # Frozen decoder layers still backpropagate into the visual tokens.
    for module in model.model.language_model.modules():
        if module.__class__.__name__ == "Qwen3VLTextDecoderLayer" and getattr(module, "gradient_checkpointing", False):
            module.training = True


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.log_every <= 0:
        raise ValueError("Epochs, learning rate, gradient norm and log interval must be positive.")
    if args.weight_decay < 0 or args.num_workers < 0:
        raise ValueError("Weight decay and worker count must be nonnegative.")
    if not 0 <= args.mri_lower_percentile < args.mri_upper_percentile <= 100:
        raise ValueError("MRI percentiles must satisfy 0 <= lower < upper <= 100.")
    if (Path(args.output_dir) / "final").resolve() == Path(args.aligned_model_path).resolve():
        raise ValueError("Output checkpoint must differ from the A2 source checkpoint.")

    accelerator, accumulation = create_accelerator(args)
    output_dir = Path(args.output_dir)
    logger, handler = make_logger(accelerator, output_dir, args, "multi")
    config = {
        "enabled": True, "input_size": 4096, "hidden_size": 512, "output_size": 4096,
        "query_tokens": 256, "num_heads": 8, "dropout": 0.1,
        "sequence_names": ["T1CA", "T2A", "ADC", "T1CS", "T2S"],
        "branch_names": ["main", "ds1", "ds2", "ds3"],
        "replaces_final_projectors": True, "logical_images_per_plane": 1,
    }
    model, processor, preprocessing, load_info = load_model_and_processor(
        args.aligned_model_path, args.base_model_path, args.dinov2_path,
        args.convnext_path, config,
    )
    if int(preprocessing["merlin_sample_images"]) != 4:
        raise ValueError("The aligned visual tower requires four real 2D slices per sequence.")
    model.requires_grad_(False)
    fusion = model.model.visual.multi_sequence_fusion
    fusion.float().requires_grad_(True)
    trainable = list(fusion.parameters())
    if not trainable or any(p.dtype != torch.float32 for p in trainable):
        raise RuntimeError("The FP32 fusion module has no trainable parameters.")
    allowed = {id(p) for p in trainable}
    if any(id(p) not in allowed for p in model.parameters() if p.requires_grad):
        raise RuntimeError("Unexpected trainable parameters outside the fusion module.")
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
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay, foreach=False)
    accelerator.print(f"records={len(dataset)} fusion_parameters={sum(p.numel() for p in trainable)} accumulation={accumulation}")
    result = train_epochs(
        accelerator=accelerator, model=model, optimizer=optimizer, loader=loader,
        args=args, accumulation=accumulation, set_mode=set_training_mode, logger=logger,
    )
    metadata = {
        "source_alignment": str(Path(args.aligned_model_path).resolve()),
        "multi_sequence_config": config,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "alignment_missing_keys": len(load_info.missing_keys),
        **{key: value for key, value in result.items() if key != "model"},
    }
    save_full_model(accelerator, result["model"], processor, output_dir / "final", metadata)
    accelerator.print(f"Saved multi-sequence checkpoint to {output_dir / 'final'}")
    close_logger(logger, handler, accelerator)


if __name__ == "__main__":
    main()
