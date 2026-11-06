"""A1/A2 single-sequence MRI alignment from the released ClinFusion checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.modeling_utils import load_sharded_checkpoint

from train_alignment.data import MRITrainingCollator, SingleSequenceMRIDataset
from train_alignment.model_loader import load_model_and_processor
from train_alignment.stage_policy import configure_stage, get_vision_wrapper, set_training_modes, trainable_parameter_summary
from train_utils import close_logger, create_accelerator, make_logger, save_full_model, train_epochs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("a1", "a2"), required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--clinfusion-path", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--dinov2-path", required=True)
    p.add_argument("--convnext-path", required=True)
    p.add_argument("--a1-checkpoint")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--effective-global-batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--pe3d-learning-rate", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--mri-lower-percentile", type=float, default=0.5)
    p.add_argument("--mri-upper-percentile", type=float, default=99.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "a2" and not args.a1_checkpoint:
        raise ValueError("A2 requires --a1-checkpoint.")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.log_every <= 0:
        raise ValueError("Epochs, learning rate, gradient norm and log interval must be positive.")
    if args.weight_decay < 0 or args.num_workers < 0:
        raise ValueError("Weight decay and worker count must be nonnegative.")
    if not 0 <= args.mri_lower_percentile < args.mri_upper_percentile <= 100:
        raise ValueError("MRI percentiles must satisfy 0 <= lower < upper <= 100.")

    accelerator, accumulation = create_accelerator(args)
    output_dir = Path(args.output_dir)
    logger, handler = make_logger(accelerator, output_dir, args, f"alignment.{args.stage}")
    model, processor, preprocessing, load_info = load_model_and_processor(
        args.clinfusion_path, args.base_model_path, args.dinov2_path, args.convnext_path
    )
    if args.stage == "a2":
        checkpoint = Path(args.a1_checkpoint)
        if not (checkpoint / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"A1 full checkpoint not found: {checkpoint}")
        load_sharded_checkpoint(model, str(checkpoint), strict=True, prefer_safe=True)

    policy, selected = configure_stage(model, args.stage)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    visual = get_vision_wrapper(model)
    visual.fusion_gradient_checkpointing = True
    if args.stage == "a2":
        visual.pe_3d.set_grad_checkpointing(True)
    set_training_modes(model, selected)
    summary = trainable_parameter_summary(model)

    collator = MRITrainingCollator(
        processor,
        volume_depth=preprocessing["merlin_volume_depth"],
        image_size=preprocessing["merlin_image_size"],
        sample_images=preprocessing["merlin_sample_images"],
        lower_percentile=args.mri_lower_percentile,
        upper_percentile=args.mri_upper_percentile,
    )
    dataset = SingleSequenceMRIDataset(args.train_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collator, pin_memory=True)
    groups = {"alignment": [], "pe3d": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups["pe3d" if args.stage == "a2" and ".pe_3d." in name else "alignment"].append(parameter)
    optimizer_groups = [{"params": groups["alignment"], "lr": args.learning_rate}]
    if groups["pe3d"]:
        optimizer_groups.append({"params": groups["pe3d"], "lr": args.pe3d_learning_rate})
    optimizer = AdamW(optimizer_groups, weight_decay=args.weight_decay, foreach=False)
    accelerator.print(f"stage={args.stage} records={len(dataset)} trainable={summary['trainable_parameters']} accumulation={accumulation}")
    result = train_epochs(
        accelerator=accelerator, model=model, optimizer=optimizer, loader=loader,
        args=args, accumulation=accumulation,
        set_mode=lambda unwrapped: set_training_modes(unwrapped, selected), logger=logger,
    )
    metadata = {
        "stage": args.stage,
        "source_clinfusion": str(Path(args.clinfusion_path).resolve()),
        "source_a1": str(Path(args.a1_checkpoint).resolve()) if args.a1_checkpoint else None,
        "trainable_parameters": summary["trainable_parameters"],
        "clinfusion_missing_keys": len(load_info.missing_keys),
        **{key: value for key, value in result.items() if key != "model"},
    }
    save_full_model(accelerator, result["model"], processor, output_dir / "final", metadata)
    accelerator.print(f"Saved {args.stage} checkpoint to {output_dir / 'final'}")
    close_logger(logger, handler, accelerator)


if __name__ == "__main__":
    main()
