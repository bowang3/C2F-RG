"""Shared two-GPU DDP training loop for the three reproduction stages."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Callable

# Each DDP rank constructs a large vision tower. Avoid sharing Triton cache files.
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / f"c2f-triton-{os.getuid()}-{os.environ.get('LOCAL_RANK', '0')}"),
)

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.optim.lr_scheduler import LambdaLR


def create_accelerator(args) -> tuple[Accelerator, int]:
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.num_processes != 2:
        raise ValueError("This reproduction uses exactly two DDP processes; launch with run.sh.")
    micro_batch = accelerator.num_processes * args.batch_size
    if micro_batch <= 0 or args.effective_global_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.effective_global_batch_size % micro_batch:
        raise ValueError("Effective global batch size must be divisible by 2 × per-GPU batch size.")
    accumulation = args.effective_global_batch_size // micro_batch
    accelerator.gradient_accumulation_steps = accumulation
    set_seed(args.seed)
    return accelerator, accumulation


def cosine_schedule(optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    if total_steps < 1 or not 0 <= warmup_ratio < 1:
        raise ValueError("Invalid number of updates or warmup ratio.")
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, multiplier)


def make_logger(accelerator: Accelerator, output_dir: Path, args, name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = None
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(output_dir / "training.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.info("Arguments: %s", json.dumps(vars(args), ensure_ascii=False))
    return logger, handler


def train_epochs(
    *, accelerator: Accelerator, model, optimizer, loader, args,
    accumulation: int, set_mode: Callable, logger,
) -> dict:
    """Train with DDP no-sync accumulation and correct final-window normalization."""
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = cosine_schedule(optimizer, total_updates, args.warmup_ratio)
    parameters = [p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters were selected.")
    step = 0
    last_loss = float("nan")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        set_mode(accelerator.unwrap_model(model))
        totals = torch.zeros(2, dtype=torch.float64, device=accelerator.device)
        window = torch.zeros_like(totals)
        remainder = len(loader) % accumulation
        for batch_index, batch in enumerate(loader):
            with accelerator.accumulate(model):
                loss = model(**batch, use_cache=False).loss
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError(f"Non-finite loss at update {step + 1}.")
                observed = loss.detach().double()
                totals += torch.stack((observed, torch.ones_like(observed)))
                window += torch.stack((observed, torch.ones_like(observed)))
                # Accelerate divides by `accumulation` even for a short last window.
                scale = (accumulation / remainder
                         if remainder and batch_index >= len(loader) - remainder else 1.0)
                accelerator.backward(loss * scale)
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
                    if not math.isfinite(float(norm)):
                        raise FloatingPointError(f"Non-finite gradient at update {step + 1}.")
                    optimizer.step()
                    if not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                if step % args.log_every == 0:
                    reduced = accelerator.reduce(window, reduction="sum")
                    mean = (reduced[0] / reduced[1].clamp_min(1)).item()
                    message = f"epoch={epoch + 1}/{args.epochs} update={step}/{total_updates} loss={mean:.6f} lr={scheduler.get_last_lr()}"
                    accelerator.print(message)
                    if accelerator.is_main_process:
                        logger.info(message)
                    window.zero_()
        reduced = accelerator.reduce(totals, reduction="sum")
        last_loss = (reduced[0] / reduced[1].clamp_min(1)).item()
        accelerator.print(f"epoch={epoch + 1}/{args.epochs} train_loss={last_loss:.6f}")
        if accelerator.is_main_process:
            logger.info("epoch=%s train_loss=%.6f", epoch + 1, last_loss)

    return {
        "model": model,
        "optimizer_updates": step,
        "updates_per_epoch": updates_per_epoch,
        "effective_global_batch_size": 2 * args.batch_size * accumulation,
        "gradient_accumulation_steps": accumulation,
        "last_train_loss": last_loss,
    }


def save_full_model(accelerator: Accelerator, model, processor, path: Path, metadata: dict) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        # Keep the custom vision configuration in the saved full checkpoint.
        unwrapped.config.vision_config = unwrapped.model.visual.config
        path.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(path, safe_serialization=True, max_shard_size="5GB")
        processor.save_pretrained(path)
        (path / "training_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    accelerator.wait_for_everyone()


def close_logger(logger, handler, accelerator: Accelerator) -> None:
    if handler is not None:
        logger.removeHandler(handler)
        handler.close()
    accelerator.end_training()
