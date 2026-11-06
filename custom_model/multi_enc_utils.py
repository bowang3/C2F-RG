"""Processor construction and fusion-mode parsing required by ClinFusion."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
from PIL import Image
from transformers import AutoProcessor


def load_new_processors(args, init_kwargs):
    processors = {}
    image_kwargs = {}
    encoders = [
        (args.new_encoder_type, args.new_encoder_path),
        (args.new_encoder_2_type, args.new_encoder_2_path),
        (args.new_encoder_3_type, args.new_encoder_3_path),
    ]
    for index, (name, path) in enumerate(encoders, start=1):
        if name == "dinov2":
            processor = AutoProcessor.from_pretrained(
                path, use_fast=args.use_fast_tokenizer, **init_kwargs,
            )
        elif name == "convnext":
            _, _, processor = open_clip.create_model_and_transforms(
                model_name="convnext_large_d_320",
                pretrained=str(Path(path) / "open_clip_pytorch_model.bin"),
                force_image_size=(args.convnext_image_size, args.convnext_image_size),
            )
        elif name == "pe_3d":
            processor = None
            image_kwargs.update(
                merlin_patch_size=args.merlin_patch_size,
                merlin_image_size=args.merlin_image_size,
                merlin_num_channels=args.merlin_num_channels,
                merlin_volume_depth=args.merlin_volume_depth,
                merlin_sample_images=args.merlin_sample_images,
                CT_volume_cache=None,
            )
        else:
            raise ValueError(f"Unsupported auxiliary encoder {name!r}.")
        processors[f"new_processor_{index}"] = processor
        processors[f"new_processor_{index}_type"] = name
    return processors, image_kwargs


def _number_after(pattern: str, value: str) -> Optional[int]:
    match = re.search(pattern, value)
    return int(match.group(1)) if match else None


def parse_cascade_mode(fusion_type: str) -> Optional[int]:
    return _number_after(r"cascade(\d+)", fusion_type)


def parse_expansion_factor(fusion_type: str) -> Optional[int]:
    return _number_after(r"expansion(\d+)", fusion_type)


def parse_local_field(fusion_type: str) -> Optional[int]:
    return _number_after(r"local(\d+)x\d+", fusion_type)


def parse_use_ffn(fusion_type: str) -> bool:
    return "ffnx" not in fusion_type


def pil_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        values = np.asarray(image, dtype=np.float32)
        lo, hi = float(values.min()), float(values.max())
        values = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
        image = Image.fromarray((values * 255).clip(0, 255).astype(np.uint8), mode="L")
    return image.convert("RGB")
