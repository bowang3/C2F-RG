"""Construct the PE-3D tower used by the ClinFusion checkpoint."""

import torch

from .mae_v2 import vit_large
from .pe import MyCLIP_ssl


def create_ssl_large_p8_3d_pe_model(ckpt_path="", patch_size=None, return_config=False):
    size = 8 if patch_size is None else patch_size
    encoder, config = vit_large(
        image_size=128,
        patch_size=size,
        depth_patch_size=size,
        use_rope=True,
        drop_path=0.2,
        pool_type="none",
        return_config=return_config,
    )
    model = MyCLIP_ssl(vision_encoder=encoder, text_encoder=None)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if any(name.startswith("module.") for name in checkpoint):
            checkpoint = {name.removeprefix("module."): value for name, value in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True)
    return model, config
