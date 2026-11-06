"""Construction of the trainable ClinFusion model and multimodal processor."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from transformers import AutoProcessor
from transformers.modeling_utils import load_sharded_checkpoint

from custom_model.image_processing_multi_enc_qwen2_vl import ImageMultiProcessor
from custom_model.loader import add_new_enc_to_model
from custom_model.model_config import ModelBuildArgs, get_config_values
from custom_model.modeling_multi_enc_qwen3_vl import (
    MultiEnc_Qwen3VLForConditionalGeneration,
    MultiEnc_Qwen3VLModel,
)
from custom_model.multi_enc_utils import load_new_processors


def load_model_and_processor(
    clinfusion_path: str,
    base_model_path: str,
    dinov2_path: str,
    convnext_path: str,
):
    """Recreate ClinFusion's custom architecture and load its released weights."""
    required = [clinfusion_path, base_model_path, dinov2_path, convnext_path]
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required model path does not exist: {path}")

    torch_dtype = torch.bfloat16
    load_kwargs = {
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "flash_attention_2",
    }
    model = MultiEnc_Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        **load_kwargs,
    )

    config_values = get_config_values(
        clinfusion_path,
        [
            "vision_config.fusion_type",
            "vision_config.vision_model_config",
            "vision_config.vision_model_config.convnext.model_cfg.vision_cfg.image_size",
        ],
        strict=True,
    )
    encoder_types = [
        name
        for name in config_values["vision_config.vision_model_config"]
        if name != "original_enc"
    ]
    encoder_paths: Dict[str, str] = {
        "dinov2": dinov2_path,
        "convnext": convnext_path,
        "pe_3d": "",
    }
    unknown = [name for name in encoder_types if name not in encoder_paths]
    if unknown:
        raise ValueError(f"Unsupported auxiliary encoders in checkpoint: {unknown}")
    if len(encoder_types) != 3:
        raise ValueError(
            "This trainer expects the released three-auxiliary ClinFusion layout "
            f"(DINOv2, ConvNeXt, PE-3D), got {encoder_types}."
        )
    pairs = [(name, encoder_paths[name]) for name in encoder_types]
    pairs += [(None, None)] * (3 - len(pairs))

    model_args = ModelBuildArgs(
        new_encoder_type=pairs[0][0],
        new_encoder_path=pairs[0][1],
        new_encoder_2_type=pairs[1][0],
        new_encoder_2_path=pairs[1][1],
        new_encoder_3_type=pairs[2][0],
        new_encoder_3_path=pairs[2][1],
        compute_dtype=torch_dtype,
        multi_enc_fusion_type=config_values["vision_config.fusion_type"],
        convnext_image_size=config_values[
            "vision_config.vision_model_config.convnext.model_cfg.vision_cfg.image_size"
        ],
    )
    preprocessing = get_config_values(
        str(Path(clinfusion_path) / "preprocessor_config.json"),
        [
            "merlin_patch_size",
            "merlin_image_size",
            "merlin_num_channels",
            "merlin_volume_depth",
            "merlin_sample_images",
        ],
        strict=True,
    )
    model_args.update(preprocessing)

    vl_model = MultiEnc_Qwen3VLModel.from_qwen_model(model.model)
    vl_model = add_new_enc_to_model(vl_model, model_args)
    model.set_model(vl_model)
    load_info = load_sharded_checkpoint(model, clinfusion_path, strict=False, prefer_safe=True)
    if load_info.unexpected_keys:
        raise RuntimeError(f"Unexpected ClinFusion checkpoint keys: {load_info.unexpected_keys[:20]}")
    parameter_names = {name for name, _ in model.named_parameters()}
    missing_parameters = [name for name in load_info.missing_keys if name in parameter_names]
    if missing_parameters:
        raise RuntimeError(
            f"ClinFusion checkpoint is missing model parameters: {missing_parameters[:20]}"
        )
    model.config.use_cache = False
    model.model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=False)
    processor_kwargs = {
        "trust_remote_code": False,
        "cache_dir": None,
        "revision": "main",
        "token": None,
    }
    new_processors, image_kwargs = load_new_processors(model_args, processor_kwargs)
    processor.image_processor = ImageMultiProcessor(
        processor.image_processor,
        **new_processors,
        **image_kwargs,
    )
    return model, processor, preprocessing, load_info
