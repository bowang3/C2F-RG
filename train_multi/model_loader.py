"""Load the single-sequence A2 checkpoint and install multi-sequence fusion."""

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
from custom_model.multi_sequence_fusion import install_multi_sequence_fusion


def load_model_and_processor(
    aligned_model_path: str,
    base_model_path: str,
    dinov2_path: str,
    convnext_path: str,
    multi_sequence_config: Dict[str, object],
):
    """Recreate ClinFusion, restore A2 weights, then add the trainable module."""
    fusion_mode = multi_sequence_config.get("fusion_mode", "by_plane")
    if fusion_mode == "all_sequences":
        if multi_sequence_config.get("logical_images_per_patient") != 1:
            raise ValueError("All-sequence fusion requires one logical image per patient.")
    elif multi_sequence_config.get("logical_images_per_plane") != 1:
        raise ValueError("Plane fusion requires one logical image per plane.")
    for path in (aligned_model_path, base_model_path, dinov2_path, convnext_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Required model path does not exist: {path}")

    torch_dtype = torch.bfloat16
    model = MultiEnc_Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )

    config_values = get_config_values(
        aligned_model_path,
        [
            "vision_config.fusion_type",
            "vision_config.vision_model_config",
            "vision_config.vision_model_config.convnext.model_cfg.vision_cfg.image_size",
        ],
        strict=True,
    )
    encoder_configs = config_values["vision_config.vision_model_config"]
    if not isinstance(encoder_configs, dict):
        raise TypeError("vision_config.vision_model_config must be an object.")

    encoder_types = [name for name in encoder_configs if name != "original_enc"]
    encoder_paths: Dict[str, str] = {
        "dinov2": dinov2_path,
        "convnext": convnext_path,
        "pe_3d": "",
    }
    unknown = [name for name in encoder_types if name not in encoder_paths]
    if unknown or len(encoder_types) != 3:
        raise ValueError(
            "Expected the A2 DINOv2/ConvNeXt/PE-3D layout; "
            f"got {encoder_types}."
        )
    pairs = [(name, encoder_paths[name]) for name in encoder_types]

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
        str(Path(aligned_model_path) / "preprocessor_config.json"),
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
    model.set_model(add_new_enc_to_model(vl_model, model_args))

    # Restore the complete A2 model before replacing its final 4096→4096 projectors.
    load_info = load_sharded_checkpoint(
        model,
        aligned_model_path,
        strict=False,
        prefer_safe=True,
    )
    if load_info.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {load_info.unexpected_keys[:20]}")
    parameter_names = {name for name, _ in model.named_parameters()}
    missing_parameters = [
        name for name in load_info.missing_keys if name in parameter_names
    ]
    if missing_parameters:
        raise RuntimeError(
            f"Checkpoint is missing model parameters: {missing_parameters[:20]}"
        )

    # 安装融合模块
    install_multi_sequence_fusion(model.model.visual, multi_sequence_config)
    model.config.multi_sequence_config = dict(multi_sequence_config)
    model.model.config.multi_sequence_config = dict(multi_sequence_config)
    model.config.use_cache = False
    model.model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=False)
    new_processors, image_kwargs = load_new_processors(
        model_args,
        {
            "trust_remote_code": False,
            "cache_dir": None,
            "revision": "main",
            "token": None,
        },
    )
    processor.image_processor = ImageMultiProcessor(
        processor.image_processor,
        **new_processors,
        **image_kwargs,
    )
    return model, processor, preprocessing, load_info
