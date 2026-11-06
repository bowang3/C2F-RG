"""Load the completed multi-sequence model and its saved image processor."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from custom_model.image_processing_multi_enc_qwen2_vl import ClinFusionProcessor
from custom_model.modeling_multi_enc_qwen3_vl import MultiEnc_Qwen3VLForConditionalGeneration
from train_multi_llm.lora import install_text_attention_lora


def load_full_model_and_processor(multi_model_path: str):
    """Restore a complete multi-sequence checkpoint without LoRA wrappers."""
    root = Path(multi_model_path)
    if not (root / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Complete multi-sequence checkpoint not found: {root}")
    with (root / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    fusion_config = config.get("multi_sequence_config", {})
    if (
        fusion_config.get("enabled") is not True
        or fusion_config.get("logical_images_per_plane") != 1
        or fusion_config.get("query_tokens") != 256
        or fusion_config.get("branch_names") != ["main", "ds1", "ds2", "ds3"]
        or fusion_config.get("sequence_names") != ["T1CA", "T2A", "ADC", "T1CS", "T2S"]
    ):
        raise ValueError("Checkpoint is not the completed one-interval-per-plane multi-sequence model.")

    model = MultiEnc_Qwen3VLForConditionalGeneration.from_pretrained(
        root,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    # PE-3D's RoPE cache is non-persistent. Under low_cpu_mem_usage it is
    # constructed on meta and is absent from the saved weight shards.
    pe_3d_rope = model.model.visual.pe_3d.visual.rope
    pe_3d_rope.init_tensors()
    if getattr(model.model.visual, "multi_sequence_fusion", None) is None:
        raise RuntimeError("The loaded model has no multi-sequence fusion module.")
    processor = ClinFusionProcessor.from_pretrained(root, local_files_only=True)
    preprocessing = {
        "merlin_volume_depth": processor.image_processor.merlin_volume_depth,
        "merlin_image_size": processor.image_processor.merlin_image_size,
        "merlin_sample_images": processor.image_processor.merlin_sample_images,
    }
    if int(preprocessing["merlin_sample_images"]) != 4:
        raise ValueError("The frozen A2 encoders require four real slices per MRI sequence.")
    return model, processor, preprocessing, fusion_config


def load_model_and_processor(
    multi_model_path: str,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
):
    """Restore the preceding full checkpoint and prepare the trainable stage."""
    model, processor, preprocessing, fusion_config = load_full_model_and_processor(multi_model_path)
    model.config.use_cache = False
    model.model.config.use_cache = False
    model.model.language_model.config.use_cache = False
    model.requires_grad_(False)
    fusion = model.model.visual.multi_sequence_fusion
    fusion.float().requires_grad_(True)
    lora_module_count = install_text_attention_lora(
        model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
    )
    return model, processor, preprocessing, fusion_config, lora_module_count
