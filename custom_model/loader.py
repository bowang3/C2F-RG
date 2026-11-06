"""Build the ClinFusion Qwen3-VL vision tower from the released components."""

from __future__ import annotations

from transformers import AutoConfig, AutoModel

from .medical_vit import create_ssl_large_p8_3d_pe_model
from .modeling_multi_enc_qwen2_5_vl import VisionWrapperConfig
from .modeling_multi_enc_qwen3_vl import Qwen3VL_VisionWrapper
from .openclip_encoder import OpenCLIPVisionTower


def add_new_enc_to_model(model, args):
    """Attach DINOv2, ConvNeXt and PE-3D in the original checkpoint order."""
    encoders = [
        (args.new_encoder_type, args.new_encoder_path),
        (args.new_encoder_2_type, args.new_encoder_2_path),
        (args.new_encoder_3_type, args.new_encoder_3_path),
    ]
    if {name for name, _ in encoders} != {"dinov2", "convnext", "pe_3d"}:
        raise ValueError(f"Expected DINOv2, ConvNeXt and PE-3D; got {encoders!r}.")

    configs = {"original_enc": model.config.vision_config.to_dict()}
    modules = {"original_enc": model.visual}
    for name, path in encoders:
        if name == "dinov2":
            configs[name] = AutoConfig.from_pretrained(path, trust_remote_code=True).to_dict()
            modules[name] = AutoModel.from_pretrained(
                path, torch_dtype=args.compute_dtype, trust_remote_code=True,
            ).to(model.device)
        elif name == "convnext":
            tower = OpenCLIPVisionTower(
                vision_tower="convnext-large-d-320", model_path=path,
                force_image_size=args.convnext_image_size,
            ).to(model.device)
            configs[name] = tower.config
            modules[name] = tower
        else:
            tower, config = create_ssl_large_p8_3d_pe_model(
                path, patch_size=args.merlin_patch_size, return_config=True,
            )
            configs[name] = config
            modules[name] = tower.to(model.device)

    wrapper_config = VisionWrapperConfig(
        vision_model_config=configs,
        llm_hidden_size=model.config.text_config.hidden_size,
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
        fusion_type=args.multi_enc_fusion_type,
    )
    model.config.vision_config = wrapper_config
    visual = Qwen3VL_VisionWrapper(config=wrapper_config, **modules).to(
        device=model.device, dtype=model.dtype,
    )
    model.set_visual(visual)
    return model
