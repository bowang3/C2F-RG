# 先把整个 ClinFusion 全冻结，再按 A1/A2 阶段只解冻指定的 3D 模块。


from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class StagePolicy:
    stage: str
    train_projector: bool = True
    train_full_pe3d: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def get_vision_wrapper(model: nn.Module) -> nn.Module:
    """Return ``model.model.visual`` while tolerating a DDP-style wrapper."""
    if hasattr(model, "module"):
        model = model.module
    try:
        visual = model.model.visual
    except AttributeError as exc:
        raise AttributeError("Expected a ClinFusion model with model.visual") from exc
    if not hasattr(visual, "pe_3d") or visual.pe_3d is None:
        raise ValueError("This ClinFusion checkpoint does not contain a PE-3D encoder.")
    return visual


def get_pe3d_fusion_index(visual: nn.Module) -> int:
    """Get PE-3D's index in the cascade FFN lists.

    Qwen3-VL appends PE-3D after all enabled 2D auxiliary encoders in
    ``processing_flags``.  Therefore its FFN index is the number of enabled 2D
    auxiliaries, independent of cascade1/cascade2 ordering.
    """
    return sum(getattr(visual, name, None) is not None for name in ("dinov2", "convnext"))


def _set_trainable(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(True)

# 找出所有PE-3D CaSL
def _pe3d_casl_modules(visual: nn.Module) -> List[Tuple[str, nn.Module]]:
    modules: List[Tuple[str, nn.Module]] = []
    for name in ("pe_3d_cross_attn", "pe_3d_cross_attn_deepstack"):
        module = getattr(visual, name, None)
        if module is not None:
            modules.append((name, module))

    fusion_index = get_pe3d_fusion_index(visual)
    fusion_ffn = getattr(visual, "fusion_ffn", None)
    if isinstance(fusion_ffn, nn.ModuleList):
        if fusion_index >= len(fusion_ffn):
            raise IndexError("PE-3D cascade FFN index is outside fusion_ffn.")
        modules.append((f"fusion_ffn.{fusion_index}", fusion_ffn[fusion_index]))

    deepstack_ffn = getattr(visual, "fusion_ffn_deepstack", None)
    if isinstance(deepstack_ffn, nn.ModuleList):
        for layer_index, layer_ffns in enumerate(deepstack_ffn):
            if fusion_index >= len(layer_ffns):
                raise IndexError("PE-3D cascade FFN index is outside fusion_ffn_deepstack.")
            modules.append(
                (f"fusion_ffn_deepstack.{layer_index}.{fusion_index}", layer_ffns[fusion_index])
            )
    return modules


def configure_stage(
    model: nn.Module,
    stage: str,
) -> Tuple[StagePolicy, List[Tuple[str, nn.Module]]]:
    """Freeze the whole model and selectively enable A1 or A2 modules."""
    stage = stage.lower()
    if stage not in {"a1", "a2"}:
        raise ValueError(f"stage must be 'a1' or 'a2', got {stage!r}.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    visual = get_vision_wrapper(model)
    selected: List[Tuple[str, nn.Module]] = []

    # 3D CaSL is trained in both stages.
    selected.extend(_pe3d_casl_modules(visual))

    # The 3D projector is trained in both A1 and A2.
    projector = getattr(visual, "pe_3d_pre_projector", None)
    if projector is None:
        raise ValueError("PE-3D projector (pe_3d_pre_projector) was not found.")
    selected.append(("pe_3d_pre_projector", projector))

    if stage == "a2":
        # A2 fine-tunes the complete PE-3D image encoder, including patch
        # embedding, positional embedding, all Transformer blocks and ln_post.
        selected.append(("pe_3d", visual.pe_3d))

    for _, module in selected:
        _set_trainable(module)

    policy = StagePolicy(
        stage=stage,
        train_projector=True,
        train_full_pe3d=stage == "a2",
    )
    assert_no_unexpected_trainables(model, selected)
    return policy, selected


def assert_no_unexpected_trainables(
    model: nn.Module, selected_modules: Sequence[Tuple[str, nn.Module]]
) -> None:
    allowed_ids = {
        id(parameter)
        for _, module in selected_modules
        for parameter in module.parameters()
    }
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in allowed_ids
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:20]}")


def set_training_modes(model: nn.Module, selected_modules: Iterable[Tuple[str, nn.Module]]) -> None:
    """Keep every frozen component deterministic while enabling selected dropout."""
    model.eval()
    for _, module in selected_modules:
        module.train()
    # Frozen decoder layers still backpropagate to the visual inputs. HF gates
    # checkpointing on this flag; keep their dropout children in eval mode.
    for module in model.modules():
        if (module.__class__.__name__ == "Qwen3VLTextDecoderLayer"
                and getattr(module, "gradient_checkpointing", False)):
            module.training = True


def trainable_parameter_summary(model: nn.Module) -> Dict[str, object]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100.0 * trainable / max(total, 1),
        "trainable_names": names,
    }
