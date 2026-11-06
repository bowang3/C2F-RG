"""Minimal LoRA for the Qwen3-VL text decoder's attention projections."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0, 1).")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        # FP32 adapter weights and gradients; the frozen base remains BF16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            update = F.linear(F.linear(self.dropout(x).float(), self.lora_A), self.lora_B)
        return base_output + (update * self.scale).to(base_output.dtype)


def install_text_attention_lora(model, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05) -> int:
    layers = model.model.language_model.layers
    expected = int(model.config.text_config.num_hidden_layers)
    if len(layers) != expected:
        raise RuntimeError(f"Expected {expected} text decoder layers, got {len(layers)}.")
    count = 0
    for layer in layers:
        attention = layer.self_attn
        for name in TARGET_NAMES:
            base = getattr(attention, name)
            if type(base) is not nn.Linear:
                raise TypeError(f"Expected nn.Linear at self_attn.{name}, got {type(base).__name__}.")
            setattr(attention, name, LoRALinear(base, rank, alpha, dropout))
            count += 1
    if count != expected * len(TARGET_NAMES):
        raise RuntimeError("The LoRA module count does not match the decoder architecture.")
    return count


def lora_parameters(model):
    for module in model.model.language_model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


@torch.no_grad()
def merge_text_attention_lora(model) -> int:
    """Fold trained LoRA deltas into base weights for a standard full checkpoint."""
    merged = 0
    for layer in model.model.language_model.layers:
        attention = layer.self_attn
        for name in TARGET_NAMES:
            module = getattr(attention, name)
            if not isinstance(module, LoRALinear):
                raise TypeError(f"Expected a LoRA layer at self_attn.{name} before merging.")
            delta = (module.lora_B @ module.lora_A) * module.scale
            module.base.weight.add_(delta.to(module.base.weight.dtype))
            setattr(attention, name, module.base)
            merged += 1
    return merged
