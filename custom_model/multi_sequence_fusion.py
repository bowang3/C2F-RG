"""Patient-level multi-sequence MRI fusion for ClinFusion.

Every MRI sequence first passes through the frozen A2 single-sequence visual
pipeline.  Main and the three DeepStack branches therefore enter this module
as 256 tokens in the 4096-dimensional ClinFusion feature space.  The module
does not assume that equal token indices in different MRI sequences refer to
the same physical position.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SEQUENCE_NAMES: Tuple[str, ...] = ("T1CA", "T2A", "ADC", "T1CS", "T2S")
AXIAL_SEQUENCE_IDS: Tuple[int, ...] = (0, 1, 2)
SAGITTAL_SEQUENCE_IDS: Tuple[int, ...] = (3, 4)
BRANCH_NAMES: Tuple[str, ...] = ("main", "ds1", "ds2", "ds3")
class DirectQueryCrossAttention(nn.Module):
    """Cross-attention where learned queries are Q without a Wq projection."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = float(dropout)

        # Q is already a learned [query_tokens, hidden_size] matrix. Only the
        # memory needs K/V projections; Wo mixes the attended heads afterward.
        self.key_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or memory.ndim != 3 or key_padding_mask.ndim != 2:
            raise ValueError("Cross-attention expects Q [B,Q,D], memory [B,M,D], mask [B,M].")
        if query.shape[0] != memory.shape[0] or memory.shape[:2] != key_padding_mask.shape:
            raise ValueError("Cross-attention batch or memory-mask shapes do not match.")
        if query.shape[-1] != self.hidden_size or memory.shape[-1] != self.hidden_size:
            raise ValueError(f"Cross-attention hidden size must be {self.hidden_size}.")

        batch_size, query_tokens, _ = query.shape
        memory_tokens = memory.shape[1]
        # No query projection: normalized learned queries are split directly
        # into heads as [B, H, Q, Dh].
        query = query.reshape(
            batch_size, query_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key_projection(memory).reshape(
            batch_size, memory_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value_projection(memory).reshape(
            batch_size, memory_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # scaled_dot_product_attention uses True for positions that may attend,
        # the inverse convention of nn.MultiheadAttention's key_padding_mask.
        attention_mask = (~key_padding_mask)[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size, query_tokens, self.hidden_size
        )
        return self.output_projection(attended)




class PlaneResampler(nn.Module):
    """Resample all available sequences in one anatomical plane."""

    def __init__(
        self,
        hidden_size: int,
        query_tokens: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_tokens = int(query_tokens)
        self.queries = nn.Parameter(torch.empty(1, self.query_tokens, hidden_size))
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = DirectQueryCrossAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(
        self,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if memory.ndim != 3 or key_padding_mask.ndim != 2:
            raise ValueError("PlaneResampler expects memory [B,M,D] and mask [B,M].")
        if memory.shape[:2] != key_padding_mask.shape:
            raise ValueError("Plane memory and key-padding mask shapes do not match.")
        if bool(key_padding_mask.all(dim=1).any()):
            raise ValueError("A missing plane must be filtered before PlaneResampler.")

        query = self.queries.expand(memory.shape[0], -1, -1)
        normalized_memory = self.memory_norm(memory)
        attended = self.cross_attention(
            self.query_norm(query),
            normalized_memory,
            key_padding_mask=key_padding_mask,
        )
        query = query + attended
        query = query + self.ffn(self.ffn_norm(query))
        return self.output_norm(query).to(dtype=memory.dtype)


class BranchFusion(nn.Module):
    """One independent fusion path for Main or a DeepStack branch."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        query_tokens: int,
        num_heads: int,
        dropout: float,
        fusion_mode: str = "by_plane",
    ) -> None:
        super().__init__()
        # These modules are shared by all MRI sequences inside this branch,
        # but are independent from the corresponding modules in other branches.
        self.input_norm = nn.LayerNorm(input_size)
        self.down_projection = nn.Linear(input_size, hidden_size)
        if fusion_mode == "by_plane":
            self.axial = PlaneResampler(hidden_size, query_tokens, num_heads, dropout)
            self.sagittal = PlaneResampler(hidden_size, query_tokens, num_heads, dropout)
        elif fusion_mode == "all_sequences":
            self.all_sequences = PlaneResampler(hidden_size, query_tokens, num_heads, dropout)
        else:
            raise ValueError(f"Unsupported MRI fusion mode: {fusion_mode}")
        self.up_projection = nn.Linear(hidden_size, output_size)


class MultiSequenceMRIFusion(nn.Module):
    """Fuse Main and three DeepStack branches by plane or into one group.

    The sequence-identity embedding is shared globally because MRI sequence
    semantics do not change with visual depth.  Every other trainable module is
    independent for Main, DS1, DS2, and DS3.  Within one branch, all sequence
    types share that branch's input norm and down/up projections.
    """

    def __init__(
        self,
        input_size: int = 4096,
        hidden_size: int = 512,
        output_size: int = 4096,
        query_tokens: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        branch_names: Sequence[str] = BRANCH_NAMES,
        fusion_mode: str = "by_plane",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if query_tokens != 256:
            raise ValueError("Multi-sequence MRI fusion requires 256 query tokens per group.")
        if fusion_mode not in ("by_plane", "all_sequences"):
            raise ValueError(f"Unsupported MRI fusion mode: {fusion_mode}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.query_tokens = int(query_tokens)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.branch_names = tuple(branch_names)
        self.fusion_mode = fusion_mode

        # One table, five distinct rows, shared by Main and all DeepStack paths.
        self.sequence_embedding = nn.Embedding(len(SEQUENCE_NAMES), self.hidden_size)
        self.branches = nn.ModuleDict(
            {
                name: BranchFusion(
                    self.input_size,
                    self.hidden_size,
                    self.output_size,
                    self.query_tokens,
                    self.num_heads,
                    self.dropout,
                    self.fusion_mode,
                )
                for name in self.branch_names
            }
        )
        nn.init.normal_(self.sequence_embedding.weight, mean=0.0, std=0.02)

    def to_config_dict(self) -> Dict[str, object]:
        config = {
            "enabled": True,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "query_tokens": self.query_tokens,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "sequence_names": list(SEQUENCE_NAMES),
            "branch_names": list(self.branch_names),
            "replaces_final_projectors": True,
            "fusion_mode": self.fusion_mode,
        }
        if self.fusion_mode == "all_sequences":
            config["logical_images_per_patient"] = 1
        else:
            config["logical_images_per_plane"] = 1
        return config

    @classmethod
    def from_config_dict(cls, config: Dict[str, object]) -> "MultiSequenceMRIFusion":
        fusion_mode = config.get("fusion_mode", "by_plane")
        if fusion_mode == "all_sequences":
            if config.get("logical_images_per_patient") != 1:
                raise ValueError("All-sequence fusion requires one logical image per patient.")
        elif config.get("logical_images_per_plane") != 1:
            raise ValueError("Plane fusion requires one logical image per plane.")
        return cls(
            input_size=int(config.get("input_size", 4096)),
            hidden_size=int(config.get("hidden_size", 512)),
            output_size=int(config.get("output_size", 4096)),
            query_tokens=int(config.get("query_tokens", 256)),
            num_heads=int(config.get("num_heads", 8)),
            dropout=float(config.get("dropout", 0.1)),
            branch_names=tuple(config.get("branch_names", BRANCH_NAMES)),
            fusion_mode=str(fusion_mode),
        )

    def _pack_sequence_slots(
        self,
        flat_features: torch.Tensor,
        sequence_token_counts: torch.Tensor,
        sequence_type_ids: torch.Tensor,
        sequence_owner_ids: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        counts = [int(value) for value in sequence_token_counts.detach().cpu().tolist()]
        types = [int(value) for value in sequence_type_ids.detach().cpu().tolist()]
        owners = [int(value) for value in sequence_owner_ids.detach().cpu().tolist()]
        if not (len(counts) == len(types) == len(owners)):
            raise ValueError("Sequence metadata tensors must have the same length.")
        if sum(counts) != int(flat_features.shape[0]):
            raise ValueError(
                "Sequence token counts do not cover the visual features: "
                f"counts={sum(counts)}, features={flat_features.shape[0]}."
            )
        if any(count != self.query_tokens for count in counts):
            raise ValueError(
                f"Every MRI sequence must contain {self.query_tokens} visual tokens; "
                f"got {counts}."
            )

        slots = flat_features.new_zeros(
            batch_size,
            len(SEQUENCE_NAMES),
            self.query_tokens,
            self.input_size,
        )
        presence = torch.zeros(
            batch_size,
            len(SEQUENCE_NAMES),
            dtype=torch.bool,
            device=flat_features.device,
        )
        offset = 0
        for count, sequence_type, owner in zip(counts, types, owners):
            if not 0 <= owner < batch_size:
                raise ValueError(f"Invalid sequence owner {owner} for batch size {batch_size}.")
            if not 0 <= sequence_type < len(SEQUENCE_NAMES):
                raise ValueError(f"Invalid MRI sequence type id: {sequence_type}.")
            if bool(presence[owner, sequence_type]):
                raise ValueError(
                    f"Patient {owner} contains duplicate {SEQUENCE_NAMES[sequence_type]} sequences."
                )
            slots[owner, sequence_type] = flat_features[offset : offset + count]
            presence[owner, sequence_type] = True
            offset += count
        return slots, presence

    @staticmethod
    def _plane_memory(
        projected_slots: torch.Tensor,
        presence: torch.Tensor,
        sequence_ids: Iterable[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = tuple(sequence_ids)
        selected = projected_slots[:, ids]
        selected_presence = presence[:, ids]
        batch_size, sequence_count, tokens, hidden_size = selected.shape
        memory = selected.reshape(batch_size, sequence_count * tokens, hidden_size)
        valid_tokens = selected_presence.unsqueeze(-1).expand(-1, -1, tokens)
        key_padding_mask = ~valid_tokens.reshape(batch_size, sequence_count * tokens)
        plane_present = selected_presence.any(dim=1)
        return memory, key_padding_mask, plane_present

    @staticmethod
    def _resample_present_planes(
        module: PlaneResampler,
        memory: torch.Tensor,
        mask: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        output = memory.new_zeros(
            memory.shape[0], module.query_tokens, memory.shape[-1]
        )
        if bool(present.any()):
            indices = present.nonzero(as_tuple=False).flatten()
            output[indices] = module(
                memory.index_select(0, indices),
                mask.index_select(0, indices),
            )
        return output

    def _fuse_branch(
        self,
        branch_name: str,
        flat_features: torch.Tensor,
        sequence_token_counts: torch.Tensor,
        sequence_type_ids: torch.Tensor,
        sequence_owner_ids: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        branch = self.branches[branch_name]
        slots, presence = self._pack_sequence_slots(
            flat_features,
            sequence_token_counts,
            sequence_type_ids,
            sequence_owner_ids,
            batch_size,
        )
        slots = branch.down_projection(branch.input_norm(slots))
        sequence_ids = torch.arange(len(SEQUENCE_NAMES), device=slots.device)
        slots = slots + self.sequence_embedding(sequence_ids)[None, :, None, :]

        if self.fusion_mode == "all_sequences":
            memory, mask, present = self._plane_memory(
                slots, presence, range(len(SEQUENCE_NAMES))
            )
            if not bool(present.all()):
                raise ValueError("Every patient needs at least one MRI sequence.")
            fused = branch.up_projection(branch.all_sequences(memory, mask))
            return fused.flatten(0, 1), present.to(dtype=torch.long)

        axial_memory, axial_mask, axial_present = self._plane_memory(
            slots, presence, AXIAL_SEQUENCE_IDS
        )
        sagittal_memory, sagittal_mask, sagittal_present = self._plane_memory(
            slots, presence, SAGITTAL_SEQUENCE_IDS
        )
        group_counts = axial_present.to(dtype=torch.long) + sagittal_present.to(
            dtype=torch.long
        )
        if bool((group_counts == 0).any()):
            missing = (group_counts == 0).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Every patient needs at least one MRI plane; missing for {missing}.")

        axial = self._resample_present_planes(
            branch.axial, axial_memory, axial_mask, axial_present
        )
        sagittal = self._resample_present_planes(
            branch.sagittal, sagittal_memory, sagittal_mask, sagittal_present
        )

        ordered_groups: List[torch.Tensor] = []
        for patient_index in range(batch_size):
            if bool(axial_present[patient_index]):
                ordered_groups.append(axial[patient_index])
            if bool(sagittal_present[patient_index]):
                ordered_groups.append(sagittal[patient_index])
        fused = branch.up_projection(torch.stack(ordered_groups, dim=0))
        return fused.flatten(0, 1), group_counts

    def forward(
        self,
        main_features: torch.Tensor,
        deepstack_features: Sequence[torch.Tensor],
        sequence_token_counts: torch.Tensor,
        sequence_type_ids: torch.Tensor,
        sequence_owner_ids: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        features = [main_features, *deepstack_features]
        if len(features) != len(self.branch_names):
            raise ValueError(
                f"Expected {len(self.branch_names)} visual branches, got {len(features)}."
            )

        fused_branches: List[torch.Tensor] = []
        expected_group_counts = None
        for branch_name, branch_features in zip(self.branch_names, features):
            fused, group_counts = self._fuse_branch(
                branch_name,
                branch_features,
                sequence_token_counts,
                sequence_type_ids,
                sequence_owner_ids,
                int(batch_size),
            )
            if expected_group_counts is None:
                expected_group_counts = group_counts
            elif not torch.equal(expected_group_counts, group_counts):
                raise RuntimeError("Main and DeepStack plane-group layouts diverged.")
            fused_branches.append(fused)
        return fused_branches[0], fused_branches[1:], expected_group_counts


def install_multi_sequence_fusion(visual: nn.Module, config: Dict[str, object]) -> nn.Module:
    """Install fusion in the visual wrapper and remove replaced projectors."""
    if not bool(config.get("enabled", True)):
        return visual
    existing = getattr(visual, "multi_sequence_fusion", None)
    if existing is None:
        module = MultiSequenceMRIFusion.from_config_dict(config)
        reference = next(visual.parameters())
        module.to(device=reference.device, dtype=reference.dtype)
        visual.multi_sequence_fusion = module

    # The branch-specific 512->4096 projections above replace these A2
    # 4096->4096 projectors.  Delete them from the module tree so they are not
    # saved, moved between devices, or counted as part of the multi model.
    for attribute in ("fusion_projector", "deepstack_fusion_projector"):
        if hasattr(visual, attribute):
            delattr(visual, attribute)
    return visual
