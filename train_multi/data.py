"""Multi-sequence cervical MRI report dataset and training collator."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from custom_model.medical_image_preprocessing import (
    preprocess_mri_image,
    sample_slices_from_volume_to_pil_fast,
)
from custom_model.multi_sequence_fusion import (
    AXIAL_SEQUENCE_IDS,
    SAGITTAL_SEQUENCE_IDS,
    SEQUENCE_NAMES,
)


SEQUENCE_TO_ID = {name: index for index, name in enumerate(SEQUENCE_NAMES)}


def sequence_name_from_path(path: str) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    sequence_name = name.rsplit("-", 1)[-1].upper()
    if sequence_name not in SEQUENCE_TO_ID:
        raise ValueError(
            f"Cannot infer MRI sequence type from {path!r}; expected one of {SEQUENCE_NAMES}."
        )
    return sequence_name


class MultiSequenceMRIDataset(Dataset):
    """Read one patient, one to five MRI sequences, and one report per record."""

    def __init__(self, jsonl_path: str, require_report: bool = True):
        self.path = Path(jsonl_path)
        self.require_report = require_report
        if not self.path.is_file():
            raise FileNotFoundError(f"Training JSONL not found: {self.path}")
        self.records: List[Dict[str, object]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.path}:{line_number}") from exc
                self.records.append(self._normalize(record, line_number))
        if not self.records:
            raise ValueError(f"No records found in {self.path}")

    def _resolve_path(self, raw_path: str, line_number: int) -> str:
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            relative_to_jsonl = self.path.parent / image_path
            image_path = relative_to_jsonl if relative_to_jsonl.exists() else image_path
        if not image_path.is_file():
            raise FileNotFoundError(
                f"MRI file at {self.path}:{line_number} not found: {image_path}"
            )
        return str(image_path.resolve())

    def _normalize(self, record: Dict[str, object], line_number: int) -> Dict[str, object]:
        messages = record.get("messages", {})
        if not isinstance(messages, dict):
            raise ValueError(f"{self.path}:{line_number} has a non-object 'messages' field.")
        nifti = messages.get("nifti")
        if not isinstance(nifti, list) or not nifti:
            raise ValueError(f"{self.path}:{line_number} requires a non-empty nifti list.")
        if len(nifti) > len(SEQUENCE_NAMES):
            raise ValueError(f"{self.path}:{line_number} has more than five MRI sequences.")

        by_type: Dict[int, str] = {}
        for raw_path in nifti:
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"{self.path}:{line_number} contains an invalid NIfTI path.")
            sequence_name = sequence_name_from_path(raw_path)
            sequence_id = SEQUENCE_TO_ID[sequence_name]
            if sequence_id in by_type:
                raise ValueError(
                    f"{self.path}:{line_number} contains duplicate {sequence_name} sequences."
                )
            by_type[sequence_id] = self._resolve_path(raw_path, line_number)

        report = record.get("GT")
        prompt = messages.get("prompt")
        if self.require_report and (not isinstance(report, str) or not report.strip()):
            raise ValueError(f"{self.path}:{line_number} is missing a non-empty report/GT.")
        if report is not None and not isinstance(report, str):
            raise ValueError(f"{self.path}:{line_number} has a non-string report/GT.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{self.path}:{line_number} has an invalid prompt.")

        ordered_ids = sorted(by_type)
        return {
            "nifti": [by_type[index] for index in ordered_ids],
            "sequence_type_ids": ordered_ids,
            "has_axial": any(index in by_type for index in AXIAL_SEQUENCE_IDS),
            "has_sagittal": any(index in by_type for index in SAGITTAL_SEQUENCE_IDS),
            "prompt": prompt.strip(),
            "report": report.strip() if report is not None else "",
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.records[index]


@dataclass
class _PreparedBatch:
    """Intermediate patient data collected before processor calls."""

    texts: List[str]
    logical_image_batches: List[List[Image.Image]]
    encoder_image_batches: List[List[Image.Image]]
    volume_batches: List[List[torch.Tensor]]
    sequence_type_ids: List[int]
    sequence_owner_ids: List[int]
    plane_counts: List[int]


class MultiSequenceMRITrainingCollator:
    """Build separate real encoder inputs and fused-plane LLM placeholders."""

    def __init__(
        self,
        processor,
        volume_depth: int,
        image_size: int,
        sample_images: int,
        query_tokens: int = 256,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
    ) -> None:
        self.processor = processor
        self.volume_depth = int(volume_depth)
        self.image_size = int(image_size)
        self.sample_images = int(sample_images)
        self.query_tokens = int(query_tokens)
        if self.query_tokens != 256:
            raise ValueError("Each fused plane must contain exactly 256 query tokens.")
        self.percentiles = (float(lower_percentile), float(upper_percentile))
        self.processor.tokenizer.padding_side = "right"
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        # Logical placeholders only need Qwen's grid/token expansion. Keep a
        # shallow processor copy so they skip DINOv2 and ConvNeXt preprocessing.
        self.logical_processor = copy.copy(self.processor)
        self.logical_processor.image_processor = getattr(
            self.processor.image_processor,
            "original_processor",
            self.processor.image_processor,
        )
        self.assistant_marker = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        if not self.assistant_marker:
            raise ValueError("Could not tokenize the assistant response marker.")
        # One logical image per fused plane. Its merged grid must have exactly
        # query_tokens positions; this size is independent of real MRI slices.
        image_processor = self.logical_processor.image_processor
        grid_side = 16 * int(image_processor.merge_size)
        self.logical_grid_shape = (1, grid_side, grid_side)
        placeholder_side = grid_side * int(image_processor.patch_size)
        self.placeholder_image = Image.new("RGB", (placeholder_side, placeholder_side), color=0)

    def _load_volume(self, path: str) -> torch.Tensor:
        return preprocess_mri_image(
            path,
            target_size=(self.volume_depth, self.image_size, self.image_size),
            lower_percentile=self.percentiles[0],
            upper_percentile=self.percentiles[1],
        ).float()

    @staticmethod
    def _last_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int:
        for start in range(len(sequence) - len(pattern), -1, -1):
            if list(sequence[start : start + len(pattern)]) == list(pattern):
                return start
        return -1

    def _logical_content(self, record: Dict[str, object]) -> List[Dict[str, str]]:
        content: List[Dict[str, str]] = []
        if bool(record["has_axial"]):
            content.append({"type": "text", "text": "Axial MRI:\n"})
            content.append({"type": "image"})
        if bool(record["has_sagittal"]):
            content.append({"type": "text", "text": "Sagittal MRI:\n"})
            content.append({"type": "image"})
        if not content:
            raise ValueError("Every patient must contain at least one MRI plane.")
        content.append({"type": "text", "text": str(record["prompt"])})
        return content

    @staticmethod
    def _plane_count(record: Dict[str, object]) -> int:
        return int(bool(record["has_axial"])) + int(bool(record["has_sagittal"]))

    def _prepare_batch(self, records: List[Dict[str, object]], *, generation: bool = False) -> _PreparedBatch:
        """Load MRI data and collect patient/sequence metadata."""
        if not records:
            raise ValueError("The collator received an empty batch.")

        prepared = _PreparedBatch([], [], [], [], [], [], [])
        for patient_index, record in enumerate(records):
            encoder_images: List[Image.Image] = []
            patient_volumes: List[torch.Tensor] = []
            paths = list(record["nifti"])
            type_ids = list(record["sequence_type_ids"])
            if len(paths) != len(type_ids):
                raise ValueError("NIfTI paths and sequence type IDs must have equal length.")

            for path, sequence_type in zip(paths, type_ids):
                volume = self._load_volume(str(path))
                patient_volumes.append(volume)
                encoder_images.extend(
                    sample_slices_from_volume_to_pil_fast(
                        volume,
                        num_slices=self.sample_images,
                        random=False,
                    )
                )
                prepared.sequence_type_ids.append(int(sequence_type))
                prepared.sequence_owner_ids.append(patient_index)

            plane_count = self._plane_count(record)
            prepared.plane_counts.append(plane_count)
            prepared.logical_image_batches.append(
                [self.placeholder_image] * plane_count
            )
            prepared.encoder_image_batches.append(encoder_images)
            prepared.volume_batches.append(patient_volumes)

            messages = [{"role": "user", "content": self._logical_content(record)}]
            if not generation:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(record["report"])}],
                })
            prepared.texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=generation,
                )
            )
        return prepared

    def _build_logical_batch(
        self,
        prepared: _PreparedBatch,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Tokenize reports and reserve LLM positions for fused plane features."""
        batch = dict(
            self.logical_processor(
                text=prepared.texts,
                images=prepared.logical_image_batches,
                padding=True,
                return_tensors="pt",
            )
        )
        logical_grid = batch.pop("image_grid_thw")
        tokenizer = self.processor.tokenizer
        start_id, image_id, end_id = tokenizer.convert_tokens_to_ids(
            ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
        )
        for row, plane_count in zip(batch["input_ids"], prepared.plane_counts):
            starts = (row == start_id).nonzero(as_tuple=False).flatten().tolist()
            if len(starts) != plane_count or int((row == image_id).sum()) != plane_count * self.query_tokens:
                raise ValueError("Each patient must have one 256-token visual interval per present plane.")
            for start in starts:
                end = start + 1 + self.query_tokens
                if end >= row.numel() or row[end] != end_id or not bool((row[start + 1:end] == image_id).all()):
                    raise ValueError("A fused plane's 256 image tokens must be contiguous within one visual interval.")
        for key in list(batch):
            if key.startswith("pixel_values"):
                del batch[key]
        return batch, logical_grid

    def _build_encoder_batch(
        self,
        prepared: _PreparedBatch,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Preprocess the real 2D slices and 3D volumes for frozen encoders."""
        encoder_inputs = dict(
            self.processor.image_processor.preprocess(
                prepared.encoder_image_batches,
                return_tensors="pt",
                volume=prepared.volume_batches,
            )
        )
        encoder_grid = encoder_inputs.pop("image_grid_thw")
        return encoder_inputs, encoder_grid

    def _sequence_token_counts(
        self,
        encoder_grid: torch.Tensor,
        sequence_count: int,
        merge_size: int,
    ) -> List[int]:
        """Group each sequence's sampled-image tokens and enforce 256 tokens."""
        per_image_tokens = (encoder_grid.prod(dim=-1) // (merge_size**2)).tolist()
        expected_images = sequence_count * self.sample_images
        if len(per_image_tokens) != expected_images:
            raise ValueError(
                "Encoder grid image count does not match the MRI sequences: "
                f"images={len(per_image_tokens)}, expected={expected_images}."
            )

        counts = [
            int(sum(per_image_tokens[start : start + self.sample_images]))
            for start in range(0, expected_images, self.sample_images)
        ]
        if any(count != self.query_tokens for count in counts):
            raise ValueError(
                f"Expected {self.query_tokens} tokens per MRI sequence, got {counts}."
            )
        return counts

    def _validate_logical_grid(
        self,
        logical_grid: torch.Tensor,
        plane_counts: Sequence[int],
        merge_size: int,
    ) -> None:
        """Require one square, query_tokens-long visual interval per plane."""
        expected_planes = sum(plane_counts)
        if tuple(logical_grid.shape) != (expected_planes, 3):
            raise ValueError(
                "Expected one logical image grid row per fused plane: "
                f"expected {(expected_planes, 3)}, got {tuple(logical_grid.shape)}."
            )
        expected_grid = logical_grid.new_tensor(self.logical_grid_shape)
        if not bool((logical_grid == expected_grid).all()):
            raise ValueError(
                f"Expected every fused-plane grid to be {self.logical_grid_shape}; "
                f"got {logical_grid.tolist()}."
            )
        per_plane_tokens = logical_grid.prod(dim=-1) // (merge_size**2)
        if not bool((per_plane_tokens == self.query_tokens).all()):
            raise ValueError(
                f"Every plane must reserve {self.query_tokens} image tokens; "
                f"got {per_plane_tokens.tolist()}."
            )

    def _build_labels(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mask the prompt, image tokens, assistant marker, and padding."""
        labels = batch["input_ids"].clone()
        for row_index in range(labels.shape[0]):
            valid_length = int(batch["attention_mask"][row_index].sum().item())
            token_ids = labels[row_index, :valid_length].tolist()
            marker_start = self._last_subsequence(token_ids, self.assistant_marker)
            if marker_start < 0:
                raise ValueError("Assistant marker not found after chat-template tokenization.")
            response_start = marker_start + len(self.assistant_marker)
            labels[row_index, :response_start] = -100
            labels[row_index, valid_length:] = -100
        return labels

    def _build_model_inputs(self, prepared: _PreparedBatch) -> Dict[str, torch.Tensor]:
        # 得到texts，logical_image_batches（占位图像），encoder_image_batches（mri切片），volume_batches，sequence_type_ids，sequence_owner_ids，plane_counts

        batch, logical_grid = self._build_logical_batch(prepared)
        # 构建给 LLM 用的文本 tokens 和融合后 每个<image>对应多少视觉位置

        encoder_inputs, encoder_grid = self._build_encoder_batch(prepared)
        # 把真正的 MRI 2D 切片和 3D volume，预处理成 ClinFusion 各视觉 Encoder 可以直接使用的输入

        batch.update(encoder_inputs)
        batch["image_grid_thw"] = logical_grid
        batch["encoder_image_grid_thw"] = encoder_grid

        merge_size = int(self.logical_processor.image_processor.merge_size)
        token_counts = self._sequence_token_counts(
            encoder_grid,
            sequence_count=len(prepared.sequence_type_ids),
            merge_size=merge_size,
        )
        self._validate_logical_grid(logical_grid, prepared.plane_counts, merge_size)

        batch["sequence_token_counts"] = torch.tensor(token_counts, dtype=torch.long)
        batch["sequence_type_ids"] = torch.tensor(
            prepared.sequence_type_ids, dtype=torch.long
        )
        batch["sequence_owner_ids"] = torch.tensor(
            prepared.sequence_owner_ids, dtype=torch.long
        )
        return batch

    def __call__(self, records: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        self.processor.tokenizer.padding_side = "right"
        prepared = self._prepare_batch(records)
        batch = self._build_model_inputs(prepared)
        batch["labels"] = self._build_labels(batch)
        return batch

    def build_generation_batch(self, records: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        """Use the training MRI input path with a user-only generation prompt."""
        self.processor.tokenizer.padding_side = "left"
        prepared = self._prepare_batch(records, generation=True)
        return self._build_model_inputs(prepared)
