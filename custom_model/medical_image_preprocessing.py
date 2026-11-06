"""MRI-only preprocessing used by all three training stages."""

from os import PathLike
from typing import List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

MRI_PLANE_ORDERS = {
    # Closest-canonical NIfTI data has axes (R/L, A/P, S/I).  The encoder
    # expects depth first and image rows before columns.
    "axial": (2, 1, 0),
    "sagittal": (0, 2, 1),
    "coronal": (1, 2, 0),
}


def get_nifti_acquisition_plane(nifti_image) -> str:
    """Infer the native slice plane from the third voxel axis of the affine.

    The result describes how the series was stored/acquired, rather than an
    arbitrary assumption that the third array dimension is always axial.
    Oblique acquisitions are assigned to their closest anatomical plane.
    """
    affine = np.asarray(nifti_image.affine, dtype=np.float64)
    slice_direction = affine[:3, 2]
    if not np.isfinite(slice_direction).all() or np.linalg.norm(slice_direction) < 1e-8:
        return "axial"
    return ("sagittal", "coronal", "axial")[int(np.argmax(np.abs(slice_direction)))]


def normalize_mri_volume(
    volume: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    foreground_threshold: float = 1e-6,
) -> np.ndarray:
    """Robustly normalize an MRI volume to ``[0, 1]`` using its foreground.

    Percentiles are computed once per 3D series, not independently per slice,
    so relative contrast across slices is preserved. NaN/Inf values and empty
    volumes are handled deterministically.
    """
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("MRI percentiles must satisfy 0 <= lower < upper <= 100.")

    data = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(data)
    foreground = finite & (np.abs(data) > foreground_threshold)
    values = data[foreground]
    if values.size < 16:
        values = data[finite]
    if values.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    low, high = np.percentile(values, (lower_percentile, upper_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(data, dtype=np.float32)

    normalized = np.nan_to_num(data, nan=low, posinf=high, neginf=low)
    normalized = np.clip(normalized, low, high)
    normalized = (normalized - low) / (high - low)
    # Preserve a clean background when the source volume uses zero padding.
    normalized[finite & (np.abs(data) <= foreground_threshold)] = 0.0
    return normalized.astype(np.float32, copy=False)


def _crop_invalid_background_3d(
    volume_dhw: np.ndarray,
    min_corner_votes: int = 6,
) -> np.ndarray:
    """Crop constant background only when at least six corners agree.

    The agreed background value may be non-zero (common in some ADC series).
    Background voxels are changed to zero before cropping so they are excluded
    from the later MRI intensity normalization. If fewer than
    ``min_corner_votes`` corners agree, the volume is returned without any
    background cropping.
    """
    data = np.asarray(volume_dhw, dtype=np.float32)
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError(f"Expected a non-empty 3D MRI volume, got shape {data.shape}.")
    if not 1 <= int(min_corner_votes) <= 8:
        raise ValueError("min_corner_votes must be between 1 and 8.")

    corners = np.asarray(
        [
            data[d, h, w]
            for d in (0, -1)
            for h in (0, -1)
            for w in (0, -1)
        ],
        dtype=np.float32,
    )
    finite_corners = corners[np.isfinite(corners)]
    if finite_corners.size < min_corner_votes:
        return np.ascontiguousarray(data)

    values, counts = np.unique(finite_corners, return_counts=True)
    best_index = int(np.argmax(counts))
    if int(counts[best_index]) < min_corner_votes:
        return np.ascontiguousarray(data)

    background_value = values[best_index]
    foreground = np.isfinite(data)
    foreground &= data != background_value
    if not foreground.any():
        return np.zeros_like(data, dtype=np.float32)

    occupied_axes = (
        foreground.any(axis=(1, 2)),
        foreground.any(axis=(0, 2)),
        foreground.any(axis=(0, 1)),
    )
    slices = []
    for occupied in occupied_axes:
        indices = np.flatnonzero(occupied)
        slices.append(slice(int(indices[0]), int(indices[-1]) + 1))

    cropped = np.array(data[tuple(slices)], dtype=np.float32, copy=True, order="C")
    cropped[cropped == background_value] = 0.0
    return cropped


def _center_crop_or_pad_3d(
    tensor: torch.Tensor,
    target_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Center-crop and symmetrically zero-pad ``(N,C,D,H,W)`` data."""
    slices = []
    for current, target in zip(tensor.shape[-3:], target_size):
        start = max((current - target) // 2, 0)
        slices.append(slice(start, start + min(current, target)))
    tensor = tensor[..., slices[0], slices[1], slices[2]]

    pads = []
    for current, target in reversed(list(zip(tensor.shape[-3:], target_size))):
        total = max(target - current, 0)
        pads.extend((total // 2, total - total // 2))
    return F.pad(tensor, pads, mode="constant", value=0.0)


def _resize_or_center_crop_depth(
    tensor: torch.Tensor,
    target_depth: int,
) -> torch.Tensor:
    """Upsample short volumes or center-crop long volumes along depth only."""
    current_depth = int(tensor.shape[-3])
    if current_depth < target_depth:
        return F.interpolate(
            tensor,
            size=(target_depth, int(tensor.shape[-2]), int(tensor.shape[-1])),
            mode="trilinear",
            align_corners=False,
        )
    if current_depth > target_depth:
        start = (current_depth - target_depth) // 2
        return tensor[..., start:start + target_depth, :, :]
    return tensor


def preprocess_mri_image(
    path_or_object,
    target_size: Tuple[int, int, int] = (64, 512, 512),
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> torch.Tensor:
    """Preprocess one MRI NIfTI into a ``(1, D, H, W)`` float tensor.

    The affine determines the acquisition plane, orientation is standardized to
    closest RAS, and axes are reordered so the acquired slice axis is depth.
    Empty outer margins are removed. Short volumes are resized along depth,
    while long volumes are center-cropped; the in-plane dimensions are then
    resized independently. Robust foreground percentile normalization to
    ``[0, 1]`` is applied last so the 2D and 3D encoder branches receive a
    consistent intensity representation.
    """
    if nib is None:
        raise ImportError("nibabel is required for MRI preprocessing.")
    if len(target_size) != 3 or any(int(size) <= 0 for size in target_size):
        raise ValueError(f"target_size must contain three positive values, got {target_size}.")

    if isinstance(path_or_object, (str, bytes, PathLike)):
        nifti_image = nib.load(path_or_object)
    elif isinstance(path_or_object, nib.spatialimages.SpatialImage):
        nifti_image = path_or_object
    else:
        raise TypeError(f"Unsupported MRI input type: {type(path_or_object)}")

    if len(nifti_image.shape) != 3:
        raise ValueError(f"Expected a 3D MRI NIfTI, got shape {nifti_image.shape}.")

    plane = get_nifti_acquisition_plane(nifti_image)
    canonical = nib.as_closest_canonical(nifti_image)
    data_xyz = canonical.get_fdata(dtype=np.float32)
    order = MRI_PLANE_ORDERS[plane]
    data_dhw = np.transpose(data_xyz, order)

    # Follow the MRI-specific geometry pipeline before intensity normalization.
    data_dhw = _crop_invalid_background_3d(data_dhw)
    tensor = torch.from_numpy(data_dhw).unsqueeze(0).unsqueeze(0)
    tensor = _resize_or_center_crop_depth(tensor, int(target_size[0]))
    # Treat depth slices as a batch so interpolation occurs only within H/W.
    tensor = tensor.squeeze(0).squeeze(0).unsqueeze(1)
    tensor = F.interpolate(
        tensor,
        size=(int(target_size[1]), int(target_size[2])),
        mode="bilinear",
        align_corners=False,
    )
    data_dhw = tensor.squeeze(1).cpu().numpy()
    data_dhw = normalize_mri_volume(
        data_dhw,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    return torch.from_numpy(data_dhw).unsqueeze(0).contiguous()

def sample_slices_from_volume_to_pil_fast(
    volume_tensor: torch.Tensor,
    num_slices: int = 4,
    random: bool = True,
    seed: Optional[int] = None,
    per_slice_minmax: bool = False,
) -> Union[List[Image.Image], List[List[Image.Image]]]:
    """
    Samples slices from a 3D volume and returns them as PIL Images (Optimized, Vectorized Version).

    Args:
        volume_tensor: 4D (C, D, H, W) or 5D (B, C, D, H, W) tensor.
        num_slices: Number of slices to sample.
        random: If True, randomly sample; otherwise, uniformly sample across depth.
        seed: Optional RNG seed for reproducible random sampling.
        per_slice_minmax: If True, min-max normalize each slice before converting.
                          If False, assume values are in [0, 1] and clip.

    Returns:
        - If input is 4D: List[PIL.Image.Image] of length num_slices.
        - If input is 5D: List[List[PIL.Image.Image]] with shape (B, num_slices).
    """
    # --- 1. Input Validation and Unified Path ---
    v = volume_tensor.float()  # Ensure float for calculations
    was_4d = False
    if v.ndim == 4:
        # (C, D, H, W) -> (1, C, D, H, W)
        v = v.unsqueeze(0)
        was_4d = True
    elif v.ndim != 5:
        raise ValueError(f"Expected 4D (C,D,H,W) or 5D (B,C,D,H,W) tensor, got shape {volume_tensor.shape}")

    B, C, D, H, W = v.shape
    device = v.device

    # --- 2. Generate Slice Indices ---
    if seed is not None and random:
        g = torch.Generator(device=device).manual_seed(seed)
    else:
        g = None

    if D <= num_slices:
        indices = torch.arange(D, device=device)
        if D < num_slices: # Pad by repeating the last slice index
            pad = indices[-1].repeat(num_slices - D)
            indices = torch.cat([indices, pad])
    else:
        if random:
            perm = torch.randperm(D, generator=g, device=device)
            indices = perm[:num_slices].sort().values
        else: # Uniformly spaced indices
            indices = torch.linspace(0, D - 1, steps=num_slices, device=device).round().long()

    # --- 3. Vectorized Slice Extraction ---
    # v is (B, C, D, H, W), indices is (S,)
    # We select along dimension 2 (Depth)
    slices = torch.index_select(v, dim=2, index=indices)  # (B, C, S, H, W)
    slices = slices.permute(0, 2, 1, 3, 4)  # (B, S, C, H, W) for easier processing

    # --- 4. Vectorized Channel and Normalization Logic ---
    # Reshape for easier per-slice operations
    slices = slices.reshape(B * num_slices, C, H, W)

    if C > 3 or C == 2: # Reduce to grayscale if not 1 or 3 channels
        slices = slices.mean(dim=1, keepdim=True) # (B*S, 1, H, W)
        C = 1
        mode = "L"
    elif C == 1:
        mode = "L"
    else: # C == 3
        mode = "RGB"

    if per_slice_minmax:
        # Vectorized min-max normalization per slice
        slice_min = slices.amin(dim=(-2, -1), keepdim=True)
        slice_max = slices.amax(dim=(-2, -1), keepdim=True)
        scale = slice_max - slice_min
        # Use torch.where to avoid division by zero
        slices = torch.where(scale > 1e-6, (slices - slice_min) / scale, torch.zeros_like(slices))
    else:
        slices = slices.clamp(0.0, 1.0)

    # --- 5. Vectorized Type Conversion and CPU Transfer ---
    # Scale to [0, 255] and convert to uint8 in one go
    slices = (slices * 255.0).round().to(torch.uint8)

    # Permute channels to the end for PIL (H, W, C) for RGB
    # And remove channel dim for grayscale (H, W)
    if mode == "RGB":
        slices = slices.permute(0, 2, 3, 1) # (B*S, H, W, 3)
    else: # mode == "L"
        slices = slices.squeeze(1) # (B*S, H, W)

    # Single, bulk transfer to CPU and convert to NumPy
    np_slices = slices.cpu().numpy()
    np_slices = np_slices.reshape(B, num_slices, *np_slices.shape[1:]) # (B, S, H, W, [C])

    # --- 6. Final (Unavoidable) Loop to Create PIL Images ---
    pil_batches = [
        [Image.fromarray(np_slices[b, s], mode=mode) for s in range(num_slices)]
        for b in range(B)
    ]

    # --- 7. Return in Original Format ---
    if was_4d:
        return pil_batches[0]
    else:
        return pil_batches
