"""Physical-space NIfTI alignment for multimodal scientific runs.

Array-shape resizing is not a valid registration operation.  This module makes
the spatial contract explicit: every source is canonicalized, checked for a
shared physical field of view, and resampled onto one reference grid.  Arrays
leave the module in the package's native ``(Z, Y, X)`` convention while all
affines and spacing metadata remain in NIfTI ``(X, Y, Z)`` convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeometryMetadata:
    source_path: str
    reference_path: str
    source_shape_xyz: tuple[int, int, int]
    output_shape_xyz: tuple[int, int, int]
    source_affine_xyz: tuple[tuple[float, ...], ...]
    output_affine_xyz: tuple[tuple[float, ...], ...]
    source_orientation: str
    output_orientation: str
    source_spacing_xyz: tuple[float, float, float]
    output_spacing_xyz: tuple[float, float, float]
    physical_overlap_fraction: float
    center_distance_mm: float
    interpolation_order: int
    transform_history: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignedNiftiVolume:
    data_zyx: np.ndarray
    geometry: GeometryMetadata


def load_nifti_on_reference_grid(
    path: str | Path,
    *,
    reference_path: str | Path,
    target_shape_zyx: tuple[int, int, int] | None = None,
    is_label: bool = False,
    minimum_overlap_fraction: float = 0.5,
    channel_index: int | None = None,
    reference_channel_index: int | None = None,
) -> AlignedNiftiVolume:
    """Load ``path`` on the canonical physical grid of ``reference_path``.

    Labels use nearest-neighbor interpolation.  Images use trilinear
    interpolation.  A non-overlapping source/reference pair is rejected before
    resampling so a plausible-looking array cannot hide a geometry mismatch.
    """

    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except Exception as exc:  # pragma: no cover - optional real-data dependency
        raise ImportError("physical-space NIfTI alignment requires nibabel") from exc

    source_path = Path(path).resolve()
    reference_path = Path(reference_path).resolve()
    if not 0.0 <= minimum_overlap_fraction <= 1.0:
        raise ValueError("minimum_overlap_fraction must be between 0 and 1")
    source_original = _select_spatial_volume(
        nib.load(str(source_path)), source_path, channel_index=channel_index, nib=nib
    )
    reference_original = _select_spatial_volume(
        nib.load(str(reference_path)),
        reference_path,
        channel_index=reference_channel_index,
        nib=nib,
    )
    _require_3d(source_original, source_path)
    _require_3d(reference_original, reference_path)

    source = nib.as_closest_canonical(source_original)
    reference = nib.as_closest_canonical(reference_original)
    overlap, center_distance = physical_alignment_report(source, reference)
    if overlap < minimum_overlap_fraction:
        raise ValueError(
            f"physical fields of view do not overlap: source={source_path}, "
            f"reference={reference_path}, overlap={overlap:.6f}"
        )

    if target_shape_zyx is None:
        output_shape_xyz = tuple(int(v) for v in reference.shape)
        output_affine = np.asarray(reference.affine, dtype=np.float64)
    else:
        output_shape_xyz = _xyz_shape(target_shape_zyx)
        output_affine = affine_for_output_shape(
            np.asarray(reference.affine, dtype=np.float64),
            tuple(int(v) for v in reference.shape),
            output_shape_xyz,
        )

    interpolation_order = 0 if is_label else 1
    aligned = resample_from_to(
        source,
        (output_shape_xyz, output_affine),
        order=interpolation_order,
        mode="constant",
        cval=0.0,
    )
    data_xyz = np.asarray(aligned.dataobj, dtype=np.float32)
    if not bool(np.isfinite(data_xyz).all()):
        raise ValueError(f"resampled volume contains non-finite values: {source_path}")
    if is_label:
        data_xyz = np.rint(data_xyz).astype(np.float32, copy=False)

    metadata = GeometryMetadata(
        source_path=str(source_path),
        reference_path=str(reference_path),
        source_shape_xyz=tuple(int(v) for v in source_original.shape),
        output_shape_xyz=output_shape_xyz,
        source_affine_xyz=_matrix_tuple(source_original.affine),
        output_affine_xyz=_matrix_tuple(output_affine),
        source_orientation="".join(str(v) for v in nib.aff2axcodes(source_original.affine)),
        output_orientation="".join(str(v) for v in nib.aff2axcodes(output_affine)),
        source_spacing_xyz=tuple(float(v) for v in source_original.header.get_zooms()[:3]),
        output_spacing_xyz=tuple(float(v) for v in nib.affines.voxel_sizes(output_affine)),
        physical_overlap_fraction=float(overlap),
        center_distance_mm=float(center_distance),
        interpolation_order=interpolation_order,
        transform_history=(
            "load:nifti",
            "canonicalize:RAS",
            f"resample:reference={reference_path.name}:order={interpolation_order}",
        ),
    )
    return AlignedNiftiVolume(
        data_zyx=np.transpose(data_xyz, (2, 1, 0)).copy(),
        geometry=metadata,
    )


def affine_for_output_shape(
    reference_affine: np.ndarray,
    reference_shape_xyz: tuple[int, int, int],
    output_shape_xyz: tuple[int, int, int],
) -> np.ndarray:
    """Preserve reference-grid corner voxel centers for a new array shape."""

    if any(int(v) <= 1 for v in output_shape_xyz):
        raise ValueError("output shape values must be greater than 1")
    scale = []
    for source_size, output_size in zip(reference_shape_xyz, output_shape_xyz):
        if source_size <= 1:
            raise ValueError("reference shape values must be greater than 1")
        scale.append((source_size - 1) / (output_size - 1))
    voxel_transform = np.eye(4, dtype=np.float64)
    voxel_transform[0, 0], voxel_transform[1, 1], voxel_transform[2, 2] = scale
    return np.asarray(reference_affine, dtype=np.float64) @ voxel_transform


def physical_alignment_report(source: Any, reference: Any) -> tuple[float, float]:
    """Return FOV overlap/min-volume and world-center distance in millimeters."""

    source_min, source_max, source_center = _world_bounds(source)
    reference_min, reference_max, reference_center = _world_bounds(reference)
    overlap_extent = np.maximum(
        0.0,
        np.minimum(source_max, reference_max) - np.maximum(source_min, reference_min),
    )
    overlap_volume = float(np.prod(overlap_extent))
    source_volume = float(np.prod(np.maximum(source_max - source_min, 0.0)))
    reference_volume = float(np.prod(np.maximum(reference_max - reference_min, 0.0)))
    denominator = min(source_volume, reference_volume)
    overlap_fraction = 1.0 if denominator == 0.0 and overlap_volume == 0.0 else overlap_volume / denominator
    center_distance = float(np.linalg.norm(source_center - reference_center))
    return overlap_fraction, center_distance


def _world_bounds(image: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = tuple(int(v) for v in image.shape[:3])
    corners = np.array(
        [
            (x, y, z)
            for x in (0, shape[0] - 1)
            for y in (0, shape[1] - 1)
            for z in (0, shape[2] - 1)
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
    world = (np.asarray(image.affine, dtype=np.float64) @ homogeneous.T).T[:, :3]
    center_voxel = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    center_world = (np.asarray(image.affine) @ np.append(center_voxel, 1.0))[:3]
    return world.min(axis=0), world.max(axis=0), center_world


def _require_3d(image: Any, path: Path) -> None:
    if len(image.shape) != 3:
        raise ValueError(f"NIfTI input must be 3D, got {image.shape}: {path}")
    if any(int(value) <= 1 for value in image.shape):
        raise ValueError(f"NIfTI dimensions must all be greater than 1, got {image.shape}: {path}")


def _select_spatial_volume(image: Any, path: Path, *, channel_index: int | None, nib: Any) -> Any:
    if len(image.shape) == 3:
        if channel_index not in (None, 0):
            raise ValueError(f"channel_index is invalid for a 3D NIfTI: {path}")
        return image
    if len(image.shape) != 4 or channel_index is None:
        raise ValueError(
            f"NIfTI input must be 3D or specify channel_index for 4D input, "
            f"got {image.shape}: {path}"
        )
    if channel_index < 0 or channel_index >= image.shape[3]:
        raise ValueError(f"channel_index {channel_index} is outside shape {image.shape}: {path}")
    data = np.asarray(image.dataobj[..., channel_index])
    return nib.Nifti1Image(data, image.affine, header=image.header.copy())


def _xyz_shape(shape_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(shape_zyx) != 3 or any(int(v) <= 1 for v in shape_zyx):
        raise ValueError("target_shape_zyx must contain three integers greater than 1")
    return tuple(int(v) for v in reversed(shape_zyx))


def _matrix_tuple(matrix: Any) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(matrix, dtype=np.float64)
    return tuple(tuple(float(value) for value in row) for row in array)
