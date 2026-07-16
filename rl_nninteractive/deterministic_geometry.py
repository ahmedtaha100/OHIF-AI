"""Deterministic geometry builders for component-selected tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import ndimage

from .adapter import Box3D, VoxelCoord

GeometryTool = Literal["point", "scribble", "lasso", "box"]
Polarity = Literal["positive", "negative"]


@dataclass(frozen=True)
class ComponentGeometry:
    tool: GeometryTool
    polarity: Polarity
    coord: VoxelCoord
    component_size: int
    component_label: int
    bbox: Box3D
    slice_axis: int
    slice_index: int
    scribble: np.ndarray
    lasso: np.ndarray

    def to_json_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "polarity": self.polarity,
            "coord_zyx": list(self.coord),
            "component_size": self.component_size,
            "component_label": self.component_label,
            "bbox": [[start, stop] for start, stop in self.bbox],
            "slice_axis": self.slice_axis,
            "slice_index": self.slice_index,
            "scribble_voxel_count": int(self.scribble.sum()),
            "lasso_voxel_count": int(self.lasso.sum()),
        }


def build_component_geometry(
    component_mask: Any,
    *,
    tool: GeometryTool,
    polarity: Polarity = "positive",
) -> ComponentGeometry:
    """Convert one 3D component mask to point/scribble/lasso/box geometry."""

    mask = _as_component_mask(component_mask)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if count != 1:
        raise ValueError("component_mask must contain exactly one connected component")
    coords = np.argwhere(labels == 1)
    coord = _representative_coord(coords)
    bbox = _bbox(coords)
    slice_axis, slice_index = _representative_slice(coords)
    slice_mask = _slice_mask(mask, slice_axis=slice_axis, slice_index=slice_index)
    scribble = _central_scribble(slice_mask)
    lasso = slice_mask.astype(np.uint8, copy=True)
    return ComponentGeometry(
        tool=tool,
        polarity=polarity,
        coord=coord,
        component_size=int(coords.shape[0]),
        component_label=1,
        bbox=bbox,
        slice_axis=slice_axis,
        slice_index=slice_index,
        scribble=scribble,
        lasso=lasso,
    )


def largest_error_component_mask(
    current_mask: Any,
    ground_truth: Any,
    *,
    polarity: Polarity,
) -> np.ndarray:
    """Return the largest FN/FP component for deterministic geometry tests."""

    current = _as_binary_volume(current_mask, "current_mask")
    target = _as_binary_volume(ground_truth, "ground_truth")
    if current.shape != target.shape:
        raise ValueError(f"mask shapes differ: {current.shape} != {target.shape}")
    error = np.logical_and(target, ~current) if polarity == "positive" else np.logical_and(current, ~target)
    labels, count = ndimage.label(error, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(error, dtype=bool)
    counts = np.bincount(labels.ravel(), minlength=count + 1)
    counts[0] = 0
    label_id = int(np.flatnonzero(counts == counts.max())[0])
    return labels == label_id


def _as_component_mask(mask: Any) -> np.ndarray:
    array = _as_binary_volume(mask, "component_mask")
    if not bool(array.any()):
        raise ValueError("component_mask must not be empty")
    return array


def _as_binary_volume(mask: Any, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D volume")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype == np.bool_:
        return array.astype(bool, copy=True)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be boolean or binary numeric")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains non-finite values")
    if not bool(np.isin(array, (0, 1)).all()):
        raise ValueError(f"{name} numeric values must be in {{0, 1}}")
    return array.astype(bool, copy=True)


def _representative_coord(coords: np.ndarray) -> VoxelCoord:
    centroid = coords.mean(axis=0)
    distances = np.square(coords - centroid).sum(axis=1)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], distances))
    chosen = coords[int(order[0])]
    return (int(chosen[0]), int(chosen[1]), int(chosen[2]))


def _bbox(coords: np.ndarray) -> Box3D:
    starts = coords.min(axis=0)
    stops = coords.max(axis=0) + 1
    return tuple((int(starts[axis]), int(stops[axis])) for axis in range(3))  # type: ignore[return-value]


def _representative_slice(coords: np.ndarray) -> tuple[int, int]:
    spans = coords.max(axis=0) - coords.min(axis=0)
    axis = int(np.argmin(spans))
    values, counts = np.unique(coords[:, axis], return_counts=True)
    max_count = counts.max()
    return axis, int(values[np.flatnonzero(counts == max_count)[0]])


def _slice_mask(mask: np.ndarray, *, slice_axis: int, slice_index: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    selector: list[slice | int] = [slice(None), slice(None), slice(None)]
    selector[slice_axis] = slice_index
    result[tuple(selector)] = mask[tuple(selector)]
    return result


def _central_scribble(slice_mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(slice_mask)
    result = np.zeros_like(slice_mask, dtype=np.uint8)
    if coords.size == 0:
        return result
    centroid = coords.mean(axis=0)
    distances = np.square(coords - centroid).sum(axis=1)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], distances))
    chosen = coords[order[: max(1, min(3, coords.shape[0]))]]
    result[tuple(chosen.T)] = 1
    return result
