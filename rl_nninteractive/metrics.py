"""Segmentation metrics for synthetic and real nnInteractive evaluation masks."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SurfaceDistances:
    pred_to_target: np.ndarray
    target_to_pred: np.ndarray

    @property
    def all(self) -> np.ndarray:
        return np.concatenate([self.pred_to_target, self.target_to_pred])


def dice_score(
    prediction: Any,
    target: Any,
    *,
    empty_score: float = 1.0,
    threshold: float | None = None,
) -> float:
    """Compute Dice on binary masks.

    Empty/empty masks return `empty_score`. Non-boolean masks require `threshold`
    so probability maps cannot be silently truthy-cast.
    """
    pred, tgt = _paired_masks(prediction, target, threshold=threshold)
    pred_count = int(pred.sum())
    target_count = int(tgt.sum())
    denominator = pred_count + target_count
    if denominator == 0:
        return float(empty_score)
    intersection = int(np.logical_and(pred, tgt).sum())
    return float(2.0 * intersection / denominator)


def surface_distances(
    prediction: Any,
    target: Any,
    *,
    spacing: float | Sequence[float] | None = None,
    threshold: float | None = None,
) -> SurfaceDistances:
    """Return directed surface distances for binary masks.

    Non-boolean masks must pass `threshold`; NaN values are always rejected.
    """
    pred, tgt = _paired_masks(prediction, target, threshold=threshold)
    voxel_spacing = _spacing_tuple(spacing, pred.ndim)

    pred_surface = _surface(pred)
    target_surface = _surface(tgt)
    pred_count = int(pred_surface.sum())
    target_count = int(target_surface.sum())

    if pred_count == 0 and target_count == 0:
        empty = np.array([], dtype=float)
        return SurfaceDistances(empty, empty)

    if target_count == 0:
        pred_to_target = np.full(pred_count, math.inf, dtype=float)
    else:
        target_distance = ndimage.distance_transform_edt(
            ~target_surface,
            sampling=voxel_spacing,
        )
        pred_to_target = target_distance[pred_surface].astype(float, copy=False)

    if pred_count == 0:
        target_to_pred = np.full(target_count, math.inf, dtype=float)
    else:
        pred_distance = ndimage.distance_transform_edt(
            ~pred_surface,
            sampling=voxel_spacing,
        )
        target_to_pred = pred_distance[target_surface].astype(float, copy=False)

    return SurfaceDistances(pred_to_target, target_to_pred)


def hd95(
    prediction: Any,
    target: Any,
    *,
    spacing: float | Sequence[float] | None = None,
    threshold: float | None = None,
) -> float:
    """Return the canonical symmetric HD95.

    This is `max(p95(pred->target), p95(target->pred))`, not a pooled percentile.
    Empty/empty masks return 0.0; single-empty masks return `math.inf`.
    Pair this with `normalized_surface_dice`, which uses the canonical pooled
    surface-size-weighted NSD convention.
    """
    distances = surface_distances(
        prediction,
        target,
        spacing=spacing,
        threshold=threshold,
    )
    if distances.pred_to_target.size == 0 and distances.target_to_pred.size == 0:
        return 0.0
    if np.isinf(distances.all).any():
        return math.inf
    return float(
        max(
            _directed_percentile95(distances.pred_to_target),
            _directed_percentile95(distances.target_to_pred),
        )
    )


def normalized_surface_dice(
    prediction: Any,
    target: Any,
    *,
    tolerance: float,
    spacing: float | Sequence[float] | None = None,
    threshold: float | None = None,
) -> float:
    """Compute canonical normalized surface Dice (NSD).

    This is the pooled, surface-size-weighted ratio of surface voxels within
    tolerance across both directed surfaces. Empty/empty masks return 1.0;
    single-empty masks return 0.0.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    distances = surface_distances(
        prediction,
        target,
        spacing=spacing,
        threshold=threshold,
    )
    all_distances = distances.all
    if all_distances.size == 0:
        return 1.0
    within_tolerance = int(np.less_equal(all_distances, tolerance).sum())
    return float(within_tolerance / all_distances.size)


def noc_at_threshold(dice_by_step: Iterable[float], threshold: float) -> int | None:
    """Return the first 1-indexed interaction whose Dice reaches `threshold`.

    The first crossing is used even when later interactions are non-monotonic.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    for step, score in enumerate(dice_by_step, start=1):
        if _finite_score(score, "dice_by_step") >= threshold:
            return step
    return None


def noc_at_85(dice_by_step: Iterable[float]) -> int | None:
    return noc_at_threshold(dice_by_step, 0.85)


def noc_at_90(dice_by_step: Iterable[float]) -> int | None:
    return noc_at_threshold(dice_by_step, 0.90)


def dice_at_steps(
    dice_by_step: Sequence[float],
    *,
    steps: Sequence[int] = (1, 3, 5),
) -> dict[int, float | None]:
    scores = [_finite_score(score, "dice_by_step") for score in dice_by_step]
    result: dict[int, float | None] = {}
    for step in steps:
        if step < 1:
            raise ValueError("steps must be 1-indexed positive integers")
        result[int(step)] = scores[step - 1] if len(scores) >= step else None
    return result


def _paired_masks(
    prediction: Any,
    target: Any,
    *,
    threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    pred = _as_bool_mask(prediction, "prediction", threshold=threshold)
    tgt = _as_bool_mask(target, "target", threshold=threshold)
    if pred.shape != tgt.shape:
        raise ValueError(f"mask shapes differ: {pred.shape} != {tgt.shape}")
    return pred, tgt


def _as_bool_mask(
    mask: Any,
    name: str,
    *,
    threshold: float | None,
) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim < 2:
        raise ValueError(f"{name} must be at least 2D")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(array.dtype, np.floating) and bool(np.isnan(array).any()):
        raise ValueError(f"{name} contains NaN")
    if array.dtype == np.bool_:
        return array
    if threshold is None:
        raise ValueError(f"{name} must be boolean or threshold must be provided")
    if math.isnan(float(threshold)):
        raise ValueError("threshold must not be NaN")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric when threshold is provided")
    numeric = array.astype(float, copy=False)
    if not bool(np.isfinite(numeric).all()):
        raise ValueError(f"{name} contains non-finite values")
    if bool((numeric < 0.0).any()) or bool((numeric > 1.0).any()):
        raise ValueError(
            f"{name} threshold input must be probability-like values in [0, 1]; "
            "convert multi-label maps to a single binary class mask first"
        )
    threshold_value = float(threshold)
    if threshold_value < 0.0 or threshold_value > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return np.greater_equal(numeric, threshold_value)


def _surface(mask: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3,) * mask.ndim, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, ~eroded)


def _spacing_tuple(
    spacing: float | Sequence[float] | None,
    ndim: int,
) -> tuple[float, ...]:
    if spacing is None:
        values = (1.0,) * ndim
    elif isinstance(spacing, (int, float)):
        values = (float(spacing),) * ndim
    else:
        values = tuple(float(value) for value in spacing)

    if len(values) != ndim:
        raise ValueError(f"spacing must have {ndim} values")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("spacing values must be finite and > 0")
    return values


def _directed_percentile95(distances: np.ndarray) -> float:
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def _finite_score(value: float, name: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} scores must be finite")
    return score
