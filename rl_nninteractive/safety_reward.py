"""Safety-shaped reward terms for local policy smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import dice_score, normalized_surface_dice


@dataclass(frozen=True)
class SafetyRewardBreakdown:
    total: float
    delta_dice: float
    delta_nsd: float
    step_cost: float
    leakage_penalty: float
    missed_target_penalty: float
    stop_reward: float

    def to_json_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "delta_dice": self.delta_dice,
            "delta_nsd": self.delta_nsd,
            "step_cost": self.step_cost,
            "leakage_penalty": self.leakage_penalty,
            "missed_target_penalty": self.missed_target_penalty,
            "stop_reward": self.stop_reward,
        }


def safety_shaped_reward(
    *,
    previous_mask: Any,
    current_mask: Any,
    ground_truth: Any,
    organ_mask: Any | None = None,
    is_stop: bool = False,
    stop_threshold: float = 0.90,
    step_cost_weight: float = 0.01,
    nsd_weight: float = 0.25,
    leakage_weight: float = 0.10,
    missed_weight: float = 0.10,
    stop_reward_weight: float = 0.25,
    nsd_tolerance: float = 1.0,
) -> SafetyRewardBreakdown:
    previous = _as_binary_volume(previous_mask, "previous_mask")
    current = _as_binary_volume(current_mask, "current_mask")
    target = _as_binary_volume(ground_truth, "ground_truth")
    if previous.shape != current.shape or current.shape != target.shape:
        raise ValueError("previous/current/ground_truth shapes must match")
    organ = np.ones_like(target, dtype=bool) if organ_mask is None else _as_binary_volume(organ_mask, "organ_mask")
    if organ.shape != target.shape:
        raise ValueError("organ_mask shape must match ground_truth")

    prev_dice = dice_score(previous, target)
    curr_dice = dice_score(current, target)
    prev_nsd = normalized_surface_dice(previous, target, tolerance=nsd_tolerance)
    curr_nsd = normalized_surface_dice(current, target, tolerance=nsd_tolerance)
    leakage_fraction = _fraction(np.logical_and(current, ~organ))
    missed_fraction = _fraction(np.logical_and(target, ~current), denominator=max(int(target.sum()), 1))
    if not is_stop:
        stop_reward = 0.0
    elif curr_dice >= stop_threshold:
        stop_reward = stop_reward_weight
    else:
        stop_reward = -stop_reward_weight
    step_cost = 0.0 if is_stop else step_cost_weight
    delta_dice = curr_dice - prev_dice
    delta_nsd = curr_nsd - prev_nsd
    leakage_penalty = leakage_weight * leakage_fraction
    missed_penalty = missed_weight * missed_fraction
    total = delta_dice + nsd_weight * delta_nsd - step_cost - leakage_penalty - missed_penalty + stop_reward
    return SafetyRewardBreakdown(
        total=float(total),
        delta_dice=float(delta_dice),
        delta_nsd=float(delta_nsd),
        step_cost=float(step_cost),
        leakage_penalty=float(leakage_penalty),
        missed_target_penalty=float(missed_penalty),
        stop_reward=float(stop_reward),
    )


def _fraction(mask: np.ndarray, *, denominator: int | None = None) -> float:
    denom = int(mask.size) if denominator is None else int(denominator)
    if denom <= 0:
        return 0.0
    return float(mask.sum() / denom)


def _as_binary_volume(mask: Any, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D volume")
    if array.dtype == np.bool_:
        return array.astype(bool, copy=True)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be boolean or binary numeric")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains non-finite values")
    if not bool(np.isin(array, (0, 1)).all()):
        raise ValueError(f"{name} numeric values must be in {{0, 1}}")
    return array.astype(bool, copy=True)
