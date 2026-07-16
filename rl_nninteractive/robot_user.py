"""Deterministic FP/FN robot-user baseline for interaction policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np
from scipy import ndimage

from .adapter import VoxelCoord
from .env import POINT_NEGATIVE, POINT_POSITIVE, STOP, RlNnInteractiveEnv

ErrorKind = Literal["false_negative", "false_positive", "none"]


@dataclass(frozen=True)
class RobotUserDecision:
    """One deterministic robot-user action derived from current mask error."""

    action_type: int
    coord: VoxelCoord
    error_kind: ErrorKind
    component_size: int

    def to_env_action(self) -> dict[str, object]:
        return {"action_type": self.action_type, "coord": self.coord}


@dataclass(frozen=True)
class RobotUserEpisode:
    """Recorded rollout of the largest-component robot user.

    If a caller-provided `max_steps` is exhausted before STOP or env truncation,
    both `terminated` and `truncated` remain false.
    """

    initial_info: Mapping[str, Any]
    final_info: Mapping[str, Any]
    decisions: tuple[RobotUserDecision, ...]
    dice_by_step: tuple[float, ...]
    total_reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class _ComponentCandidate:
    action_type: int
    coord: VoxelCoord
    error_kind: ErrorKind
    component_size: int


def largest_component_robot_action(
    current_mask: Any,
    ground_truth: Any,
) -> RobotUserDecision:
    """Return the next largest-component FP/FN robot-user action.

    The policy chooses the largest connected component across false negatives
    and false positives. Ties prefer false negatives, matching the usual
    correction priority of filling missed target before removing equal-sized
    leakage. The selected point is the component voxel nearest the component
    centroid, with lexicographic tie-breaking for reproducibility.
    """

    current = _as_binary_volume(current_mask, name="current_mask")
    target = _as_binary_volume(ground_truth, name="ground_truth")
    if current.shape != target.shape:
        raise ValueError(f"mask shapes differ: {current.shape} != {target.shape}")

    false_negative = np.logical_and(target, ~current)
    false_positive = np.logical_and(current, ~target)
    fn_candidate = _largest_component_candidate(
        false_negative,
        action_type=POINT_POSITIVE,
        error_kind="false_negative",
    )
    fp_candidate = _largest_component_candidate(
        false_positive,
        action_type=POINT_NEGATIVE,
        error_kind="false_positive",
    )

    if fn_candidate is None and fp_candidate is None:
        return RobotUserDecision(
            action_type=STOP,
            coord=(0, 0, 0),
            error_kind="none",
            component_size=0,
        )
    if fp_candidate is None:
        assert fn_candidate is not None
        return _to_decision(fn_candidate)
    if fn_candidate is None:
        return _to_decision(fp_candidate)
    if fn_candidate.component_size >= fp_candidate.component_size:
        return _to_decision(fn_candidate)
    return _to_decision(fp_candidate)


def run_largest_component_robot_user(
    env: RlNnInteractiveEnv,
    *,
    image: Any,
    ground_truth: Any,
    initial_point: Sequence[int] | None = None,
    initial_include: bool = True,
    max_steps: int | None = None,
) -> RobotUserEpisode:
    """Roll the deterministic robot user through an `RlNnInteractiveEnv`.

    `dice_by_step` records only point interactions, not the terminal STOP.
    """

    target = _as_binary_volume(ground_truth, name="ground_truth")
    step_limit = env.max_interactions + 1 if max_steps is None else _as_positive_int(max_steps)
    reset_options: dict[str, Any] = {"image": image, "ground_truth": target}
    if initial_point is not None:
        reset_options["initial_point"] = initial_point
        reset_options["initial_include"] = initial_include

    obs, info = env.reset(options=reset_options)
    initial_info = dict(info)
    final_info = dict(info)
    decisions: list[RobotUserDecision] = []
    dice_by_step: list[float] = []
    total_reward = 0.0
    terminated = False
    truncated = False

    for _ in range(step_limit):
        decision = largest_component_robot_action(obs["mask"], target)
        decisions.append(decision)
        obs, reward, terminated, truncated, info = env.step(decision.to_env_action())
        final_info = dict(info)
        total_reward += float(reward)
        if decision.action_type != STOP:
            dice_by_step.append(float(info["dice"]))
        if terminated or truncated:
            break

    return RobotUserEpisode(
        initial_info=initial_info,
        final_info=final_info,
        decisions=tuple(decisions),
        dice_by_step=tuple(dice_by_step),
        total_reward=float(total_reward),
        terminated=terminated,
        truncated=truncated,
    )


def _largest_component_candidate(
    error_mask: np.ndarray,
    *,
    action_type: int,
    error_kind: ErrorKind,
) -> _ComponentCandidate | None:
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, component_count = ndimage.label(error_mask, structure=structure)
    if component_count == 0:
        return None

    counts = np.bincount(labels.ravel(), minlength=component_count + 1)
    counts[0] = 0
    component_size = int(counts.max())
    label_id = int(np.flatnonzero(counts == component_size)[0])
    return _ComponentCandidate(
        action_type=action_type,
        coord=_representative_coord(labels, label_id),
        error_kind=error_kind,
        component_size=component_size,
    )


def _representative_coord(labels: np.ndarray, label_id: int) -> VoxelCoord:
    coords = np.argwhere(labels == label_id)
    centroid = coords.mean(axis=0)
    distances = np.square(coords - centroid).sum(axis=1)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], distances))
    chosen = coords[int(order[0])]
    return (int(chosen[0]), int(chosen[1]), int(chosen[2]))


def _to_decision(candidate: _ComponentCandidate) -> RobotUserDecision:
    return RobotUserDecision(
        action_type=candidate.action_type,
        coord=candidate.coord,
        error_kind=candidate.error_kind,
        component_size=candidate.component_size,
    )


def _as_binary_volume(mask: Any, *, name: str) -> np.ndarray:
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


def _as_positive_int(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("max_steps must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError("max_steps must be >= 1")
    return result
