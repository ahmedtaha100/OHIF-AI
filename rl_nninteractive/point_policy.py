"""Point-only BC and lightweight DQN-style policy utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import ndimage

from .adapter import NnInteractiveSession, VoxelCoord
from .env import POINT_NEGATIVE, POINT_POSITIVE, STOP, RlNnInteractiveEnv
from .evaluation import evaluate_interaction_trajectory
from .robot_user import largest_component_robot_action, run_largest_component_robot_user
from .toy_dataset import ToySegmentationCase

CandidateKind = Literal["false_negative", "false_positive", "stop"]
FEATURE_DIM = 12
SessionFactory = Callable[[], NnInteractiveSession]


@dataclass(frozen=True)
class PointActionCandidate:
    action_type: int
    coord: VoxelCoord
    error_kind: CandidateKind
    component_size: int
    rank: int

    def to_env_action(self) -> dict[str, object]:
        return {"action_type": self.action_type, "coord": self.coord}

    def to_json_dict(self) -> dict[str, object]:
        return {
            "action_type": self.action_type,
            "coord_zyx": list(self.coord),
            "error_kind": self.error_kind,
            "component_size": self.component_size,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class BehaviorCloningSample:
    case_name: str
    step_index: int
    candidates: tuple[PointActionCandidate, ...]
    candidate_features: np.ndarray
    label_index: int


@dataclass(frozen=True)
class PointPolicy:
    weights: np.ndarray

    @classmethod
    def zeros(cls) -> "PointPolicy":
        return cls(weights=np.zeros(FEATURE_DIM, dtype=np.float64))

    def scores(self, candidate_features: np.ndarray) -> np.ndarray:
        features = np.asarray(candidate_features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"candidate_features must have shape (n, {FEATURE_DIM})")
        return features @ self.weights

    def choose_index(self, candidate_features: np.ndarray) -> int:
        return int(np.argmax(self.scores(candidate_features)))

    def choose(
        self,
        candidates: Sequence[PointActionCandidate],
        candidate_features: np.ndarray,
    ) -> PointActionCandidate:
        if not candidates:
            raise ValueError("candidates must not be empty")
        return candidates[self.choose_index(candidate_features)]


@dataclass(frozen=True)
class PolicyEpisode:
    case_name: str
    dice_by_step: tuple[float, ...]
    final_dice: float
    decisions: tuple[PointActionCandidate, ...]
    terminated: bool
    truncated: bool

    def evaluation_row(self) -> dict[str, object]:
        return evaluate_interaction_trajectory(
            self.case_name,
            self.dice_by_step,
            final_dice=self.final_dice,
        ).to_json_dict()


def point_action_candidates(
    current_mask: Any,
    ground_truth: Any,
    *,
    top_k: int = 3,
    include_stop: bool = True,
) -> tuple[PointActionCandidate, ...]:
    """Return top-k FP/FN component centroid actions plus optional STOP."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    current = _as_binary_volume(current_mask, name="current_mask")
    target = _as_binary_volume(ground_truth, name="ground_truth")
    if current.shape != target.shape:
        raise ValueError(f"mask shapes differ: {current.shape} != {target.shape}")

    false_negative = np.logical_and(target, ~current)
    false_positive = np.logical_and(current, ~target)
    raw_candidates = _component_candidates(
        false_negative,
        action_type=POINT_POSITIVE,
        error_kind="false_negative",
    ) + _component_candidates(
        false_positive,
        action_type=POINT_NEGATIVE,
        error_kind="false_positive",
    )
    raw_candidates.sort(
        key=lambda candidate: (
            -candidate.component_size,
            0 if candidate.error_kind == "false_negative" else 1,
            candidate.coord,
        )
    )
    selected = [
        PointActionCandidate(
            action_type=candidate.action_type,
            coord=candidate.coord,
            error_kind=candidate.error_kind,
            component_size=candidate.component_size,
            rank=index,
        )
        for index, candidate in enumerate(raw_candidates[:top_k])
    ]
    if include_stop:
        selected.append(
            PointActionCandidate(
                action_type=STOP,
                coord=(0, 0, 0),
                error_kind="stop",
                component_size=0,
                rank=len(selected),
            )
        )
    return tuple(selected)


def collect_behavior_cloning_samples(
    cases: Sequence[ToySegmentationCase],
    *,
    max_interactions: int = 5,
    top_k: int = 3,
    session_factory: SessionFactory | None = None,
) -> tuple[BehaviorCloningSample, ...]:
    samples: list[BehaviorCloningSample] = []
    for case in cases:
        env = _make_env(
            case.ground_truth.shape,
            max_interactions=max_interactions,
            session_factory=session_factory,
        )
        try:
            obs, _ = env.reset(
                options={
                    "image": case.image,
                    "ground_truth": case.ground_truth,
                    "initial_point": case.initial_point,
                    "initial_include": case.initial_include,
                }
            )
            for step_index in range(max_interactions + 1):
                candidates = point_action_candidates(obs["mask"], case.ground_truth, top_k=top_k)
                features = candidate_feature_matrix(
                    candidates=candidates,
                    current_mask=obs["mask"],
                    step_index=step_index,
                    max_steps=max_interactions,
                )
                label = _label_index(
                    candidates,
                    largest_component_robot_action(obs["mask"], case.ground_truth),
                )
                samples.append(
                    BehaviorCloningSample(
                        case_name=case.name,
                        step_index=step_index,
                        candidates=candidates,
                        candidate_features=features,
                        label_index=label,
                    )
                )
                chosen = candidates[label]
                obs, _, terminated, truncated, _ = env.step(chosen.to_env_action())
                if terminated or truncated:
                    break
        finally:
            env.close()
    return tuple(samples)


def train_behavior_cloning(
    samples: Sequence[BehaviorCloningSample],
    *,
    epochs: int = 8,
    learning_rate: float = 0.25,
) -> PointPolicy:
    """Train a deterministic linear candidate scorer by multiclass perceptron."""

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    weights = PointPolicy.zeros().weights.copy()
    for _ in range(epochs):
        for sample in samples:
            predicted = int(np.argmax(sample.candidate_features @ weights))
            if predicted != sample.label_index:
                weights += learning_rate * (
                    sample.candidate_features[sample.label_index]
                    - sample.candidate_features[predicted]
                )
    return PointPolicy(weights=weights)


def fine_tune_dqn(
    policy: PointPolicy,
    cases: Sequence[ToySegmentationCase],
    *,
    episodes: int = 12,
    max_interactions: int = 5,
    top_k: int = 3,
    learning_rate: float = 0.10,
    gamma: float = 0.90,
    epsilon: float = 0.10,
    seed: int = 20260705,
    session_factory: SessionFactory | None = None,
) -> PointPolicy:
    """Run a tiny deterministic DQN-style linear Q update for local smoke tests."""

    if episodes < 0:
        raise ValueError("episodes must be >= 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if not cases:
        raise ValueError("cases must not be empty")

    rng = np.random.default_rng(seed)
    weights = policy.weights.astype(np.float64, copy=True)
    env: RlNnInteractiveEnv | None = None
    try:
        for episode_index in range(episodes):
            case = cases[episode_index % len(cases)]
            if env is None or env.volume_shape != case.ground_truth.shape:
                if env is not None:
                    env.close()
                env = _make_env(
                    case.ground_truth.shape,
                    max_interactions=max_interactions,
                    session_factory=session_factory,
                )
            obs, _ = env.reset(
                options={
                    "image": case.image,
                    "ground_truth": case.ground_truth,
                    "initial_point": case.initial_point,
                    "initial_include": case.initial_include,
                }
            )
            for step_index in range(max_interactions + 1):
                candidates = point_action_candidates(obs["mask"], case.ground_truth, top_k=top_k)
                features = candidate_feature_matrix(
                    candidates,
                    current_mask=obs["mask"],
                    step_index=step_index,
                    max_steps=max_interactions,
                )
                if rng.random() < epsilon:
                    action_index = int(rng.integers(0, len(candidates)))
                else:
                    action_index = int(np.argmax(features @ weights))
                obs_next, reward, terminated, truncated, _ = env.step(
                    candidates[action_index].to_env_action()
                )
                if terminated or truncated:
                    target = float(reward)
                else:
                    next_candidates = point_action_candidates(
                        obs_next["mask"],
                        case.ground_truth,
                        top_k=top_k,
                    )
                    next_features = candidate_feature_matrix(
                        next_candidates,
                        current_mask=obs_next["mask"],
                        step_index=step_index + 1,
                        max_steps=max_interactions,
                    )
                    target = float(reward) + gamma * float(np.max(next_features @ weights))
                prediction = float(features[action_index] @ weights)
                weights += learning_rate * (target - prediction) * features[action_index]
                obs = obs_next
                if terminated or truncated:
                    break
    finally:
        if env is not None:
            env.close()
    return PointPolicy(weights=weights)


def rollout_point_policy(
    policy: PointPolicy,
    case: ToySegmentationCase,
    *,
    max_interactions: int = 5,
    top_k: int = 3,
    session_factory: SessionFactory | None = None,
) -> PolicyEpisode:
    env = _make_env(
        case.ground_truth.shape,
        max_interactions=max_interactions,
        session_factory=session_factory,
    )
    try:
        obs, info = env.reset(
            options={
                "image": case.image,
                "ground_truth": case.ground_truth,
                "initial_point": case.initial_point,
                "initial_include": case.initial_include,
            }
        )
        final_info = dict(info)
        dice_by_step: list[float] = []
        decisions: list[PointActionCandidate] = []
        terminated = False
        truncated = False
        for step_index in range(max_interactions + 1):
            candidates = point_action_candidates(obs["mask"], case.ground_truth, top_k=top_k)
            features = candidate_feature_matrix(
                candidates,
                current_mask=obs["mask"],
                step_index=step_index,
                max_steps=max_interactions,
            )
            decision = policy.choose(candidates, features)
            decisions.append(decision)
            obs, _, terminated, truncated, info = env.step(decision.to_env_action())
            final_info = dict(info)
            if decision.action_type != STOP:
                dice_by_step.append(float(info["dice"]))
            if terminated or truncated:
                break
        return PolicyEpisode(
            case_name=case.name,
            dice_by_step=tuple(dice_by_step),
            final_dice=float(final_info["dice"]),
            decisions=tuple(decisions),
            terminated=terminated,
            truncated=truncated,
        )
    finally:
        env.close()


def rollout_robot_user(
    case: ToySegmentationCase,
    *,
    max_interactions: int = 5,
    session_factory: SessionFactory | None = None,
) -> PolicyEpisode:
    env = _make_env(
        case.ground_truth.shape,
        max_interactions=max_interactions,
        session_factory=session_factory,
    )
    try:
        episode = run_largest_component_robot_user(
            env,
            image=case.image,
            ground_truth=case.ground_truth,
            initial_point=case.initial_point,
            initial_include=case.initial_include,
        )
        decisions = tuple(
            PointActionCandidate(
                action_type=decision.action_type,
                coord=decision.coord,
                error_kind=(
                    "false_negative"
                    if decision.error_kind == "false_negative"
                    else "false_positive"
                    if decision.error_kind == "false_positive"
                    else "stop"
                ),
                component_size=decision.component_size,
                rank=index,
            )
            for index, decision in enumerate(episode.decisions)
        )
        return PolicyEpisode(
            case_name=case.name,
            dice_by_step=episode.dice_by_step,
            final_dice=float(episode.final_info["dice"]),
            decisions=decisions,
            terminated=episode.terminated,
            truncated=episode.truncated,
        )
    finally:
        env.close()


def candidate_feature_matrix(
    candidates: Sequence[PointActionCandidate],
    *,
    current_mask: Any,
    step_index: int,
    max_steps: int,
) -> np.ndarray:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    mask = _as_binary_volume(current_mask, name="current_mask")
    mask_fraction = float(mask.mean())
    volume_shape = mask.shape
    return np.vstack(
        [
            _candidate_features(
                candidate,
                volume_shape=volume_shape,
                mask_fraction=mask_fraction,
                step_fraction=min(float(step_index) / float(max_steps), 1.0),
            )
            for candidate in candidates
        ]
    )


def _make_env(
    volume_shape: Sequence[int],
    *,
    max_interactions: int,
    session_factory: SessionFactory | None,
) -> RlNnInteractiveEnv:
    if session_factory is None:
        return RlNnInteractiveEnv(volume_shape, max_interactions=max_interactions)
    return RlNnInteractiveEnv(
        volume_shape,
        max_interactions=max_interactions,
        session_factory=session_factory,
    )


def _candidate_features(
    candidate: PointActionCandidate,
    *,
    volume_shape: tuple[int, int, int],
    mask_fraction: float,
    step_fraction: float,
) -> np.ndarray:
    denominators = np.maximum(np.asarray(volume_shape, dtype=np.float64) - 1.0, 1.0)
    coord = np.asarray(candidate.coord, dtype=np.float64) / denominators
    volume_size = float(np.prod(volume_shape))
    return np.asarray(
        [
            1.0,
            1.0 if candidate.action_type == STOP else 0.0,
            1.0 if candidate.action_type == POINT_POSITIVE else 0.0,
            1.0 if candidate.action_type == POINT_NEGATIVE else 0.0,
            coord[0],
            coord[1],
            coord[2],
            float(candidate.component_size) / volume_size,
            1.0 if candidate.error_kind == "false_negative" else 0.0,
            1.0 if candidate.error_kind == "false_positive" else 0.0,
            mask_fraction,
            step_fraction,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class _RawCandidate:
    action_type: int
    coord: VoxelCoord
    error_kind: CandidateKind
    component_size: int


def _component_candidates(
    error_mask: np.ndarray,
    *,
    action_type: int,
    error_kind: CandidateKind,
) -> list[_RawCandidate]:
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, component_count = ndimage.label(error_mask, structure=structure)
    results: list[_RawCandidate] = []
    for label_id in range(1, component_count + 1):
        coords = np.argwhere(labels == label_id)
        if coords.size == 0:
            continue
        results.append(
            _RawCandidate(
                action_type=action_type,
                coord=_representative_coord(coords),
                error_kind=error_kind,
                component_size=int(coords.shape[0]),
            )
        )
    return results


def _representative_coord(coords: np.ndarray) -> VoxelCoord:
    centroid = coords.mean(axis=0)
    distances = np.square(coords - centroid).sum(axis=1)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], distances))
    chosen = coords[int(order[0])]
    return (int(chosen[0]), int(chosen[1]), int(chosen[2]))


def _label_index(
    candidates: Sequence[PointActionCandidate],
    robot_decision: Any,
) -> int:
    for index, candidate in enumerate(candidates):
        if robot_decision.action_type == STOP and candidate.action_type == STOP:
            return index
        if (
            candidate.action_type == robot_decision.action_type
            and candidate.coord == robot_decision.coord
        ):
            return index
    raise RuntimeError("robot decision missing from candidate set")


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
