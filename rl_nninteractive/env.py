"""Gymnasium environment for synthetic RL-over-nnInteractive experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .adapter import Box3D, NnInteractiveSession, VoxelCoord, as_voxel_coord
from .metrics import dice_score
from .mock_adapter import MockNnInteractiveSession

STOP = 0
POINT_POSITIVE = 1
POINT_NEGATIVE = 2


class RlNnInteractiveEnv(gym.Env):
    """Minimal Gymnasium env around a frozen nnInteractive-like session.

    The default session is the mock adapter. Rewards are immediate Dice deltas
    against a provided binary ground-truth mask, so this environment is for
    synthetic/unit-test wiring until real dataset/evaluation units land.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        volume_shape: Sequence[int],
        *,
        max_interactions: int = 5,
        session_factory: Callable[[], NnInteractiveSession] = MockNnInteractiveSession,
    ) -> None:
        super().__init__()
        self.volume_shape = _as_volume_shape(volume_shape)
        self.max_interactions = _as_positive_int(max_interactions, name="max_interactions")
        self._session_factory = session_factory
        self._session: NnInteractiveSession | None = None
        self._image: np.ndarray | None = None
        self._ground_truth: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._dice = 0.0
        self._steps = 0
        self._done = False

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, *self.volume_shape),
                    dtype=np.float32,
                ),
                "mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=self.volume_shape,
                    dtype=np.uint8,
                ),
                "step": spaces.Box(
                    low=0,
                    high=self.max_interactions,
                    shape=(1,),
                    dtype=np.int32,
                ),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(3),
                "coord": spaces.MultiDiscrete(np.asarray(self.volume_shape, dtype=np.int64)),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if options is None:
            raise ValueError("reset requires options with image and ground_truth")
        if "image" not in options or "ground_truth" not in options:
            raise ValueError("reset options must include image and ground_truth")

        image = _as_image(options["image"], self.volume_shape)
        ground_truth = _as_binary_volume(options["ground_truth"], self.volume_shape)
        initial_point = options.get("initial_point")
        point: VoxelCoord | None = None
        include_interaction = True
        if initial_point is not None:
            point = _checked_coord(initial_point, self.volume_shape)
            include_interaction = bool(options.get("initial_include", True))

        session = self._session
        image_changed = (
            session is None
            or self._image is None
            or not np.array_equal(self._image, image)
        )
        if session is None:
            session = self._session_factory()
        else:
            session.reset_interactions()
        if image_changed:
            session.set_image(image)
        session.set_target_buffer(np.zeros(self.volume_shape, dtype=np.uint8))

        changed_bbox: Box3D | None = None
        if point is not None:
            result = session.add_point_interaction(
                point,
                include_interaction=include_interaction,
            )
            changed_bbox = result.changed_bbox

        mask = self._read_session_target_buffer(session)
        dice = dice_score(mask.astype(bool), ground_truth)

        self._image = image
        self._ground_truth = ground_truth
        self._session = session
        self._steps = 0
        self._done = False
        self._mask = mask
        self._dice = dice

        info = self._info(
            previous_dice=self._dice,
            reward=0.0,
            changed_bbox=changed_bbox,
            done_reason=None,
        )
        info["initial_seed_used"] = initial_point is not None
        return self._observation(), info

    def step(
        self,
        action: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._session is None or self._ground_truth is None:
            raise RuntimeError("reset must be called before step")
        if self._done:
            raise RuntimeError("step called after episode ended; call reset")
        if "action_type" not in action:
            raise ValueError("action requires action_type")
        action_type = _checked_action_type(action["action_type"])
        previous_dice = self._dice
        changed_bbox: Box3D | None = None
        terminated = False
        truncated = False
        done_reason: str | None = None

        if action_type == STOP:
            reward = 0.0
            terminated = True
            done_reason = "stop"
        elif action_type in (POINT_POSITIVE, POINT_NEGATIVE):
            if "coord" not in action:
                raise ValueError("point actions require coord")
            coord = _checked_coord(action["coord"], self.volume_shape)
            include_interaction = action_type == POINT_POSITIVE
            try:
                result = self._session.add_point_interaction(
                    coord,
                    include_interaction=include_interaction,
                )
                mask = self._read_target_buffer()
                dice = dice_score(mask.astype(bool), self._ground_truth)
            except Exception:
                self._done = True
                raise
            changed_bbox = result.changed_bbox
            self._steps += 1
            self._mask = mask
            self._dice = dice
            reward = self._dice - previous_dice
            if self._steps >= self.max_interactions:
                truncated = True
                done_reason = "max_interactions"
        self._done = terminated or truncated
        info = self._info(
            previous_dice=previous_dice,
            reward=reward,
            changed_bbox=changed_bbox,
            done_reason=done_reason,
        )
        return self._observation(), float(reward), terminated, truncated, info

    def _read_target_buffer(self) -> np.ndarray:
        assert self._session is not None
        return self._read_session_target_buffer(self._session)

    def _read_session_target_buffer(self, session: NnInteractiveSession) -> np.ndarray:
        buffer = session.target_buffer
        if hasattr(buffer, "detach") and hasattr(buffer, "cpu"):
            array = buffer.detach().cpu().numpy()
        else:
            array = np.asarray(buffer)
        if array.shape != self.volume_shape:
            raise RuntimeError(f"target_buffer shape {array.shape} != {self.volume_shape}")
        if array.dtype == np.bool_:
            return array.astype(np.uint8, copy=True)
        if not np.issubdtype(array.dtype, np.number):
            raise RuntimeError("target_buffer must be boolean or binary numeric")
        if not bool(np.isfinite(array).all()):
            raise RuntimeError("target_buffer contains non-finite values")
        if not bool(np.isin(array, (0, 1)).all()):
            raise RuntimeError("target_buffer numeric values must be in {0, 1}")
        return array.astype(np.uint8, copy=True)

    def close(self) -> None:
        if self._session is not None and hasattr(self._session, "close"):
            self._session.close()
        self._session = None
        super().close()

    def _observation(self) -> dict[str, np.ndarray]:
        assert self._image is not None
        assert self._mask is not None
        return {
            "image": self._image.copy(),
            "mask": self._mask.copy(),
            "step": np.asarray([self._steps], dtype=np.int32),
        }

    def _info(
        self,
        *,
        previous_dice: float,
        reward: float,
        changed_bbox: Box3D | None,
        done_reason: str | None,
    ) -> dict[str, Any]:
        assert self._mask is not None
        return {
            "dice": float(self._dice),
            "previous_dice": float(previous_dice),
            "reward_delta_dice": float(reward),
            "steps": int(self._steps),
            "changed_bbox": changed_bbox,
            "mask_sum": int(self._mask.sum()),
            "done_reason": done_reason,
        }


def _as_volume_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError("volume_shape must have exactly 3 values")
    return tuple(_as_positive_int(value, name="volume_shape") for value in shape)


def _as_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} values must be positive integers")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} values must be positive")
    return result


def _as_image(image: Any, volume_shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    expected = (1, *volume_shape)
    if array.shape != expected:
        raise ValueError(f"image shape must be {expected}, got {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise ValueError("image contains non-finite values")
    return array.copy()


def _as_binary_volume(mask: Any, volume_shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != volume_shape:
        raise ValueError(f"ground_truth shape must be {volume_shape}, got {array.shape}")
    if array.dtype == np.bool_:
        return array.copy()
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("ground_truth must be boolean or binary numeric")
    if not bool(np.isfinite(array).all()):
        raise ValueError("ground_truth contains non-finite values")
    if not bool(np.isin(array, (0, 1)).all()):
        raise ValueError("ground_truth numeric values must be in {0, 1}")
    return array.astype(bool, copy=True)


def _checked_coord(coord: Sequence[int], volume_shape: tuple[int, int, int]) -> VoxelCoord:
    point = as_voxel_coord(coord)
    if any(point[axis] >= volume_shape[axis] for axis in range(3)):
        raise ValueError("coord values must be inside volume_shape")
    return point


def _checked_action_type(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("action_type must be an integer")
    action_type = int(value)
    if action_type not in (STOP, POINT_POSITIVE, POINT_NEGATIVE):
        raise ValueError("action_type must be STOP, POINT_POSITIVE, or POINT_NEGATIVE")
    return action_type
