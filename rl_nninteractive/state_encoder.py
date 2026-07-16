"""State-channel encoding for point-policy MVP experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EncodedState:
    channels: np.ndarray
    channel_names: tuple[str, ...]
    step_fraction: float


def encode_state_channels(
    *,
    image: Any,
    current_mask: Any,
    positive_prompt_history: Any | None = None,
    negative_prompt_history: Any | None = None,
    step_index: int = 0,
    max_steps: int = 1,
) -> EncodedState:
    """Stack image, current mask, prompt-history, and step channels.

    The output is a light 3D state tensor for the point-only MVP:
    `(image, current_mask, positive_prompt_history, negative_prompt_history,
    step_fraction)`.
    """

    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 4 or image_array.shape[0] != 1:
        raise ValueError("image must have shape (1, z, y, x)")
    volume_shape = image_array.shape[1:]
    if not bool(np.isfinite(image_array).all()):
        raise ValueError("image contains non-finite values")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if step_index < 0:
        raise ValueError("step_index must be >= 0")

    mask = _as_binary_volume(current_mask, "current_mask", volume_shape)
    positive = _history_or_zeros(positive_prompt_history, "positive_prompt_history", volume_shape)
    negative = _history_or_zeros(negative_prompt_history, "negative_prompt_history", volume_shape)
    step_fraction = min(float(step_index) / float(max_steps), 1.0)
    step_channel = np.full(volume_shape, step_fraction, dtype=np.float32)

    channels = np.stack(
        [
            image_array[0],
            mask.astype(np.float32),
            positive.astype(np.float32),
            negative.astype(np.float32),
            step_channel,
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return EncodedState(
        channels=channels,
        channel_names=(
            "image",
            "current_mask",
            "positive_prompt_history",
            "negative_prompt_history",
            "step_fraction",
        ),
        step_fraction=step_fraction,
    )


def _history_or_zeros(value: Any | None, name: str, shape: tuple[int, int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    return _as_binary_volume(value, name, shape)


def _as_binary_volume(value: Any, name: str, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    if array.dtype == np.bool_:
        return array.astype(bool, copy=True)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be boolean or binary numeric")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains non-finite values")
    if not bool(np.isin(array, (0, 1)).all()):
        raise ValueError(f"{name} numeric values must be in {{0, 1}}")
    return array.astype(bool, copy=True)
