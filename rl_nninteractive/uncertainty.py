"""Uncertainty-channel helpers for small-data smoke tests."""

from __future__ import annotations

from typing import Any

import numpy as np


def tta_disagreement_channel(predictions: Any) -> np.ndarray:
    """Return voxelwise disagreement from a stack of binary TTA predictions."""

    stack = np.asarray(predictions)
    if stack.ndim != 4:
        raise ValueError("predictions must have shape (n, z, y, x)")
    if stack.shape[0] < 2:
        raise ValueError("at least two predictions are required")
    if stack.dtype != np.bool_:
        if not np.issubdtype(stack.dtype, np.number):
            raise ValueError("predictions must be boolean or binary numeric")
        if not bool(np.isfinite(stack).all()):
            raise ValueError("predictions contain non-finite values")
        if not bool(np.isin(stack, (0, 1)).all()):
            raise ValueError("predictions must be binary")
        stack = stack.astype(bool)
    probability = stack.astype(np.float32).mean(axis=0)
    return (2.0 * np.minimum(probability, 1.0 - probability)).astype(np.float32)


def append_uncertainty_channel(state_channels: Any, uncertainty: Any) -> np.ndarray:
    channels = np.asarray(state_channels, dtype=np.float32)
    channel = np.asarray(uncertainty, dtype=np.float32)
    if channels.ndim != 4:
        raise ValueError("state_channels must have shape (c, z, y, x)")
    if channel.shape != channels.shape[1:]:
        raise ValueError(f"uncertainty shape {channel.shape} != {channels.shape[1:]}")
    if not bool(np.isfinite(channel).all()):
        raise ValueError("uncertainty contains non-finite values")
    return np.concatenate([channels, channel[None, ...]], axis=0)
