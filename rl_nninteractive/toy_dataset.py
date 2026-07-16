"""Fixed small synthetic cases for local proof-of-pipeline runs.

These cases are code-created binary masks for wiring and regression tests.
They are not a public benchmark and must not be reported as real results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SplitName = Literal["train", "val", "all"]


@dataclass(frozen=True)
class ToySegmentationCase:
    name: str
    image: np.ndarray
    ground_truth: np.ndarray
    initial_point: tuple[int, int, int] | None = None
    initial_include: bool = True


def synthetic_toy_cases(split: SplitName = "all") -> tuple[ToySegmentationCase, ...]:
    """Return a deterministic tiny split for BC/RL smoke tests."""

    train = (
        _case("train_single_center", [(2, 2, 2)]),
        _case("train_two_adjacent", [(2, 2, 2), (2, 2, 3)]),
        _case("train_three_l_shape", [(1, 1, 1), (1, 1, 2), (1, 2, 1)]),
        _case("train_false_positive_cleanup", [(3, 3, 3)], initial_point=(0, 0, 0)),
    )
    val = (
        _case("val_two_diagonal", [(2, 3, 2), (3, 3, 2)]),
        _case("val_three_line", [(4, 1, 1), (4, 1, 2), (4, 1, 3)]),
        _case("val_single_corner", [(1, 4, 4)]),
    )
    if split == "train":
        return train
    if split == "val":
        return val
    if split == "all":
        return train + val
    raise ValueError("split must be train, val, or all")


def _case(
    name: str,
    voxels: list[tuple[int, int, int]],
    *,
    initial_point: tuple[int, int, int] | None = None,
) -> ToySegmentationCase:
    shape = (6, 6, 6)
    ground_truth = np.zeros(shape, dtype=bool)
    image = np.zeros((1, *shape), dtype=np.float32)
    for z, y, x in voxels:
        ground_truth[z, y, x] = True
        image[0, z, y, x] = 1.0
    if initial_point is not None:
        image[0, initial_point[0], initial_point[1], initial_point[2]] = 0.5
    return ToySegmentationCase(
        name=name,
        image=image,
        ground_truth=ground_truth,
        initial_point=initial_point,
        initial_include=True,
    )
