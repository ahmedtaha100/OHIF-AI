"""Typed nnInteractive session interface used by the RL environment layer.

Coordinate conventions are inferred from the caller-side usage in
`basic_infer.py`, which reverses OHIF coordinates before calling nnInteractive.
The scaffold uses `VoxelCoord` as `(z, y, x)` and `Box3D` as
`((z_start, z_stop), (y_start, y_stop), (x_start, x_stop))` with stop-exclusive
axis ranges. The real adapter smoke-test unit must verify this against the
installed nnInteractive package before any real evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, TypeAlias

MaskArray: TypeAlias = Any
ImageArray: TypeAlias = Any
VoxelCoord: TypeAlias = tuple[int, int, int]
AxisRange: TypeAlias = tuple[int, int]
Box3D: TypeAlias = tuple[AxisRange, AxisRange, AxisRange]


@dataclass(frozen=True)
class InteractionResult:
    """Metadata returned by an interaction call.

    `changed_bbox` is best-effort metadata for patch-local reads. Environment
    code must fall back to `target_buffer` when an adapter returns `None`.
    """

    changed_bbox: Box3D | None = None


class NnInteractiveSession(Protocol):
    """Protocol for a frozen nnInteractive session treated as an RL environment."""

    target_buffer: MaskArray

    def set_image(self, image: ImageArray) -> None:
        """Set the fixed image for the current episode."""

    def set_target_buffer(self, mask: MaskArray) -> None:
        """Set or replace the mutable target mask buffer."""

    def add_point_interaction(
        self,
        coord: VoxelCoord,
        *,
        include_interaction: bool,
    ) -> InteractionResult:
        """Apply a positive or negative point interaction."""

    def add_bbox_interaction(
        self,
        box: Box3D,
        *,
        include_interaction: bool,
    ) -> InteractionResult:
        """Apply a 3D bounding-box interaction."""

    def add_scribble_interaction(
        self,
        scribble_image: MaskArray,
        *,
        include_interaction: bool,
        interaction_bbox: Box3D | None = None,
    ) -> InteractionResult:
        """Apply a scribble interaction mask."""

    def add_lasso_interaction(
        self,
        lasso_image: MaskArray,
        *,
        include_interaction: bool,
        interaction_bbox: Box3D | None = None,
    ) -> InteractionResult:
        """Apply a lasso interaction mask."""

    def reset_interactions(self) -> None:
        """Clear interaction history for a new episode."""


def as_voxel_coord(coord: Sequence[int]) -> VoxelCoord:
    """Normalize a coordinate to a 3D integer voxel coordinate."""
    if len(coord) != 3:
        raise ValueError("coord must have exactly 3 values")
    result = (
        _as_integer(coord[0], "coord"),
        _as_integer(coord[1], "coord"),
        _as_integer(coord[2], "coord"),
    )
    if any(value < 0 for value in result):
        raise ValueError("coord values must be non-negative")
    return result


def as_box3d(box: Sequence[Sequence[int]]) -> Box3D:
    """Normalize a bbox to nnInteractive's per-axis [[min, max], ...] shape."""
    if len(box) != 3:
        raise ValueError("box must have exactly 3 axis ranges")
    return (
        _as_axis_range(box[0]),
        _as_axis_range(box[1]),
        _as_axis_range(box[2]),
    )


def _as_axis_range(axis_range: Sequence[int]) -> AxisRange:
    if len(axis_range) != 2:
        raise ValueError("axis ranges must have exactly 2 values")
    result = (
        _as_integer(axis_range[0], "axis range"),
        _as_integer(axis_range[1], "axis range"),
    )
    if result[0] < 0 or result[1] < 0:
        raise ValueError("axis range values must be non-negative")
    if result[1] <= result[0]:
        raise ValueError("axis range stop must be greater than start")
    return result


def _as_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} values must be integers")
    return int(value)
