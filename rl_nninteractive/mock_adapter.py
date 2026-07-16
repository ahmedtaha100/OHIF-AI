"""Clearly labeled mock nnInteractive adapter for CI and unit tests.

This class does not run nnInteractive. It applies deterministic binary-mask
edits so downstream environment code can be tested without checkpoints, GPUs, or
medical data. It is not representative of real nnInteractive dynamics: there is
no neural forward pass, no 3D propagation from prompts, no interaction-history
decay, no logits/probabilities, and no changed-patch inference timing.
"""

from __future__ import annotations

import numpy as np

from .adapter import Box3D, ImageArray, InteractionResult, MaskArray, VoxelCoord


class MockNnInteractiveSession:
    """Mock implementation of the local `NnInteractiveSession` Protocol."""

    def __init__(self) -> None:
        self.image: ImageArray | None = None
        self.target_buffer: MaskArray | None = None
        self.interactions: list[tuple[str, object, bool]] = []

    def set_image(self, image: ImageArray) -> None:
        self.image = np.asarray(image).copy()

    def set_target_buffer(self, mask: MaskArray) -> None:
        self.target_buffer = np.asarray(mask, dtype=np.uint8).copy()

    def add_point_interaction(
        self,
        coord: VoxelCoord,
        *,
        include_interaction: bool,
    ) -> InteractionResult:
        self._require_target()
        z, y, x = coord
        self.target_buffer[z, y, x] = 1 if include_interaction else 0
        changed_bbox = ((z, z + 1), (y, y + 1), (x, x + 1))
        self.interactions.append(("point", coord, include_interaction))
        return InteractionResult(changed_bbox=changed_bbox)

    def add_bbox_interaction(
        self,
        box: Box3D,
        *,
        include_interaction: bool,
    ) -> InteractionResult:
        self._require_target()
        z_range, y_range, x_range = box
        self.target_buffer[
            z_range[0] : z_range[1],
            y_range[0] : y_range[1],
            x_range[0] : x_range[1],
        ] = 1 if include_interaction else 0
        self.interactions.append(("bbox", box, include_interaction))
        return InteractionResult(changed_bbox=box)

    def add_scribble_interaction(
        self,
        scribble_image: MaskArray,
        *,
        include_interaction: bool,
        interaction_bbox: Box3D | None = None,
    ) -> InteractionResult:
        return self._apply_mask_interaction(
            "scribble",
            scribble_image,
            include_interaction=include_interaction,
            interaction_bbox=interaction_bbox,
        )

    def add_lasso_interaction(
        self,
        lasso_image: MaskArray,
        *,
        include_interaction: bool,
        interaction_bbox: Box3D | None = None,
    ) -> InteractionResult:
        return self._apply_mask_interaction(
            "lasso",
            lasso_image,
            include_interaction=include_interaction,
            interaction_bbox=interaction_bbox,
        )

    def reset_interactions(self) -> None:
        self.interactions.clear()

    def _apply_mask_interaction(
        self,
        kind: str,
        interaction_mask: MaskArray,
        *,
        include_interaction: bool,
        interaction_bbox: Box3D | None,
    ) -> InteractionResult:
        self._require_target()
        mask = np.asarray(interaction_mask, dtype=bool)
        if mask.shape != self.target_buffer.shape:
            raise ValueError("interaction mask shape must match target_buffer")
        self.target_buffer[mask] = 1 if include_interaction else 0
        changed_bbox = interaction_bbox if interaction_bbox is not None else _bbox_from_mask(mask)
        self.interactions.append((kind, changed_bbox, include_interaction))
        return InteractionResult(changed_bbox=changed_bbox)

    def _require_target(self) -> None:
        if self.target_buffer is None:
            raise RuntimeError("target_buffer must be set before interactions")


def _bbox_from_mask(mask: np.ndarray) -> Box3D | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    starts = coords.min(axis=0)
    stops = coords.max(axis=0) + 1
    return (
        (int(starts[0]), int(stops[0])),
        (int(starts[1]), int(stops[1])),
        (int(starts[2]), int(stops[2])),
    )
