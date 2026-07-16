"""Multi-tool action candidates and adapter dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .adapter import NnInteractiveSession
from .deterministic_geometry import (
    ComponentGeometry,
    GeometryTool,
    Polarity,
    build_component_geometry,
    largest_error_component_mask,
)

ToolName = Literal["point", "scribble", "lasso", "box", "stop"]


@dataclass(frozen=True)
class MultiToolAction:
    tool: ToolName
    polarity: Polarity | None
    geometry: ComponentGeometry | None
    confidence: float = 1.0

    def to_json_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "polarity": self.polarity,
            "confidence": float(self.confidence),
            "geometry": None if self.geometry is None else self.geometry.to_json_dict(),
        }


def multi_tool_candidates(
    current_mask: Any,
    ground_truth: Any,
    *,
    tools: tuple[GeometryTool, ...] = ("point", "scribble", "lasso", "box"),
) -> tuple[MultiToolAction, ...]:
    """Build one deterministic candidate per tool for the largest FN/FP error."""

    actions: list[MultiToolAction] = []
    for polarity in ("positive", "negative"):
        component = largest_error_component_mask(current_mask, ground_truth, polarity=polarity)
        if not bool(component.any()):
            continue
        for tool in tools:
            actions.append(
                MultiToolAction(
                    tool=tool,
                    polarity=polarity,
                    geometry=build_component_geometry(component, tool=tool, polarity=polarity),
                )
            )
    actions.append(MultiToolAction(tool="stop", polarity=None, geometry=None))
    return tuple(actions)


def dispatch_multi_tool_action(
    session: NnInteractiveSession,
    action: MultiToolAction,
) -> object | None:
    """Apply a multi-tool action to an nnInteractive-like session."""

    if action.tool == "stop":
        return None
    if action.geometry is None or action.polarity is None:
        raise ValueError("non-stop action requires geometry and polarity")
    include = action.polarity == "positive"
    if action.tool == "point":
        return session.add_point_interaction(action.geometry.coord, include_interaction=include)
    if action.tool == "box":
        return session.add_bbox_interaction(action.geometry.bbox, include_interaction=include)
    if action.tool == "scribble":
        return session.add_scribble_interaction(
            action.geometry.scribble.astype(np.uint8, copy=True),
            include_interaction=include,
        )
    if action.tool == "lasso":
        return session.add_lasso_interaction(
            action.geometry.lasso.astype(np.uint8, copy=True),
            include_interaction=include,
        )
    raise ValueError(f"unknown tool: {action.tool}")
