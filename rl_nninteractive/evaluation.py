"""Interaction evaluation summaries wired to the metric library."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .metrics import dice_at_steps, noc_at_85, noc_at_90

DEFAULT_DICE_STEPS = (1, 3, 5)


@dataclass(frozen=True)
class InteractionEvaluation:
    """NoC/Dice summary for one interaction trajectory."""

    name: str
    point_interaction_count: int
    final_dice: float | None
    noc_at_85: int | None
    noc_at_90: int | None
    dice_at: dict[int, float | None]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "point_interaction_count": self.point_interaction_count,
            "final_dice": self.final_dice,
            "noc_at_85": self.noc_at_85,
            "noc_at_90": self.noc_at_90,
            "dice_at": {str(step): value for step, value in self.dice_at.items()},
        }


def evaluate_interaction_trajectory(
    name: str,
    dice_by_step: Iterable[float],
    *,
    final_dice: float | None = None,
    dice_steps: Sequence[int] = DEFAULT_DICE_STEPS,
) -> InteractionEvaluation:
    """Evaluate one point-interaction Dice trajectory."""

    scores = [_finite_score(score, "dice_by_step") for score in dice_by_step]
    resolved_final = _resolve_final_dice(scores, final_dice)
    return InteractionEvaluation(
        name=name,
        point_interaction_count=len(scores),
        final_dice=resolved_final,
        noc_at_85=noc_at_85(scores),
        noc_at_90=noc_at_90(scores),
        dice_at=dice_at_steps(scores, steps=dice_steps),
    )


def summarize_interaction_evaluations(
    evaluations: Iterable[InteractionEvaluation],
) -> dict[str, object]:
    """Aggregate evaluation rows without hiding per-case values."""

    rows = list(evaluations)
    final_dice_values = [
        evaluation.final_dice
        for evaluation in rows
        if evaluation.final_dice is not None
    ]
    return {
        "case_count": len(rows),
        "mean_final_dice": _mean(final_dice_values),
        "mean_point_interactions": _mean(
            [evaluation.point_interaction_count for evaluation in rows]
        ),
        "reached_85_count": sum(evaluation.noc_at_85 is not None for evaluation in rows),
        "reached_90_count": sum(evaluation.noc_at_90 is not None for evaluation in rows),
        "rows": [evaluation.to_json_dict() for evaluation in rows],
    }


def _resolve_final_dice(scores: list[float], final_dice: float | None) -> float | None:
    if final_dice is not None:
        return _finite_score(final_dice, "final_dice")
    if not scores:
        return None
    return scores[-1]


def _finite_score(value: float, name: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} scores must be finite")
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{name} scores must be in [0, 1]")
    return score


def _mean(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
