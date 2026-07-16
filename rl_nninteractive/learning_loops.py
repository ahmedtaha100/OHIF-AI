"""DAgger, preference-learning, and STOP calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .interaction_log import InteractionLogEvent


@dataclass(frozen=True)
class DAggerSample:
    case_id: str
    step_index: int
    accepted: bool
    tool: str


@dataclass(frozen=True)
class PreferencePair:
    preferred_tool: str
    rejected_tool: str
    weight: float


def dagger_samples_from_logs(events: Iterable[InteractionLogEvent]) -> tuple[DAggerSample, ...]:
    return tuple(
        DAggerSample(
            case_id=event.case_id,
            step_index=event.step_index,
            accepted=event.decision in ("accepted", "edited"),
            tool=event.tool,
        )
        for event in events
    )


def preference_pairs_from_logs(events: Iterable[InteractionLogEvent]) -> tuple[PreferencePair, ...]:
    pairs: list[PreferencePair] = []
    for event in events:
        if event.decision != "edited" or event.final_prompt is None:
            continue
        final_tool = str(event.final_prompt.get("tool", event.tool))
        if final_tool != event.tool:
            pairs.append(PreferencePair(preferred_tool=final_tool, rejected_tool=event.tool, weight=1.0))
    return tuple(pairs)


def calibrate_stop_threshold(
    final_dice_scores: Iterable[float],
    *,
    default_threshold: float = 0.90,
) -> float:
    scores = sorted(float(score) for score in final_dice_scores)
    if not scores:
        return float(default_threshold)
    if any(score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("final_dice_scores must be in [0, 1]")
    index = max(0, int(0.25 * (len(scores) - 1)))
    return float(max(default_threshold, scores[index]))
