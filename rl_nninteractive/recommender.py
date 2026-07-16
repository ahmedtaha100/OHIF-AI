"""OHIF-facing prompt recommender payload helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .multitool import MultiToolAction, multi_tool_candidates


@dataclass(frozen=True)
class PromptSuggestion:
    case_id: str
    action: MultiToolAction
    confidence: float
    reason: str
    requires_review: bool = True

    def to_json_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "action": self.action.to_json_dict(),
            "confidence": float(self.confidence),
            "reason": self.reason,
            "requires_review": self.requires_review,
        }


def recommend_next_prompt(
    *,
    case_id: str,
    current_mask: Any,
    ground_truth_for_mock: Any,
) -> PromptSuggestion:
    """Return a deterministic ghost-prompt suggestion for mock smoke tests.

    `ground_truth_for_mock` is intentionally named to prevent confusing this
    helper with a production inference hook. Real OHIF use must replace it with
    an uncertainty/error model output.
    """

    candidates = multi_tool_candidates(current_mask, ground_truth_for_mock)
    action = next((candidate for candidate in candidates if candidate.tool != "stop"), candidates[-1])
    confidence = 0.5 if action.tool == "stop" else min(0.99, 0.5 + 0.05 * (action.geometry.component_size if action.geometry else 0))
    return PromptSuggestion(
        case_id=case_id,
        action=action,
        confidence=confidence,
        reason="largest_mock_error_component",
        requires_review=True,
    )
