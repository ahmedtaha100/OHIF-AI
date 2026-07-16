"""Interaction logging schema for clinician-in-the-loop learning loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Decision = Literal["accepted", "rejected", "edited"]


@dataclass(frozen=True)
class InteractionLogEvent:
    case_id: str
    step_index: int
    tool: str
    decision: Decision
    proposed_prompt: dict[str, Any]
    final_prompt: dict[str, Any] | None
    elapsed_ms: int
    organ: str | None = None
    target: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be >= 0")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be >= 0")
        if self.decision == "edited" and self.final_prompt is None:
            raise ValueError("edited events require final_prompt")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "step_index": self.step_index,
            "tool": self.tool,
            "decision": self.decision,
            "proposed_prompt": self.proposed_prompt,
            "final_prompt": self.final_prompt,
            "elapsed_ms": self.elapsed_ms,
            "organ": self.organ,
            "target": self.target,
            "failure_reason": self.failure_reason,
        }
