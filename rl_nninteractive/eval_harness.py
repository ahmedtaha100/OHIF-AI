"""Evaluation and ablation scaffolds for large-run handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_entropy: bool
    use_multi_tool: bool
    use_safety_reward: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "use_entropy": self.use_entropy,
            "use_multi_tool": self.use_multi_tool,
            "use_safety_reward": self.use_safety_reward,
        }


def default_ablation_grid() -> tuple[AblationConfig, ...]:
    return (
        AblationConfig("point_only_no_entropy", False, False, False),
        AblationConfig("point_only_entropy", True, False, False),
        AblationConfig("multi_tool_no_entropy", False, True, True),
        AblationConfig("multi_tool_entropy", True, True, True),
    )


def classify_failure(
    prediction: np.ndarray,
    target: np.ndarray,
    organ_mask: np.ndarray | None = None,
) -> tuple[str, ...]:
    pred = np.asarray(prediction).astype(bool)
    tgt = np.asarray(target).astype(bool)
    if pred.shape != tgt.shape:
        raise ValueError("prediction and target shapes must match")
    failures: list[str] = []
    if bool(np.logical_and(tgt, ~pred).any()):
        failures.append("missed_target")
    if organ_mask is not None and bool(np.logical_and(pred, ~np.asarray(organ_mask).astype(bool)).any()):
        failures.append("leakage_outside_organ")
    if bool(np.logical_and(pred, ~tgt).any()):
        failures.append("false_positive")
    return tuple(failures or ["none"])


def summarize_ablation_results(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    materialized = list(rows)
    return {
        "ablation_count": len({row["ablation"] for row in materialized}) if materialized else 0,
        "case_count": len(materialized),
        "rows": materialized,
    }
