"""Phase 2-4 local code-surface smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .deterministic_geometry import build_component_geometry, largest_error_component_mask
from .eval_harness import classify_failure, default_ablation_grid, summarize_ablation_results
from .interaction_log import InteractionLogEvent
from .learning_loops import (
    calibrate_stop_threshold,
    dagger_samples_from_logs,
    preference_pairs_from_logs,
)
from .multitool import dispatch_multi_tool_action, multi_tool_candidates
from .recommender import recommend_next_prompt
from .safety_reward import safety_shaped_reward
from .state_encoder import encode_state_channels
from .toy_dataset import synthetic_toy_cases
from .uncertainty import append_uncertainty_channel, tta_disagreement_channel
from .mock_adapter import MockNnInteractiveSession


def run_phase2_smoke(*, output_dir: Path) -> dict[str, Any]:
    case = synthetic_toy_cases("val")[0]
    current = np.zeros_like(case.ground_truth)
    component = largest_error_component_mask(current, case.ground_truth, polarity="positive")
    geometries = [
        build_component_geometry(component, tool=tool, polarity="positive").to_json_dict()
        for tool in ("point", "scribble", "lasso", "box")
    ]
    encoded = encode_state_channels(image=case.image, current_mask=current)
    disagreement = tta_disagreement_channel(
        np.stack([current, case.ground_truth, np.logical_or(current, case.ground_truth)])
    )
    state_with_uncertainty = append_uncertainty_channel(encoded.channels, disagreement)

    actions = multi_tool_candidates(current, case.ground_truth)
    session = MockNnInteractiveSession()
    session.set_image(case.image)
    session.set_target_buffer(np.zeros_like(case.ground_truth, dtype=np.uint8))
    dispatched = dispatch_multi_tool_action(session, actions[0])
    reward = safety_shaped_reward(
        previous_mask=current,
        current_mask=session.target_buffer,
        ground_truth=case.ground_truth,
        organ_mask=np.ones_like(case.ground_truth),
    )
    suggestion = recommend_next_prompt(
        case_id=case.name,
        current_mask=current,
        ground_truth_for_mock=case.ground_truth,
    )
    log_event = InteractionLogEvent(
        case_id=case.name,
        step_index=0,
        tool=suggestion.action.tool,
        decision="accepted",
        proposed_prompt=suggestion.to_json_dict(),
        final_prompt=None,
        elapsed_ms=125,
        organ="synthetic",
        target="synthetic",
    )
    dagger = dagger_samples_from_logs([log_event])
    preferences = preference_pairs_from_logs([log_event])
    stop_threshold = calibrate_stop_threshold([0.88, 0.91, 0.95])
    failures = classify_failure(session.target_buffer, case.ground_truth, np.ones_like(case.ground_truth))
    ablations = default_ablation_grid()
    ablation_summary = summarize_ablation_results(
        {
            "ablation": ablation.name,
            "case": case.name,
            "failure": failures[0],
        }
        for ablation in ablations
    )

    result = {
        "status": "phase2-4 code smoke complete",
        "claim": "synthetic/mock code-surface smoke only; not a real benchmark or reader study",
        "case": case.name,
        "geometry_tools": [geometry["tool"] for geometry in geometries],
        "candidate_count": len(actions),
        "dispatched_changed_bbox": None if dispatched is None else dispatched.changed_bbox,
        "state_channel_count_with_uncertainty": int(state_with_uncertainty.shape[0]),
        "reward": reward.to_json_dict(),
        "suggestion": suggestion.to_json_dict(),
        "log_event": log_event.to_json_dict(),
        "dagger_sample_count": len(dagger),
        "preference_pair_count": len(preferences),
        "stop_threshold": stop_threshold,
        "failure_taxonomy": failures,
        "ablation_grid": [ablation.to_json_dict() for ablation in ablations],
        "ablation_summary": ablation_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase2_smoke_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl_nninteractive/phase2_smoke"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run_phase2_smoke(output_dir=args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
