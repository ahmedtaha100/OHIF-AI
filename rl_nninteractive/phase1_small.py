"""Small-scale Phase 1 point-policy proof-of-pipeline runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import evaluate_interaction_trajectory, summarize_interaction_evaluations
from .point_policy import (
    collect_behavior_cloning_samples,
    fine_tune_dqn,
    rollout_point_policy,
    rollout_robot_user,
    train_behavior_cloning,
)
from .state_encoder import encode_state_channels
from .toy_dataset import synthetic_toy_cases


def run_phase1_small(
    *,
    output_dir: Path,
    max_interactions: int = 5,
    top_k: int = 3,
    bc_epochs: int = 12,
    dqn_episodes: int = 24,
    seed: int = 20260705,
) -> dict[str, Any]:
    """Run the local synthetic Phase 1 proof and write JSON artifacts."""

    train_cases = synthetic_toy_cases("train")
    val_cases = synthetic_toy_cases("val")
    state_encoder_summary = _state_encoder_summary(
        [*train_cases, *val_cases],
        max_interactions=max_interactions,
    )
    samples = collect_behavior_cloning_samples(
        train_cases,
        max_interactions=max_interactions,
        top_k=top_k,
    )
    bc_policy = train_behavior_cloning(samples, epochs=bc_epochs)
    policy = fine_tune_dqn(
        bc_policy,
        train_cases,
        episodes=dqn_episodes,
        max_interactions=max_interactions,
        top_k=top_k,
        seed=seed,
    )

    heuristic_episodes = [
        rollout_robot_user(case, max_interactions=max_interactions) for case in val_cases
    ]
    policy_episodes = [
        rollout_point_policy(
            policy,
            case,
            max_interactions=max_interactions,
            top_k=top_k,
        )
        for case in val_cases
    ]
    heuristic_evaluations = [
        episode.evaluation_row() for episode in heuristic_episodes
    ]
    policy_evaluations = [episode.evaluation_row() for episode in policy_episodes]
    comparison_rows = _comparison_rows(heuristic_evaluations, policy_evaluations)
    heuristic_summary = summarize_interaction_evaluations(
        _evaluation_objects(heuristic_episodes)
    )
    policy_summary = summarize_interaction_evaluations(_evaluation_objects(policy_episodes))
    noc_comparison = {
        "noc_at_85": _noc_comparison(policy_summary, heuristic_summary, "noc_at_85"),
        "noc_at_90": _noc_comparison(policy_summary, heuristic_summary, "noc_at_90"),
    }
    beats_heuristic = all(
        comparison["comparable"] and comparison["policy_mean"] < comparison["heuristic_mean"]
        for comparison in noc_comparison.values()
    )

    result: dict[str, Any] = {
        "status": "phase1 small-scale proof complete",
        "claim": (
            "synthetic/mock proof-of-pipeline only; not a real benchmark, "
            "training claim, or clinical result"
        ),
        "dataset": "synthetic_toy_v1",
        "train_case_count": len(train_cases),
        "validation_case_count": len(val_cases),
        "behavior_cloning_sample_count": len(samples),
        "state_encoder": state_encoder_summary,
        "max_interactions": max_interactions,
        "top_k": top_k,
        "bc_epochs": bc_epochs,
        "dqn_episodes": dqn_episodes,
        "seed": seed,
        "heuristic_summary": heuristic_summary,
        "policy_summary": policy_summary,
        "comparison_rows": comparison_rows,
        "noc_comparison": noc_comparison,
        "go_no_go": {
            "beats_heuristic_on_mean_noc85_and_noc90": beats_heuristic,
            "decision": "continue_point_policy"
            if beats_heuristic
            else "pivot_required_keep_tool_select_and_stop_work_active",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase1_small_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _state_encoder_summary(cases: list[Any], *, max_interactions: int) -> dict[str, Any]:
    channel_names: tuple[str, ...] | None = None
    for case in cases:
        encoded = encode_state_channels(
            image=case.image,
            current_mask=np.zeros_like(case.ground_truth),
            step_index=0,
            max_steps=max_interactions,
        )
        if channel_names is None:
            channel_names = encoded.channel_names
        elif channel_names != encoded.channel_names:
            raise RuntimeError("state encoder channel names changed across cases")
    return {
        "case_count": len(cases),
        "channel_count": len(channel_names or ()),
        "channel_names": list(channel_names or ()),
    }


def _evaluation_objects(episodes: list[Any]) -> list[Any]:
    return [
        evaluate_interaction_trajectory(
            episode.case_name,
            episode.dice_by_step,
            final_dice=episode.final_dice,
        )
        for episode in episodes
    ]


def _comparison_rows(
    heuristic_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for heuristic, policy in zip(heuristic_rows, policy_rows, strict=True):
        rows.append(
            {
                "case": heuristic["name"],
                "heuristic_noc_at_85": heuristic["noc_at_85"],
                "heuristic_noc_at_90": heuristic["noc_at_90"],
                "heuristic_final_dice": heuristic["final_dice"],
                "policy_noc_at_85": policy["noc_at_85"],
                "policy_noc_at_90": policy["noc_at_90"],
                "policy_final_dice": policy["final_dice"],
            }
        )
    return rows


def _noc_comparison(
    policy_summary: dict[str, Any],
    heuristic_summary: dict[str, Any],
    field_name: str,
) -> dict[str, Any]:
    policy_values = _noc_values(policy_summary, field_name)
    heuristic_values = _noc_values(heuristic_summary, field_name)
    comparable = len(policy_values) == len(heuristic_values) and bool(policy_values)
    return {
        "comparable": comparable,
        "policy_mean": float(sum(policy_values) / len(policy_values)) if comparable else None,
        "heuristic_mean": float(sum(heuristic_values) / len(heuristic_values))
        if comparable
        else None,
        "policy_values": policy_values,
        "heuristic_values": heuristic_values,
    }


def _noc_values(summary: dict[str, Any], field_name: str) -> list[int]:
    values: list[int] = []
    for row in summary["rows"]:
        value = row[field_name]
        if value is None:
            return []
        values.append(int(value))
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl_nninteractive/phase1_small"),
    )
    parser.add_argument("--max-interactions", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--dqn-episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_phase1_small(
        output_dir=args.output_dir,
        max_interactions=args.max_interactions,
        top_k=args.top_k,
        bc_epochs=args.bc_epochs,
        dqn_episodes=args.dqn_episodes,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
