"""Manifest-driven Phase 1 point-policy runner for remote nnInteractive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset_manifest import load_manifest_cases
from .evaluation import evaluate_interaction_trajectory, summarize_interaction_evaluations
from .point_policy import (
    collect_behavior_cloning_samples,
    fine_tune_dqn,
    rollout_point_policy,
    rollout_robot_user,
    train_behavior_cloning,
)

DEFAULT_DQN_EPISODES = 256
DEFAULT_MAX_REMOTE_ENV_STEPS = 10_000


def run_phase1_real(
    *,
    dataset_manifest: Path,
    server_url: str,
    output_dir: Path,
    max_interactions: int = 5,
    top_k: int = 3,
    bc_epochs: int = 12,
    dqn_episodes: int = DEFAULT_DQN_EPISODES,
    max_remote_env_steps: int = DEFAULT_MAX_REMOTE_ENV_STEPS,
    allow_large_run: bool = False,
    seed: int = 20260705,
    api_key: str | None = None,
    dry_run_manifest: bool = False,
) -> dict[str, Any]:
    train_cases = load_manifest_cases(dataset_manifest, split="train")
    validation_cases = load_manifest_cases(dataset_manifest, split="val")
    if not train_cases:
        raise ValueError("dataset manifest must contain at least one train case")
    if not validation_cases:
        raise ValueError("dataset manifest must contain at least one val case")
    estimated_remote_env_steps = _estimate_remote_env_steps(
        train_case_count=len(train_cases),
        validation_case_count=len(validation_cases),
        max_interactions=max_interactions,
        dqn_episodes=dqn_episodes,
    )

    result: dict[str, Any] = {
        "status": "phase1 real manifest validated"
        if dry_run_manifest
        else "phase1 remote point-policy run complete",
        "claim": (
            "large-run output only counts as a real result if the manifest "
            "contains public/de-identified data with valid provenance and GT"
        ),
        "dataset_manifest": str(dataset_manifest),
        "server_url": server_url,
        "train_case_count": len(train_cases),
        "validation_case_count": len(validation_cases),
        "max_interactions": max_interactions,
        "top_k": top_k,
        "bc_epochs": bc_epochs,
        "dqn_episodes": dqn_episodes,
        "estimated_remote_env_steps_upper_bound": estimated_remote_env_steps,
        "max_remote_env_steps": max_remote_env_steps,
        "allow_large_run": allow_large_run,
        "seed": seed,
    }
    if dry_run_manifest:
        _write_result(result, output_dir)
        return result
    _enforce_remote_budget(
        estimated_remote_env_steps=estimated_remote_env_steps,
        max_remote_env_steps=max_remote_env_steps,
        allow_large_run=allow_large_run,
    )

    session_factory = _remote_session_factory(server_url=server_url, api_key=api_key)
    samples = collect_behavior_cloning_samples(
        train_cases,
        max_interactions=max_interactions,
        top_k=top_k,
        session_factory=session_factory,
    )
    bc_policy = train_behavior_cloning(samples, epochs=bc_epochs)
    policy = fine_tune_dqn(
        bc_policy,
        train_cases,
        episodes=dqn_episodes,
        max_interactions=max_interactions,
        top_k=top_k,
        seed=seed,
        session_factory=session_factory,
    )

    heuristic_episodes = [
        rollout_robot_user(
            case,
            max_interactions=max_interactions,
            session_factory=session_factory,
        )
        for case in validation_cases
    ]
    policy_episodes = [
        rollout_point_policy(
            policy,
            case,
            max_interactions=max_interactions,
            top_k=top_k,
            session_factory=session_factory,
        )
        for case in validation_cases
    ]
    result.update(
        {
            "behavior_cloning_sample_count": len(samples),
            "heuristic_summary": summarize_interaction_evaluations(
                [
                    evaluate_interaction_trajectory(
                        episode.case_name,
                        episode.dice_by_step,
                        final_dice=episode.final_dice,
                    )
                    for episode in heuristic_episodes
                ]
            ),
            "policy_summary": summarize_interaction_evaluations(
                [
                    evaluate_interaction_trajectory(
                        episode.case_name,
                        episode.dice_by_step,
                        final_dice=episode.final_dice,
                    )
                    for episode in policy_episodes
                ]
            ),
        }
    )
    _write_result(result, output_dir)
    return result


def _estimate_remote_env_steps(
    *,
    train_case_count: int,
    validation_case_count: int,
    max_interactions: int,
    dqn_episodes: int,
) -> int:
    """Upper-bound remote point calls for the current point-only runner."""

    if max_interactions < 1:
        raise ValueError("max_interactions must be >= 1")
    if dqn_episodes < 0:
        raise ValueError("dqn_episodes must be >= 0")
    bc_steps = train_case_count * (max_interactions + 1)
    dqn_steps = dqn_episodes * (max_interactions + 1)
    validation_steps = validation_case_count * 2 * (max_interactions + 1)
    return int(bc_steps + dqn_steps + validation_steps)


def _enforce_remote_budget(
    *,
    estimated_remote_env_steps: int,
    max_remote_env_steps: int,
    allow_large_run: bool,
) -> None:
    if max_remote_env_steps < 1:
        raise ValueError("max_remote_env_steps must be >= 1")
    if estimated_remote_env_steps > max_remote_env_steps and not allow_large_run:
        raise ValueError(
            "estimated remote env steps "
            f"({estimated_remote_env_steps}) exceed --max-remote-env-steps "
            f"({max_remote_env_steps}); pass --allow-large-run only after "
            "real-sized-volume and parallel-session throughput are measured"
        )


def _remote_session_factory(*, server_url: str, api_key: str | None):
    try:
        from nnInteractive.inference.remote.remote_session import (
            nnInteractiveRemoteInferenceSession,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("nnInteractive remote client is unavailable; run `make setup-real`.") from exc

    def factory():
        return nnInteractiveRemoteInferenceSession(server_url=server_url, api_key=api_key)

    return factory


def _write_result(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase1_real_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:1527")
    parser.add_argument("--api-key")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl_nninteractive/phase1_real"),
    )
    parser.add_argument("--max-interactions", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--dqn-episodes", type=int, default=DEFAULT_DQN_EPISODES)
    parser.add_argument("--max-remote-env-steps", type=int, default=DEFAULT_MAX_REMOTE_ENV_STEPS)
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--require-remote", action="store_true")
    parser.add_argument("--dry-run-manifest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.require_remote and not args.dry_run_manifest:
        parser.error("pass --require-remote for real server execution or --dry-run-manifest")
    result = run_phase1_real(
        dataset_manifest=args.dataset_manifest,
        server_url=args.server_url,
        api_key=args.api_key,
        output_dir=args.output_dir,
        max_interactions=args.max_interactions,
        top_k=args.top_k,
        bc_epochs=args.bc_epochs,
        dqn_episodes=args.dqn_episodes,
        max_remote_env_steps=args.max_remote_env_steps,
        allow_large_run=args.allow_large_run,
        seed=args.seed,
        dry_run_manifest=args.dry_run_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
