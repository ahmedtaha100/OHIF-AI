"""Blackwell handoff runbook generator.

This module does not claim benchmark results. It records the exact local
commands and remaining GPU-gated work so the large-run phase can start from a
versioned, test-covered handoff instead of an ad hoc note.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_THROUGHPUT_SUMMARY = Path("artifacts/rl_nninteractive/throughput_remote/summary.json")
DEFAULT_PHASE1_DQN_EPISODES = 256
DEFAULT_MAX_INTERACTIONS = 5


@dataclass(frozen=True)
class RemainingRun:
    plan_item: str
    why_gpu_required: str
    commands: tuple[str, ...]
    expected_vram_gb: str
    env_count: str
    gpu_hours_estimate: str
    workload_env_steps: int | None = None

    def to_json_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "plan_item": self.plan_item,
            "why_gpu_required": self.why_gpu_required,
            "commands": list(self.commands),
            "expected_vram_gb": self.expected_vram_gb,
            "env_count": self.env_count,
            "gpu_hours_estimate": self.gpu_hours_estimate,
        }
        if self.workload_env_steps is not None:
            payload["workload_env_steps"] = self.workload_env_steps
        return payload


def build_blackwell_handoff(
    *,
    server_url: str,
    dataset_manifest: Path,
    output_dir: Path,
    model: str = "nnInteractive_v1.0",
    host: str = "127.0.0.1",
    port: int = 1527,
    device: str = "cuda:0",
    max_sessions: int = 1,
    env_count: int = 1,
    interactions_storage: str = "blosc2",
    throughput_summary: Path = DEFAULT_THROUGHPUT_SUMMARY,
    phase1_dqn_episodes: int = DEFAULT_PHASE1_DQN_EPISODES,
    max_interactions: int = DEFAULT_MAX_INTERACTIONS,
) -> dict[str, object]:
    """Build a JSON-friendly handoff for the remaining GPU-gated work."""

    measured = _load_throughput_summary(throughput_summary)
    server_command = (
        f".venv\\Scripts\\nninteractive-server.exe --model {model} "
        f"--host {host} --port {port} --device {device} "
        f"--max-sessions {max_sessions} --no-torch-compile "
        f"--interactions-storage {interactions_storage}"
    )
    throughput_command = (
        f"$env:NNINTERACTIVE_SERVER_URL='{server_url}'; "
        "make throughput-remote"
    )
    manifest = str(dataset_manifest)
    large_train_command = (
        f"$env:NNINTERACTIVE_SERVER_URL='{server_url}'; "
        "make phase1-real"
    )
    phase2_smoke_command = (
        "python -m rl_nninteractive.phase2_smoke "
        f"--output-dir {output_dir / 'phase2_4_code_smoke'}"
    )
    phase1_steps = phase1_dqn_episodes * (max_interactions + 1)
    eval_steps = 3 * 20 * 2 * (max_interactions + 1)
    ablation_steps = 4 * phase1_steps
    total_workload_steps = phase1_steps + eval_steps + ablation_steps

    remaining = (
        RemainingRun(
            plan_item="Phase 0 throughput harness vs the remote inference server",
            why_gpu_required=(
                "Needs a CUDA-backed nninteractive-server and must be repeated "
                "on a real-sized public/de-identified volume before its number "
                "is used for rental budgeting."
            ),
            commands=(server_command, throughput_command),
            expected_vram_gb="6-10 GB for one nnInteractive model plus session buffers",
            env_count=f"{max_sessions} validated session(s); increase only after throughput-remote N>1 passes",
            gpu_hours_estimate=_estimate_gpu_hours(8, measured),
            workload_env_steps=8,
        ),
        RemainingRun(
            plan_item="Large-scale RL training run",
            why_gpu_required=(
                "A useful run needs many remote env steps over public GT cases. "
                "Small synthetic BC/DQN proof is complete, but large online "
                "runs must stay capped until real-volume throughput is measured."
            ),
            commands=(
                server_command,
                f"# Use public/de-identified dataset manifest: {manifest}",
                large_train_command,
            ),
            expected_vram_gb="12-16 GB minimum; 24+ GB preferred for parallel env sessions",
            env_count=f"{env_count} requested env session(s), capped by validated server throughput",
            gpu_hours_estimate=_estimate_gpu_hours(phase1_steps, measured),
            workload_env_steps=phase1_steps,
        ),
        RemainingRun(
            plan_item="Phase 4 multi-tumor evaluation vs all baselines",
            why_gpu_required=(
                "Requires >=3 public/de-identified datasets/modalities, repeated "
                "nnInteractive inference, paired baseline runs, and real GT masks. "
                "Mock/public-fixture tests cannot be counted as results."
            ),
            commands=(
                server_command,
                f"# Populate {manifest} with public cases, split ids, spacing, and checksums.",
                "make phase1-real",
                phase2_smoke_command,
                "# Then run the real evaluation driver once dataset adapters are filled.",
            ),
            expected_vram_gb="16-24 GB for 4-8 remote sessions; 32+ GB preferred",
            env_count=f"{env_count} requested eval session(s), only after N>1 validation",
            gpu_hours_estimate=_estimate_gpu_hours(eval_steps, measured),
            workload_env_steps=eval_steps,
        ),
        RemainingRun(
            plan_item="Phase 4 ablation sweeps",
            why_gpu_required=(
                "Ablations require multiple full training/evaluation passes "
                "over the same fixed splits: entropy on/off, point-only vs "
                "multi-tool, safety reward on/off, and STOP calibration."
            ),
            commands=(
                server_command,
                "python -m rl_nninteractive.phase2_smoke "
                f"--output-dir {output_dir / 'ablation_code_smoke'}",
                "# Launch full ablations from the same real manifest after the first large run passes.",
            ),
            expected_vram_gb="24+ GB preferred; larger GPUs reduce wall-clock by higher session count",
            env_count=f"{env_count} requested session(s), not assumed",
            gpu_hours_estimate=_estimate_gpu_hours(ablation_steps, measured),
            workload_env_steps=ablation_steps,
        ),
        RemainingRun(
            plan_item="Phase 4 failure taxonomy, safety analysis, reproducibility, and write-up",
            why_gpu_required=(
                "These depend on the real multi-dataset outputs and cannot be "
                "completed honestly from synthetic/mock artifacts."
            ),
            commands=(
                "# After large eval: aggregate failure labels with rl_nninteractive.eval_harness.",
                "# Then freeze seeds/configs/manifests and write the final REAL results document.",
            ),
            expected_vram_gb="No new VRAM beyond the completed large eval outputs",
            env_count="N/A after result artifacts exist",
            gpu_hours_estimate=_estimate_gpu_hours(0, measured),
            workload_env_steps=0,
        ),
    )
    return {
        "status": "blackwell handoff ready",
        "claim": "runbook only; no large-scale result is claimed",
        "server_url": server_url,
        "dataset_manifest": manifest,
        "output_dir": str(output_dir),
        "server_start_command": server_command,
        "throughput_summary": _throughput_summary_metadata(measured, throughput_summary),
        "remaining_runs": [item.to_json_dict() for item in remaining],
        "total_gpu_hours_estimate": _estimate_gpu_hours(total_workload_steps, measured),
    }


def _load_throughput_summary(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"throughput summary must be a JSON object: {path}")
    return payload


def _measured_steps_per_sec(summary: dict[str, object] | None) -> float | None:
    if summary is None:
        return None
    value = summary.get("aggregate_env_steps_per_sec", summary.get("env_steps_per_sec"))
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _estimate_gpu_hours(workload_env_steps: int, summary: dict[str, object] | None) -> str:
    steps_per_sec = _measured_steps_per_sec(summary)
    if workload_env_steps == 0:
        return "0 GPU-hours unless audit reruns are required"
    if steps_per_sec is None:
        return "unestimated; run make throughput-remote on CUDA first"
    hours = workload_env_steps / (steps_per_sec * 3600.0)
    estimate = "<0.001" if hours < 0.001 else f">= {hours:.3f}"
    return (
        f"{estimate} GPU-hours at measured {steps_per_sec:.3f} "
        "toy point-steps/sec; not rental-ready until real-sized-volume and N>1 "
        "parallel-session throughput are measured"
    )


def _throughput_summary_metadata(
    summary: dict[str, object] | None,
    path: Path,
) -> dict[str, object]:
    if summary is None:
        return {"path": str(path), "status": "missing"}
    return {
        "path": str(path),
        "status": "loaded",
        "image_shape": summary.get("image_shape"),
        "image_label": summary.get("image_label"),
        "parallel_sessions_completed": summary.get("parallel_sessions_completed", 1),
        "aggregate_env_steps_per_sec": summary.get(
            "aggregate_env_steps_per_sec",
            summary.get("env_steps_per_sec"),
        ),
        "limitations": summary.get("measurement_limitations", []),
    }


def write_handoff_files(handoff: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blackwell_handoff.json").write_text(
        json.dumps(handoff, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "blackwell_handoff.md").write_text(
        _to_markdown(handoff),
        encoding="utf-8",
    )


def _to_markdown(handoff: dict[str, object]) -> str:
    lines = [
        "# Blackwell Handoff",
        "",
        f"Status: {handoff['status']}",
        f"Claim: {handoff['claim']}",
        (
            "Safety: not rental-ready until real-sized-volume throughput and "
            "parallel-session scaling are measured"
        ),
        f"Server URL: `{handoff['server_url']}`",
        f"Dataset manifest: `{handoff['dataset_manifest']}`",
        f"Total estimate: {handoff['total_gpu_hours_estimate']}",
        "",
        "## Server",
        "",
        f"```powershell\n{handoff['server_start_command']}\n```",
    ]
    for item in handoff["remaining_runs"]:  # type: ignore[index]
        lines.extend(
            [
                "",
                f"## {item['plan_item']}",
                "",
                f"Why GPU: {item['why_gpu_required']}",
                f"Expected VRAM: {item['expected_vram_gb']}",
                f"Parallelism: {item['env_count']}",
                f"Estimate: {item['gpu_hours_estimate']}",
                "",
                "Commands:",
                "",
            ]
        )
        for command in item["commands"]:
            lines.append(f"```powershell\n{command}\n```")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:1527")
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("manifests/blackwell_datasets.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl_nninteractive/blackwell_handoff"),
    )
    parser.add_argument("--model", default="nnInteractive_v1.0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1527)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--env-count", type=int, default=1)
    parser.add_argument("--interactions-storage", default="blosc2")
    parser.add_argument("--throughput-summary", type=Path, default=DEFAULT_THROUGHPUT_SUMMARY)
    parser.add_argument("--phase1-dqn-episodes", type=int, default=DEFAULT_PHASE1_DQN_EPISODES)
    parser.add_argument("--max-interactions", type=int, default=DEFAULT_MAX_INTERACTIONS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    handoff = build_blackwell_handoff(
        server_url=args.server_url,
        dataset_manifest=args.dataset_manifest,
        output_dir=args.output_dir,
        model=args.model,
        host=args.host,
        port=args.port,
        device=args.device,
        max_sessions=args.max_sessions,
        env_count=args.env_count,
        interactions_storage=args.interactions_storage,
        throughput_summary=args.throughput_summary,
        phase1_dqn_episodes=args.phase1_dqn_episodes,
        max_interactions=args.max_interactions,
    )
    write_handoff_files(handoff, args.output_dir)
    print(json.dumps(handoff, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
