"""Gated throughput harness for nnInteractive interaction sessions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import NnInteractiveSession, VoxelCoord, as_voxel_coord
from .real_adapter import find_nibabel_test_image, load_nifti_image


@dataclass(frozen=True)
class ThroughputResult:
    env_steps: int
    elapsed_sec: float
    env_steps_per_sec: float
    warmup_steps: int
    mask_sum: int
    image_shape: tuple[int, int, int, int]
    target_shape: tuple[int, int, int]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "env_steps": self.env_steps,
            "elapsed_sec": self.elapsed_sec,
            "env_steps_per_sec": self.env_steps_per_sec,
            "warmup_steps": self.warmup_steps,
            "mask_sum": self.mask_sum,
            "image_shape": list(self.image_shape),
            "target_shape": list(self.target_shape),
            "timed_operation": "add_point_interaction_only",
            "set_image_timed": False,
            "target_buffer_read_timed": False,
        }


def measure_point_throughput(
    session: NnInteractiveSession,
    *,
    image: Any,
    points: Sequence[Sequence[int]],
    iterations: int,
    warmup_steps: int = 1,
) -> ThroughputResult:
    """Measure point-interaction calls per second for one session."""

    image_array = _as_image4d(image)
    target_shape = image_array.shape[1:]
    checked_points = tuple(_checked_point(point, target_shape) for point in points)
    if not checked_points:
        raise ValueError("points must include at least one coordinate")
    iterations = _as_nonnegative_int(iterations, name="iterations")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    warmup_steps = _as_nonnegative_int(warmup_steps, name="warmup_steps")

    session.set_image(image_array)
    session.set_target_buffer(np.zeros(target_shape, dtype=np.uint8))

    for step in range(warmup_steps):
        session.add_point_interaction(
            checked_points[step % len(checked_points)],
            include_interaction=True,
        )

    started = time.perf_counter()
    for step in range(iterations):
        session.add_point_interaction(
            checked_points[step % len(checked_points)],
            include_interaction=True,
        )
    elapsed = time.perf_counter() - started
    if elapsed <= 0:
        raise RuntimeError("throughput timer did not advance")

    mask = _target_buffer_numpy(session.target_buffer)
    if mask.shape != target_shape:
        raise RuntimeError(f"target_buffer shape {mask.shape} != {target_shape}")
    return ThroughputResult(
        env_steps=iterations,
        elapsed_sec=float(elapsed),
        env_steps_per_sec=float(iterations / elapsed),
        warmup_steps=warmup_steps,
        mask_sum=int(mask.sum()),
        image_shape=tuple(int(value) for value in image_array.shape),
        target_shape=tuple(int(value) for value in target_shape),
    )


def run_remote_point_throughput(
    *,
    server_url: str,
    image: Any,
    points: Sequence[Sequence[int]],
    iterations: int,
    warmup_steps: int,
    parallel_sessions: int = 1,
    server_max_sessions: int | None = None,
    image_label: str | None = None,
    api_key: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Run throughput against an existing nnInteractive remote server."""

    try:
        from nnInteractive.inference.remote.remote_session import (
            nnInteractiveRemoteInferenceSession,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("nnInteractive remote client is unavailable; run `make setup-real`.") from exc

    parallel_sessions = _as_positive_int(parallel_sessions, name="parallel_sessions")
    if server_max_sessions is not None:
        server_max_sessions = _as_positive_int(server_max_sessions, name="server_max_sessions")
        if parallel_sessions > server_max_sessions:
            raise ValueError(
                "parallel_sessions exceeds server_max_sessions; start "
                "nninteractive-server with --max-sessions >= --parallel-sessions"
            )

    def run_one(index: int) -> dict[str, object]:
        session = nnInteractiveRemoteInferenceSession(server_url=server_url, api_key=api_key)
        result = measure_point_throughput(
            session,
            image=image,
            points=points,
            iterations=iterations,
            warmup_steps=warmup_steps,
        )
        payload = result.to_json_dict()
        payload["session_index"] = index
        return payload

    started = time.perf_counter()
    if parallel_sessions == 1:
        session_results = [run_one(0)]
    else:
        with ThreadPoolExecutor(max_workers=parallel_sessions) as executor:
            session_results = list(executor.map(run_one, range(parallel_sessions)))
    wall_elapsed = time.perf_counter() - started
    if wall_elapsed <= 0:
        raise RuntimeError("throughput wall timer did not advance")
    total_steps = sum(int(item["env_steps"]) for item in session_results)
    first = session_results[0]
    summary: dict[str, object] = {
        "status": "remote throughput complete",
        "claim": (
            "throughput harness result only; verify server device separately; "
            "toy/single-session results are not rental-ready training budgets"
        ),
        "server_url": server_url,
        "image_label": image_label or "unspecified",
        "parallel_sessions_requested": parallel_sessions,
        "server_max_sessions": server_max_sessions,
        "parallel_sessions_completed": len(session_results),
        "aggregate_env_steps": total_steps,
        "aggregate_elapsed_sec": float(wall_elapsed),
        "aggregate_env_steps_per_sec": float(total_steps / wall_elapsed),
        "env_steps": int(first["env_steps"]),
        "elapsed_sec": float(first["elapsed_sec"]),
        "env_steps_per_sec": float(first["env_steps_per_sec"]),
        "warmup_steps": int(first["warmup_steps"]),
        "mask_sum": int(first["mask_sum"]),
        "image_shape": first["image_shape"],
        "target_shape": first["target_shape"],
        "timed_operation": first["timed_operation"],
        "set_image_timed": first["set_image_timed"],
        "target_buffer_read_timed": first["target_buffer_read_timed"],
        "session_results": session_results,
        "measurement_limitations": [
            "timer excludes set_image and initial set_target_buffer",
            "aggregate_env_steps_per_sec includes per-session set_image and warmup wall time; per-session env_steps_per_sec does not",
            "timer excludes metric/candidate generation done by the Gym env",
            "toy fixture measurements must not be extrapolated to tumor volumes without a real-sized-volume run",
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure remote nnInteractive env-step throughput.")
    parser.add_argument(
        "--require-remote",
        action="store_true",
        help="Acknowledge that an existing nninteractive-server is required.",
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("NNINTERACTIVE_SERVER_URL", "http://127.0.0.1:1527"),
    )
    parser.add_argument("--api-key", default=os.environ.get("NN_INTERACTIVE_API_KEY"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--use-nibabel-test-image", action="store_true")
    parser.add_argument("--point", action="append", help="z,y,x point; may be repeated.")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--parallel-sessions", type=int, default=1)
    parser.add_argument("--server-max-sessions", type=int)
    parser.add_argument("--label")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl_nninteractive/throughput_remote"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.require_remote and os.environ.get("RL_NNINTERACTIVE_REQUIRE_REMOTE") != "1":
        parser.error("pass --require-remote or set RL_NNINTERACTIVE_REQUIRE_REMOTE=1")
    try:
        image = _resolve_image(args)
        target_shape = image.shape[1:]
        points = [parse_point(text) for text in args.point] if args.point else _default_points(target_shape)
    except ValueError as exc:
        parser.error(str(exc))
    summary = run_remote_point_throughput(
        server_url=args.server_url,
        api_key=args.api_key,
        image=image,
        points=points,
        iterations=args.iterations,
        warmup_steps=args.warmup_steps,
        parallel_sessions=args.parallel_sessions,
        server_max_sessions=args.server_max_sessions,
        image_label=args.label or _image_label(args),
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


def parse_point(text: str) -> VoxelCoord:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("point must be formatted as z,y,x")
    return as_voxel_coord(tuple(int(part) for part in parts))


def _resolve_image(args: argparse.Namespace) -> np.ndarray:
    if args.image and args.use_nibabel_test_image:
        raise ValueError("use either --image or --use-nibabel-test-image, not both")
    if args.use_nibabel_test_image:
        return load_nifti_image(find_nibabel_test_image())[None]
    if args.image:
        return load_nifti_image(args.image)[None]
    raise ValueError("throughput requires --image or --use-nibabel-test-image")


def _image_label(args: argparse.Namespace) -> str:
    if args.use_nibabel_test_image:
        return "nibabel_anatomical_fixture"
    if args.image:
        return str(args.image)
    return "unspecified"


def _default_points(target_shape: tuple[int, int, int]) -> tuple[VoxelCoord, ...]:
    center = tuple(int(dim // 2) for dim in target_shape)
    points: list[VoxelCoord] = [as_voxel_coord(center)]
    for axis in (2, 1, 0):
        if center[axis] + 1 < target_shape[axis]:
            values = list(center)
            values[axis] += 1
            points.append(as_voxel_coord(tuple(values)))
            break
    return tuple(points)


def _as_image4d(image: Any) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError(f"image must have shape (1, z, y, x), got {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise ValueError("image contains non-finite values")
    return array.copy()


def _checked_point(point: Sequence[int], target_shape: tuple[int, int, int]) -> VoxelCoord:
    coord = as_voxel_coord(point)
    if any(coord[axis] >= target_shape[axis] for axis in range(3)):
        raise ValueError("point values must be inside image shape")
    return coord


def _target_buffer_numpy(target_buffer: Any) -> np.ndarray:
    if hasattr(target_buffer, "detach") and hasattr(target_buffer, "cpu"):
        return target_buffer.detach().cpu().numpy()
    return np.asarray(target_buffer)


def _as_nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _as_positive_int(value: int, *, name: str) -> int:
    result = _as_nonnegative_int(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
