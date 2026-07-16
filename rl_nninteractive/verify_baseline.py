"""Verification harness for the deterministic robot-user baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import VoxelCoord
from .env import POINT_NEGATIVE, POINT_POSITIVE, STOP, RlNnInteractiveEnv
from .evaluation import evaluate_interaction_trajectory
from .real_adapter import find_nibabel_test_image, load_nifti_image
from .robot_user import RobotUserEpisode, run_largest_component_robot_user


@dataclass(frozen=True)
class BaselineVerificationCase:
    name: str
    image: np.ndarray
    ground_truth: np.ndarray
    image_source: str
    ground_truth_source: str
    initial_point: VoxelCoord | None = None
    initial_include: bool = True


ACTION_NAMES = {
    STOP: "STOP",
    POINT_POSITIVE: "POINT_POSITIVE",
    POINT_NEGATIVE: "POINT_NEGATIVE",
}


def make_synthetic_tumor_cases() -> tuple[BaselineVerificationCase, ...]:
    """Return three tiny synthetic tumor-mask cases for deterministic CI."""

    single = np.zeros((3, 3, 3), dtype=bool)
    single[1, 1, 1] = True

    adjacent = np.zeros((3, 3, 3), dtype=bool)
    adjacent[1, 1, 1] = True
    adjacent[1, 1, 2] = True

    mixed = np.zeros((4, 4, 4), dtype=bool)
    mixed[2, 2, 2] = True
    mixed[2, 2, 3] = True

    return (
        BaselineVerificationCase(
            name="synthetic_single_voxel_tumor",
            image=_image_from_ground_truth(single),
            ground_truth=single,
            image_source="synthetic",
            ground_truth_source="synthetic binary tumor mask",
        ),
        BaselineVerificationCase(
            name="synthetic_adjacent_two_voxel_tumor",
            image=_image_from_ground_truth(adjacent),
            ground_truth=adjacent,
            image_source="synthetic",
            ground_truth_source="synthetic binary tumor mask",
        ),
        BaselineVerificationCase(
            name="synthetic_missed_tumor_plus_initial_false_positive",
            image=_image_from_ground_truth(mixed),
            ground_truth=mixed,
            image_source="synthetic",
            ground_truth_source="synthetic binary tumor mask",
            initial_point=(0, 0, 0),
        ),
    )


def public_image_verification_case(
    image: Any,
    *,
    image_source: str,
) -> BaselineVerificationCase:
    """Build a public-image verification case with synthetic GT.

    The image is real/public input data, but the GT is a tiny synthetic mask so
    this remains a wiring verification, not a real tumor benchmark.
    """

    image_3d = np.asarray(image, dtype=np.float32)
    if image_3d.ndim != 3:
        raise ValueError(f"public verification image must be 3D, got {image_3d.shape}")
    if not bool(np.isfinite(image_3d).all()):
        raise ValueError("public verification image contains non-finite values")
    if any(dim < 2 for dim in image_3d.shape):
        raise ValueError("public verification image dimensions must be >= 2")

    center = tuple(int(dim // 2) for dim in image_3d.shape)
    neighbor = _adjacent_coord(center, image_3d.shape)
    ground_truth = np.zeros(image_3d.shape, dtype=bool)
    ground_truth[center] = True
    ground_truth[neighbor] = True

    return BaselineVerificationCase(
        name="public_nibabel_anatomical_synthetic_gt",
        image=image_3d[None],
        ground_truth=ground_truth,
        image_source=image_source,
        ground_truth_source="synthetic center two-voxel mask for baseline verification only",
        initial_point=center,
    )


def load_public_nibabel_verification_case() -> BaselineVerificationCase:
    image_path = find_nibabel_test_image()
    image = load_nifti_image(image_path)
    return public_image_verification_case(image, image_source=str(image_path))


def run_verification_case(
    case: BaselineVerificationCase,
    *,
    max_interactions: int = 16,
) -> dict[str, Any]:
    env = RlNnInteractiveEnv(case.ground_truth.shape, max_interactions=max_interactions)
    episode = run_largest_component_robot_user(
        env,
        image=case.image,
        ground_truth=case.ground_truth,
        initial_point=case.initial_point,
        initial_include=case.initial_include,
    )
    return _episode_summary(case, episode)


def run_baseline_verification(
    *,
    include_public_nibabel: bool = False,
    output_dir: Path | None = None,
    max_interactions: int = 16,
) -> dict[str, Any]:
    cases = list(make_synthetic_tumor_cases())
    if include_public_nibabel:
        cases.append(load_public_nibabel_verification_case())

    results = [
        run_verification_case(case, max_interactions=max_interactions)
        for case in cases
    ]
    summary: dict[str, Any] = {
        "status": "baseline verification complete",
        "claim": (
            "synthetic/mock robot-user verification; public image case uses "
            "synthetic GT and is not a tumor benchmark or clinical result"
        ),
        "case_count": len(results),
        "all_cases_passed": all(result["passed"] for result in results),
        "results": results,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the largest-component robot-user baseline."
    )
    parser.add_argument(
        "--include-public-nibabel",
        action="store_true",
        help=(
            "Also run the public Nibabel anatomical image with a synthetic "
            "center GT mask. Requires `make setup-real`."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl_nninteractive/baseline_verification"),
    )
    parser.add_argument("--max-interactions", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_baseline_verification(
        include_public_nibabel=args.include_public_nibabel,
        output_dir=args.output_dir,
        max_interactions=args.max_interactions,
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_cases_passed"]:
        return 1
    return 0


def _episode_summary(
    case: BaselineVerificationCase,
    episode: RobotUserEpisode,
) -> dict[str, Any]:
    final_dice = float(episode.final_info["dice"])
    evaluation = evaluate_interaction_trajectory(
        case.name,
        episode.dice_by_step,
        final_dice=final_dice,
    )
    decisions = [
        {
            "action": ACTION_NAMES[decision.action_type],
            "coord_zyx": list(decision.coord),
            "error_kind": decision.error_kind,
            "component_size": decision.component_size,
        }
        for decision in episode.decisions
    ]
    return {
        "name": case.name,
        "image_source": case.image_source,
        "ground_truth_source": case.ground_truth_source,
        "image_shape": list(case.image.shape),
        "ground_truth_shape": list(case.ground_truth.shape),
        "initial_point": list(case.initial_point) if case.initial_point else None,
        "initial_dice": float(episode.initial_info["dice"]),
        "final_dice": final_dice,
        "dice_by_step": list(episode.dice_by_step),
        "total_reward": episode.total_reward,
        "terminated": episode.terminated,
        "truncated": episode.truncated,
        "decision_count": len(episode.decisions),
        "point_interaction_count": len(episode.dice_by_step),
        "evaluation": evaluation.to_json_dict(),
        "decisions": decisions,
        "passed": bool(final_dice == 1.0 and episode.terminated and not episode.truncated),
    }


def _image_from_ground_truth(ground_truth: np.ndarray) -> np.ndarray:
    image = np.zeros((1, *ground_truth.shape), dtype=np.float32)
    image[0, ground_truth] = 1.0
    return image


def _adjacent_coord(coord: VoxelCoord, shape: tuple[int, int, int]) -> VoxelCoord:
    for axis in (2, 1, 0):
        if coord[axis] + 1 < shape[axis]:
            values = list(coord)
            values[axis] += 1
            return (values[0], values[1], values[2])
        if coord[axis] - 1 >= 0:
            values = list(coord)
            values[axis] -= 1
            return (values[0], values[1], values[2])
    raise ValueError("could not find adjacent coordinate")


if __name__ == "__main__":
    raise SystemExit(main())
