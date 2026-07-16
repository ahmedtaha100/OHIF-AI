#!/usr/bin/env python
"""Freeze AutoPET candidate masks/manifests, then score the sealed test on request."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from rl_nninteractive.metrics import dice_score, hd95, normalized_surface_dice


ACTIONS = (
    "prompt_r1_replace",
    "prompt_r2_replace",
    "resenc_intersection_r1",
    "resenc_intersection_r2",
    "resenc_union_r1",
    "resenc_union_r2",
)


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, cache: dict[Path, str] | None = None) -> str:
    resolved = path.resolve()
    if cache is not None and resolved in cache:
        return cache[resolved]
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    result = digest.hexdigest()
    if cache is not None:
        cache[resolved] = result
    return result


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def load_mask(path: Path, reference: nib.Nifti1Image | None = None) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj) > 0
    if reference is not None:
        if image.shape != reference.shape:
            raise ValueError(f"shape mismatch: {path}: {image.shape} != {reference.shape}")
        if not np.allclose(image.affine, reference.affine, atol=1e-4):
            raise ValueError(f"affine mismatch: {path}")
    return image, data


def save_mask(mask: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header), str(path))


def candidate_paths(root: Path, case_id: str) -> dict[str, Path]:
    case_root = root / "cases" / case_id
    return {
        "keep_resenc": root / "batches" / "resenc" / "prediction" / f"{case_id}.nii.gz",
        "prompt_r1_replace": root / "batches" / "round1" / "prediction" / f"{case_id}.nii.gz",
        "prompt_r2_replace": root / "batches" / "round2" / "prediction" / f"{case_id}.nii.gz",
        "resenc_intersection_r1": case_root / "candidates" / "resenc_intersection_r1.nii.gz",
        "resenc_intersection_r2": case_root / "candidates" / "resenc_intersection_r2.nii.gz",
        "resenc_union_r1": case_root / "candidates" / "resenc_union_r1.nii.gz",
        "resenc_union_r2": case_root / "candidates" / "resenc_union_r2.nii.gz",
    }


def generate_fusions(root: Path, case_id: str) -> dict[str, Path]:
    paths = candidate_paths(root, case_id)
    reference, resenc = load_mask(paths["keep_resenc"])
    _, round1 = load_mask(paths["prompt_r1_replace"], reference)
    _, round2 = load_mask(paths["prompt_r2_replace"], reference)
    generated = {
        "resenc_intersection_r1": resenc & round1,
        "resenc_intersection_r2": resenc & round2,
        "resenc_union_r1": resenc | round1,
        "resenc_union_r2": resenc | round2,
    }
    for action, mask in generated.items():
        save_mask(mask, reference, paths[action])
        load_mask(paths[action], reference)
    return paths


def safe_metric(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def mask_metrics(prediction: np.ndarray, ground_truth: np.ndarray, spacing: tuple[float, ...]) -> dict[str, Any]:
    voxel_ml = float(np.prod(spacing) / 1000.0)
    false_positive = prediction & ~ground_truth
    false_negative = ground_truth & ~prediction
    return {
        "dice": float(dice_score(prediction, ground_truth)),
        "nsd_2mm": float(normalized_surface_dice(prediction, ground_truth, tolerance=2.0, spacing=spacing)),
        "hd95_mm": safe_metric(hd95(prediction, ground_truth, spacing=spacing)),
        "positive_voxels": int(prediction.sum()),
        "fpv_ml": float(false_positive.sum() * voxel_ml),
        "fnv_ml": float(false_negative.sum() * voxel_ml),
    }


def score_case(root: Path, study: dict[str, Any]) -> dict[str, Any]:
    case_id = study["case_id"]
    paths = candidate_paths(root, case_id)
    ground_truth_path = root / "cases" / case_id / "prepared" / "labels" / f"{case_id}.nii.gz"
    ground_truth_image, ground_truth = load_mask(ground_truth_path)
    spacing = tuple(float(value) for value in ground_truth_image.header.get_zooms()[:3])
    metrics = {}
    for action, path in paths.items():
        _, prediction = load_mask(path, ground_truth_image)
        metrics[action] = mask_metrics(prediction, ground_truth, spacing)
    oracle_action = max(metrics, key=lambda action: metrics[action]["dice"])
    return {
        "case_id": case_id,
        "patient_id": study["patient_id"],
        "tracer": study["tracer"],
        "split": study["split"],
        "prior_exposed": True,
        "metrics": metrics,
        "hindsight_oracle_action": oracle_action,
        "hindsight_oracle_dice": metrics[oracle_action]["dice"],
    }


def prompt_metadata(root: Path, case_id: str, round_index: int, cache: dict[Path, str]) -> tuple[dict[str, Any], Path, str]:
    path = root / "cases" / case_id / f"round{round_index}" / "simulated_error_clicks.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = {
        "round_index": round_index,
        "foreground_xyz": payload.get("foreground_xyz"),
        "background_xyz": payload.get("background_xyz"),
        "foreground_count": payload.get("cumulative_foreground_clicks", 0),
        "background_count": payload.get("cumulative_background_clicks", 0),
        "new_foreground_count": int(payload.get("foreground_xyz") is not None),
        "new_background_count": int(payload.get("background_xyz") is not None),
    }
    return metadata, path, sha256_file(path, cache)


def role_fields(role: str, path: Path, cache: dict[Path, str]) -> dict[str, Any]:
    return {
        f"{role}_path": str(path.resolve()),
        f"{role}_sha256": sha256_file(path, cache),
    }


def freeze_manifest(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    contract = json.loads(args.split_contract.read_text(encoding="utf-8"))
    amendment = json.loads(args.protocol_amendment.read_text(encoding="utf-8"))
    cache: dict[Path, str] = {}
    records = []
    pretest_metrics = []
    for study in contract["studies"]:
        case_id = study["case_id"]
        paths = generate_fusions(root, case_id)
        case_root = root / "cases" / case_id
        ct_path = case_root / "prepared" / "input" / f"{case_id}_0000.nii.gz"
        pet_path = case_root / "prepared" / "input" / f"{case_id}_0001.nii.gz"
        ground_truth_path = case_root / "prepared" / "labels" / f"{case_id}.nii.gz"
        current_path = paths["keep_resenc"]
        round_metadata = {
            index: prompt_metadata(root, case_id, index, cache)
            for index in (1, 2)
        }
        for action in ACTIONS:
            round_index = 1 if "r1" in action else 2
            metadata, simulator_path, simulator_sha = round_metadata[round_index]
            record = {
                "case_id": case_id,
                "patient_id": study["patient_id"],
                "tracer": study["tracer"],
                "split": study["split"],
                "transition_id": f"resenc_to_{action}",
                "action_id": action,
                "action": (
                    "intersection" if "intersection" in action
                    else "union" if "union" in action
                    else "replace"
                ),
                "round_index": round_index,
                "prior_exposure": True,
                "external_validation_eligible": False,
                "prompt_metadata": metadata,
                "prompt_simulator_manifest_path": str(simulator_path.resolve()),
                "prompt_simulator_manifest_sha256": simulator_sha,
                **role_fields("pet", pet_path, cache),
                **role_fields("ct", ct_path, cache),
                **role_fields("current_mask", current_path, cache),
                **role_fields("proposed_mask", paths[action], cache),
                **role_fields("ground_truth", ground_truth_path, cache),
            }
            records.append(record)
        # Outcome scoring is deliberately deferred to the grouped evaluator so
        # candidate generation and manifest hashing do not open any split.

    repo_files = (
        "scripts/autopet_resenc_l_batch.py",
        "scripts/prepare_autopet_nnunet_input.py",
        "scripts/simulate_autopetv_error_clicks.py",
        "scripts/finalize_frozen_route_cohort.py",
        "scripts/train_prompt_update_edl.py",
        "scripts/evaluate_prompt_routes.py",
        "rl_nninteractive/prompt_update_edl.py",
        "rl_nninteractive/route_policy_eval.py",
    )
    split_counts = {
        split: {
            "patients": len(set(contract["patient_split"][split])),
            "studies": sum(study["split"] == split for study in contract["studies"]),
            "candidate_records": sum(record["split"] == split for record in records),
        }
        for split in ("train", "calibration", "policy_validation", "test")
    }
    import nnunetv2

    nnunet_root = Path(nnunetv2.__file__).resolve().parent
    predictor_source = nnunet_root / "inference" / "predict_from_raw_data.py"
    payload = {
        "schema_version": 1,
        "frozen_at": iso_now(),
        "status": "FROZEN_CANDIDATE_MANIFEST_TEST_OUTCOMES_SEALED",
        "claim_boundary": "Exploratory evidential candidate gate/KEEP selector under GT-based robot-user prompts; not learned STOP efficacy, online RL benefit, external validation, or clinical generalization.",
        "prior_exposed": True,
        "external_validation_eligible": False,
        "efficacy_claim_eligible": False,
        "independent_unit": "patient",
        "counts": {
            "patients": len(contract["patient_split"]["train"] + contract["patient_split"]["calibration"] + contract["patient_split"]["policy_validation"] + contract["patient_split"]["test"]),
            "studies": len(contract["studies"]),
            "candidate_records": len(records),
            "by_split": split_counts,
        },
        "base_action": "keep_resenc",
        "candidate_actions": list(ACTIONS),
        "selection_rule": "Highest predicted utility among EDL-ACCEPT candidates; otherwise KEEP_RESENC. STOP and REJECT_CONTINUE both map to KEEP in this fixed menu.",
        "ground_truth_boundary": {
            "direct_feature_use": False,
            "robot_user_prompt_use": True,
            "offline_label_and_scoring_use": True,
            "declaration": "GT determines simulated error clicks and offline labels, so proposals are indirectly oracle-prompt dependent. GT arrays and metrics are not segmentation or EDL inputs.",
            "same_protocol_all_splits": True,
        },
        "test_seal": {
            "status": "SEALED_UNTIL_POLICY_THRESHOLDS_FROZEN",
            "artifact_generation_caveat": "ResEnc inference generated a report containing per-case labels/metrics, but test fields were not inspected. This manifest build hashes test GT and proposals without aggregating or exposing test outcomes.",
        },
        "protocol": {
            "split_contract_path": str(args.split_contract.resolve()),
            "split_contract_sha256": sha256_file(args.split_contract, cache),
            "clarification_amendment_path": str(args.protocol_amendment.resolve()),
            "clarification_amendment_sha256": sha256_file(args.protocol_amendment, cache),
        },
        "generator_config": {
            "resenc_model": "AutoPET III ResEnc-L",
            "resenc_fold": 0,
            "resenc_tta": False,
            "prompt_model": "official AutoPET V 4-channel PlainConvUNet",
            "prompt_fold": 0,
            "prompt_tta": False,
            "prompt_rounds": 2,
            "simulator": "deepest-voxel-in-largest-current-error-component",
            "fusion": "voxelwise boolean intersection/union on the exact PET grid",
        },
        "provenance": {
            "command": " ".join(sys.argv),
            "ohif_ai_commit": git_value(args.repo_root, "rev-parse", "HEAD"),
            "ohif_ai_worktree_dirty": bool(git_value(args.repo_root, "status", "--porcelain")),
            "autopet3_repo_commit": git_value(args.autopet3_repo, "rev-parse", "HEAD"),
            "autopetv_repo_commit": git_value(args.autopetv_repo, "rev-parse", "HEAD"),
            "resenc_predictor_source_sha256": sha256_file(
                args.autopet3_repo / "nnunetv2" / "inference" / "predict_from_raw_data.py",
                cache,
            ),
            "prompt_nnunet_runtime": {
                "version": importlib.metadata.version("nnunetv2"),
                "module_root": str(nnunet_root),
                "predictor_source": str(predictor_source),
                "predictor_source_sha256": sha256_file(predictor_source, cache),
            },
            "resenc_checkpoint": {
                "path": str(args.resenc_checkpoint.resolve()),
                "sha256": sha256_file(args.resenc_checkpoint, cache),
                "bytes": args.resenc_checkpoint.stat().st_size,
            },
            "autopetv_checkpoint": {
                "path": str(args.autopetv_checkpoint.resolve()),
                "sha256": sha256_file(args.autopetv_checkpoint, cache),
                "bytes": args.autopetv_checkpoint.stat().st_size,
            },
            "code_sha256": {
                relative: sha256_file(args.repo_root / relative, cache)
                for relative in repo_files
            },
        },
        "records": records,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics_path = root / "candidate_metrics_pretest.json"
    metrics_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "OUTCOME_SCORING_DEFERRED_ALL_SPLITS_SEALED",
        "independent_unit": "patient",
        "cases": pretest_metrics,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest": str(args.output_manifest.resolve()),
        "manifest_sha256": sha256_file(args.output_manifest),
        "records": len(records),
        "metrics": str(metrics_path.resolve()),
        "test_outcomes_opened": False,
    }))


def score_all(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    contract = json.loads(args.split_contract.read_text(encoding="utf-8"))
    cases = [score_case(root, study) for study in contract["studies"]]
    output = root / "candidate_metrics_all_after_threshold_freeze.json"
    output.write_text(json.dumps({
        "schema_version": 1,
        "opened_at": iso_now(),
        "status": "EXPLORATORY_INTERNAL_PRIOR_EXPOSED",
        "independent_unit": "patient",
        "patients": 8,
        "studies": 16,
        "external_validation_eligible": False,
        "efficacy_claim_eligible": False,
        "cases": cases,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "metrics": str(output.resolve()),
        "metrics_sha256": sha256_file(output),
        "cases": len(cases),
        "test_outcomes_opened": True,
    }))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "score-all"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-contract", type=Path, required=True)
    parser.add_argument("--protocol-amendment", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--autopet3-repo", type=Path, required=True)
    parser.add_argument("--autopetv-repo", type=Path, required=True)
    parser.add_argument("--resenc-checkpoint", type=Path, required=True)
    parser.add_argument("--autopetv-checkpoint", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "freeze":
        freeze_manifest(args)
    else:
        score_all(args)


if __name__ == "__main__":
    main()
