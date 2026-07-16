#!/usr/bin/env python
"""Run the frozen 30-patient fusion-only AutoPET rescue cohort.

This additive runner deliberately separates contract freezing from inference.
The held-out test patients are allowed to drive deterministic robot-user clicks,
but no test overlap metric or utility is computed or written here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nibabel as nib
import numpy as np
from scipy import ndimage


PATIENT_SPLIT = {
    "train": tuple(f"train_{index:04d}" for index in range(1, 13)),
    "calibration": tuple(f"train_{index:04d}" for index in range(13, 17)),
    "policy_validation": tuple(f"train_{index:04d}" for index in range(17, 25)),
    "test": tuple(f"train_{index:04d}" for index in range(25, 31)),
}
TRACERS = ("FDG", "PSMA")
CANDIDATE_ROUTES = (
    "KEEP",
    "r1_intersection",
    "r2_intersection",
    "r1_union",
    "r2_union",
)
PROPOSAL_ROUTES = CANDIDATE_ROUTES[1:]
PRIOR_OUTCOMES_OPENED_PATIENTS = tuple(
    f"train_{index:04d}" for index in range(1, 9)
)
DEFAULT_REUSE_ROOT = Path(
    os.environ.get(
        "OHIF_AI_REUSE_ROOT",
        "artifacts/frozen_route_cohort_2026-07-15",
    )
)
DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "OHIF_AI_RESENC_MODEL_DIR",
        "artifacts/models/autopet3/resenc_l_fold0",
    )
)
DEFAULT_NNUNET_RESULTS = Path(
    os.environ.get(
        "OHIF_AI_NNUNET_RESULTS",
        "artifacts/models/autopetv/nnUNet_results",
    )
)
EXPECTED_RESENC_FOLD0_SHA256 = (
    "45cc19b791b39cde87842a3f30672e165cdbf1b1ff8d5d57fdc52ee49f920fde"
)
EXPECTED_AUTOPETV_FOLD0_SHA256 = (
    "4f47a4bbbbddc86575dc815a363f816891222fc40a550f882539784838ef9948"
)


def iso_now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def append_event(root: Path, event: str, **payload: object) -> None:
    record = {"timestamp": iso_now(), "event": event, **payload}
    path = root / "execution_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"existing destination hash mismatch: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"copy verification failed: {source} -> {destination}")


def load_contract(root: Path) -> dict:
    path = root / "fusion_only_v2_contract.json"
    if not path.is_file():
        raise FileNotFoundError(f"freeze contract before inference: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if expected != observed:
        raise RuntimeError(f"contract hash mismatch: {observed} != {expected}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("candidate_routes") != list(CANDIDATE_ROUTES):
        raise RuntimeError("fusion-only candidate menu drift")
    return contract


def require_test_opening_clearance(root: Path, operation: str) -> dict:
    """Fail closed unless an independently hashed deployment clearance exists."""

    clearance_path = root / "test_opening_clearance.json"
    short_path = root / "test_open_clearance.json"
    observed_hashes = []
    for path in (clearance_path, short_path):
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise RuntimeError(
                f"test opening is sealed: clearance artifact/hash sidecar absent: {path}"
            )
        observed = sha256_file(path)
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        if observed != expected:
            raise RuntimeError(f"test opening clearance hash mismatch: {path}")
        observed_hashes.append(observed)
    if clearance_path.read_bytes() != short_path.read_bytes():
        raise RuntimeError("test opening canonical and short clearance artifacts differ")
    observed_clearance_sha = observed_hashes[0]
    clearance = json.loads(clearance_path.read_text(encoding="utf-8"))
    if clearance.get("schema_version") != 1:
        raise RuntimeError("unsupported test clearance schema")
    if clearance.get("status") != "AUDIT_CLEARED":
        raise RuntimeError("test opening clearance status is not AUDIT_CLEARED")
    audit = clearance.get("independent_audit")
    if not isinstance(audit, dict) or audit.get("status") != "PASS":
        raise RuntimeError("independent audit decision is not PASS")
    if not audit.get("reviewer") or not audit.get("reviewed_at"):
        raise RuntimeError("independent audit identity/timestamp absent")
    audit_report = Path(str(audit.get("report_path", "")))
    if not audit_report.is_file() or audit.get("report_sha256") != sha256_file(audit_report):
        raise RuntimeError("independent audit report/hash mismatch")
    allowed_operations = clearance.get("allowed_operations")
    if not isinstance(allowed_operations, list) or operation not in allowed_operations:
        raise RuntimeError(f"test operation not authorized by clearance: {operation}")
    if clearance.get("test_patient_ids") != list(PATIENT_SPLIT["test"]):
        raise RuntimeError("test patient IDs differ from frozen split")
    linked = {
        "contract_sha256": root / "fusion_only_v2_contract.json",
        "safety_amendment_sha256": root / "fusion_only_v2_safety_amendment_v1.json",
        "test_seal_sha256": root / "test_seal.json",
        "test_seal_amendment_sha256": root / "test_seal_amendment_v1.json",
    }
    for field, path in linked.items():
        if clearance.get(field) != sha256_file(path):
            raise RuntimeError(f"test clearance {field} mismatch")
    dev_manifest = Path(str(clearance.get("dev_manifest_path", "")))
    if not dev_manifest.is_file() or clearance.get("dev_manifest_sha256") != sha256_file(dev_manifest):
        raise RuntimeError("frozen development manifest/hash mismatch")
    bundle_path = Path(str(clearance.get("selector_bundle_path", "")))
    if not bundle_path.is_file():
        raise RuntimeError("frozen deployment bundle is absent")
    if clearance.get("selector_bundle_sha256") != sha256_file(bundle_path):
        raise RuntimeError("frozen deployment bundle hash mismatch")
    bundle_sidecar = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
    if not bundle_sidecar.is_file():
        raise RuntimeError("frozen deployment bundle hash sidecar absent")
    bundle_sidecar_sha = sha256_file(bundle_sidecar)
    if clearance.get("selector_bundle_sidecar_sha256") != bundle_sidecar_sha:
        raise RuntimeError("deployment bundle sidecar hash mismatch")
    if bundle_sidecar.read_text(encoding="utf-8").split()[0] != clearance.get(
        "selector_bundle_sha256"
    ):
        raise RuntimeError("deployment bundle sidecar does not bind bundle hash")
    checkpoint_path = Path(str(clearance.get("edl_checkpoint_path", "")))
    if not checkpoint_path.is_file() or clearance.get("edl_checkpoint_sha256") != sha256_file(checkpoint_path):
        raise RuntimeError("frozen EDL checkpoint/hash mismatch")
    code_hashes = clearance.get("code_hashes")
    required_code = {
        "fusion_only_runner",
        "fusion_only_finalizer",
        "route_policy_eval",
        "prompt_update_edl",
    }
    if not isinstance(code_hashes, dict) or set(code_hashes) != required_code:
        raise RuntimeError("test clearance code hash inventory is incomplete or has drift")
    for role, fingerprint in code_hashes.items():
        if not isinstance(fingerprint, dict):
            raise RuntimeError(f"invalid code fingerprint for {role}")
        code_path = Path(str(fingerprint.get("path", "")))
        if not code_path.is_file() or fingerprint.get("sha256") != sha256_file(code_path):
            raise RuntimeError(f"test clearance code hash mismatch: {role}")
    selected_policies = clearance.get("selected_policies")
    if not isinstance(selected_policies, dict) or set(selected_policies) != {
        "edl_accept_gate",
        "full_information_linear_ridge_comparator",
    }:
        raise RuntimeError("test clearance selected policies are absent or incomplete")
    for name, policy in selected_policies.items():
        if not isinstance(policy, dict) or policy.get("deployment_decision") not in {
            "SELECT_ROUTE_OR_KEEP",
            "KEEP_ALL",
        }:
            raise RuntimeError(f"invalid frozen deployment decision: {name}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("status") != "FROZEN_BEFORE_TEST_OPENING":
        raise RuntimeError("deployment bundle is not frozen before test opening")
    if bundle.get("test_outcomes_opened") is not False:
        raise RuntimeError("deployment bundle does not prove sealed test outcomes")
    if bundle.get("dev_manifest_sha256") != clearance["dev_manifest_sha256"]:
        raise RuntimeError("deployment bundle/dev manifest cross-link mismatch")
    if bundle.get("edl_checkpoint_sha256") != clearance["edl_checkpoint_sha256"]:
        raise RuntimeError("deployment bundle/checkpoint cross-link mismatch")
    if bundle.get("code_hashes") != code_hashes:
        raise RuntimeError("deployment bundle/code inventory cross-link mismatch")
    if bundle.get("selected_policies") != selected_policies:
        raise RuntimeError("deployment bundle/selected-policy cross-link mismatch")
    return {
        "clearance_path": str(clearance_path.resolve()),
        "clearance_sha256": observed_clearance_sha,
        "dev_manifest_path": str(dev_manifest.resolve()),
        "dev_manifest_sha256": clearance["dev_manifest_sha256"],
        "selector_bundle_path": str(bundle_path.resolve()),
        "selector_bundle_sha256": clearance["selector_bundle_sha256"],
        "selector_bundle_sidecar_sha256": clearance[
            "selector_bundle_sidecar_sha256"
        ],
    }


def self_test_clearance_guard(report_root: Path) -> dict:
    """Exercise the valid, missing, and mismatched clearance branches."""

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        linked = {}
        for name in (
            "fusion_only_v2_contract.json",
            "fusion_only_v2_safety_amendment_v1.json",
            "test_seal.json",
            "test_seal_amendment_v1.json",
        ):
            path = root / name
            path.write_text("{}\n", encoding="utf-8")
            linked[name] = sha256_file(path)
        dev_manifest = root / "dev_manifest.json"
        dev_manifest.write_text("{}\n", encoding="utf-8")
        checkpoint = root / "edl_checkpoint.pt"
        checkpoint.write_bytes(b"frozen-edl-checkpoint")
        audit_report = root / "audit_report.json"
        audit_report.write_text('{"status":"PASS"}\n', encoding="utf-8")
        script_path = Path(__file__).resolve()
        code_hashes = {
            role: {"path": str(script_path), "sha256": sha256_file(script_path)}
            for role in (
                "fusion_only_runner",
                "fusion_only_finalizer",
                "route_policy_eval",
                "prompt_update_edl",
            )
        }
        selected_policies = {
            "edl_accept_gate": {"deployment_decision": "KEEP_ALL"},
            "full_information_linear_ridge_comparator": {
                "deployment_decision": "KEEP_ALL"
            },
        }
        bundle = {
            "status": "FROZEN_BEFORE_TEST_OPENING",
            "test_outcomes_opened": False,
            "dev_manifest_sha256": sha256_file(dev_manifest),
            "edl_checkpoint_sha256": sha256_file(checkpoint),
            "code_hashes": code_hashes,
            "selected_policies": selected_policies,
        }
        bundle_path = root / "selector_bundle.json"
        bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")
        bundle_sidecar = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
        bundle_sidecar.write_text(
            f"{sha256_file(bundle_path)}  {bundle_path.name}\n", encoding="utf-8"
        )
        clearance = {
            "schema_version": 1,
            "status": "AUDIT_CLEARED",
            "independent_audit": {
                "status": "PASS",
                "reviewer": "self-test-independent-role",
                "reviewed_at": iso_now(),
                "report_path": str(audit_report),
                "report_sha256": sha256_file(audit_report),
            },
            "allowed_operations": ["stage-round1-test"],
            "test_patient_ids": list(PATIENT_SPLIT["test"]),
            "contract_sha256": linked["fusion_only_v2_contract.json"],
            "safety_amendment_sha256": linked[
                "fusion_only_v2_safety_amendment_v1.json"
            ],
            "test_seal_sha256": linked["test_seal.json"],
            "test_seal_amendment_sha256": linked["test_seal_amendment_v1.json"],
            "dev_manifest_path": str(dev_manifest),
            "dev_manifest_sha256": sha256_file(dev_manifest),
            "selector_bundle_path": str(bundle_path),
            "selector_bundle_sha256": sha256_file(bundle_path),
            "selector_bundle_sidecar_sha256": sha256_file(bundle_sidecar),
            "edl_checkpoint_path": str(checkpoint),
            "edl_checkpoint_sha256": sha256_file(checkpoint),
            "code_hashes": code_hashes,
            "selected_policies": selected_policies,
        }
        encoded = json.dumps(clearance, indent=2, sort_keys=True) + "\n"
        for clearance_path in (
            root / "test_opening_clearance.json",
            root / "test_open_clearance.json",
        ):
            clearance_path.write_text(encoded, encoding="utf-8")
            sidecar = clearance_path.with_suffix(clearance_path.suffix + ".sha256")
            sidecar.write_text(
                f"{sha256_file(clearance_path)}  {clearance_path.name}\n",
                encoding="utf-8",
            )
        require_test_opening_clearance(root, "stage-round1-test")
        probe_calls = {"ground_truth_path": 0, "nib_load": 0}
        original_ground_truth_path = globals()["ground_truth_path"]
        original_nib_load = nib.load

        def probe_ground_truth_path(*args, **kwargs):
            probe_calls["ground_truth_path"] += 1
            raise AssertionError("ground_truth_path reached before clearance")

        def probe_nib_load(*args, **kwargs):
            probe_calls["nib_load"] += 1
            raise AssertionError("nib.load reached before clearance")

        globals()["ground_truth_path"] = probe_ground_truth_path
        nib.load = probe_nib_load
        canonical = root / "test_opening_clearance.json"
        canonical_sidecar = canonical.with_suffix(canonical.suffix + ".sha256")
        canonical_sidecar.write_text(f"{'0' * 64}  {canonical.name}\n", encoding="utf-8")
        mismatch_rejected = False
        try:
            stage_prompt_round(root, 1, "test")
        except RuntimeError:
            mismatch_rejected = True
        for path in (
            root / "test_opening_clearance.json",
            root / "test_open_clearance.json",
        ):
            sidecar = path.with_suffix(path.suffix + ".sha256")
            if sidecar.exists():
                sidecar.unlink()
            if path.exists():
                path.unlink()
        missing_rejected = False
        try:
            stage_prompt_round(root, 1, "test")
        except RuntimeError:
            missing_rejected = True
        finally:
            globals()["ground_truth_path"] = original_ground_truth_path
            nib.load = original_nib_load
        if not mismatch_rejected or not missing_rejected:
            raise AssertionError("clearance guard did not fail closed")
        if probe_calls != {"ground_truth_path": 0, "nib_load": 0}:
            raise AssertionError(f"test stage touched GT before clearance: {probe_calls}")
        result = {
            "valid_clearance_accepted": True,
            "mismatched_hash_rejected": mismatch_rejected,
            "missing_clearance_rejected": missing_rejected,
            "stage_probe_ground_truth_path_calls": probe_calls["ground_truth_path"],
            "stage_probe_nib_load_calls": probe_calls["nib_load"],
            "stage_probe_gt_touched_before_clearance": False,
            "runner": file_fingerprint(Path(__file__).resolve()),
            "finished_at": iso_now(),
        }
        report_path = report_root / "clearance_guard_stage_probe_v2.json"
        report_sha = write_frozen_json(report_path, result)
        return {**result, "report_path": str(report_path), "report_sha256": report_sha}


def write_frozen_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        existing = path.read_bytes()
        if existing != encoded:
            raise RuntimeError(f"refusing to mutate frozen artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected_sidecar = f"{digest}  {path.name}\n"
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise RuntimeError(f"frozen hash sidecar mismatch: {sidecar}")
    if not sidecar.exists():
        sidecar.write_text(expected_sidecar, encoding="utf-8")
    return digest


def split_for(patient_id: str) -> str:
    matches = [split for split, patients in PATIENT_SPLIT.items() if patient_id in patients]
    if len(matches) != 1:
        raise RuntimeError(f"patient must occur in exactly one split: {patient_id}")
    return matches[0]


def freeze_contract(root: Path, source_root: Path) -> dict:
    root = root.resolve()
    source_root = source_root.resolve()
    studies = []
    for index in range(1, 31):
        patient_id = f"train_{index:04d}"
        split = split_for(patient_id)
        for tracer in TRACERS:
            case_id = f"{patient_id}_{tracer}"
            source_dir = source_root / patient_id / tracer
            sources = {
                role: source_dir / filename
                for role, filename in (
                    ("ct", "CT.nii.gz"),
                    ("pet", "PET.nii.gz"),
                    ("ground_truth", "TTB.nii.gz"),
                )
            }
            missing = [str(path) for path in sources.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"{case_id} missing required source(s): {missing}")
            studies.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "tracer": tracer,
                    "split": split,
                    "prior_exposed": True,
                    "external_validation_eligible": False,
                    "source_paths": {key: str(path.resolve()) for key, path in sources.items()},
                }
            )

    patients = [patient for members in PATIENT_SPLIT.values() for patient in members]
    if len(patients) != 30 or len(set(patients)) != 30:
        raise RuntimeError("split contract must contain exactly 30 disjoint patients")
    if len(studies) != 60:
        raise RuntimeError("contract requires all 60 paired FDG/PSMA studies")

    payload = {
        "schema_version": 2,
        "frozen_at": iso_now(),
        "status": "FROZEN_BEFORE_NEW_INFERENCE",
        "purpose": "Fusion-only internal prior-exposed rescue experiment for PI triage.",
        "claim_boundary": "Exploratory internal prior-exposed evidence only; not external validation or clinical readiness.",
        "patient_split": {key: list(value) for key, value in PATIENT_SPLIT.items()},
        "studies": studies,
        "selection_unit": "study",
        "baseline_route": "KEEP",
        "candidate_routes": list(CANDIDATE_ROUTES),
        "proposal_routes": list(PROPOSAL_ROUTES),
        "direct_replacement_forbidden": True,
        "expected_manifest_records": 240,
        "prior_exposed": True,
        "external_validation_eligible": False,
        "development_due_to_prior_outcomes_patient_ids": list(
            PRIOR_OUTCOMES_OPENED_PATIENTS
        ),
        "prior_outcomes_opened_patient_ids": list(PRIOR_OUTCOMES_OPENED_PATIENTS),
        "new_test_patient_ids": list(PATIENT_SPLIT["test"]),
        "test_opening_rule": "Test labels may drive deterministic robot-user clicks, but test metrics/utilities remain sealed until model, calibration, route policy, and thresholds are frozen and independently audited.",
        "ground_truth_boundary": "Ground truth is permitted only for deterministic robot-user clicks before finalization and for the single later offline labeling/evaluation pass. It is never a segmentation or policy feature.",
        "reuse_rule": "Reuse hash-verified trajectories for train_0001 through train_0008; generate identical-protocol trajectories for train_0009 through train_0030.",
        "model_protocol": {
            "initial_mask": "AutoPET III ResEnc-L fold 0 checkpoint_final.pth, no TTA",
            "prompt_refinement": "Official AutoPET V 4-channel PlainConvUNet fold 0 checkpoint_final.pth, no TTA, two cumulative deterministic error-click rounds",
            "fusion": "Exact-grid voxelwise ResEnc intersection/union with R1/R2 prompt masks",
            "h100_required": False,
        },
    }
    contract_path = root / "fusion_only_v2_contract.json"
    contract_hash = write_frozen_json(contract_path, payload)
    seal = {
        "schema_version": 1,
        "created_at": payload["frozen_at"],
        "status": "SEALED_UNTIL_THRESHOLDS_FROZEN_AND_AUDIT_CLEARS_OPENING",
        "contract_path": str(contract_path),
        "contract_sha256": contract_hash,
        "test_patient_ids": list(PATIENT_SPLIT["test"]),
        "prohibited_before_opening": [
            "test Dice/NSD/utility computation",
            "test-derived threshold or route selection",
            "test aggregate outcome report",
        ],
        "permitted_before_opening": [
            "geometry validation",
            "file hashing",
            "ResEnc and prompt inference",
            "deterministic robot-user click generation",
            "fusion mask construction",
        ],
    }
    seal_path = root / "test_seal.json"
    seal_hash = write_frozen_json(seal_path, seal)
    return {
        "contract_path": str(contract_path),
        "contract_sha256": contract_hash,
        "test_seal_path": str(seal_path),
        "test_seal_sha256": seal_hash,
        "patients": 30,
        "studies": 60,
        "expected_manifest_records": 240,
    }


def freeze_safety_amendment(root: Path) -> dict:
    """Freeze the pre-inference safety rule without mutating the v2 contract."""

    root = root.resolve()
    contract_path = root / "fusion_only_v2_contract.json"
    seal_path = root / "test_seal.json"
    contract_sha = sha256_file(contract_path)
    seal_sha = sha256_file(seal_path)
    amendment = {
        "schema_version": 1,
        "frozen_at": iso_now(),
        "status": "FROZEN_BEFORE_NEW_INFERENCE",
        "amendment_type": "safety-selection-and-ground-truth-boundary",
        "supersedes_conflicting_wording_only": True,
        "original_contract_path": str(contract_path.resolve()),
        "original_contract_sha256": contract_sha,
        "split_or_candidate_menu_changed": False,
        "safety_selection_rule": {
            "version": 1,
            "selection_split": "policy_validation",
            "selection_split_exclusive": True,
            "applies_to": [
                "edl_accept_gate",
                "full_information_linear_ridge_comparator",
            ],
            "selection_unit": "study decisions clustered by patient",
            "patient_utility": "unweighted mean realized utility across that patient's studies",
            "bootstrap": {
                "method": "patient-cluster percentile",
                "seed": 20260715,
                "resamples": 10000,
                "procedure": "resample P independent patients with replacement and compute the mean of patient utilities",
                "ci_quantiles": [0.025, 0.975],
            },
            "safe_iff": {
                "harmful_action_rate_all_studies_lte": 0.05,
                "bootstrap_95pct_lower_bound_mean_patient_utility_strictly_gt": 0.0,
            },
            "harmful_study_definition": "selected proposal delta_dice < -1e-12",
            "keep_definition": {"harmful": False, "utility": 0.0},
            "edl_grid": {
                "accept_probability": "linspace(0.35, 0.80, 10)",
                "max_accept_vacuity": "linspace(0.30, 0.90, 7)",
                "min_predicted_utility": [-0.01, 0.0, 0.01, 0.02],
                "grid_points": 280,
            },
            "ridge_grid": {
                "lambdas": [0.01, 0.1, 1.0, 10.0],
                "per_lambda_threshold_grid": "[min_score-epsilon, every unique policy-validation score, max_score+epsilon]",
                "epsilon": "max(1, max_abs_score) * 1e-9",
            },
            "safe_candidate_optimization_order": [
                "patient_mean_utility",
                "bootstrap_ci_lower_bound",
                "study_mean_utility",
                "lower_harmful_study_rate",
                "coverage_and_deterministic_tie_fields",
            ],
            "zero_safe_candidates_fallback": "KEEP_ALL",
            "threshold_extrapolation_forbidden": True,
            "required_report_fields": [
                "grid_size",
                "eligible_grid_points_or_models",
                "independent_patient_count",
                "study_count",
                "coverage",
                "harmful_action_rate_all_studies",
                "harmful_action_rate_when_covered",
                "patient_mean_utility",
                "bootstrap_ci",
                "deployment_decision",
                "fallback_reason",
            ],
        },
        "test_label_boundary": {
            "fit_calibration_threshold_apis": "test labels unavailable",
            "opening": "single score pass only after model, calibration, and safety thresholds are frozen",
        },
        "ground_truth_boundary_clarification": "Candidate proposals are INDIRECTLY ground-truth-dependent because robot-user corrections are generated from ground truth. Deployable features exclude direct ground-truth arrays, metrics, and scalars, but the evidence remains offline oracle-assisted and cannot support external or live efficacy claims.",
    }
    amendment_path = root / "fusion_only_v2_safety_amendment_v1.json"
    amendment_sha = write_frozen_json(amendment_path, amendment)
    seal_amendment = {
        "schema_version": 1,
        "frozen_at": amendment["frozen_at"],
        "status": "SEALED_UNTIL_SAFETY_RULE_FROZEN_AND_INDEPENDENT_AUDIT_CLEARS_OPENING",
        "original_test_seal_path": str(seal_path.resolve()),
        "original_test_seal_sha256": seal_sha,
        "original_contract_path": str(contract_path.resolve()),
        "original_contract_sha256": contract_sha,
        "safety_amendment_path": str(amendment_path.resolve()),
        "safety_amendment_sha256": amendment_sha,
        "test_patient_ids": list(PATIENT_SPLIT["test"]),
        "opening_preconditions": [
            "EDL and ridge candidates selected only on policy_validation under the frozen safety rule",
            "deployment artifacts and hashes frozen before any test label API is invoked",
            "independent audit explicitly clears the sole test opening",
        ],
        "test_outcome_pass_limit": 1,
    }
    seal_amendment_path = root / "test_seal_amendment_v1.json"
    seal_amendment_sha = write_frozen_json(seal_amendment_path, seal_amendment)
    return {
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha,
        "safety_amendment_path": str(amendment_path),
        "safety_amendment_sha256": amendment_sha,
        "test_seal_path": str(seal_path),
        "test_seal_sha256": seal_sha,
        "test_seal_amendment_path": str(seal_amendment_path),
        "test_seal_amendment_sha256": seal_amendment_sha,
    }


def reuse_verified_development(root: Path, reuse_root: Path) -> dict:
    contract = load_contract(root)
    started = time.perf_counter()
    copied = []
    for study in contract["studies"]:
        if study["patient_id"] not in PRIOR_OUTCOMES_OPENED_PATIENTS:
            continue
        case_id = study["case_id"]
        pairs = []
        for relative in (
            Path("cases") / case_id / "prepared" / "input_manifest.json",
            Path("cases") / case_id / "prepared" / "input" / f"{case_id}_0000.nii.gz",
            Path("cases") / case_id / "prepared" / "input" / f"{case_id}_0001.nii.gz",
            Path("cases") / case_id / "prepared" / "labels" / f"{case_id}.nii.gz",
            Path("cases") / case_id / "round1" / "simulated_error_clicks.json",
            Path("cases") / case_id / "round1" / "input" / f"{case_id}_0000.nii.gz",
            Path("cases") / case_id / "round1" / "input" / f"{case_id}_0001.nii.gz",
            Path("cases") / case_id / "round1" / "input" / f"{case_id}_0002.nii.gz",
            Path("cases") / case_id / "round1" / "input" / f"{case_id}_0003.nii.gz",
            Path("cases") / case_id / "round2" / "simulated_error_clicks.json",
            Path("cases") / case_id / "round2" / "input" / f"{case_id}_0000.nii.gz",
            Path("cases") / case_id / "round2" / "input" / f"{case_id}_0001.nii.gz",
            Path("cases") / case_id / "round2" / "input" / f"{case_id}_0002.nii.gz",
            Path("cases") / case_id / "round2" / "input" / f"{case_id}_0003.nii.gz",
            Path("batches") / "resenc" / "prediction" / f"{case_id}.nii.gz",
            Path("batches") / "round1" / "prediction" / f"{case_id}.nii.gz",
            Path("batches") / "round2" / "prediction" / f"{case_id}.nii.gz",
        ):
            source = reuse_root / relative
            destination = root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            link_or_copy(source, destination)
            pairs.append(
                {
                    "relative_path": str(relative),
                    "sha256": sha256_file(destination),
                    "reuse_source": str(source.resolve()),
                }
            )
        copied.append({"case_id": case_id, "files": pairs})
    report = {
        "schema_version": 1,
        "status": "PASS",
        "finished_at": iso_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "source_root": str(reuse_root.resolve()),
        "cases": copied,
        "case_count": len(copied),
        "outcome_metrics_copied": False,
    }
    path = root / "reuse_verified_0001_0008_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    append_event(root, "reuse_verified_complete", report_path=str(path), cases=len(copied), elapsed_seconds=report["elapsed_seconds"])
    return report


def freeze_reuse_provenance(
    root: Path,
    reuse_root: Path,
    model_dir: Path,
    nnunet_results: Path,
) -> dict:
    """Bind reused trajectories to their authoritative reports and checkpoints."""

    load_contract(root)
    reuse_report = root / "reuse_verified_0001_0008_report.json"
    if not reuse_report.is_file():
        raise FileNotFoundError(reuse_report)
    prior_artifacts = {
        "prior_frozen_route_manifest": reuse_root / "frozen_route_manifest.json",
        "prior_resenc_batch_report": reuse_root / "resenc_batch_report.json",
        "prior_execution_summary": reuse_root / "execution_summary.md",
        "prior_round1_inference_log": reuse_root / "logs" / "predict_round1_batch16.log",
        "prior_round2_inference_log": reuse_root / "logs" / "predict_round2_batch16.log",
    }
    for path in prior_artifacts.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    click_manifests = []
    for index in range(1, 9):
        for tracer in TRACERS:
            case_id = f"train_{index:04d}_{tracer}"
            for round_index in (1, 2):
                path = (
                    reuse_root
                    / "cases"
                    / case_id
                    / f"round{round_index}"
                    / "simulated_error_clicks.json"
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                click_manifests.append(
                    {
                        "case_id": case_id,
                        "round_index": round_index,
                        **file_fingerprint(path),
                    }
                )
    resenc_checkpoint = model_dir / "fold_0" / "checkpoint_final.pth"
    prompt_checkpoints = sorted(nnunet_results.rglob("checkpoint_final.pth"))
    if len(prompt_checkpoints) != 1:
        raise RuntimeError(f"expected one AutoPET V final checkpoint, found {len(prompt_checkpoints)}")
    prompt_checkpoint = prompt_checkpoints[0]
    observed_resenc = sha256_file(resenc_checkpoint)
    observed_prompt = sha256_file(prompt_checkpoint)
    if observed_resenc != EXPECTED_RESENC_FOLD0_SHA256:
        raise RuntimeError("ResEnc checkpoint hash drift")
    if observed_prompt != EXPECTED_AUTOPETV_FOLD0_SHA256:
        raise RuntimeError("AutoPET V checkpoint hash drift")
    payload = {
        "schema_version": 1,
        "frozen_at": iso_now(),
        "status": "FROZEN_HASH_VERIFIED_REUSE_PROVENANCE",
        "reuse_patient_ids": list(PRIOR_OUTCOMES_OPENED_PATIENTS),
        "reuse_case_count": 16,
        "reused_round_count": 2,
        "destination_reuse_report": file_fingerprint(reuse_report),
        "authoritative_prior_artifacts": {
            role: file_fingerprint(path) for role, path in prior_artifacts.items()
        },
        "authoritative_click_manifests": click_manifests,
        "model_checkpoints": {
            "autopet3_resenc_l_fold0": file_fingerprint(resenc_checkpoint),
            "autopetv_plainconv_fold0": file_fingerprint(prompt_checkpoint),
        },
        "expected_checkpoint_sha256": {
            "autopet3_resenc_l_fold0": EXPECTED_RESENC_FOLD0_SHA256,
            "autopetv_plainconv_fold0": EXPECTED_AUTOPETV_FOLD0_SHA256,
        },
        "verification": {
            "destination_equals_source_hashes": True,
            "authoritative_prior_manifest_bound": True,
            "prompt_inference_logs_bound": True,
            "per_case_click_manifests_bound": True,
            "current_checkpoint_hashes_match_expected": True,
            "outcome_metrics_reused_in_v2": False,
        },
    }
    path = root / "reuse_provenance_binding_v1.json"
    digest = write_frozen_json(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "click_manifests": len(click_manifests),
        "checkpoint_hashes_match": True,
    }


def prepare_new_cases(root: Path, repo: Path) -> dict:
    contract = load_contract(root)
    started = time.perf_counter()
    cases = []
    for study in contract["studies"]:
        if study["patient_id"] in PRIOR_OUTCOMES_OPENED_PATIENTS:
            continue
        case_id = study["case_id"]
        prepared = root / "cases" / case_id / "prepared"
        manifest = prepared / "input_manifest.json"
        command = [
            sys.executable,
            str(repo / "scripts" / "prepare_autopet_nnunet_input.py"),
            "--case-dir",
            str(Path(study["source_paths"]["ct"]).parent.parent),
            "--tracer",
            study["tracer"],
            "--output-dir",
            str(prepared),
            "--case-name",
            case_id,
        ]
        case_started = time.perf_counter()
        if not manifest.is_file():
            completed = subprocess.run(command, cwd=repo, text=True, capture_output=True)
            if completed.returncode != 0:
                append_event(root, "prepare_failed", case_id=case_id, returncode=completed.returncode, stderr=completed.stderr[-4000:])
                raise RuntimeError(f"preparation failed for {case_id}: {completed.stderr}")
        label = prepared / "labels" / f"{case_id}.nii.gz"
        if study["split"] == "test":
            sealed_label = root / "sealed" / "test_labels" / f"{case_id}.nii.gz"
            if label.is_file() and not sealed_label.exists():
                sealed_label.parent.mkdir(parents=True, exist_ok=True)
                label.replace(sealed_label)
            elif label.is_file() and sealed_label.is_file():
                if sha256_file(label) != sha256_file(sealed_label):
                    raise RuntimeError(f"sealed test label mismatch: {case_id}")
                label.unlink()
            label = sealed_label
        if not label.is_file():
            raise FileNotFoundError(label)
        inputs = []
        for channel in (0, 1):
            source = prepared / "input" / f"{case_id}_{channel:04d}.nii.gz"
            destination = root / "batches" / "resenc" / "input_new" / source.name
            link_or_copy(source, destination)
            inputs.append(file_fingerprint(source))
        cases.append(
            {
                "case_id": case_id,
                "split": study["split"],
                "elapsed_seconds": time.perf_counter() - case_started,
                "manifest": file_fingerprint(manifest),
                "inputs": inputs,
                "ground_truth_storage": "sealed_test_labels" if study["split"] == "test" else "development_prepared_labels",
                "ground_truth_sha256": sha256_file(label),
                "outcome_metrics_computed": False,
            }
        )
        if len(cases) % 8 == 0:
            append_event(root, "prepare_heartbeat", cases_complete=len(cases), cases_total=44, elapsed_seconds=time.perf_counter() - started)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "started_at": iso_now(),
        "finished_at": iso_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "cases": cases,
        "case_count": len(cases),
        "test_outcomes_opened": False,
    }
    path = root / "prepare_new_0009_0030_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    append_event(root, "prepare_complete", report_path=str(path), cases=len(cases), elapsed_seconds=report["elapsed_seconds"])
    return report


def run_resenc_no_score(root: Path, model_dir: Path) -> dict:
    contract = load_contract(root)
    del contract
    import torch

    input_dir = root / "batches" / "resenc" / "input_new"
    output_dir = root / "batches" / "resenc" / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [path.name[:-12] for path in sorted(input_dir.glob("*_0000.nii.gz"))]
    if len(cases) != 44:
        raise RuntimeError(f"expected 44 new ResEnc cases, found {len(cases)}")
    for case_id in cases:
        for channel in (0, 1):
            path = input_dir / f"{case_id}_{channel:04d}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_at": iso_now(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cases_requested": cases,
        "test_outcome_fields": "SEALED_AND_NOT_COMPUTED",
        "label_paths_loaded": False,
    }
    try:
        from nnunetv2 import __file__ as nnunet_module_file
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        report["nnunet_module"] = str(Path(nnunet_module_file).resolve())
        checkpoint = model_dir / "fold_0" / "checkpoint_final.pth"
        report["checkpoint"] = file_fingerprint(checkpoint)
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device("cuda"),
            verbose=False,
            allow_tqdm=False,
            verbose_preprocessing=False,
        )
        init_started = time.perf_counter()
        predictor.initialize_from_trained_model_folder(
            str(model_dir), [0], "checkpoint_final.pth"
        )
        report["model_initialization_seconds"] = time.perf_counter() - init_started
        inference_started = time.perf_counter()
        predictor.predict_from_files(
            str(input_dir),
            str(output_dir),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        torch.cuda.synchronize()
        report["inference_seconds"] = time.perf_counter() - inference_started
        report["gpu"].update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        report["cases"] = [
            {
                "case_id": case_id,
                "inputs": [
                    file_fingerprint(input_dir / f"{case_id}_0000.nii.gz"),
                    file_fingerprint(input_dir / f"{case_id}_0001.nii.gz"),
                ],
                "prediction": file_fingerprint(output_dir / f"{case_id}.nii.gz"),
                "outcome_metrics_computed": False,
            }
            for case_id in cases
        ]
        if len(list(output_dir.glob("train_*.nii.gz"))) != 60:
            raise RuntimeError("ResEnc output cohort must contain 60 predictions including 16 reused")
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        report["finished_at"] = iso_now()
        report["elapsed_seconds"] = time.perf_counter() - started
        path = root / "resenc_new_0009_0030_no_score_report.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        append_event(root, "resenc_finished", status=report["status"], report_path=str(path), elapsed_seconds=report["elapsed_seconds"])
    return report


def ground_truth_path(root: Path, study: dict) -> Path:
    case_id = study["case_id"]
    if study["split"] == "test":
        return root / "sealed" / "test_labels" / f"{case_id}.nii.gz"
    return root / "cases" / case_id / "prepared" / "labels" / f"{case_id}.nii.gz"


def _deepest_largest_component(mask: np.ndarray) -> tuple[int, int, int] | None:
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    component = components == int(np.argmax(sizes))
    distances = ndimage.distance_transform_edt(component)
    return tuple(int(value) for value in np.unravel_index(int(np.argmax(distances)), mask.shape))


def _load_prior_heatmap(source_input: Path, case_id: str, channel: int, shape: tuple[int, ...]) -> np.ndarray:
    path = source_input / f"{case_id}_{channel:04d}.nii.gz"
    if not path.is_file():
        return np.zeros(shape, dtype=np.float32)
    data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if data.shape != shape:
        raise RuntimeError(f"prompt heatmap shape mismatch: {path}")
    return data


def stage_prompt_round(root: Path, round_index: int, scope: str) -> dict:
    if round_index not in (1, 2):
        raise ValueError("round_index must be 1 or 2")
    if scope not in ("development", "test"):
        raise ValueError("scope must be development or test")
    clearance = None
    if scope == "test":
        clearance = require_test_opening_clearance(root, f"stage-round{round_index}-test")
        if round_index == 2:
            round1_report = root / "prompt_round1_test_0025_0030_no_score_report.json"
            if not round1_report.is_file():
                raise RuntimeError("round2 test staging requires completed round1 test inference report")
            round1_payload = json.loads(round1_report.read_text(encoding="utf-8"))
            if round1_payload.get("status") != "PASS" or round1_payload.get("case_count", len(round1_payload.get("cases", []))) != 12:
                raise RuntimeError("round1 test inference provenance is incomplete")
    contract = load_contract(root)
    started = time.perf_counter()
    staged = []
    scope_slug = "dev" if scope == "development" else "test"
    batch_input = root / "batches" / f"round{round_index}" / f"input_{scope_slug}"
    expected_cases = 32 if scope == "development" else 12
    for study in contract["studies"]:
        if study["patient_id"] in PRIOR_OUTCOMES_OPENED_PATIENTS:
            continue
        if scope == "development" and study["split"] == "test":
            continue
        if scope == "test" and study["split"] != "test":
            continue
        case_id = study["case_id"]
        case_root = root / "cases" / case_id
        if round_index == 1:
            source_input = case_root / "prepared" / "input"
            prediction_path = root / "batches" / "resenc" / "prediction" / f"{case_id}.nii.gz"
        else:
            source_input = case_root / "round1" / "input"
            prediction_path = root / "batches" / "round1" / "prediction" / f"{case_id}.nii.gz"
        output_root = case_root / f"round{round_index}"
        output_input = output_root / "input"
        output_input.mkdir(parents=True, exist_ok=True)
        ct = source_input / f"{case_id}_0000.nii.gz"
        pet = source_input / f"{case_id}_0001.nii.gz"
        gt_path = ground_truth_path(root, study)
        for path in (ct, pet, prediction_path, gt_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        reference = nib.load(str(pet))
        prediction_image = nib.load(str(prediction_path))
        gt_image = nib.load(str(gt_path))
        if prediction_image.shape != reference.shape or gt_image.shape != reference.shape:
            raise RuntimeError(f"prompt geometry shape mismatch: {case_id}")
        if not np.allclose(prediction_image.affine, reference.affine, atol=1e-4):
            raise RuntimeError(f"prompt prediction affine mismatch: {case_id}")
        if not np.allclose(gt_image.affine, reference.affine, atol=1e-4):
            raise RuntimeError(f"prompt GT affine mismatch: {case_id}")
        prediction = np.asarray(prediction_image.dataobj) > 0
        ground_truth = np.asarray(gt_image.dataobj) > 0
        false_negative = ground_truth & ~prediction
        false_positive = prediction & ~ground_truth
        foreground = _deepest_largest_component(false_negative)
        background = _deepest_largest_component(false_positive)
        fg_map = _load_prior_heatmap(source_input, case_id, 2, reference.shape)
        bg_map = _load_prior_heatmap(source_input, case_id, 3, reference.shape)
        previous_foreground = int(np.count_nonzero(fg_map))
        previous_background = int(np.count_nonzero(bg_map))
        if foreground is not None:
            fg_map[foreground] = 1.0
        if background is not None:
            bg_map[background] = 1.0

        outputs = []
        for channel, source in ((0, ct), (1, pet)):
            destination = output_input / f"{case_id}_{channel:04d}.nii.gz"
            link_or_copy(source, destination)
            outputs.append(destination)
        for channel, data in ((2, fg_map), (3, bg_map)):
            destination = output_input / f"{case_id}_{channel:04d}.nii.gz"
            header = reference.header.copy()
            header.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(data.astype(np.float32), reference.affine, header), str(destination))
            outputs.append(destination)
        for output in outputs:
            link_or_copy(output, batch_input / output.name)
        payload = {
            "schema_version": 2,
            "case": case_id,
            "round_index": round_index,
            "simulator": "deepest-voxel-in-largest-current-error-component",
            "candidate_proposal_oracle_dependency": "INDIRECT_GT_DEPENDENCE_VIA_ROBOT_USER_CLICKS",
            "deployable_features_exclude_direct_gt_arrays_metrics_scalars": True,
            "foreground_xyz": list(foreground) if foreground is not None else None,
            "background_xyz": list(background) if background is not None else None,
            "previous_foreground_clicks": previous_foreground,
            "previous_background_clicks": previous_background,
            "cumulative_foreground_clicks": int(np.count_nonzero(fg_map)),
            "cumulative_background_clicks": int(np.count_nonzero(bg_map)),
            "sources": {
                "prediction": file_fingerprint(prediction_path),
                "ground_truth_sha256": sha256_file(gt_path),
                "ground_truth_storage": "SEALED_TEST_LABEL" if study["split"] == "test" else "DEVELOPMENT_LABEL",
            },
            "outputs": {path.name: file_fingerprint(path) for path in outputs},
            "outcome_metrics_computed": False,
            "test_error_volume_counts_written": False,
        }
        manifest = output_root / "simulated_error_clicks.json"
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        staged.append(
            {
                "case_id": case_id,
                "split": study["split"],
                "manifest": file_fingerprint(manifest),
                "output_channels": {path.name: sha256_file(path) for path in outputs},
                "outcome_metrics_computed": False,
            }
        )
        if len(staged) % 8 == 0:
            append_event(root, f"round{round_index}_{scope_slug}_stage_heartbeat", cases_complete=len(staged), cases_total=expected_cases, elapsed_seconds=time.perf_counter() - started)
    if len(staged) != expected_cases:
        raise RuntimeError(
            f"expected {expected_cases} {scope} cases for round {round_index}, got {len(staged)}"
        )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "round_index": round_index,
        "scope": scope,
        "case_count": len(staged),
        "elapsed_seconds": time.perf_counter() - started,
        "cases": staged,
        "test_outcomes_opened": False,
        "test_opening_clearance": clearance,
    }
    range_slug = "0009_0024" if scope == "development" else "0025_0030"
    path = root / f"prompt_round{round_index}_{scope_slug}_{range_slug}_stage_no_score_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    append_event(root, f"round{round_index}_{scope_slug}_stage_complete", report_path=str(path), cases=len(staged), elapsed_seconds=report["elapsed_seconds"])
    return report


def run_prompt_no_score(root: Path, round_index: int, nnunet_results: Path, scope: str) -> dict:
    load_contract(root)
    if scope not in ("development", "test"):
        raise ValueError("scope must be development or test")
    clearance = None
    if scope == "test":
        clearance = require_test_opening_clearance(root, f"infer-round{round_index}-test")
        stage_report = root / f"prompt_round{round_index}_test_0025_0030_stage_no_score_report.json"
        if not stage_report.is_file():
            raise RuntimeError(f"test round {round_index} inference requires staged prompt provenance")
        stage_payload = json.loads(stage_report.read_text(encoding="utf-8"))
        if stage_payload.get("status") != "PASS" or stage_payload.get("case_count") != 12:
            raise RuntimeError(f"test round {round_index} stage report is incomplete")
    scope_slug = "dev" if scope == "development" else "test"
    expected_cases = 32 if scope == "development" else 12
    input_dir = root / "batches" / f"round{round_index}" / f"input_{scope_slug}"
    output_dir = root / "batches" / f"round{round_index}" / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [path.name[:-12] for path in sorted(input_dir.glob("*_0000.nii.gz"))]
    if len(cases) != expected_cases:
        raise RuntimeError(f"expected {expected_cases} {scope} prompt cases for round {round_index}, found {len(cases)}")
    for case_id in cases:
        for channel in range(4):
            path = input_dir / f"{case_id}_{channel:04d}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
    executable = Path(sys.executable).with_name("nnUNetv2_predict.exe")
    command = [
        str(executable),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        "998",
        "-c",
        "3d_fullres",
        "-f",
        "0",
        "-chk",
        "checkpoint_final.pth",
        "--disable_tta",
        "--not_on_device",
        "-npp",
        "1",
        "-nps",
        "1",
    ]
    checkpoints = sorted(nnunet_results.rglob("checkpoint_final.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected exactly one AutoPET V final checkpoint, found {len(checkpoints)}")
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "status": "RUNNING",
        "round_index": round_index,
        "scope": scope,
        "started_at": iso_now(),
        "command": command,
        "cases_requested": cases,
        "checkpoint": file_fingerprint(checkpoints[0]),
        "test_outcome_fields": "SEALED_AND_NOT_COMPUTED",
        "label_paths_loaded_by_inference": False,
        "test_opening_clearance": clearance,
    }
    env = os.environ.copy()
    env["nnUNet_results"] = str(nnunet_results.resolve())
    log_path = root / "logs" / f"prompt_round{round_index}_{scope_slug}_inference.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            time.sleep(30)
            append_event(root, f"round{round_index}_{scope_slug}_inference_heartbeat", elapsed_seconds=time.perf_counter() - started, process_id=process.pid)
        returncode = process.returncode
    try:
        if returncode != 0:
            raise RuntimeError(f"AutoPET V round {round_index} inference failed with exit code {returncode}; see {log_path}")
        report["cases"] = [
            {
                "case_id": case_id,
                "prediction": file_fingerprint(output_dir / f"{case_id}.nii.gz"),
                "outcome_metrics_computed": False,
            }
            for case_id in cases
        ]
        expected_total = 48 if scope == "development" else 60
        if len(list(output_dir.glob("train_*.nii.gz"))) != expected_total:
            raise RuntimeError(f"round {round_index} output cohort must contain {expected_total} predictions after {scope} batch")
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        report["finished_at"] = iso_now()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["log"] = file_fingerprint(log_path)
        range_slug = "0009_0024" if scope == "development" else "0025_0030"
        path = root / f"prompt_round{round_index}_{scope_slug}_{range_slug}_no_score_report.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        append_event(root, f"round{round_index}_{scope_slug}_inference_finished", status=report["status"], report_path=str(path), elapsed_seconds=report["elapsed_seconds"])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze-contract",
            "freeze-safety-amendment",
            "reuse-verified",
            "freeze-reuse-provenance",
            "prepare-new",
            "resenc-new",
            "stage-round1-dev",
            "infer-round1-dev",
            "stage-round2-dev",
            "infer-round2-dev",
            "stage-round1-test",
            "infer-round1-test",
            "stage-round2-test",
            "infer-round2-test",
            "self-test-clearance",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--reuse-root", type=Path, default=DEFAULT_REUSE_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--nnunet-results", type=Path, default=DEFAULT_NNUNET_RESULTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.command == "freeze-contract":
        if args.source_root is None:
            raise ValueError("--source-root is required for freeze-contract")
        result = freeze_contract(root, args.source_root)
    elif args.command == "freeze-safety-amendment":
        result = freeze_safety_amendment(root)
    elif args.command == "reuse-verified":
        result = reuse_verified_development(root, args.reuse_root)
    elif args.command == "freeze-reuse-provenance":
        result = freeze_reuse_provenance(
            root,
            args.reuse_root.resolve(),
            args.model_dir.resolve(),
            args.nnunet_results.resolve(),
        )
    elif args.command == "prepare-new":
        result = prepare_new_cases(root, Path(__file__).resolve().parents[1])
    elif args.command == "resenc-new":
        result = run_resenc_no_score(root, args.model_dir.resolve())
    elif args.command == "stage-round1-dev":
        result = stage_prompt_round(root, 1, "development")
    elif args.command == "infer-round1-dev":
        result = run_prompt_no_score(root, 1, args.nnunet_results.resolve(), "development")
    elif args.command == "stage-round2-dev":
        result = stage_prompt_round(root, 2, "development")
    elif args.command == "infer-round2-dev":
        result = run_prompt_no_score(root, 2, args.nnunet_results.resolve(), "development")
    elif args.command == "stage-round1-test":
        result = stage_prompt_round(root, 1, "test")
    elif args.command == "infer-round1-test":
        result = run_prompt_no_score(root, 1, args.nnunet_results.resolve(), "test")
    elif args.command == "stage-round2-test":
        result = stage_prompt_round(root, 2, "test")
    elif args.command == "infer-round2-test":
        result = run_prompt_no_score(root, 2, args.nnunet_results.resolve(), "test")
    elif args.command == "self-test-clearance":
        result = self_test_clearance_guard(root)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
