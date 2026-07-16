#!/usr/bin/env python
"""Freeze fusion-only development/test manifests and immutable selector bundles."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from run_fusion_only_cohort_v2 import (
    CANDIDATE_ROUTES,
    PROPOSAL_ROUTES,
    file_fingerprint,
    ground_truth_path,
    iso_now,
    load_contract,
    require_test_opening_clearance,
    sha256_file,
    write_frozen_json,
)


UTILITY_CONFIG = {
    "nsd_weight": 0.0,
    "interaction_cost": 0.0,
    "accept_margin": 0.0,
    "primary_utility": "delta_dice",
    "nsd_tolerance_mm": 2.0,
    "nsd_role": "descriptive_only",
}


def load_mask(
    path: Path, reference: nib.Nifti1Image | None = None
) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    mask = np.asarray(image.dataobj) > 0
    if reference is not None:
        if image.shape != reference.shape:
            raise RuntimeError(
                f"shape mismatch: {path}: {image.shape} != {reference.shape}"
            )
        if not np.allclose(image.affine, reference.affine, atol=1e-4):
            raise RuntimeError(f"affine mismatch: {path}")
    return image, mask


def save_mask(path: Path, mask: np.ndarray, reference: nib.Nifti1Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header), str(path)
    )


def candidate_paths(root: Path, case_id: str) -> dict[str, Path]:
    candidate_root = root / "cases" / case_id / "candidates" / "fusion_only"
    return {
        "KEEP": root / "batches" / "resenc" / "prediction" / f"{case_id}.nii.gz",
        "r1_intersection": candidate_root / "r1_intersection.nii.gz",
        "r2_intersection": candidate_root / "r2_intersection.nii.gz",
        "r1_union": candidate_root / "r1_union.nii.gz",
        "r2_union": candidate_root / "r2_union.nii.gz",
    }


def generate_fusions(root: Path, case_id: str) -> dict[str, Path]:
    paths = candidate_paths(root, case_id)
    reference, resenc = load_mask(paths["KEEP"])
    prompt_masks = {}
    for round_index in (1, 2):
        path = (
            root
            / "batches"
            / f"round{round_index}"
            / "prediction"
            / f"{case_id}.nii.gz"
        )
        _, prompt_masks[round_index] = load_mask(path, reference)
    generated = {
        "r1_intersection": resenc & prompt_masks[1],
        "r2_intersection": resenc & prompt_masks[2],
        "r1_union": resenc | prompt_masks[1],
        "r2_union": resenc | prompt_masks[2],
    }
    for route_id, mask in generated.items():
        destination = paths[route_id]
        if destination.exists():
            _, existing = load_mask(destination, reference)
            if not np.array_equal(existing, mask):
                raise RuntimeError(f"existing fusion mask drift: {destination}")
        else:
            save_mask(destination, mask, reference)
        load_mask(destination, reference)
    return paths


def prompt_metadata(root: Path, case_id: str, round_index: int) -> tuple[dict, Path]:
    path = (
        root / "cases" / case_id / f"round{round_index}" / "simulated_error_clicks.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = {
        "round_index": round_index,
        "foreground_xyz": payload.get("foreground_xyz"),
        "background_xyz": payload.get("background_xyz"),
        "foreground_count": int(payload.get("cumulative_foreground_clicks", 0)),
        "background_count": int(payload.get("cumulative_background_clicks", 0)),
        "new_foreground_count": int(payload.get("foreground_xyz") is not None),
        "new_background_count": int(payload.get("background_xyz") is not None),
        "candidate_proposal_oracle_dependency": "INDIRECT_GT_DEPENDENCE_VIA_ROBOT_USER_CLICKS",
    }
    return metadata, path


def role_fields(role: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        f"{role}_path": str(path.resolve()),
        f"{role}_sha256": sha256_file(path),
    }


def build_records(
    root: Path, studies: list[dict], *, allow_test_gt: bool
) -> list[dict]:
    records = []
    for study in studies:
        if study["split"] == "test" and not allow_test_gt:
            raise RuntimeError("development record builder refuses test ground truth")
        case_id = study["case_id"]
        paths = generate_fusions(root, case_id)
        case_root = root / "cases" / case_id
        ct_path = case_root / "prepared" / "input" / f"{case_id}_0000.nii.gz"
        pet_path = case_root / "prepared" / "input" / f"{case_id}_0001.nii.gz"
        current_path = paths["KEEP"]
        gt_path = ground_truth_path(root, study)
        for route_id in PROPOSAL_ROUTES:
            round_index = int(route_id[1])
            action = route_id.split("_", 1)[1]
            metadata, simulator_path = prompt_metadata(root, case_id, round_index)
            records.append(
                {
                    "case_id": case_id,
                    "patient_id": study["patient_id"],
                    "tracer": study["tracer"],
                    "split": study["split"],
                    "transition_id": f"{case_id}::{route_id}",
                    "action_id": route_id,
                    "route_id": route_id,
                    "action": action,
                    "round_index": round_index,
                    "prior_exposure": True,
                    "external_validation_eligible": False,
                    "prompt_metadata": metadata,
                    "prompt_simulator_manifest_path": str(simulator_path.resolve()),
                    "prompt_simulator_manifest_sha256": sha256_file(simulator_path),
                    **role_fields("pet", pet_path),
                    **role_fields("ct", ct_path),
                    **role_fields("current_mask", current_path),
                    **role_fields("proposed_mask", paths[route_id]),
                    **role_fields("ground_truth", gt_path),
                }
            )
    return records


def validate_record_contract(
    records: list[dict], expected_records: int, allowed_splits: set[str]
) -> None:
    if len(records) != expected_records:
        raise RuntimeError(f"expected {expected_records} records, found {len(records)}")
    seen = set()
    by_case: dict[str, set[str]] = {}
    patient_split: dict[str, str] = {}
    for record in records:
        if record["split"] not in allowed_splits:
            raise RuntimeError(f"forbidden split in manifest: {record['split']}")
        uid = (record["case_id"], record["route_id"])
        if uid in seen:
            raise RuntimeError(f"duplicate route record: {uid}")
        seen.add(uid)
        by_case.setdefault(record["case_id"], set()).add(record["route_id"])
        prior = patient_split.setdefault(record["patient_id"], record["split"])
        if prior != record["split"]:
            raise RuntimeError(f"patient split leakage: {record['patient_id']}")
    expected_routes = set(PROPOSAL_ROUTES)
    for case_id, routes in by_case.items():
        if routes != expected_routes:
            raise RuntimeError(f"route menu mismatch for {case_id}: {routes}")


def provenance(root: Path) -> dict:
    repo = Path(__file__).resolve().parents[1]
    paths = {
        "fusion_only_runner": repo / "scripts" / "run_fusion_only_cohort_v2.py",
        "fusion_only_finalizer": Path(__file__).resolve(),
        "route_policy_eval": repo / "rl_nninteractive" / "route_policy_eval.py",
        "prompt_update_edl": repo / "rl_nninteractive" / "prompt_update_edl.py",
        "evaluate_prompt_routes_cli": repo / "scripts" / "evaluate_prompt_routes.py",
        "train_prompt_update_edl_cli": repo / "scripts" / "train_prompt_update_edl.py",
    }
    return {
        "code_hashes": {role: file_fingerprint(path) for role, path in paths.items()},
        "contract": file_fingerprint(root / "fusion_only_v2_contract.json"),
        "safety_amendment": file_fingerprint(
            root / "fusion_only_v2_safety_amendment_v1.json"
        ),
        "test_seal": file_fingerprint(root / "test_seal.json"),
        "test_seal_amendment": file_fingerprint(root / "test_seal_amendment_v1.json"),
        "reuse_provenance": file_fingerprint(root / "reuse_provenance_binding_v1.json"),
    }


def freeze_development(root: Path) -> dict:
    contract = load_contract(root)
    studies = [study for study in contract["studies"] if study["split"] != "test"]
    if len(studies) != 48:
        raise RuntimeError(f"expected 48 development studies, found {len(studies)}")
    records = build_records(root, studies, allow_test_gt=False)
    validate_record_contract(
        records,
        expected_records=192,
        allowed_splits={"train", "calibration", "policy_validation"},
    )
    payload = {
        "schema_version": 1,
        "frozen_at": iso_now(),
        "status": "FROZEN_DEVELOPMENT",
        "candidate_routes": list(CANDIDATE_ROUTES),
        "baseline_route": "KEEP",
        "proposal_routes": list(PROPOSAL_ROUTES),
        "direct_replacement_forbidden": True,
        "selection_unit": "study decisions clustered by patient",
        "utility_config": UTILITY_CONFIG,
        "counts": {
            "patients": 24,
            "studies": 48,
            "records": 192,
            "test_records": 0,
        },
        "test_outcomes_opened": False,
        "ground_truth_boundary": "Proposals are indirectly GT-dependent through deterministic robot-user clicks; deployable features contain no direct GT arrays, metrics, or scalars.",
        "provenance": provenance(root),
        "records": records,
    }
    path = root / "fusion_only_development_manifest.json"
    digest = write_frozen_json(path, payload)
    return {"path": str(path.resolve()), "sha256": digest, "records": len(records)}


def freeze_test_open(root: Path, deployment_plan: Path) -> dict:
    clearance = require_test_opening_clearance(root, "freeze-test-manifest")
    contract = load_contract(root)
    studies = [study for study in contract["studies"] if study["split"] == "test"]
    if len(studies) != 12:
        raise RuntimeError(f"expected 12 test studies, found {len(studies)}")
    dev_manifest = Path(clearance["dev_manifest_path"])
    deployment_plan = deployment_plan.resolve()
    deployment_sha = sha256_file(deployment_plan)
    bundle = json.loads(
        Path(clearance["selector_bundle_path"]).read_text(encoding="utf-8")
    )
    if bundle.get("route_policy_deployment_sha256") != deployment_sha:
        raise RuntimeError("selector bundle/deployment plan hash mismatch")
    records = build_records(root, studies, allow_test_gt=True)
    validate_record_contract(records, expected_records=48, allowed_splits={"test"})
    payload = {
        "schema_version": 1,
        "frozen_at": iso_now(),
        "status": "FROZEN_TEST_OPEN",
        "candidate_routes": list(CANDIDATE_ROUTES),
        "baseline_route": "KEEP",
        "proposal_routes": list(PROPOSAL_ROUTES),
        "direct_replacement_forbidden": True,
        "development_manifest_sha256": sha256_file(dev_manifest),
        "deployment_sha256": deployment_sha,
        "utility_config": UTILITY_CONFIG,
        "counts": {"patients": 6, "studies": 12, "records": 48},
        "test_opening_clearance": clearance,
        "provenance": provenance(root),
        "records": records,
    }
    path = root / "fusion_only_test_open_manifest.json"
    digest = write_frozen_json(path, payload)
    return {"path": str(path.resolve()), "sha256": digest, "records": len(records)}


def freeze_selector_bundle(
    root: Path,
    dev_manifest: Path,
    edl_checkpoint: Path,
    deployment_plan: Path,
) -> dict:
    dev_manifest = dev_manifest.resolve()
    edl_checkpoint = edl_checkpoint.resolve()
    deployment_plan = deployment_plan.resolve()
    dev = json.loads(dev_manifest.read_text(encoding="utf-8"))
    if (
        dev.get("status") != "FROZEN_DEVELOPMENT"
        or dev.get("counts", {}).get("records") != 192
    ):
        raise RuntimeError(
            "selector bundle requires exact frozen 192-record development manifest"
        )
    plan = json.loads(deployment_plan.read_text(encoding="utf-8"))
    selected_policies = validate_deployment_plan(
        plan,
        dev_manifest=dev_manifest,
        edl_checkpoint=edl_checkpoint,
        deployment_plan=deployment_plan,
    )
    prov = provenance(root)
    code_hashes = {
        role: fingerprint
        for role, fingerprint in prov["code_hashes"].items()
        if role
        in {
            "fusion_only_runner",
            "fusion_only_finalizer",
            "route_policy_eval",
            "prompt_update_edl",
        }
    }
    payload = {
        "schema_version": 1,
        "frozen_at": iso_now(),
        "status": "FROZEN_BEFORE_TEST_OPENING",
        "test_outcomes_opened": False,
        "dev_manifest_path": str(dev_manifest),
        "dev_manifest_sha256": sha256_file(dev_manifest),
        "edl_checkpoint_path": str(edl_checkpoint),
        "edl_checkpoint_sha256": sha256_file(edl_checkpoint),
        "route_policy_deployment_path": str(deployment_plan),
        "route_policy_deployment_sha256": sha256_file(deployment_plan),
        "route_policy_deployment_sidecar": file_fingerprint(
            deployment_plan.with_suffix(".sha256")
        ),
        "selected_policies": selected_policies,
        "utility_config": UTILITY_CONFIG,
        "code_hashes": code_hashes,
        "contract_sha256": prov["contract"]["sha256"],
        "safety_amendment_sha256": prov["safety_amendment"]["sha256"],
        "test_seal_sha256": prov["test_seal"]["sha256"],
        "test_seal_amendment_sha256": prov["test_seal_amendment"]["sha256"],
        "reuse_provenance_sha256": prov["reuse_provenance"]["sha256"],
    }
    path = root / "fusion_only_selector_bundle.json"
    digest = write_frozen_json(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "selected_policies": selected_policies,
    }


def _require_hex64(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise RuntimeError(f"{field} must be 64 lowercase hexadecimal characters")
    return text


def validate_deployment_plan(
    plan: dict,
    *,
    dev_manifest: Path,
    edl_checkpoint: Path,
    deployment_plan: Path,
) -> dict[str, dict]:
    if (
        plan.get("schema_version") != 1
        or plan.get("artifact_type") != "frozen_route_policy_deployment"
        or plan.get("status") != "FROZEN_DEVELOPMENT"
    ):
        raise RuntimeError("deployment plan schema/artifact/status mismatch")
    manifest_hash = sha256_file(dev_manifest)
    development = plan.get("development_manifest")
    if not isinstance(development, dict):
        raise RuntimeError("deployment plan development manifest block absent")
    if development.get("sha256") != manifest_hash:
        raise RuntimeError("deployment plan/development manifest hash mismatch")
    if (
        development.get("record_count") != 192
        or development.get("test_records_opened") != 0
    ):
        raise RuntimeError(
            "deployment plan did not freeze exact 192-record no-test development cohort"
        )
    if development.get("candidate_routes") != list(CANDIDATE_ROUTES):
        raise RuntimeError("deployment plan fusion candidate menu mismatch")
    if development.get("splits_opened") != [
        "train",
        "calibration",
        "policy_validation",
    ]:
        raise RuntimeError("deployment plan development split opening mismatch")
    route_contract = plan.get("route_contract")
    if not isinstance(route_contract, dict) or route_contract.get(
        "candidate_routes"
    ) != list(CANDIDATE_ROUTES):
        raise RuntimeError("deployment route contract menu mismatch")
    utility = plan.get("utility_definition")
    expected_utility = {
        "delta_dice_weight": 1.0,
        "nsd_tolerance_mm": 2.0,
        "nsd_weight": 0.0,
        "interaction_cost": 0.0,
        "accept_margin": 0.0,
    }
    if not isinstance(utility, dict) or any(
        float(utility.get(key, float("nan"))) != expected
        for key, expected in expected_utility.items()
    ):
        raise RuntimeError(
            "deployment utility definition differs from frozen Dice-only flags"
        )
    safety = plan.get("policy_safety_contract")
    if not isinstance(safety, dict):
        raise RuntimeError("deployment safety contract absent")
    bootstrap = safety.get("bootstrap")
    if (
        safety.get("selection_split") != "policy_validation"
        or float(safety.get("max_harmful_study_rate", -1)) != 0.05
        or not isinstance(bootstrap, dict)
        or bootstrap.get("method") != "patient-cluster percentile bootstrap"
        or bootstrap.get("samples") != 10000
        or bootstrap.get("seed") != 20260715
        or safety.get("applies_to")
        != ["edl_accept_gate", "full_information_linear_ridge_comparator"]
        or safety.get("fallback") != "KEEP_ALL"
        or safety.get("ridge_lambdas") != [0.01, 0.1, 1.0, 10.0]
    ):
        raise RuntimeError("deployment safety constants differ from frozen amendment")
    edl_grid = safety.get("edl_grid")
    if (
        not isinstance(edl_grid, dict)
        or edl_grid.get("candidate_grid_size") != 280
        or edl_grid.get("min_predicted_utility") != [-0.01, 0.0, 0.01, 0.02]
    ):
        raise RuntimeError("deployment EDL grid differs from frozen amendment")
    if safety.get("tie_break_order") != [
        "patient_mean_realized_utility",
        "patient_cluster_bootstrap_95_ci_lower",
        "study_mean_realized_utility",
        "lower_harmful_study_rate",
        "coverage_then_deterministic_model_fields",
    ]:
        raise RuntimeError("deployment safety tie order drift")
    edl = plan.get("edl")
    ridge = plan.get("full_information_linear_ridge")
    if not isinstance(edl, dict) or not isinstance(ridge, dict):
        raise RuntimeError("deployment EDL/ridge blocks absent")
    checkpoint_hash = sha256_file(edl_checkpoint)
    if edl.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("deployment plan/EDL checkpoint hash mismatch")
    if edl.get("manifest_sha256") != manifest_hash:
        raise RuntimeError(
            "EDL checkpoint was not trained on exact development manifest"
        )
    thresholds = edl.get("deployed_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {
        "accept_probability",
        "max_accept_vacuity",
        "min_predicted_utility",
    }:
        raise RuntimeError("EDL deployed thresholds are absent or ambiguous")
    selection = edl.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeError("EDL policy-validation selection report absent")
    edl_decision = selection.get("deployment_decision")
    expected_keep = edl_decision == "KEEP_ALL"
    if edl_decision not in {"KEEP_ALL", "SELECT_ROUTE_OR_KEEP"}:
        raise RuntimeError("invalid EDL deployment decision")
    if bool(edl.get("deploy_keep_all")) != expected_keep:
        raise RuntimeError("EDL KEEP_ALL flag/decision mismatch")
    config_sha = _require_hex64(edl.get("config_sha256"), "edl.config_sha256")
    code_sha = _require_hex64(edl.get("code_sha256"), "edl.code_sha256")
    model = ridge.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("ridge selected model absent")
    ridge_decision = model.get("deployment_decision")
    if ridge_decision not in {"KEEP_ALL", "SELECT_ROUTE_OR_KEEP"}:
        raise RuntimeError("invalid ridge deployment decision")
    coefficients_sha = _require_hex64(
        model.get("coefficients_sha256"), "ridge.coefficients_sha256"
    )
    normalized = {
        "edl_accept_gate": {
            "deployment_decision": edl_decision,
            "deploy_keep_all": expected_keep,
            "thresholds": thresholds,
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": config_sha,
            "code_sha256": code_sha,
        },
        "full_information_linear_ridge_comparator": {
            "deployment_decision": ridge_decision,
            "deploy_keep_all": ridge_decision == "KEEP_ALL",
            "ridge_lambda": float(model["ridge_lambda"]),
            "accept_threshold": float(model["accept_threshold"]),
            "coefficients_sha256": coefficients_sha,
        },
    }
    if plan.get("selected_policies") != normalized:
        raise RuntimeError(
            "deployment selected_policies disagrees with EDL/ridge blocks"
        )
    control = plan.get("test_open_control")
    if not isinstance(control, dict) or control.get("pass_limit") != 1:
        raise RuntimeError("deployment one-pass test opening control absent")
    for key in (
        "attempt_receipt_path",
        "completion_receipt_path",
        "failure_receipt_path",
    ):
        if not Path(str(control.get(key, ""))).is_absolute():
            raise RuntimeError(f"deployment test receipt path is not absolute: {key}")
    sidecar = deployment_plan.with_suffix(".sha256")
    if not sidecar.is_file():
        raise RuntimeError("route policy deployment sidecar absent")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256_file(deployment_plan):
        raise RuntimeError("route policy deployment sidecar mismatch")
    return normalized


def self_test_test_guard(report_root: Path) -> dict:
    """Prove missing/mismatched clearance fails before GT resolution or image I/O."""

    calls = {"ground_truth_path": 0, "nib_load": 0}
    original_ground_truth_path = globals()["ground_truth_path"]
    original_nib_load = nib.load

    def probe_ground_truth_path(*args, **kwargs):
        calls["ground_truth_path"] += 1
        raise AssertionError("ground_truth_path reached before clearance")

    def probe_nib_load(*args, **kwargs):
        calls["nib_load"] += 1
        raise AssertionError("nib.load reached before clearance")

    globals()["ground_truth_path"] = probe_ground_truth_path
    nib.load = probe_nib_load
    missing_rejected = False
    mismatch_rejected = False
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            try:
                freeze_test_open(root, root / "missing_deployment.json")
            except RuntimeError:
                missing_rejected = True
            payload = '{"schema_version":1,"status":"AUDIT_CLEARED"}\n'
            for name in ("test_opening_clearance.json", "test_open_clearance.json"):
                path = root / name
                path.write_text(payload, encoding="utf-8")
                path.with_suffix(path.suffix + ".sha256").write_text(
                    f"{'0' * 64}  {name}\n", encoding="utf-8"
                )
            try:
                freeze_test_open(root, root / "missing_deployment.json")
            except RuntimeError:
                mismatch_rejected = True
    finally:
        globals()["ground_truth_path"] = original_ground_truth_path
        nib.load = original_nib_load
    if (
        not missing_rejected
        or not mismatch_rejected
        or calls
        != {
            "ground_truth_path": 0,
            "nib_load": 0,
        }
    ):
        raise AssertionError(
            f"test finalizer did not fail closed: missing={missing_rejected}, "
            f"mismatch={mismatch_rejected}, calls={calls}"
        )
    result = {
        "schema_version": 1,
        "finished_at": iso_now(),
        "missing_clearance_rejected": True,
        "mismatched_clearance_hash_rejected": True,
        "ground_truth_path_calls_before_clearance": 0,
        "nib_load_calls_before_clearance": 0,
        "finalizer": file_fingerprint(Path(__file__).resolve()),
    }
    path = report_root / "fusion_only_finalizer_guard_probe_v2.json"
    digest = write_frozen_json(path, result)
    return {**result, "report_path": str(path), "report_sha256": digest}


def self_test_selector_integration(report_root: Path) -> dict:
    """Exercise the exact phase-A deployment schema and a broken cross-link."""

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name in (
            "fusion_only_v2_contract.json",
            "fusion_only_v2_safety_amendment_v1.json",
            "test_seal.json",
            "test_seal_amendment_v1.json",
            "reuse_provenance_binding_v1.json",
        ):
            (root / name).write_text("{}\n", encoding="utf-8")
        dev_manifest = root / "fusion_only_development_manifest.json"
        dev_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "FROZEN_DEVELOPMENT",
                    "candidate_routes": list(CANDIDATE_ROUTES),
                    "counts": {"records": 192, "test_records": 0},
                    "records": [],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint = root / "edl_checkpoint.pt"
        checkpoint.write_bytes(b"synthetic-phase-a-edl-checkpoint")
        manifest_sha = sha256_file(dev_manifest)
        checkpoint_sha = sha256_file(checkpoint)
        config_sha = "1" * 64
        coefficient_sha = "2" * 64
        code_sha = "3" * 64
        edl_thresholds = {
            "accept_probability": 0.8,
            "max_accept_vacuity": 0.3,
            "min_predicted_utility": 0.02,
        }
        selected_policies = {
            "edl_accept_gate": {
                "deployment_decision": "KEEP_ALL",
                "deploy_keep_all": True,
                "thresholds": edl_thresholds,
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "code_sha256": code_sha,
            },
            "full_information_linear_ridge_comparator": {
                "deployment_decision": "SELECT_ROUTE_OR_KEEP",
                "deploy_keep_all": False,
                "ridge_lambda": 0.1,
                "accept_threshold": 0.01,
                "coefficients_sha256": coefficient_sha,
            },
        }
        plan = {
            "schema_version": 1,
            "artifact_type": "frozen_route_policy_deployment",
            "status": "FROZEN_DEVELOPMENT",
            "development_manifest": {
                "path": str(dev_manifest),
                "sha256": manifest_sha,
                "record_count": 192,
                "candidate_routes": list(CANDIDATE_ROUTES),
                "splits_opened": ["train", "calibration", "policy_validation"],
                "test_records_opened": 0,
            },
            "route_contract": {"candidate_routes": list(CANDIDATE_ROUTES)},
            "utility_definition": {
                "delta_dice_weight": 1.0,
                "nsd_tolerance_mm": 2.0,
                "nsd_weight": 0.0,
                "interaction_cost": 0.0,
                "accept_margin": 0.0,
            },
            "policy_safety_contract": {
                "selection_split": "policy_validation",
                "max_harmful_study_rate": 0.05,
                "bootstrap": {
                    "method": "patient-cluster percentile bootstrap",
                    "samples": 10000,
                    "seed": 20260715,
                },
                "applies_to": [
                    "edl_accept_gate",
                    "full_information_linear_ridge_comparator",
                ],
                "fallback": "KEEP_ALL",
                "edl_grid": {
                    "accept_probability": [],
                    "max_accept_vacuity": [],
                    "min_predicted_utility": [-0.01, 0.0, 0.01, 0.02],
                    "candidate_grid_size": 280,
                },
                "ridge_lambdas": [0.01, 0.1, 1.0, 10.0],
                "tie_break_order": [
                    "patient_mean_realized_utility",
                    "patient_cluster_bootstrap_95_ci_lower",
                    "study_mean_realized_utility",
                    "lower_harmful_study_rate",
                    "coverage_then_deterministic_model_fields",
                ],
            },
            "edl": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": manifest_sha,
                "config_sha256": config_sha,
                "code_sha256": code_sha,
                "deployed_thresholds": edl_thresholds,
                "selection": {"deployment_decision": "KEEP_ALL"},
                "deploy_keep_all": True,
            },
            "full_information_linear_ridge": {
                "model": {
                    "deployment_decision": "SELECT_ROUTE_OR_KEEP",
                    "ridge_lambda": 0.1,
                    "accept_threshold": 0.01,
                    "coefficients_sha256": coefficient_sha,
                },
                "fit_report": {},
            },
            "selected_policies": selected_policies,
            "test_open_control": {
                "pass_limit": 1,
                "attempt_receipt_path": str(root / "attempt.json"),
                "completion_receipt_path": str(root / "completion.json"),
                "failure_receipt_path": str(root / "failure.json"),
            },
        }
        deployment = root / "route_policy_deployment.json"
        deployment.write_text(
            json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8"
        )
        deployment.with_suffix(".sha256").write_text(
            f"{sha256_file(deployment)}  {deployment.name}\n", encoding="utf-8"
        )
        bundle = freeze_selector_bundle(root, dev_manifest, checkpoint, deployment)
        bundle_payload = json.loads(Path(bundle["path"]).read_text(encoding="utf-8"))
        if bundle_payload["selected_policies"] != selected_policies:
            raise AssertionError("selector bundle policy mapping drift")
        if (
            bundle_payload["selected_policies"]["edl_accept_gate"]["code_sha256"]
            != code_sha
        ):
            raise AssertionError("selector bundle dropped EDL code provenance")
        plan["edl"]["code_sha256"] = "invalid"
        broken_code_hash_rejected = False
        try:
            validate_deployment_plan(
                plan,
                dev_manifest=dev_manifest,
                edl_checkpoint=checkpoint,
                deployment_plan=deployment,
            )
        except RuntimeError as exc:
            broken_code_hash_rejected = "edl.code_sha256" in str(exc)
        if not broken_code_hash_rejected:
            raise AssertionError("invalid EDL code provenance hash was accepted")
        plan["edl"]["code_sha256"] = code_sha
        plan["development_manifest"]["sha256"] = "0" * 64
        deployment.write_text(
            json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8"
        )
        deployment.with_suffix(".sha256").write_text(
            f"{sha256_file(deployment)}  {deployment.name}\n", encoding="utf-8"
        )
        broken_crosslink_rejected = False
        try:
            validate_deployment_plan(
                plan,
                dev_manifest=dev_manifest,
                edl_checkpoint=checkpoint,
                deployment_plan=deployment,
            )
        except RuntimeError:
            broken_crosslink_rejected = True
        if not broken_crosslink_rejected:
            raise AssertionError("broken development-manifest cross-link was accepted")
    result = {
        "schema_version": 1,
        "finished_at": iso_now(),
        "actual_phase_a_schema_accepted": True,
        "selected_policies_exactly_mapped": True,
        "edl_code_sha256_preserved": True,
        "invalid_edl_code_sha256_rejected": True,
        "broken_development_manifest_crosslink_rejected": True,
        "test_outcomes_opened": False,
        "finalizer": file_fingerprint(Path(__file__).resolve()),
    }
    path = report_root / "fusion_only_selector_integration_probe.json"
    digest = write_frozen_json(path, result)
    return {**result, "report_path": str(path), "report_sha256": digest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze-development",
            "freeze-selector-bundle",
            "freeze-test-open",
            "self-test-test-guard",
            "self-test-selector-integration",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path)
    parser.add_argument("--edl-checkpoint", type=Path)
    parser.add_argument("--deployment-plan", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.command == "freeze-development":
        result = freeze_development(root)
    elif args.command == "freeze-selector-bundle":
        for name in ("dev_manifest", "edl_checkpoint", "deployment_plan"):
            if getattr(args, name) is None:
                raise ValueError(f"--{name.replace('_', '-')} is required")
        result = freeze_selector_bundle(
            root, args.dev_manifest, args.edl_checkpoint, args.deployment_plan
        )
    elif args.command == "freeze-test-open":
        if args.deployment_plan is None:
            raise ValueError("--deployment-plan is required")
        result = freeze_test_open(root, args.deployment_plan)
    elif args.command == "self-test-test-guard":
        result = self_test_test_guard(root)
    elif args.command == "self-test-selector-integration":
        result = self_test_selector_integration(root)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
