"""Post-failure, development-only EDL veto for safe fusion routes.

This module is deliberately isolated from the original frozen route deployment.
It freezes a post-hoc zero-observed-harm consensus/PET screen and lets a newly
cross-fitted EDL head *veto* (never change) the screen-selected route. Robot-user
prompts and proposal masks are indirectly ground-truth-derived; policy feature
construction never directly accesses ground truth or outcome metrics. The
standalone CLI has no test-scoring mode. A future sealed-test integration must
run through the existing canonical one-shot evaluator and receipt path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch

from .prompt_update_edl import (
    FEATURE_ORDER,
    FeatureNormalizer,
    EvidentialUtilityHead,
    _temperature_scale,
    calibrate_temperature,
    dice_score,
    evidential_utility_loss,
    extract_update_features,
)


PROTOCOL_SCHEMA_VERSION = 6
FEATURE_TABLE_SCHEMA_VERSION = 1
FREEZE_SCHEMA_VERSION = 1
DEVELOPMENT_SPLITS = frozenset({"train", "calibration", "policy_validation"})
TRACERS = ("FDG", "PSMA")
ROUTES = ("r1_intersection", "r2_intersection", "r1_union", "r2_union")
QUANTILE_GRID = tuple(float(value) for value in np.linspace(0.1, 0.9, 9))
BOOTSTRAP_SAMPLES = 10_000
SEED = 20260715
HARM_TOLERANCE = 1e-12
TEST_MAX_HARM_RATE = 0.05
INNER_HARM_RATE = 0.0
MINIMUM_INNER_COVERAGE = 4
RULE_POOL_SIZE = 15
STABLE_SPLIT_PREFIX = "fusion-edl-calibration-v1::"
EDL_HIDDEN = 48
EDL_EPOCHS = 300
EDL_LR = 3e-3
EDL_WEIGHT_DECAY = 1e-4
EDL_GATE = {
    "accept_probability": 0.5,
    "max_accept_vacuity": 0.6,
    "min_predicted_utility": 0.0,
}
EXPECTED_CASE_COUNT = 48
EXPECTED_PATIENT_COUNT = 24
EXPECTED_RECORD_COUNT = 192
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_ORDER)}


@dataclass(frozen=True)
class HybridCandidate:
    """One label-free fusion proposal plus an optional development outcome."""

    case_id: str
    patient_id: str
    split: str
    tracer: str
    route: str
    features: np.ndarray
    round_agreement_dice: float
    delta_dice: float | None = None

    @property
    def uid(self) -> str:
        return f"{self.case_id}::{self.route}"

    @property
    def changed(self) -> bool:
        return bool(self.feature("changed_volume_fraction") > 0.0)

    def feature(self, name: str) -> float:
        if name == "round_agreement_dice":
            return float(self.round_agreement_dice)
        try:
            return float(self.features[FEATURE_INDEX[name]])
        except KeyError as exc:
            raise ValueError(f"unknown hybrid feature: {name}") from exc

    def development_delta(self) -> float:
        if self.delta_dice is None:
            raise ValueError(f"development outcome is unavailable for {self.uid}")
        return float(self.delta_dice)


@dataclass(frozen=True)
class TrainedEDL:
    model: EvidentialUtilityHead
    normalizer: FeatureNormalizer
    temperature: float
    fit_patients: tuple[str, ...]
    calibration_patients: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical_json(value), encoding="utf-8", newline="\n")


def _write_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    return digest


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj), np.asarray(image.affine, dtype=np.float64)


def _binary_dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = int(left.sum()) + int(right.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(left, right).sum() / denominator)


def _require_same_grid(
    case_id: str,
    role: str,
    array: np.ndarray,
    affine: np.ndarray,
    reference_array: np.ndarray,
    reference_affine: np.ndarray,
) -> None:
    if array.shape != reference_array.shape or not np.allclose(
        affine, reference_affine, atol=1e-4
    ):
        raise ValueError(f"{case_id}/{role} is not on the PET grid")


def validate_development_manifest_metadata(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Reject test or malformed metadata before resolving a single file path."""

    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("development manifest schema_version must be 1")
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("development manifest has no records")
    observed_splits = {str(record.get("split", "")) for record in records}
    if observed_splits != DEVELOPMENT_SPLITS:
        raise ValueError(
            "hybrid development freeze requires exactly train/calibration/"
            f"policy_validation and zero test records; got {sorted(observed_splits)}"
        )
    if bool(payload.get("test_outcomes_opened", False)):
        raise ValueError("development manifest reports opened test outcomes")
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"expected {EXPECTED_RECORD_COUNT} proposal records, got {len(records)}"
        )

    patients = {str(record.get("patient_id", "")) for record in records}
    cases = {str(record.get("case_id", "")) for record in records}
    if "" in patients or len(patients) != EXPECTED_PATIENT_COUNT:
        raise ValueError("development manifest must contain exactly 24 patient IDs")
    if "" in cases or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("development manifest must contain exactly 48 case IDs")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["case_id"])].append(record)
    for case_id, group in grouped.items():
        routes = {str(record.get("route_id", "")) for record in group}
        if len(group) != len(ROUTES) or routes != set(ROUTES):
            raise ValueError(f"{case_id} does not contain the exact four-route menu")
        tracers = {str(record.get("tracer", "")) for record in group}
        patients_in_group = {str(record.get("patient_id", "")) for record in group}
        if len(tracers) != 1 or not tracers.issubset(TRACERS):
            raise ValueError(f"{case_id} has an invalid tracer binding")
        if len(patients_in_group) != 1:
            raise ValueError(f"{case_id} mixes patient IDs")
        for record in group:
            route = str(record["route_id"])
            action = str(record.get("action", ""))
            round_index = int(record.get("round_index", -1))
            if route != f"r{round_index}_{action}":
                raise ValueError(f"{case_id}/{route} action-round fields disagree")
            for role in ("pet", "ct", "current_mask", "proposed_mask", "ground_truth"):
                digest = str(record.get(f"{role}_sha256", ""))
                if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
                    raise ValueError(f"{case_id}/{route} lacks a valid {role} SHA-256")

    cases_by_patient: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for group in grouped.values():
        cases_by_patient[str(group[0]["patient_id"])].append(group[0])
    for patient_id, patient_cases in cases_by_patient.items():
        if len(patient_cases) != 2 or {
            str(record["tracer"]) for record in patient_cases
        } != set(TRACERS):
            raise ValueError(f"{patient_id} must have exactly one FDG and one PSMA case")
    return records


def build_development_feature_table(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    volume_loader: Callable[[Path], tuple[np.ndarray, np.ndarray]] = _load_nifti,
) -> dict[str, Any]:
    """Build the GT-free feature table and development-only Dice labels.

    Metadata guards run before any path resolution, hashing, or volume load.  The
    optional loader exists so tests can prove this ordering fail-closed.
    """

    path = Path(manifest_path).resolve()
    observed_hash = sha256_file(path)
    if observed_hash != str(expected_manifest_sha256).lower():
        raise ValueError("development manifest SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = validate_development_manifest_metadata(payload)
    root = path.parent
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["case_id"])].append(record)

    hash_cache: dict[Path, str] = {}

    def verify(role_path: Path, expected: str) -> None:
        if role_path not in hash_cache:
            hash_cache[role_path] = sha256_file(role_path)
        if hash_cache[role_path] != expected.lower():
            raise ValueError(f"SHA-256 mismatch for {role_path.name}")

    rows: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        group = sorted(grouped[case_id], key=lambda record: str(record["route_id"]))
        first = group[0]
        common: dict[str, tuple[Path, str]] = {}
        for role in ("pet", "ct", "current_mask", "ground_truth"):
            paths = {_resolve(root, str(record[f"{role}_path"])) for record in group}
            hashes = {str(record[f"{role}_sha256"]).lower() for record in group}
            if len(paths) != 1 or len(hashes) != 1:
                raise ValueError(f"{case_id} routes do not share one {role}")
            common[role] = (next(iter(paths)), next(iter(hashes)))

        loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role, (role_path, expected_hash) in common.items():
            verify(role_path, expected_hash)
            loaded[role] = volume_loader(role_path)
        pet, pet_affine = loaded["pet"]
        for role in ("ct", "current_mask", "ground_truth"):
            array, affine = loaded[role]
            _require_same_grid(case_id, role, array, affine, pet, pet_affine)
        pet = np.asarray(pet, dtype=np.float32)
        ct = np.asarray(loaded["ct"][0], dtype=np.float32)
        current = np.asarray(loaded["current_mask"][0]) > 0.5
        ground_truth = np.asarray(loaded["ground_truth"][0]) > 0.5
        baseline_dice = dice_score(current, ground_truth)

        proposals: dict[str, np.ndarray] = {}
        pending: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray]] = []
        for record in group:
            route = str(record["route_id"])
            proposal_path = _resolve(root, str(record["proposed_mask_path"]))
            verify(proposal_path, str(record["proposed_mask_sha256"]).lower())
            proposal_array, proposal_affine = volume_loader(proposal_path)
            _require_same_grid(
                case_id,
                route,
                proposal_array,
                proposal_affine,
                pet,
                pet_affine,
            )
            proposed = np.asarray(proposal_array) > 0.5
            proposals[route] = proposed
            features = extract_update_features(
                pet,
                ct,
                current,
                proposed,
                prompt_metadata=record.get("prompt_metadata"),
            )
            pending.append((record, proposed, features))

        agreement = {
            action: _binary_dice(
                proposals[f"r1_{action}"], proposals[f"r2_{action}"]
            )
            for action in ("intersection", "union")
        }
        for record, proposed, features in pending:
            route = str(record["route_id"])
            action = "intersection" if "intersection" in route else "union"
            candidate_dice = dice_score(proposed, ground_truth)
            rows.append(
                {
                    "case_id": case_id,
                    "patient_id": str(record["patient_id"]),
                    "split": str(record["split"]),
                    "tracer": str(record["tracer"]),
                    "route": route,
                    "features": {
                        name: float(value)
                        for name, value in zip(FEATURE_ORDER, features, strict=True)
                    },
                    "round_agreement_dice": float(agreement[action]),
                    "delta_dice": float(candidate_dice - baseline_dice),
                }
            )
    return {
        "schema_version": FEATURE_TABLE_SCHEMA_VERSION,
        "artifact_type": "edl_hybrid_development_feature_table",
        "status": "DEVELOPMENT_ONLY_NO_TEST",
        "manifest_sha256": observed_hash,
        "feature_order": list(FEATURE_ORDER),
        "counts": {
            "patients": EXPECTED_PATIENT_COUNT,
            "studies": EXPECTED_CASE_COUNT,
            "records": EXPECTED_RECORD_COUNT,
            "test_records": 0,
        },
        "feature_provenance": {
            "direct_ground_truth_and_outcomes_forbidden": True,
            "indirect_ground_truth_dependence_via_robot_prompts_and_proposals": True,
            "development_only_label": "delta_dice",
        },
        "rows": rows,
    }


def build_label_free_test_rows(
    test_manifest: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    volume_loader: Callable[[Path], tuple[np.ndarray, np.ndarray]] = _load_nifti,
) -> list[HybridCandidate]:
    """Build sealed-test policy inputs without touching any ground-truth field.

    The canonical outer one-shot transaction must call this only after it has
    atomically claimed the frozen receipt and established its process-secret
    receipt token. This helper deliberately has no receipt-path override and
    never reads, resolves, validates, hashes, or loads ``ground_truth*`` values.
    """

    manifest = Path(test_manifest).resolve()
    manifest_hash = sha256_file(manifest)
    if expected_manifest_sha256 is not None and manifest_hash != str(
        expected_manifest_sha256
    ).lower():
        raise ValueError("test manifest SHA-256 mismatch")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("test route manifest schema_version must be 1")
    records = list(payload.get("records", []))
    if len(records) != 48 or {str(record.get("split", "")) for record in records} != {
        "test"
    }:
        raise ValueError("hybrid test feature builder requires 48 test-only records")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        case_id = str(record.get("case_id", ""))
        patient_id = str(record.get("patient_id", ""))
        tracer = str(record.get("tracer", ""))
        route = str(record.get("route_id", ""))
        if not case_id or not patient_id or tracer not in TRACERS or route not in ROUTES:
            raise ValueError("invalid label-free test record metadata")
        round_index = int(record.get("round_index", -1))
        action = str(record.get("action", ""))
        if route != f"r{round_index}_{action}":
            raise ValueError(f"{case_id}/{route} action-round fields disagree")
        for role in ("pet", "ct", "current_mask", "proposed_mask"):
            digest = str(record.get(f"{role}_sha256", ""))
            if len(digest) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in digest
            ):
                raise ValueError(f"{case_id}/{route} lacks a valid {role} SHA-256")
        grouped[case_id].append(record)
    if len(grouped) != 12:
        raise ValueError("hybrid test feature builder requires exactly 12 studies")
    patients = {str(record["patient_id"]) for record in records}
    if len(patients) != 6:
        raise ValueError("hybrid test feature builder requires exactly 6 patients")
    for case_id, group in grouped.items():
        if len(group) != 4 or {str(record["route_id"]) for record in group} != set(
            ROUTES
        ):
            raise ValueError(f"{case_id} lacks the exact fusion menu")
        if len({str(record["patient_id"]) for record in group}) != 1 or len(
            {str(record["tracer"]) for record in group}
        ) != 1:
            raise ValueError(f"{case_id} mixes patient or tracer IDs")
    cases_by_patient: dict[str, set[str]] = defaultdict(set)
    for group in grouped.values():
        cases_by_patient[str(group[0]["patient_id"])].add(str(group[0]["tracer"]))
    if any(tracers != set(TRACERS) for tracers in cases_by_patient.values()):
        raise ValueError("each test patient must have one FDG and one PSMA study")

    root = manifest.parent
    hash_cache: dict[Path, str] = {}

    def verify(path: Path, expected: str) -> None:
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        if hash_cache[path] != expected.lower():
            raise ValueError(f"SHA-256 mismatch for {path.name}")

    candidates: list[HybridCandidate] = []
    for case_id in sorted(grouped):
        group = sorted(grouped[case_id], key=lambda record: str(record["route_id"]))
        common: dict[str, tuple[Path, str]] = {}
        for role in ("pet", "ct", "current_mask"):
            paths = {_resolve(root, str(record[f"{role}_path"])) for record in group}
            hashes = {str(record[f"{role}_sha256"]).lower() for record in group}
            if len(paths) != 1 or len(hashes) != 1:
                raise ValueError(f"{case_id} routes do not share one {role}")
            common[role] = (next(iter(paths)), next(iter(hashes)))
        loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role, (path, expected_hash) in common.items():
            verify(path, expected_hash)
            loaded[role] = volume_loader(path)
        pet, pet_affine = loaded["pet"]
        for role in ("ct", "current_mask"):
            array, affine = loaded[role]
            _require_same_grid(case_id, role, array, affine, pet, pet_affine)
        pet_array = np.asarray(pet, dtype=np.float32)
        ct_array = np.asarray(loaded["ct"][0], dtype=np.float32)
        current = np.asarray(loaded["current_mask"][0]) > 0.5
        proposals: dict[str, np.ndarray] = {}
        pending: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray]] = []
        for record in group:
            route = str(record["route_id"])
            proposal_path = _resolve(root, str(record["proposed_mask_path"]))
            verify(proposal_path, str(record["proposed_mask_sha256"]).lower())
            proposal_array, proposal_affine = volume_loader(proposal_path)
            _require_same_grid(
                case_id, route, proposal_array, proposal_affine, pet_array, pet_affine
            )
            proposed = np.asarray(proposal_array) > 0.5
            proposals[route] = proposed
            features = extract_update_features(
                pet_array,
                ct_array,
                current,
                proposed,
                prompt_metadata=record.get("prompt_metadata"),
            )
            pending.append((record, proposed, features))
        agreement = {
            action: _binary_dice(
                proposals[f"r1_{action}"], proposals[f"r2_{action}"]
            )
            for action in ("intersection", "union")
        }
        for record, _proposed, features in pending:
            route = str(record["route_id"])
            action = "intersection" if "intersection" in route else "union"
            candidates.append(
                HybridCandidate(
                    case_id=case_id,
                    patient_id=str(record["patient_id"]),
                    split="test",
                    tracer=str(record["tracer"]),
                    route=route,
                    features=np.asarray(features, dtype=np.float32),
                    round_agreement_dice=float(agreement[action]),
                    delta_dice=None,
                )
            )
    validate_candidate_menu(candidates, require_outcomes=False)
    return candidates


def build_test_feature_rows(
    test_manifest: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    volume_loader: Callable[[Path], tuple[np.ndarray, np.ndarray]] = _load_nifti,
) -> list[HybridCandidate]:
    """Compatibility alias for the single canonical label-free implementation."""

    return build_label_free_test_rows(
        test_manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        volume_loader=volume_loader,
    )


def candidates_from_feature_table(payload: Mapping[str, Any]) -> list[HybridCandidate]:
    if int(payload.get("schema_version", -1)) != FEATURE_TABLE_SCHEMA_VERSION:
        raise ValueError("hybrid feature table schema mismatch")
    if tuple(payload.get("feature_order", ())) != tuple(FEATURE_ORDER):
        raise ValueError("hybrid feature order mismatch")
    rows = list(payload.get("rows", []))
    if len(rows) != EXPECTED_RECORD_COUNT:
        raise ValueError("hybrid feature table must contain 192 rows")
    if {str(row.get("split", "")) for row in rows} != DEVELOPMENT_SPLITS:
        raise ValueError("hybrid feature table contains a non-development split")
    result: list[HybridCandidate] = []
    for row in rows:
        features = np.asarray(
            [float(row["features"][name]) for name in FEATURE_ORDER], dtype=np.float32
        )
        if not bool(np.isfinite(features).all()):
            raise ValueError(f"non-finite features for {row.get('case_id')}")
        agreement = float(row["round_agreement_dice"])
        delta = float(row["delta_dice"])
        if not math.isfinite(agreement) or not math.isfinite(delta):
            raise ValueError(f"non-finite hybrid row for {row.get('case_id')}")
        result.append(
            HybridCandidate(
                case_id=str(row["case_id"]),
                patient_id=str(row["patient_id"]),
                split=str(row["split"]),
                tracer=str(row["tracer"]),
                route=str(row["route"]),
                features=features,
                round_agreement_dice=agreement,
                delta_dice=delta,
            )
        )
    validate_candidate_menu(result, require_outcomes=True)
    return result


def validate_candidate_menu(
    candidates: Sequence[HybridCandidate], *, require_outcomes: bool
) -> list[list[HybridCandidate]]:
    grouped: dict[str, list[HybridCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.route not in ROUTES or candidate.tracer not in TRACERS:
            raise ValueError(f"invalid fusion candidate {candidate.uid}")
        if candidate.features.shape != (len(FEATURE_ORDER),):
            raise ValueError(f"wrong feature dimension for {candidate.uid}")
        if require_outcomes and candidate.delta_dice is None:
            raise ValueError(f"development outcome missing for {candidate.uid}")
        grouped[candidate.case_id].append(candidate)
    groups = [grouped[case_id] for case_id in sorted(grouped)]
    for group in groups:
        if len(group) != len(ROUTES) or {candidate.route for candidate in group} != set(
            ROUTES
        ):
            raise ValueError(f"{group[0].case_id} lacks the exact fusion menu")
        if len({candidate.patient_id for candidate in group}) != 1 or len(
            {candidate.tracer for candidate in group}
        ) != 1:
            raise ValueError(f"{group[0].case_id} mixes patient or tracer IDs")
    return groups


def _rule_identity(rule: Mapping[str, Any] | None) -> tuple[Any, ...]:
    """Return the exact, non-rounded tuple used for identity and ranking."""

    if rule is None:
        return ("", "", ())
    return (
        str(rule["tracer"]),
        str(rule["route"]),
        tuple(
            (
                str(condition["feature"]),
                str(condition["op"]),
                float(condition["threshold"]),
            )
            for condition in rule["conditions"]
        ),
    )


def rule_signature(rule: Mapping[str, Any] | None) -> str:
    """Human-readable full-precision signature; never used as rule identity."""

    if rule is None:
        return "KEEP"
    conditions = "&".join(
        f"{condition['feature']}{condition['op']}{float(condition['threshold'])!r}"
        for condition in rule["conditions"]
    )
    return f"{rule['tracer']}:{rule['route']}:{conditions}"


def _condition_passes(candidate: HybridCandidate, condition: Mapping[str, Any]) -> bool:
    observed = candidate.feature(str(condition["feature"]))
    threshold = float(condition["threshold"])
    operator = str(condition["op"])
    if operator == ">=":
        return observed >= threshold
    if operator == "<=":
        return observed <= threshold
    raise ValueError(f"unsupported condition operator: {operator}")


def apply_rule(
    rule: Mapping[str, Any] | None,
    group: Sequence[HybridCandidate],
) -> HybridCandidate | None:
    if rule is None or group[0].tracer != str(rule["tracer"]):
        return None
    candidate = next(
        item for item in group if item.route == str(rule["route"])
    )
    if not candidate.changed:
        return None
    return candidate if all(
        _condition_passes(candidate, condition)
        for condition in rule["conditions"]
    ) else None


def apply_rule_set(
    rules: Sequence[Mapping[str, Any] | None],
    groups: Sequence[Sequence[HybridCandidate]],
) -> list[HybridCandidate | None]:
    choices: list[HybridCandidate | None] = []
    for group in groups:
        matched = [
            choice
            for rule in rules
            if (choice := apply_rule(rule, group)) is not None
        ]
        if len(matched) > 1:
            raise RuntimeError("rule set selected multiple routes for one study")
        choices.append(matched[0] if matched else None)
    return choices


def policy_objective(
    groups: Sequence[Sequence[HybridCandidate]],
    choices: Sequence[HybridCandidate | None],
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    if len(groups) != len(choices):
        raise ValueError("groups and choices differ in length")
    realized = np.asarray(
        [0.0 if choice is None else choice.development_delta() for choice in choices],
        dtype=np.float64,
    )
    covered = np.asarray([choice is not None for choice in choices], dtype=bool)
    harmful = realized < -HARM_TOLERANCE
    patients = sorted({group[0].patient_id for group in groups})
    by_patient: dict[str, list[float]] = defaultdict(list)
    for group, utility in zip(groups, realized, strict=True):
        by_patient[group[0].patient_id].append(float(utility))
    patient_utility = np.asarray(
        [float(np.mean(by_patient[patient])) for patient in patients],
        dtype=np.float64,
    )
    if bootstrap:
        rng = np.random.default_rng(SEED)
        indices = rng.integers(
            0,
            len(patients),
            size=(BOOTSTRAP_SAMPLES, len(patients)),
            dtype=np.int64,
        )
        bootstrap_means = patient_utility[indices].mean(axis=1)
        lower = float(np.quantile(bootstrap_means, 0.025, method="linear"))
        upper = float(np.quantile(bootstrap_means, 0.975, method="linear"))
    else:
        lower = float("nan")
        upper = float("nan")
    report = {
        "patient_count": len(patients),
        "study_count": len(groups),
        "patient_mean": float(patient_utility.mean()),
        "study_mean": float(realized.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "harm_count": int(harmful.sum()),
        "harm_rate": float(harmful.mean()),
        "coverage_count": int(covered.sum()),
        "coverage": float(covered.mean()),
        "wins": int((realized > HARM_TOLERANCE).sum()),
        "ties_or_keep": int((np.abs(realized) <= HARM_TOLERANCE).sum()),
        "losses": int(harmful.sum()),
    }
    report["passes_test_gate"] = bool(
        bootstrap
        and report["harm_rate"] <= TEST_MAX_HARM_RATE + HARM_TOLERANCE
        and lower > 0.0
    )
    report["passes_zero_observed_harm_gate"] = bool(
        bootstrap and report["harm_count"] == 0 and lower > 0.0
    )
    return report


def _rule_rank(
    report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any] | None],
) -> tuple[Any, ...]:
    return (
        float(report["patient_mean"]),
        float(report["ci_lower"]),
        float(report["study_mean"]),
        -float(report["harm_rate"]),
        float(report["coverage"]),
        tuple(_rule_identity(rule) for rule in rules),
    )


def _condition_families(route: str) -> tuple[tuple[tuple[str, str], ...], ...]:
    if "union" in route:
        uptake = (
            ("added_pet_robust_mean", ">="),
            ("added_pet_robust_p90", ">="),
        )
    else:
        uptake = (
            ("removed_pet_robust_mean", "<="),
            ("removed_pet_robust_p90", "<="),
        )
    return tuple(
        (("round_agreement_dice", ">="), uptake_feature)
        for uptake_feature in uptake
    )


def enumerate_rules(
    groups: Sequence[Sequence[HybridCandidate]],
) -> list[dict[str, Any]]:
    rows = [candidate for group in groups for candidate in group]
    rules: list[dict[str, Any]] = []
    for tracer in TRACERS:
        for route in ROUTES:
            matching = [
                candidate
                for candidate in rows
                if candidate.tracer == tracer and candidate.route == route
            ]
            if not matching:
                raise ValueError(f"empty screen stratum for {tracer}/{route}")
            for family in _condition_families(route):
                threshold_grids = [
                    sorted(
                        {
                            float(value)
                            for value in np.quantile(
                                np.asarray(
                                    [candidate.feature(feature) for candidate in matching],
                                    dtype=np.float64,
                                ),
                                QUANTILE_GRID,
                                method="linear",
                            )
                        }
                    )
                    for feature, _operator in family
                ]
                for left in threshold_grids[0]:
                    for right in threshold_grids[1]:
                        rules.append(
                            {
                                "tracer": tracer,
                                "route": route,
                                "conditions": [
                                    {
                                        "feature": feature,
                                        "op": operator,
                                        "threshold": threshold,
                                    }
                                    for (feature, operator), threshold in zip(
                                        family, (left, right), strict=True
                                    )
                                ],
                            }
                        )
    return rules


def fit_safe_rule_set(
    groups: Sequence[Sequence[HybridCandidate]],
) -> tuple[tuple[dict[str, Any] | None, ...], dict[str, Any], dict[str, Any]]:
    """Fit the exact zero-observed-harm consensus-plus-uptake screen."""

    safe_by_tracer: dict[
        str, list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    evaluated = 0
    fast_survivors = 0
    for rule in enumerate_rules(groups):
        choices = apply_rule_set((rule,), groups)
        fast = policy_objective(groups, choices, bootstrap=False)
        evaluated += 1
        if (
            fast["coverage_count"] < MINIMUM_INNER_COVERAGE
            or fast["harm_count"] > 0
            or fast["patient_mean"] <= 0.0
        ):
            continue
        fast_survivors += 1
        report = policy_objective(groups, choices, bootstrap=True)
        if report["passes_zero_observed_harm_gate"]:
            safe_by_tracer[str(rule["tracer"])].append(
                (_rule_rank(report, (rule,)), rule, report)
            )
    for tracer in safe_by_tracer:
        safe_by_tracer[tracer].sort(key=lambda item: item[0], reverse=True)

    pools: dict[str, list[dict[str, Any] | None]] = {}
    for tracer in TRACERS:
        unique: list[dict[str, Any] | None] = [None]
        seen: set[tuple[Any, ...]] = set()
        for _key, rule, _report in safe_by_tracer.get(tracer, []):
            identity = _rule_identity(rule)
            if identity not in seen:
                unique.append(rule)
                seen.add(identity)
            if len(unique) >= RULE_POOL_SIZE + 1:
                break
        pools[tracer] = unique

    combined: list[
        tuple[
            tuple[Any, ...],
            tuple[dict[str, Any] | None, ...],
            dict[str, Any],
        ]
    ] = []
    for fdg_rule in pools["FDG"]:
        for psma_rule in pools["PSMA"]:
            if fdg_rule is None and psma_rule is None:
                continue
            rules = (fdg_rule, psma_rule)
            report = policy_objective(
                groups, apply_rule_set(rules, groups), bootstrap=True
            )
            if report["passes_zero_observed_harm_gate"]:
                combined.append((_rule_rank(report, rules), rules, report))
    if combined:
        combined.sort(key=lambda item: item[0], reverse=True)
        _rank, selected, report = combined[0]
    else:
        selected = (None,)
        report = policy_objective(
            groups, apply_rule_set(selected, groups), bootstrap=True
        )
    diagnostics = {
        "enumerated_rules": evaluated,
        "fast_survivors": fast_survivors,
        "safe_single_rules": {
            tracer: len(safe_by_tracer.get(tracer, [])) for tracer in TRACERS
        },
        "pool_sizes_including_keep": {
            tracer: len(pools[tracer]) for tracer in TRACERS
        },
        "safe_combined_rule_sets": len(combined),
        "selected_signatures": [rule_signature(rule) for rule in selected],
        "identity_semantics": "exact float tuples; rounded signatures are display only",
    }
    return selected, report, diagnostics


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def stable_patient_partition(
    patients: Sequence[str], *, calibration_count: int = 5
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ranked = sorted(
        {str(patient) for patient in patients},
        key=lambda patient: (
            sha256(f"{STABLE_SPLIT_PREFIX}{patient}".encode("utf-8")).hexdigest(),
            patient,
        ),
    )
    if calibration_count <= 0 or len(ranked) <= calibration_count:
        raise ValueError("stable EDL partition leaves an empty fit/calibration pool")
    return tuple(ranked[calibration_count:]), tuple(ranked[:calibration_count])


def _feature_matrix(candidates: Sequence[HybridCandidate]) -> np.ndarray:
    if not candidates:
        raise ValueError("EDL candidate pool is empty")
    matrix = np.asarray([candidate.features for candidate in candidates], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_ORDER):
        raise ValueError("EDL feature matrix has the wrong shape")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("EDL feature matrix contains non-finite values")
    return matrix


def _raw_edl_outputs(
    trained: TrainedEDL, candidates: Sequence[HybridCandidate]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = _feature_matrix(candidates)
    trained.model.eval()
    with torch.no_grad():
        alpha, utility = trained.model(
            torch.from_numpy(trained.normalizer.transform(features)).to("cpu")
        )
        probability = (alpha[:, 1] / alpha.sum(dim=-1)).cpu().numpy()
        vacuity = (2.0 / alpha.sum(dim=-1)).cpu().numpy()
    return (
        probability.astype(np.float64),
        vacuity.astype(np.float64),
        utility.cpu().numpy().astype(np.float64),
    )


def train_edl(
    fit_candidates: Sequence[HybridCandidate],
    calibration_candidates: Sequence[HybridCandidate],
) -> TrainedEDL:
    """Train and temperature-calibrate the exact deterministic CPU EDL head."""

    if any(candidate.delta_dice is None for candidate in fit_candidates):
        raise ValueError("EDL fitting requires development outcomes")
    if any(candidate.delta_dice is None for candidate in calibration_candidates):
        raise ValueError("EDL calibration requires development outcomes")
    fit_patients = tuple(sorted({candidate.patient_id for candidate in fit_candidates}))
    calibration_patients = tuple(
        sorted({candidate.patient_id for candidate in calibration_candidates})
    )
    if set(fit_patients) & set(calibration_patients):
        raise ValueError("EDL fit/calibration patient leakage")
    _seed_everything()
    fit_features = _feature_matrix(fit_candidates)
    normalizer = FeatureNormalizer.fit(fit_features)
    train_x = torch.from_numpy(normalizer.transform(fit_features)).to("cpu")
    labels = torch.tensor(
        [int(candidate.development_delta() > 0.0) for candidate in fit_candidates],
        dtype=torch.long,
        device="cpu",
    )
    utility_target = torch.tensor(
        [candidate.development_delta() for candidate in fit_candidates],
        dtype=torch.float32,
        device="cpu",
    )
    model = EvidentialUtilityHead(in_dim=len(FEATURE_ORDER), hidden=EDL_HIDDEN).to("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=EDL_LR, weight_decay=EDL_WEIGHT_DECAY
    )
    model.train()
    for epoch in range(EDL_EPOCHS):
        optimizer.zero_grad()
        alpha, utility = model(train_x)
        loss = evidential_utility_loss(
            alpha,
            utility,
            labels,
            utility_target,
            anneal=min(1.0, (epoch + 1) / 100.0),
            utility_weight=2.0,
        )
        loss.backward()
        optimizer.step()
    uncalibrated = TrainedEDL(
        model=model,
        normalizer=normalizer,
        temperature=1.0,
        fit_patients=fit_patients,
        calibration_patients=calibration_patients,
    )
    probability, _vacuity, _utility = _raw_edl_outputs(
        uncalibrated, calibration_candidates
    )
    calibration_labels = np.asarray(
        [int(candidate.development_delta() > 0.0) for candidate in calibration_candidates],
        dtype=np.int64,
    )
    temperature = calibrate_temperature(probability, calibration_labels)
    return TrainedEDL(
        model=model,
        normalizer=normalizer,
        temperature=float(temperature),
        fit_patients=fit_patients,
        calibration_patients=calibration_patients,
    )


def score_edl(
    trained: TrainedEDL,
    candidates: Sequence[HybridCandidate],
) -> dict[str, dict[str, float]]:
    probability, vacuity, utility = _raw_edl_outputs(trained, candidates)
    probability = _temperature_scale(probability, trained.temperature)
    return {
        candidate.uid: {
            "p_accept": float(p_accept),
            "vacuity": float(candidate_vacuity),
            "predicted_utility": float(predicted_utility),
            "temperature": float(trained.temperature),
        }
        for candidate, p_accept, candidate_vacuity, predicted_utility in zip(
            candidates, probability, vacuity, utility, strict=True
        )
    }


def edl_gate(score: Mapping[str, float], *, changed: bool) -> bool:
    return bool(
        changed
        and float(score["p_accept"]) >= EDL_GATE["accept_probability"]
        and float(score["vacuity"]) <= EDL_GATE["max_accept_vacuity"]
        and float(score["predicted_utility"]) >= EDL_GATE["min_predicted_utility"]
    )


def _serialize_choices(
    groups: Sequence[Sequence[HybridCandidate]],
    choices: Sequence[HybridCandidate | None],
) -> dict[str, Any]:
    route_counts = Counter(
        "KEEP" if choice is None else choice.route for choice in choices
    )
    accepted = [
        {
            "case_id": group[0].case_id,
            "patient_id": group[0].patient_id,
            "tracer": group[0].tracer,
            "route": choice.route,
            "delta_dice": choice.development_delta(),
        }
        for group, choice in zip(groups, choices, strict=True)
        if choice is not None
    ]
    return {
        "route_counts": dict(sorted(route_counts.items())),
        "accepted": accepted,
        "route_by_case": {
            group[0].case_id: "KEEP" if choice is None else choice.route
            for group, choice in zip(groups, choices, strict=True)
        },
    }


def nested_development_replay(
    candidates: Sequence[HybridCandidate],
) -> dict[str, Any]:
    """Run the frozen 24-fold patient-LOPO screen + fixed-route EDL veto."""

    groups = validate_candidate_menu(candidates, require_outcomes=True)
    patients = sorted({group[0].patient_id for group in groups})
    if len(patients) != EXPECTED_PATIENT_COUNT or len(groups) != EXPECTED_CASE_COUNT:
        raise ValueError("nested replay requires exactly 24 patients and 48 studies")
    pure_by_case: dict[str, HybridCandidate | None] = {}
    hybrid_by_case: dict[str, HybridCandidate | None] = {}
    fold_details: list[dict[str, Any]] = []
    for outer_patient in patients:
        inner_groups = [group for group in groups if group[0].patient_id != outer_patient]
        outer_groups = [group for group in groups if group[0].patient_id == outer_patient]
        rules, inner_report, diagnostics = fit_safe_rule_set(inner_groups)
        pure_choices = apply_rule_set(rules, outer_groups)
        inner_patients = sorted({group[0].patient_id for group in inner_groups})
        fit_patients, calibration_patients = stable_patient_partition(inner_patients)
        inner_candidates = [candidate for group in inner_groups for candidate in group]
        trained = train_edl(
            [
                candidate
                for candidate in inner_candidates
                if candidate.patient_id in fit_patients
            ],
            [
                candidate
                for candidate in inner_candidates
                if candidate.patient_id in calibration_patients
            ],
        )
        outer_candidates = [candidate for group in outer_groups for candidate in group]
        scores = score_edl(trained, outer_candidates)
        studies: list[dict[str, Any]] = []
        for group, pure_choice in zip(outer_groups, pure_choices, strict=True):
            case_id = group[0].case_id
            pure_by_case[case_id] = pure_choice
            hybrid_choice = None
            if pure_choice is not None and edl_gate(
                scores[pure_choice.uid], changed=pure_choice.changed
            ):
                hybrid_choice = pure_choice
            hybrid_by_case[case_id] = hybrid_choice
            studies.append(
                {
                    "case_id": case_id,
                    "tracer": group[0].tracer,
                    "pure_route": "KEEP" if pure_choice is None else pure_choice.route,
                    "hybrid_route": "KEEP" if hybrid_choice is None else hybrid_choice.route,
                    "scores": {
                        candidate.route: scores[candidate.uid] for candidate in group
                    },
                }
            )
        fold_details.append(
            {
                "outer_patient": outer_patient,
                "fit_patients": list(fit_patients),
                "calibration_patients": list(calibration_patients),
                "screen_rules": [
                    None if rule is None else dict(rule) for rule in rules
                ],
                "screen_rule_signatures": [rule_signature(rule) for rule in rules],
                "inner_screen_objective": inner_report,
                "inner_screen_diagnostics": diagnostics,
                "temperature": trained.temperature,
                "studies": studies,
            }
        )
    pure_choices = [pure_by_case[group[0].case_id] for group in groups]
    hybrid_choices = [hybrid_by_case[group[0].case_id] for group in groups]
    pure_summary = _serialize_choices(groups, pure_choices)
    hybrid_summary = _serialize_choices(groups, hybrid_choices)
    pure_summary["objective"] = policy_objective(groups, pure_choices, bootstrap=True)
    hybrid_summary["objective"] = policy_objective(groups, hybrid_choices, bootstrap=True)
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "artifact_type": "edl_hybrid_nested_development_replay",
        "status": "POST_HOC_DEVELOPMENT_REPLAY_NOT_CONFIRMATORY",
        "feature_provenance": {
            "direct_ground_truth_and_outcomes_forbidden_for_policy_inputs": True,
            "indirect_ground_truth_dependence_via_robot_prompts_and_proposals": True,
            "development_outcomes_used_only_for_replay_fit_and_evaluation": True,
        },
        "primary_hybrid": hybrid_summary,
        "secondary_pure_screen": pure_summary,
        "fold_details": fold_details,
    }


def fit_full_development_models(
    candidates: Sequence[HybridCandidate],
) -> tuple[tuple[dict[str, Any] | None, ...], TrainedEDL, dict[str, Any]]:
    groups = validate_candidate_menu(candidates, require_outcomes=True)
    rules, screen_report, diagnostics = fit_safe_rule_set(groups)
    patients = sorted({candidate.patient_id for candidate in candidates})
    fit_patients, calibration_patients = stable_patient_partition(patients)
    trained = train_edl(
        [candidate for candidate in candidates if candidate.patient_id in fit_patients],
        [
            candidate
            for candidate in candidates
            if candidate.patient_id in calibration_patients
        ],
    )
    return rules, trained, {
        "screen_objective": screen_report,
        "screen_diagnostics": diagnostics,
        "fit_patients": list(fit_patients),
        "calibration_patients": list(calibration_patients),
        "temperature": trained.temperature,
    }


def _checkpoint_payload(
    trained: TrainedEDL, *, protocol_sha256: str, development_manifest_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "artifact_type": "edl_fixed_route_hybrid_checkpoint",
        "protocol_sha256": protocol_sha256,
        "development_manifest_sha256": development_manifest_sha256,
        "feature_order": list(FEATURE_ORDER),
        "architecture": {"input": len(FEATURE_ORDER), "hidden": EDL_HIDDEN},
        "normalizer": {
            "mean": trained.normalizer.mean,
            "std": trained.normalizer.std,
        },
        "temperature": trained.temperature,
        "fit_patients": list(trained.fit_patients),
        "calibration_patients": list(trained.calibration_patients),
        "state_dict": trained.model.state_dict(),
    }


def load_edl_checkpoint(checkpoint: str | Path | Mapping[str, Any]) -> TrainedEDL:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    else:
        payload = dict(checkpoint)
    if tuple(payload.get("feature_order", ())) != tuple(FEATURE_ORDER):
        raise ValueError("hybrid checkpoint feature order mismatch")
    architecture = dict(payload.get("architecture", {}))
    if architecture != {"input": len(FEATURE_ORDER), "hidden": EDL_HIDDEN}:
        raise ValueError("hybrid checkpoint architecture mismatch")
    normalizer_payload = dict(payload["normalizer"])
    normalizer = FeatureNormalizer(
        mean=np.asarray(normalizer_payload["mean"], dtype=np.float32),
        std=np.asarray(normalizer_payload["std"], dtype=np.float32),
    )
    if normalizer.mean.shape != (len(FEATURE_ORDER),) or normalizer.std.shape != (
        len(FEATURE_ORDER),
    ):
        raise ValueError("hybrid checkpoint normalizer shape mismatch")
    model = EvidentialUtilityHead(in_dim=len(FEATURE_ORDER), hidden=EDL_HIDDEN).to("cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    temperature = float(payload["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("invalid hybrid checkpoint temperature")
    return TrainedEDL(
        model=model,
        normalizer=normalizer,
        temperature=temperature,
        fit_patients=tuple(str(value) for value in payload.get("fit_patients", ())),
        calibration_patients=tuple(
            str(value) for value in payload.get("calibration_patients", ())
        ),
    )


def select_frozen_policy_routes(
    candidates: Sequence[HybridCandidate],
    *,
    pure_policy: str | Path | Mapping[str, Any],
    edl_checkpoint: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen screen and EDL veto using label-free candidate inputs."""

    groups = validate_candidate_menu(candidates, require_outcomes=False)
    if isinstance(pure_policy, (str, Path)):
        policy_payload = json.loads(Path(pure_policy).read_text(encoding="utf-8"))
    else:
        policy_payload = dict(pure_policy)
    rules = tuple(policy_payload.get("rules", ()))
    if not rules or any(
        rule is not None and str(rule.get("tracer", "")) not in TRACERS
        for rule in rules
    ):
        raise ValueError("invalid pure hybrid-screen policy")
    pure_choices = apply_rule_set(rules, groups)
    trained = load_edl_checkpoint(edl_checkpoint)
    all_candidates = [candidate for group in groups for candidate in group]
    scores = score_edl(trained, all_candidates)
    hybrid_choices: list[HybridCandidate | None] = []
    for pure_choice in pure_choices:
        if pure_choice is not None and edl_gate(
            scores[pure_choice.uid], changed=pure_choice.changed
        ):
            hybrid_choices.append(pure_choice)
        else:
            hybrid_choices.append(None)
    result: dict[str, Any] = {}
    for group, pure_choice, hybrid_choice in zip(
        groups, pure_choices, hybrid_choices, strict=True
    ):
        selected_score = None if pure_choice is None else scores[pure_choice.uid]
        result[group[0].case_id] = {
            "patient_id": group[0].patient_id,
            "tracer": group[0].tracer,
            "pure_screen_route": "KEEP" if pure_choice is None else pure_choice.route,
            "edl_hybrid_route": (
                "KEEP" if hybrid_choice is None else hybrid_choice.route
            ),
            "edl_gate_pass": hybrid_choice is not None,
            "confidence": (
                None if selected_score is None else selected_score["p_accept"]
            ),
            "p_accept": (
                None if selected_score is None else selected_score["p_accept"]
            ),
            "vacuity": None if selected_score is None else selected_score["vacuity"],
            "predicted_utility": (
                None
                if selected_score is None
                else selected_score["predicted_utility"]
            ),
            "route_scores": {
                candidate.route: scores[candidate.uid] for candidate in group
            },
        }
    return result


def _canonicalize_reused_feature_table(
    source_path: Path,
    *,
    expected_source_sha256: str,
    expected_manifest_sha256: str,
    expected_deployment_sha256: str,
) -> dict[str, Any]:
    observed_source_sha256 = sha256_file(source_path)
    if observed_source_sha256 != expected_source_sha256.lower():
        raise ValueError("reused feature-table SHA-256 mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if set(source) != {"feature_order", "rows", "scope"}:
        raise ValueError("reused feature table has an unexpected top-level schema")
    if tuple(source["feature_order"]) != tuple(FEATURE_ORDER):
        raise ValueError("reused feature table feature order mismatch")
    scope = dict(source["scope"])
    if (
        str(scope.get("manifest_sha256", "")) != expected_manifest_sha256
        or str(scope.get("deployment_sha256", "")) != expected_deployment_sha256
        or int(scope.get("patients", -1)) != EXPECTED_PATIENT_COUNT
        or int(scope.get("studies", -1)) != EXPECTED_CASE_COUNT
        or int(scope.get("records", -1)) != EXPECTED_RECORD_COUNT
        or bool(scope.get("test_opened", True))
        or set(scope.get("splits", ())) != DEVELOPMENT_SPLITS
    ):
        raise ValueError("reused feature table scope/hash binding mismatch")
    source_rows = list(source["rows"])
    if len(source_rows) != EXPECTED_RECORD_COUNT:
        raise ValueError("reused feature table record count mismatch")
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        features = dict(row.get("features", {}))
        if set(features) != set(FEATURE_ORDER):
            raise ValueError(f"feature inventory mismatch for {row.get('case_id')}")
        feature_values = np.asarray(
            [float(features[name]) for name in FEATURE_ORDER], dtype=np.float32
        )
        numeric = np.asarray(
            [
                *feature_values.astype(np.float64),
                float(row["round_agreement_dice"]),
                float(row["delta"]),
            ],
            dtype=np.float64,
        )
        if not bool(np.isfinite(numeric).all()):
            raise ValueError(f"non-finite reused feature row for {row.get('case_id')}")
        rows.append(
            {
                "case_id": str(row["case_id"]),
                "patient_id": str(row["patient_id"]),
                "split": str(row["split"]),
                "tracer": str(row["tracer"]),
                "route": str(row["route"]),
                "features": {
                    name: float(value)
                    for name, value in zip(FEATURE_ORDER, feature_values, strict=True)
                },
                "round_agreement_dice": float(row["round_agreement_dice"]),
                "delta_dice": float(row["delta"]),
            }
        )
    payload = {
        "schema_version": FEATURE_TABLE_SCHEMA_VERSION,
        "artifact_type": "edl_hybrid_development_feature_table",
        "status": "DEVELOPMENT_ONLY_NO_TEST",
        "manifest_sha256": expected_manifest_sha256,
        "feature_order": list(FEATURE_ORDER),
        "counts": {
            "patients": EXPECTED_PATIENT_COUNT,
            "studies": EXPECTED_CASE_COUNT,
            "records": EXPECTED_RECORD_COUNT,
            "test_records": 0,
        },
        "feature_provenance": {
            "direct_ground_truth_and_outcomes_forbidden_for_policy_inputs": True,
            "indirect_ground_truth_dependence_via_robot_prompts_and_proposals": True,
            "development_only_label": "delta_dice",
        },
        "rows": rows,
    }
    candidates_from_feature_table(payload)
    return payload


def _compare_feature_tables(
    reused: Mapping[str, Any], independently_recomputed: Mapping[str, Any]
) -> dict[str, Any]:
    reused_map = {
        (str(row["case_id"]), str(row["route"])): row for row in reused["rows"]
    }
    recomputed_map = {
        (str(row["case_id"]), str(row["route"])): row
        for row in independently_recomputed["rows"]
    }
    if set(reused_map) != set(recomputed_map):
        raise ValueError("reused/recomputed feature-table row IDs differ")
    maximum_feature_difference = 0.0
    maximum_scalar_difference = 0.0
    for key in sorted(reused_map):
        left = reused_map[key]
        right = recomputed_map[key]
        for field in ("patient_id", "split", "tracer", "route"):
            if left[field] != right[field]:
                raise ValueError(f"reused/recomputed metadata differ for {key}/{field}")
        left_features = np.asarray(
            [left["features"][name] for name in FEATURE_ORDER], dtype=np.float64
        )
        right_features = np.asarray(
            [right["features"][name] for name in FEATURE_ORDER], dtype=np.float64
        )
        feature_difference = float(np.max(np.abs(left_features - right_features)))
        maximum_feature_difference = max(maximum_feature_difference, feature_difference)
        for field in ("round_agreement_dice", "delta_dice"):
            maximum_scalar_difference = max(
                maximum_scalar_difference, abs(float(left[field]) - float(right[field]))
            )
    if maximum_feature_difference > 0.0 or maximum_scalar_difference > 1e-12:
        raise ValueError(
            "reused feature table differs from independent full-volume recomputation"
        )
    return {
        "independently_recomputed_records": len(recomputed_map),
        "independently_recomputed_studies": len(
            {case_id for case_id, _route in recomputed_map}
        ),
        "maximum_feature_absolute_difference": maximum_feature_difference,
        "maximum_agreement_or_delta_absolute_difference": maximum_scalar_difference,
        "validation_status": "EXACT_MATCH",
    }


def _assert_close(observed: float, expected: float, label: str) -> None:
    if abs(float(observed) - float(expected)) > 1e-12:
        raise ValueError(
            f"development replay target mismatch for {label}: "
            f"{observed!r} != {expected!r}"
        )


def assert_replay_targets(
    replay: Mapping[str, Any],
    protocol: Mapping[str, Any],
    expected_replay: Mapping[str, Any],
) -> None:
    targets = dict(protocol["development_replay_targets"])
    mappings = (
        ("primary_hybrid", "primary_hybrid"),
        ("secondary_pure_screen", "secondary_pure_screen"),
    )
    for result_key, target_key in mappings:
        result = dict(replay[result_key])
        observed = dict(result["objective"])
        expected = dict(targets[target_key])
        scalar_fields = {
            "patient_mean_delta_dice": "patient_mean",
            "study_mean_delta_dice": "study_mean",
            "harm_rate": "harm_rate",
            "coverage": "coverage",
        }
        for target_field, observed_field in scalar_fields.items():
            _assert_close(
                float(observed[observed_field]),
                float(expected[target_field]),
                f"{target_key}.{target_field}",
            )
        for index, bound in enumerate(("ci_lower", "ci_upper")):
            _assert_close(
                float(observed[bound]),
                float(expected["patient_bootstrap_95_ci"][index]),
                f"{target_key}.{bound}",
            )
        for field in ("harm_count", "coverage_count", "wins", "losses"):
            if int(observed[field]) != int(expected[field]):
                raise ValueError(f"development replay count mismatch: {target_key}.{field}")
        if "route_counts" in expected and result["route_counts"] != expected["route_counts"]:
            raise ValueError(f"development replay route-count mismatch: {target_key}")

    methods = dict(expected_replay.get("methods", {}))
    prototype_keys = {
        "primary_hybrid": "edl_default_gate_on_fixed_screen_route",
        "secondary_pure_screen": "pure_zero_harm_consensus_uptake",
    }
    for replay_key, prototype_key in prototype_keys.items():
        prototype = dict(methods.get(prototype_key, {}))
        accepted = {
            str(row["case_id"]): str(row["route"])
            for row in prototype.get("accepted", ())
        }
        observed = {
            case_id: route
            for case_id, route in replay[replay_key]["route_by_case"].items()
            if route != "KEEP"
        }
        if observed != accepted:
            raise ValueError(f"exact development route choices differ: {replay_key}")


def _verify_protocol(
    protocol_path: Path, *, experiment_root: Path, repo_root: Path
) -> tuple[dict[str, Any], str]:
    protocol_hash = sha256_file(protocol_path)
    sidecar = protocol_path.with_name(protocol_path.name + ".sha256")
    expected_sidecar = f"{protocol_hash}  {protocol_path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise ValueError("protocol sidecar is absent or invalid")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        int(protocol.get("schema_version", -1)) != PROTOCOL_SCHEMA_VERSION
        or protocol.get("status") != "POST_FAILURE_EXPLORATORY_PROTOCOL_FROZEN"
    ):
        raise ValueError("authoritative v6 hybrid protocol is not frozen")
    bindings = dict(protocol.get("hash_bindings", {}))
    if len(bindings) != 8:
        raise ValueError("hybrid protocol must bind exactly eight parent artifacts")
    for role, binding in bindings.items():
        root = repo_root if role in {"prompt_update_edl_code", "route_policy_eval_code"} else experiment_root
        path = _resolve(root, str(binding["path"]))
        if not path.is_file() or sha256_file(path) != str(binding["sha256"]):
            raise ValueError(f"hybrid parent hash binding failed: {role}")
    return protocol, protocol_hash


def _file_fingerprint(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _artifact_fingerprint(path: Path) -> dict[str, str]:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"artifact sidecar is absent: {sidecar}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sidecar_path": str(sidecar.resolve()),
        "sidecar_sha256": sha256_file(sidecar),
    }


def _artifact_fingerprint_at_final_path(
    temporary_path: Path, final_directory: Path
) -> dict[str, str]:
    """Hash temporary bytes while binding the post-rename canonical path."""

    temporary_sidecar = temporary_path.with_name(temporary_path.name + ".sha256")
    if not temporary_sidecar.is_file():
        raise FileNotFoundError(f"artifact sidecar is absent: {temporary_sidecar}")
    final_path = (final_directory / temporary_path.name).resolve()
    final_sidecar = final_path.with_name(final_path.name + ".sha256")
    return {
        "path": str(final_path),
        "sha256": sha256_file(temporary_path),
        "sidecar_path": str(final_sidecar),
        "sidecar_sha256": sha256_file(temporary_sidecar),
    }


def freeze_development_artifacts(
    *,
    development_manifest: Path,
    failed_deployment: Path,
    protocol_path: Path,
    source_feature_table: Path,
    source_feature_table_sha256: str,
    source_feature_builder: Path,
    expected_replay_path: Path,
    output_directory: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Reproduce and atomically freeze the isolated development-only artifacts."""

    experiment_root = protocol_path.resolve().parent
    repo_root = repo_root.resolve()
    if output_directory.name != "edl_hybrid_development_freeze":
        raise ValueError("hybrid development output directory name is frozen")
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {output_directory}")
    protocol, protocol_sha256 = _verify_protocol(
        protocol_path.resolve(), experiment_root=experiment_root, repo_root=repo_root
    )
    bindings = dict(protocol["hash_bindings"])
    manifest_sha256 = sha256_file(development_manifest)
    deployment_sha256 = sha256_file(failed_deployment)
    if manifest_sha256 != bindings["development_manifest"]["sha256"]:
        raise ValueError("development manifest is not the protocol-bound manifest")
    if deployment_sha256 != bindings["failed_deployment"]["sha256"]:
        raise ValueError("failed deployment is not the protocol-bound deployment")
    reused = _canonicalize_reused_feature_table(
        source_feature_table.resolve(),
        expected_source_sha256=source_feature_table_sha256,
        expected_manifest_sha256=manifest_sha256,
        expected_deployment_sha256=deployment_sha256,
    )
    recomputed = build_development_feature_table(
        development_manifest.resolve(), expected_manifest_sha256=manifest_sha256
    )
    recomputation = _compare_feature_tables(reused, recomputed)
    candidates = candidates_from_feature_table(reused)
    replay = nested_development_replay(candidates)
    expected_replay_sha256 = sha256_file(expected_replay_path)
    expected_replay = json.loads(expected_replay_path.read_text(encoding="utf-8"))
    assert_replay_targets(replay, protocol, expected_replay)
    rules, trained, full_fit = fit_full_development_models(candidates)

    parent = output_directory.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".edl_hybrid_freeze_", dir=parent))
    try:
        feature_path = temporary / "edl_hybrid_development_features.json"
        _write_json(feature_path, reused)
        _write_sidecar(feature_path)

        provenance_path = temporary / "edl_hybrid_feature_reuse_provenance.json"
        provenance = {
            "schema_version": FREEZE_SCHEMA_VERSION,
            "artifact_type": "edl_hybrid_feature_reuse_provenance",
            "status": "VALIDATED_EXACT_FULL_RECOMPUTATION",
            "source_feature_table": _file_fingerprint(source_feature_table.resolve()),
            "source_feature_builder": _file_fingerprint(source_feature_builder.resolve()),
            "development_manifest": _file_fingerprint(development_manifest.resolve()),
            "failed_deployment": _file_fingerprint(failed_deployment.resolve()),
            "validation": recomputation,
            "test_records_accessed": 0,
        }
        _write_json(provenance_path, provenance)
        _write_sidecar(provenance_path)

        report_path = temporary / "edl_hybrid_development_report.json"
        report = {
            **replay,
            "protocol_sha256": protocol_sha256,
            "development_manifest_sha256": manifest_sha256,
            "source_replay": {
                "path": str(expected_replay_path.resolve()),
                "sha256": expected_replay_sha256,
                "exact_route_choices_verified": True,
            },
            "full_development_fit": full_fit,
            "full_development_rules": list(rules),
            "reproduction_tolerance": 1e-12,
            "reproduction_status": "PASS",
        }
        _write_json(report_path, report)
        _write_sidecar(report_path)

        checkpoint_path = temporary / "edl_fixed_route_hybrid.pt"
        torch.save(
            _checkpoint_payload(
                trained,
                protocol_sha256=protocol_sha256,
                development_manifest_sha256=manifest_sha256,
            ),
            checkpoint_path,
            _use_new_zipfile_serialization=False,
        )
        checkpoint_sha256 = _write_sidecar(checkpoint_path)

        pure_policy_path = temporary / "pure_consensus_uptake_policy.json"
        pure_policy = {
            "schema_version": FREEZE_SCHEMA_VERSION,
            "artifact_type": "pure_zero_harm_consensus_uptake_policy",
            "status": "FROZEN_DEVELOPMENT_POST_HOC",
            "protocol_sha256": protocol_sha256,
            "development_manifest_sha256": manifest_sha256,
            "rules": list(rules),
            "fallback": "KEEP",
            "feature_provenance": report["feature_provenance"],
        }
        _write_json(pure_policy_path, pure_policy)
        pure_policy_sha256 = _write_sidecar(pure_policy_path)

        hybrid_policy_path = temporary / "edl_fixed_route_hybrid_policy.json"
        hybrid_policy = {
            "schema_version": FREEZE_SCHEMA_VERSION,
            "artifact_type": "edl_fixed_route_hybrid_policy",
            "status": "FROZEN_DEVELOPMENT_POST_HOC",
            "protocol_sha256": protocol_sha256,
            "development_manifest_sha256": manifest_sha256,
            "pure_policy_sha256": pure_policy_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "rules": list(rules),
            "gate": dict(EDL_GATE),
            "semantics": "EDL may only veto the pure screen route; fallback KEEP",
        }
        _write_json(hybrid_policy_path, hybrid_policy)
        _write_sidecar(hybrid_policy_path)

        failed_bundle_path = temporary / "failed_selector_bundle.json"
        shutil.copyfile(expected_replay_path.resolve(), failed_bundle_path)
        _write_sidecar(failed_bundle_path)

        code_paths = {
            "hybrid_policy_module": repo_root / "rl_nninteractive" / "edl_fusion_hybrid.py",
            "hybrid_freeze_cli": repo_root / "scripts" / "freeze_edl_fusion_hybrid.py",
            "hybrid_policy_tests": repo_root / "tests" / "test_edl_fusion_hybrid.py",
            "hybrid_test_orchestrator": repo_root / "scripts" / "run_edl_hybrid_test_once.py",
            "hybrid_orchestrator_tests": repo_root / "tests" / "test_run_edl_hybrid_test_once.py",
            "fusion_only_runner": repo_root / "scripts" / "run_fusion_only_cohort_v2.py",
            "fusion_only_finalizer": repo_root / "scripts" / "finalize_fusion_only_cohort_v2.py",
            "route_policy_eval": repo_root / "rl_nninteractive" / "route_policy_eval.py",
            "prompt_update_edl": repo_root / "rl_nninteractive" / "prompt_update_edl.py",
        }
        missing_code = [role for role, path in code_paths.items() if not path.is_file()]
        if missing_code:
            raise FileNotFoundError(f"required hybrid code roles are absent: {missing_code}")
        code_hashes = {role: _file_fingerprint(path) for role, path in code_paths.items()}
        code_inventory_path = temporary / "edl_hybrid_code_inventory.json"
        _write_json(
            code_inventory_path,
            {
                "schema_version": FREEZE_SCHEMA_VERSION,
                "artifact_type": "edl_hybrid_code_inventory",
                "protocol_sha256": protocol_sha256,
                "code_hashes": code_hashes,
            },
        )
        _write_sidecar(code_inventory_path)

        artifact_paths = {
            "hybrid_development_features": feature_path,
            "hybrid_development_report": report_path,
            "pure_screen_policy": pure_policy_path,
            "hybrid_policy": hybrid_policy_path,
            "hybrid_edl_checkpoint": checkpoint_path,
            "hybrid_code_inventory": code_inventory_path,
        }
        deployment_path = temporary / "edl_hybrid_deployment.json"
        _write_json(
            deployment_path,
            {
                "schema_version": 6,
                "artifact_type": "edl_hybrid_frozen_deployment",
                "status": "FROZEN_BEFORE_TEST_OPENING",
                "test_outcomes_opened": False,
                "protocol": _file_fingerprint(protocol_path.resolve()),
                "artifact_bindings": {
                    role: _artifact_fingerprint_at_final_path(
                        path, output_directory.resolve()
                    )
                    for role, path in artifact_paths.items()
                },
                "parent_hash_bindings": dict(protocol["hash_bindings"]),
                "code_hashes": code_hashes,
            },
        )
        _write_sidecar(deployment_path)
        if ".edl_hybrid_freeze_" in deployment_path.read_text(encoding="utf-8"):
            raise RuntimeError("deployment leaked a temporary freeze path")
        os.replace(temporary, output_directory.resolve())
        frozen_deployment_path = output_directory.resolve() / deployment_path.name
        frozen_text = frozen_deployment_path.read_text(encoding="utf-8")
        if ".edl_hybrid_freeze_" in frozen_text:
            raise RuntimeError("frozen deployment contains a temporary path")
        frozen_deployment = json.loads(frozen_text)
        for role, binding in frozen_deployment["artifact_bindings"].items():
            artifact_path = Path(str(binding["path"]))
            sidecar_path = Path(str(binding["sidecar_path"]))
            if (
                artifact_path.parent != output_directory.resolve()
                or sidecar_path.parent != output_directory.resolve()
                or sha256_file(artifact_path) != str(binding["sha256"])
                or sha256_file(sidecar_path) != str(binding["sidecar_sha256"])
            ):
                raise RuntimeError(f"post-rename artifact binding failed: {role}")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_directory": str(output_directory.resolve()),
        "protocol_sha256": protocol_sha256,
        "replay_status": "PASS",
        "primary_hybrid": replay["primary_hybrid"]["objective"],
        "secondary_pure_screen": replay["secondary_pure_screen"]["objective"],
        "h100_required": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze development-only EDL fixed-route hybrid artifacts"
    )
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--failed-deployment", type=Path, required=True)
    parser.add_argument("--protocol-amendment", type=Path, required=True)
    parser.add_argument("--source-feature-table", type=Path, required=True)
    parser.add_argument("--source-feature-table-sha256", required=True)
    parser.add_argument("--source-feature-builder", type=Path, required=True)
    parser.add_argument("--expected-replay", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = freeze_development_artifacts(
        development_manifest=arguments.development_manifest,
        failed_deployment=arguments.failed_deployment,
        protocol_path=arguments.protocol_amendment,
        source_feature_table=arguments.source_feature_table,
        source_feature_table_sha256=str(arguments.source_feature_table_sha256),
        source_feature_builder=arguments.source_feature_builder,
        expected_replay_path=arguments.expected_replay,
        output_directory=arguments.out_dir,
        repo_root=arguments.repo,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
