"""Frozen offline evaluation of ResEnc/AutoPET prompt-update routing policies.

The evaluator consumes the strict schema-v1 trajectory manifest accepted by
``prompt_update_edl.examples_from_manifest``.  Each record additionally names
one composition action (replace/intersection/union) and prompt round (1/2).
Those additive fields may be supplied as ``action``/``round_index`` or by the
documented aliases parsed by :func:`parse_route_metadata`.

All learned decisions obey the patient-disjoint contract:

* train: fit the non-evidential linear contextual-bandit reward model;
* calibration: calibrate the non-evidential utility-score scale (the EDL
  checkpoint supplies its temperature calibration from this split);
* policy_validation: freeze grouped-menu ACCEPT/KEEP thresholds and choose the
  non-evidential ridge value;
* test: evaluate the already-frozen policies exactly once.

Ground truth builds offline outcomes and drives the upstream deterministic
robot-user prompts, so candidate proposals are indirectly oracle-dependent.
No direct GT array, metric, or scalar enters the deployable policy feature
vector produced by ``prompt_update_edl``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch

from .metrics import dice_score, normalized_surface_dice
from .prompt_update_edl import (
    FEATURE_ORDER,
    UpdateExample,
    examples_from_manifest,
    load_checkpoint_bundle,
    sha256_file,
    validate_split_contract,
)


ACTIONS = ("replace", "intersection", "union")
ROUNDS = (1, 2)
ROUTE_IDS = tuple(
    f"r{round_index}_{action}" for round_index in ROUNDS for action in ACTIONS
)
BASELINE_ROUTE = "KEEP"
DEFAULT_CANDIDATE_ROUTES = (BASELINE_ROUTE, *ROUTE_IDS)
HARM_TOLERANCE = 1e-12
SAFETY_MAX_HARMFUL_STUDY_RATE = 0.05
SAFETY_BOOTSTRAP_SAMPLES = 10_000
SAFETY_BOOTSTRAP_SEED = 20260715
EDL_ACCEPT_PROBABILITY_GRID = tuple(
    float(value) for value in np.linspace(0.35, 0.8, 10)
)
EDL_MAX_VACUITY_GRID = tuple(float(value) for value in np.linspace(0.3, 0.9, 7))
EDL_MIN_UTILITY_GRID = (-0.01, 0.0, 0.01, 0.02)


@dataclass(frozen=True)
class RouteCandidate:
    """One GT-free candidate action plus offline outcome labels."""

    case_id: str
    patient_id: str
    transition_id: str
    split: str
    prior_exposure: bool
    action: str
    round_index: int
    features: np.ndarray
    utility: float
    baseline_dice: float
    candidate_dice: float
    baseline_nsd: float
    candidate_nsd: float

    @property
    def route_id(self) -> str:
        return f"r{self.round_index}_{self.action}"

    @property
    def uid(self) -> str:
        return f"{self.case_id}::{self.route_id}"

    @property
    def delta_dice(self) -> float:
        return float(self.candidate_dice - self.baseline_dice)

    @property
    def delta_nsd(self) -> float:
        return float(self.candidate_nsd - self.baseline_nsd)


@dataclass(frozen=True)
class EvidentialScore:
    p_accept: float
    vacuity: float
    predicted_utility: float
    accepted: bool
    confidence: float


@dataclass(frozen=True)
class LinearContextualBandit:
    """Full-information linear ridge comparator with a frozen safety gate."""

    ridge_lambda: float
    threshold: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    coefficients: np.ndarray
    deploy_keep_all: bool = False

    def predict(self, candidates: Sequence[RouteCandidate]) -> np.ndarray:
        if not candidates:
            return np.asarray([], dtype=np.float64)
        x = _design_matrix(candidates, self.feature_mean, self.feature_std)
        return (x @ self.coefficients).astype(np.float64)

    def report(self) -> dict[str, Any]:
        digest = hashlib.sha256(
            np.asarray(self.coefficients, dtype=np.float64).tobytes()
        ).hexdigest()
        return {
            "model_type": "linear_contextual_bandit_ridge_utility",
            "methodology": "full_information_linear_ridge_comparator",
            "ridge_lambda": float(self.ridge_lambda),
            "accept_threshold": float(self.threshold),
            "deployment_decision": (
                "KEEP_ALL" if self.deploy_keep_all else "SELECT_ROUTE_OR_KEEP"
            ),
            "design_dimension": int(len(self.coefficients)),
            "coefficients_sha256": digest,
            "coefficients": self.coefficients.astype(float).tolist(),
            "feature_mean": self.feature_mean.astype(float).tolist(),
            "feature_std": self.feature_std.astype(float).tolist(),
        }


def parse_route_metadata(record: Mapping[str, Any]) -> tuple[str, int]:
    """Parse the additive action/round fields without changing schema-v1."""

    raw_action = next(
        (
            record.get(key)
            for key in ("action", "composition", "route_action", "mask_operation")
            if record.get(key) is not None
        ),
        None,
    )
    transition = str(record.get("transition_id", ""))
    if raw_action is None:
        lowered = transition.lower()
        matches = [action for action in ACTIONS if action in lowered]
        if len(matches) == 1:
            raw_action = matches[0]
    action_aliases = {
        "replacement": "replace",
        "replace_mask": "replace",
        "intersect": "intersection",
        "and": "intersection",
        "or": "union",
    }
    action = action_aliases.get(
        str(raw_action).strip().lower(), str(raw_action).strip().lower()
    )
    if action not in ACTIONS:
        raise ValueError(
            f"record {transition!r} must identify action in {list(ACTIONS)} via "
            "action/composition/route_action/mask_operation or transition_id"
        )

    raw_round = next(
        (
            record.get(key)
            for key in ("round_index", "prompt_round", "round")
            if record.get(key) is not None
        ),
        None,
    )
    if raw_round is None and isinstance(record.get("prompt_metadata"), Mapping):
        metadata = record["prompt_metadata"]
        raw_round = metadata.get("round_index", metadata.get("round"))
    if raw_round is None:
        match = re.search(r"(?:^|[_-])(?:round|r)([12])(?:$|[_-])", transition.lower())
        if match:
            raw_round = match.group(1)
    try:
        round_index = int(raw_round)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"record {transition!r} must identify prompt round 1/2 via "
            "round_index/prompt_round/round/prompt_metadata or transition_id"
        ) from exc
    if round_index not in ROUNDS:
        raise ValueError(
            f"record {transition!r} round_index must be 1 or 2, got {round_index}"
        )
    return action, round_index


def _normalize_candidate_routes(
    candidate_routes: Sequence[str] | None,
) -> tuple[tuple[str, ...], str]:
    if candidate_routes is None:
        return DEFAULT_CANDIDATE_ROUTES, "legacy_six_route_default"
    if isinstance(candidate_routes, (str, bytes)):
        raise ValueError("candidate_routes must be a sequence of exact route IDs")
    menu = tuple(candidate_routes)
    if not menu or any(not isinstance(route_id, str) for route_id in menu):
        raise ValueError("candidate_routes must be a non-empty sequence of strings")
    duplicates = sorted({route_id for route_id in menu if menu.count(route_id) > 1})
    if duplicates:
        raise ValueError(f"candidate_routes contains duplicates: {duplicates}")
    if BASELINE_ROUTE not in menu:
        raise ValueError(
            f"candidate_routes must contain baseline route {BASELINE_ROUTE!r}"
        )
    unknown = sorted(set(menu) - set(DEFAULT_CANDIDATE_ROUTES))
    if unknown:
        raise ValueError(f"candidate_routes contains unknown route IDs: {unknown}")
    if len(menu) == 1:
        raise ValueError("candidate_routes must declare at least one proposal route")
    return menu, "manifest_candidate_routes"


def validate_route_contract(
    candidates: Sequence[RouteCandidate],
    *,
    require_all_routes: bool = True,
    candidate_routes: Sequence[str] | None = None,
    required_splits: Sequence[str] | None = None,
    claim_external_validation: bool = False,
    minimum_test_patients: int = 20,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one route candidate is required")
    route_menu, route_menu_source = _normalize_candidate_routes(candidate_routes)
    proposal_routes = tuple(
        route_id for route_id in route_menu if route_id != BASELINE_ROUTE
    )
    examples: list[UpdateExample] = []
    for candidate in candidates:
        features = np.asarray(candidate.features, dtype=np.float32)
        if features.shape != (len(FEATURE_ORDER),):
            raise ValueError(
                f"features for {candidate.uid} must have shape ({len(FEATURE_ORDER)},), "
                f"got {features.shape}"
            )
        if candidate.action not in ACTIONS or candidate.round_index not in ROUNDS:
            raise ValueError(
                f"invalid route {candidate.route_id} for {candidate.case_id}"
            )
        numeric = (
            candidate.utility,
            candidate.baseline_dice,
            candidate.candidate_dice,
            candidate.baseline_nsd,
            candidate.candidate_nsd,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError(f"non-finite metric for {candidate.uid}")
        examples.append(
            UpdateExample(
                case_id=candidate.case_id,
                patient_id=candidate.patient_id,
                transition_id=candidate.transition_id,
                split=candidate.split,
                prior_exposure=candidate.prior_exposure,
                features=features,
                accept_label=int(candidate.utility > 0.0),
                utility=float(candidate.utility),
                delta_dice=float(candidate.delta_dice),
                delta_nsd=float(candidate.delta_nsd),
            )
        )
    split_contract = validate_split_contract(
        examples,
        require_all_splits=required_splits is None,
        claim_external_validation=claim_external_validation,
        minimum_test_patients=minimum_test_patients,
    )
    if required_splits is not None:
        required = tuple(str(split) for split in required_splits)
        unknown_required = sorted(
            set(required) - {"train", "calibration", "policy_validation", "test"}
        )
        if unknown_required:
            raise ValueError(f"unknown required splits: {unknown_required}")
        missing_splits = [
            split
            for split in required
            if int(split_contract["split_counts"].get(split, 0)) == 0
        ]
        if missing_splits:
            raise ValueError(f"required route splits are empty: {missing_splits}")

    case_groups = _group_by_case(candidates)
    missing_by_case: dict[str, list[str]] = {}
    for case_id, group in case_groups.items():
        patients = {candidate.patient_id for candidate in group}
        splits = {candidate.split for candidate in group}
        if len(patients) != 1 or len(splits) != 1:
            raise ValueError(f"case {case_id} has inconsistent patient/split metadata")
        route_ids = [candidate.route_id for candidate in group]
        duplicates = sorted(
            {route for route in route_ids if route_ids.count(route) > 1}
        )
        if duplicates:
            raise ValueError(f"case {case_id} has duplicate routes: {duplicates}")
        missing = sorted(set(proposal_routes) - set(route_ids))
        extra = sorted(set(route_ids) - set(proposal_routes))
        if extra:
            if candidate_routes is None:
                raise ValueError(f"case {case_id} has unknown routes: {extra}")
            raise ValueError(
                f"case {case_id} has undeclared routes {extra}; exact manifest "
                f"candidate_routes are {list(route_menu)}"
            )
        if (candidate_routes is not None or require_all_routes) and missing:
            missing_by_case[case_id] = missing
        baseline_dice = np.asarray(
            [candidate.baseline_dice for candidate in group], dtype=float
        )
        baseline_nsd = np.asarray(
            [candidate.baseline_nsd for candidate in group], dtype=float
        )
        if float(np.ptp(baseline_dice)) > 1e-10 or float(np.ptp(baseline_nsd)) > 1e-10:
            raise ValueError(f"case {case_id} does not use one common ResEnc baseline")
    if missing_by_case:
        if candidate_routes is not None:
            raise ValueError(
                "every study must contain exactly the declared candidate route menu; "
                f"missing proposal routes: {missing_by_case}"
            )
        raise ValueError(
            f"fixed-route evaluation requires all six routes per case: {missing_by_case}"
        )

    return {
        **split_contract,
        "case_counts": {
            split: len(
                {
                    candidate.case_id
                    for candidate in candidates
                    if candidate.split == split
                }
            )
            for split in ("train", "calibration", "policy_validation", "test")
        },
        "candidate_routes": list(route_menu),
        "baseline_route": BASELINE_ROUTE,
        "proposal_routes": list(proposal_routes),
        "required_routes": list(proposal_routes),
        "route_menu_source": route_menu_source,
        "all_cases_complete": not missing_by_case,
    }


def load_route_manifest(
    manifest_path: str | Path,
    *,
    nsd_tolerance_mm: float = 2.0,
    nsd_weight: float = 0.0,
    interaction_cost: float = 0.0,
    accept_margin: float = 0.0,
    exact_splits: Sequence[str] | None = None,
) -> tuple[list[RouteCandidate], dict[str, Any]]:
    """Load/verify prompt-update records and recompute route Dice/NSD."""

    if nsd_tolerance_mm < 0:
        raise ValueError("nsd_tolerance_mm must be >= 0")
    manifest = Path(manifest_path).resolve()
    examples, payload = examples_from_manifest(
        manifest,
        nsd_weight=nsd_weight,
        interaction_cost=interaction_cost,
        accept_margin=accept_margin,
        exact_splits=exact_splits,
    )
    records = list(payload.get("records", []))
    if len(records) != len(examples):
        raise ValueError(
            "manifest parser returned a different number of examples than records"
        )
    example_map: dict[tuple[str, str], UpdateExample] = {}
    record_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record, example in zip(records, examples, strict=True):
        for role in ("pet", "ct", "current_mask", "proposed_mask", "ground_truth"):
            expected_hash = str(record.get(f"{role}_sha256", ""))
            if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
                raise ValueError(
                    f"strict route manifest requires {role}_sha256 for "
                    f"{record.get('case_id')}/{record.get('transition_id')}"
                )
        if record.get("totseg_path"):
            expected_hash = str(record.get("totseg_sha256", ""))
            if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
                raise ValueError(
                    f"strict route manifest requires totseg_sha256 for "
                    f"{record.get('case_id')}/{record.get('transition_id')}"
                )
        key = (str(record["case_id"]), str(record["transition_id"]))
        if key in example_map:
            raise ValueError(f"duplicate case_id/transition_id in manifest: {key}")
        example_map[key] = example
        record_map[key] = record

    grouped_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in example_map:
        grouped_keys[key[0]].append(key)
    root = manifest.parent
    candidates: list[RouteCandidate] = []
    file_hash_cache: dict[Path, str] = {}

    def file_hash(path: Path) -> str:
        if path not in file_hash_cache:
            file_hash_cache[path] = sha256_file(path)
        return file_hash_cache[path]

    for case_id in sorted(grouped_keys):
        keys = grouped_keys[case_id]
        case_records = [record_map[key] for key in keys]
        pet_paths = [
            _resolve_path(root, str(record["pet_path"])) for record in case_records
        ]
        ct_paths = [
            _resolve_path(root, str(record["ct_path"])) for record in case_records
        ]
        current_paths = [
            _resolve_path(root, str(record["current_mask_path"]))
            for record in case_records
        ]
        ground_truth_paths = [
            _resolve_path(root, str(record["ground_truth_path"]))
            for record in case_records
        ]
        pet_hashes = {file_hash(path) for path in pet_paths}
        ct_hashes = {file_hash(path) for path in ct_paths}
        current_hashes = {file_hash(path) for path in current_paths}
        ground_truth_hashes = {file_hash(path) for path in ground_truth_paths}
        if len(pet_hashes) != 1:
            raise ValueError(f"case {case_id} routes do not share one PET volume")
        if len(ct_hashes) != 1:
            raise ValueError(f"case {case_id} routes do not share one CT volume")
        if len(current_hashes) != 1:
            raise ValueError(
                f"case {case_id} routes do not share one ResEnc current mask"
            )
        if len(ground_truth_hashes) != 1:
            raise ValueError(
                f"case {case_id} routes do not share one ground-truth mask"
            )

        current_img, current = _load_binary(current_paths[0])
        gt_img, ground_truth = _load_binary(ground_truth_paths[0])
        _require_same_grid(case_id, "current_mask", current_img, gt_img)
        spacing = tuple(float(value) for value in gt_img.header.get_zooms()[:3])
        baseline_dice = dice_score(current, ground_truth)
        baseline_nsd = normalized_surface_dice(
            current,
            ground_truth,
            tolerance=nsd_tolerance_mm,
            spacing=spacing,
        )

        for key in sorted(keys):
            record = record_map[key]
            example = example_map[key]
            action, round_index = parse_route_metadata(record)
            proposed_path = _resolve_path(root, str(record["proposed_mask_path"]))
            proposed_img, proposed = _load_binary(proposed_path)
            _require_same_grid(case_id, "proposed_mask", proposed_img, gt_img)
            candidate_dice = dice_score(proposed, ground_truth)
            candidate_nsd = normalized_surface_dice(
                proposed,
                ground_truth,
                tolerance=nsd_tolerance_mm,
                spacing=spacing,
            )
            if not math.isclose(
                float(example.delta_dice),
                float(candidate_dice - baseline_dice),
                abs_tol=1e-8,
            ):
                raise ValueError(
                    f"Dice recomputation mismatch for {case_id}/{example.transition_id}"
                )
            recomputed_delta_nsd = float(candidate_nsd - baseline_nsd)
            if example.delta_nsd is not None and not math.isclose(
                float(example.delta_nsd),
                recomputed_delta_nsd,
                abs_tol=1e-8,
            ):
                raise ValueError(
                    f"NSD recomputation mismatch for {case_id}/{example.transition_id}"
                )
            if nsd_weight != 0.0 and example.delta_nsd is None:
                raise ValueError(
                    "nsd_weight is non-zero but the manifest omits delta_nsd for "
                    f"{case_id}/{example.transition_id}"
                )
            candidates.append(
                RouteCandidate(
                    case_id=example.case_id,
                    patient_id=example.patient_id,
                    transition_id=example.transition_id,
                    split=example.split,
                    prior_exposure=example.prior_exposure,
                    action=action,
                    round_index=round_index,
                    features=np.asarray(example.features, dtype=np.float32),
                    utility=float(example.utility),
                    baseline_dice=float(baseline_dice),
                    candidate_dice=float(candidate_dice),
                    baseline_nsd=float(baseline_nsd),
                    candidate_nsd=float(candidate_nsd),
                )
            )
    return candidates, payload


def score_edl_candidates(
    candidates: Sequence[RouteCandidate],
    *,
    model: Any,
    normalizer: Any,
    checkpoint: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
    force_keep_all: bool = False,
) -> dict[str, EvidentialScore]:
    """Score candidate features on CPU using the frozen compatible EDL head."""

    if not candidates:
        return {}
    features = np.stack(
        [np.asarray(candidate.features, dtype=np.float32) for candidate in candidates]
    )
    x = torch.from_numpy(normalizer.transform(features)).to("cpu")
    model = model.to("cpu")
    model.eval()
    with torch.no_grad():
        alpha, utility = model(x)
        raw_probability = alpha[:, 1] / alpha.sum(dim=-1)
        vacuity = 2.0 / alpha.sum(dim=-1)
    p = np.clip(raw_probability.cpu().numpy().astype(np.float64), 1e-6, 1.0 - 1e-6)
    temperature = float(checkpoint["calibration"]["temperature"])
    logits = np.log(p / (1.0 - p)) / temperature
    probability = 1.0 / (1.0 + np.exp(-logits))
    vac = vacuity.cpu().numpy().astype(np.float64)
    predicted = utility.cpu().numpy().astype(np.float64)
    deployed_thresholds = checkpoint["thresholds"] if thresholds is None else thresholds
    changed_index = FEATURE_ORDER.index("changed_volume_fraction")
    changed = features[:, changed_index] > 0.0
    accepted = (
        (not force_keep_all)
        & changed
        & (probability >= float(deployed_thresholds["accept_probability"]))
        & (vac <= float(deployed_thresholds["max_accept_vacuity"]))
        & (predicted >= float(deployed_thresholds["min_predicted_utility"]))
    )
    return {
        candidate.uid: EvidentialScore(
            p_accept=float(probability[index]),
            vacuity=float(vac[index]),
            predicted_utility=float(predicted[index]),
            accepted=bool(accepted[index]),
            confidence=float(probability[index] * (1.0 - vac[index])),
        )
        for index, candidate in enumerate(candidates)
    }


def select_grouped_edl_thresholds(
    candidates: Sequence[RouteCandidate],
    scores: Mapping[str, EvidentialScore],
    *,
    max_harmful_study_rate: float = SAFETY_MAX_HARMFUL_STUDY_RATE,
    bootstrap_samples: int = SAFETY_BOOTSTRAP_SAMPLES,
    seed: int = SAFETY_BOOTSTRAP_SEED,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Freeze EDL ACCEPT thresholds on grouped policy-validation menus.

    Each grid point first forms a candidate menu per study, chooses the
    highest predicted-utility ACCEPT candidate, and otherwise keeps ResEnc.
    A threshold may deploy only when its policy-validation harmful-study rate
    is at most the prespecified cap and the patient-cluster bootstrap lower
    bound for mean utility is strictly positive. Otherwise deployment is
    frozen to KEEP for every study.
    """

    if not candidates or {candidate.split for candidate in candidates} != {
        "policy_validation"
    }:
        raise ValueError(
            "grouped EDL thresholds require only non-empty policy_validation candidates"
        )
    missing = [candidate.uid for candidate in candidates if candidate.uid not in scores]
    if missing:
        raise ValueError(f"missing EDL policy-validation scores for {missing}")
    _validate_safety_parameters(max_harmful_study_rate, bootstrap_samples)
    for candidate in candidates:
        score = scores[candidate.uid]
        if not all(
            math.isfinite(float(value))
            for value in (score.p_accept, score.vacuity, score.predicted_utility)
        ):
            raise ValueError(f"non-finite EDL score for {candidate.uid}")
    groups = _ordered_case_groups(candidates)
    patient_ids, bootstrap_indices = _patient_bootstrap_plan(
        groups, samples=bootstrap_samples, seed=seed
    )
    trials: list[dict[str, Any]] = []
    ranked: list[
        tuple[
            tuple[float, ...],
            dict[str, float],
            dict[str, Any],
        ]
    ] = []
    for probability_threshold in EDL_ACCEPT_PROBABILITY_GRID:
        for maximum_vacuity in EDL_MAX_VACUITY_GRID:
            for minimum_utility in EDL_MIN_UTILITY_GRID:
                choices: list[RouteCandidate | None] = []
                for group in groups:
                    accepted = [
                        candidate
                        for candidate in group
                        if candidate.features[
                            FEATURE_ORDER.index("changed_volume_fraction")
                        ]
                        > 0.0
                        and scores[candidate.uid].p_accept >= probability_threshold
                        and scores[candidate.uid].vacuity <= maximum_vacuity
                        and scores[candidate.uid].predicted_utility >= minimum_utility
                    ]
                    choices.append(
                        max(
                            accepted,
                            key=lambda candidate: (
                                scores[candidate.uid].predicted_utility,
                                scores[candidate.uid].p_accept,
                                -scores[candidate.uid].vacuity,
                                candidate.route_id,
                            ),
                        )
                        if accepted
                        else None
                    )
                objective = _deployment_objective(
                    groups,
                    choices,
                    patient_ids=patient_ids,
                    bootstrap_indices=bootstrap_indices,
                    max_harmful_study_rate=max_harmful_study_rate,
                )
                thresholds = {
                    "accept_probability": float(probability_threshold),
                    "max_accept_vacuity": float(maximum_vacuity),
                    "min_predicted_utility": float(minimum_utility),
                }
                trial = {**thresholds, **objective}
                trials.append(trial)
                key = (
                    float(objective["patient_mean_realized_utility"]),
                    float(
                        objective[
                            "patient_cluster_bootstrap_95_ci_mean_realized_utility"
                        ]["lower"]
                    ),
                    float(objective["mean_realized_utility"]),
                    -float(objective["harmful_action_rate_all_studies"]),
                    float(objective["coverage"]),
                    float(probability_threshold),
                    -float(maximum_vacuity),
                    float(minimum_utility),
                )
                ranked.append((key, thresholds, objective))
    ranked.sort(key=lambda item: item[0], reverse=True)
    eligible = [
        item for item in ranked if bool(item[2]["safety_constraints_satisfied"])
    ]
    fallback_reason: str | None = None
    if eligible:
        selected = eligible[0]
        deployed_thresholds = selected[1]
        selected_objective = selected[2]
        safety_deployed = True
        deployment_decision = "SELECT_ROUTE_OR_KEEP"
    else:
        maximum_predicted = max(
            float(scores[candidate.uid].predicted_utility) for candidate in candidates
        )
        margin = max(1.0, abs(maximum_predicted)) * 1e-9
        deployed_thresholds = {
            "accept_probability": 1.0,
            "max_accept_vacuity": 0.0,
            "min_predicted_utility": maximum_predicted + margin,
        }
        selected_objective = _deployment_objective(
            groups,
            [None] * len(groups),
            patient_ids=patient_ids,
            bootstrap_indices=bootstrap_indices,
            max_harmful_study_rate=max_harmful_study_rate,
        )
        safety_deployed = False
        deployment_decision = "KEEP_ALL"
        fallback_reason = (
            "no policy-validation EDL threshold satisfied both the harmful-study "
            "rate cap and strictly positive patient-cluster bootstrap lower bound"
        )
    candidate_count = len(groups[0])
    candidate_count_label = "six" if candidate_count == 6 else str(candidate_count)
    return deployed_thresholds, {
        "selection_unit": (f"grouped per-study {candidate_count_label}-candidate menu"),
        "selection_options_per_study": candidate_count + 1,
        "baseline_option": BASELINE_ROUTE,
        "selection_split": "policy_validation",
        "independent_patient_count": len(patient_ids),
        "study_count": len(groups),
        "candidate_record_count": len(candidates),
        "threshold_grid_points": len(trials),
        "candidate_grid_size": len(trials),
        "eligible_grid_points": len(eligible),
        "safety_deployed": safety_deployed,
        "deployment_decision": deployment_decision,
        "fallback_reason": fallback_reason,
        "safety_gate": _safety_gate_report(
            max_harmful_study_rate=max_harmful_study_rate,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "selected_objective": selected_objective,
        "top_10": [
            {**thresholds, **objective}
            for _key, thresholds, objective in (eligible or ranked)[:10]
        ],
        "top_10_unconstrained": [
            {**thresholds, **objective} for _key, thresholds, objective in ranked[:10]
        ],
    }


def fit_linear_contextual_bandit(
    candidates: Sequence[RouteCandidate],
    *,
    ridge_lambdas: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    max_harmful_study_rate: float = SAFETY_MAX_HARMFUL_STUDY_RATE,
    bootstrap_samples: int = SAFETY_BOOTSTRAP_SAMPLES,
    seed: int = SAFETY_BOOTSTRAP_SEED,
) -> tuple[LinearContextualBandit, dict[str, Any]]:
    """Fit/calibrate/select a full-information ridge comparator.

    Test candidates are rejected at the API boundary so their outcome labels
    cannot enter fitting, calibration, model selection, or safety selection.
    """

    if any(candidate.split == "test" for candidate in candidates):
        raise ValueError(
            "fit_linear_contextual_bandit accepts development splits only; "
            "test labels must remain inaccessible to fit/selection APIs"
        )
    unexpected_splits = sorted(
        {
            candidate.split
            for candidate in candidates
            if candidate.split not in {"train", "calibration", "policy_validation"}
        }
    )
    if unexpected_splits:
        raise ValueError(f"unexpected development splits: {unexpected_splits}")
    _validate_safety_parameters(max_harmful_study_rate, bootstrap_samples)

    train = [candidate for candidate in candidates if candidate.split == "train"]
    calibration = [
        candidate for candidate in candidates if candidate.split == "calibration"
    ]
    policy_validation = [
        candidate for candidate in candidates if candidate.split == "policy_validation"
    ]
    if not train or not calibration or not policy_validation:
        raise ValueError(
            "bandit requires non-empty train/calibration/policy_validation splits"
        )
    lambdas = tuple(float(value) for value in ridge_lambdas)
    if not lambdas or any(not math.isfinite(value) or value <= 0 for value in lambdas):
        raise ValueError("ridge_lambdas must be finite and > 0")

    train_features = np.stack([candidate.features for candidate in train]).astype(
        np.float64
    )
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-6] = 1.0
    x_train = _design_matrix(train, mean, std)
    y_train = np.asarray([candidate.utility for candidate in train], dtype=np.float64)
    selection_rows: list[dict[str, Any]] = []
    fitted: list[tuple[tuple[float, ...], LinearContextualBandit, dict[str, Any]]] = []

    for ridge_lambda in lambdas:
        penalty = np.eye(x_train.shape[1], dtype=np.float64) * ridge_lambda
        penalty[-1, -1] = 0.0
        raw_coefficients = (
            np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y_train
        )
        provisional = LinearContextualBandit(
            ridge_lambda=ridge_lambda,
            threshold=0.0,
            feature_mean=mean.copy(),
            feature_std=std.copy(),
            coefficients=raw_coefficients,
        )
        raw_calibration_scores = provisional.predict(calibration)
        calibration_targets = np.asarray(
            [candidate.utility for candidate in calibration], dtype=np.float64
        )
        calibration_design = np.column_stack(
            [
                raw_calibration_scores,
                np.ones(len(raw_calibration_scores), dtype=np.float64),
            ]
        )
        calibration_parameters = (
            np.linalg.pinv(calibration_design) @ calibration_targets
        )
        slope, intercept = (
            float(calibration_parameters[0]),
            float(calibration_parameters[1]),
        )
        coefficients = raw_coefficients * slope
        coefficients[-1] += intercept
        calibrated_scores = calibration_design @ calibration_parameters
        model = LinearContextualBandit(
            ridge_lambda=ridge_lambda,
            threshold=0.0,
            feature_mean=mean.copy(),
            feature_std=std.copy(),
            coefficients=coefficients,
        )
        validation_scores = model.predict(policy_validation)
        threshold, validation_objective = _calibrate_bandit_threshold(
            policy_validation,
            validation_scores,
            max_harmful_study_rate=max_harmful_study_rate,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        model = LinearContextualBandit(
            ridge_lambda=ridge_lambda,
            threshold=threshold,
            feature_mean=mean.copy(),
            feature_std=std.copy(),
            coefficients=coefficients,
            deploy_keep_all=not bool(validation_objective["safety_deployed"]),
        )
        row = {
            "ridge_lambda": ridge_lambda,
            "calibration": {
                "method": "affine_least_squares_on_candidate_utility",
                "slope": slope,
                "intercept": intercept,
                "mean_absolute_error": float(
                    np.mean(np.abs(calibrated_scores - calibration_targets))
                ),
            },
            "policy_validation": {
                "grouped_menu_threshold": threshold,
                **validation_objective,
            },
        }
        selection_rows.append(row)
        key = (
            float(validation_objective["patient_mean_realized_utility"]),
            float(
                validation_objective[
                    "patient_cluster_bootstrap_95_ci_mean_realized_utility"
                ]["lower"]
            ),
            float(validation_objective["mean_realized_utility"]),
            -float(validation_objective["harmful_action_rate_all_studies"]),
            float(validation_objective["coverage"]),
            -ridge_lambda,
        )
        fitted.append((key, model, row))
    eligible = [item for item in fitted if not item[1].deploy_keep_all]
    fallback_reason: str | None = None
    if eligible:
        _key, selected, selected_row = max(eligible, key=lambda item: item[0])
        deployment_decision = "SELECT_ROUTE_OR_KEEP"
    else:
        _key, selected, selected_row = min(
            fitted, key=lambda item: float(item[1].ridge_lambda)
        )
        deployment_decision = "KEEP_ALL"
        fallback_reason = (
            "no ridge/threshold policy-validation candidate satisfied both the "
            "harmful-study rate cap and strictly positive patient-cluster "
            "bootstrap lower bound"
        )
    return selected, {
        "methodology": "full_information_linear_ridge_comparator",
        "selection_contract": {
            "train": "fit feature normalization and ridge coefficients",
            "calibration": "fit affine utility-score calibration for each ridge value",
            "policy_validation": (
                "select only grouped-menu thresholds/ridge values passing the "
                "prespecified patient-cluster safety gate"
            ),
            "test": "rejected by fit_linear_contextual_bandit API",
        },
        "safety_gate": _safety_gate_report(
            max_harmful_study_rate=max_harmful_study_rate,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "candidate_model_grid_size": len(selection_rows),
        "threshold_candidate_grid_size": int(
            sum(
                int(row["policy_validation"]["candidate_grid_size"])
                for row in selection_rows
            )
        ),
        "eligible_model_count": len(eligible),
        "independent_patient_count": int(
            selected_row["policy_validation"]["independent_patient_count"]
        ),
        "study_count": int(selected_row["policy_validation"]["study_count"]),
        "deployment_decision": deployment_decision,
        "fallback_reason": fallback_reason,
        "candidate_models": selection_rows,
        "selected": selected.report(),
        "selected_policy_validation": selected_row["policy_validation"],
    }


def evaluate_test_policies(
    candidates: Sequence[RouteCandidate],
    *,
    edl_scores: Mapping[str, EvidentialScore],
    bandit: LinearContextualBandit,
    candidate_routes: Sequence[str] | None = None,
    edl_deploy_keep_all: bool = False,
    bootstrap_samples: int = 10_000,
    seed: int = 20260715,
) -> dict[str, Any]:
    """Evaluate frozen policies on test; no policy selection occurs here."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    test = [candidate for candidate in candidates if candidate.split == "test"]
    if not test:
        raise ValueError("test split is empty")
    groups = _group_by_case(test)
    route_menu, _route_menu_source = _normalize_candidate_routes(candidate_routes)
    proposal_routes = tuple(
        route_id for route_id in route_menu if route_id != BASELINE_ROUTE
    )
    for case_id, group in groups.items():
        route_ids = [candidate.route_id for candidate in group]
        if len(route_ids) != len(set(route_ids)) or set(route_ids) != set(
            proposal_routes
        ):
            raise ValueError(
                f"test study {case_id} does not exactly match candidate route menu "
                f"{list(route_menu)}"
            )
    policies: dict[str, dict[str, RouteCandidate | None]] = {}
    policies["keep_resenc"] = {case_id: None for case_id in groups}
    for route_id in proposal_routes:
        policies[f"fixed_{route_id}"] = {
            case_id: _route_lookup(group)[route_id] for case_id, group in groups.items()
        }
    policies["hindsight_oracle"] = {
        case_id: _oracle_or_keep(group) for case_id, group in groups.items()
    }
    policies["edl_accept_best_utility"] = {}
    for case_id, group in groups.items():
        missing = [
            candidate.uid for candidate in group if candidate.uid not in edl_scores
        ]
        if missing:
            raise ValueError(f"missing EDL scores for {missing}")
        accepted = (
            []
            if edl_deploy_keep_all
            else [
                candidate for candidate in group if edl_scores[candidate.uid].accepted
            ]
        )
        policies["edl_accept_best_utility"][case_id] = (
            max(
                accepted,
                key=lambda candidate: (
                    edl_scores[candidate.uid].predicted_utility,
                    edl_scores[candidate.uid].p_accept,
                    -edl_scores[candidate.uid].vacuity,
                    candidate.route_id,
                ),
            )
            if accepted
            else None
        )
    bandit_predictions = bandit.predict(test)
    bandit_score_map = {
        candidate.uid: float(score)
        for candidate, score in zip(test, bandit_predictions, strict=True)
    }
    policies["linear_contextual_bandit"] = (
        {case_id: None for case_id in groups}
        if bandit.deploy_keep_all
        else {
            case_id: _positive_or_keep(
                group, bandit_score_map, threshold=bandit.threshold
            )
            for case_id, group in groups.items()
        }
    )

    per_study: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    per_patient: list[dict[str, Any]] = []
    for policy_index, (policy, choices) in enumerate(policies.items()):
        rows: list[dict[str, Any]] = []
        for case_id in sorted(groups):
            group = groups[case_id]
            choice = choices[case_id]
            confidence: float | None = None
            p_accept: float | None = None
            vacuity: float | None = None
            predicted_utility: float | None = None
            if choice is not None and policy == "edl_accept_best_utility":
                score = edl_scores[choice.uid]
                confidence = score.confidence
                p_accept = score.p_accept
                vacuity = score.vacuity
                predicted_utility = score.predicted_utility
            elif choice is not None and policy == "linear_contextual_bandit":
                predicted_utility = bandit_score_map[choice.uid]
                confidence = predicted_utility - bandit.threshold
            rows.append(
                _choice_row(
                    policy,
                    group[0],
                    choice,
                    confidence=confidence,
                    p_accept=p_accept,
                    vacuity=vacuity,
                    predicted_utility=predicted_utility,
                )
            )
        per_study.extend(rows)
        patient_rows = _patient_rows(policy, rows)
        per_patient.extend(patient_rows)
        summary = _policy_summary(
            rows, patient_rows, bootstrap_samples, seed + policy_index
        )
        if policy in {"edl_accept_best_utility", "linear_contextual_bandit"}:
            summary["risk_coverage"] = _risk_coverage(rows)
        summaries[policy] = summary
    return {
        "scope": "frozen_test",
        "candidate_routes": list(route_menu),
        "test_label_evaluation_passes": 1,
        "study_count": len(groups),
        "patient_count": len({group[0].patient_id for group in groups.values()}),
        "policies": summaries,
        "per_study": per_study,
        "per_patient": per_patient,
    }


def write_evaluation_outputs(
    report: Mapping[str, Any], *, output_dir: str | Path
) -> dict[str, str]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "route_policy_report.json"
    study_path = destination / "route_policy_per_study.csv"
    patient_path = destination / "route_policy_per_patient.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(study_path, list(report["test_evaluation"]["per_study"]))
    _write_csv(patient_path, list(report["test_evaluation"]["per_patient"]))
    return {
        "report": str(report_path),
        "per_study_csv": str(study_path),
        "per_patient_csv": str(patient_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("legacy-one-shot", "freeze-development", "score-test"),
        default="legacy-one-shot",
        help=(
            "fusion-v2 must use freeze-development then score-test; "
            "legacy-one-shot is retained only for old exploratory reports"
        ),
    )
    parser.add_argument(
        "--manifest", required=True, help="frozen strict schema-v1 route manifest"
    )
    parser.add_argument(
        "--edl-checkpoint", required=True, help="compatible prompt_update_edl.pt"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nsd-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--nsd-weight", type=float, default=0.0)
    parser.add_argument("--interaction-cost", type=float, default=0.0)
    parser.add_argument("--accept-margin", type=float, default=0.0)
    parser.add_argument("--ridge-lambdas", default="0.01,0.1,1,10")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--claim-external-validation", action="store_true")
    parser.add_argument("--minimum-test-patients", type=int, default=20)
    parser.add_argument(
        "--deployment-plan",
        help="frozen route_policy_deployment.json required by score-test",
    )
    parser.add_argument(
        "--deployment-sha256",
        help="expected SHA-256 of --deployment-plan required by score-test",
    )
    args = parser.parse_args(argv)

    if args.phase == "freeze-development":
        return _freeze_development(args)
    if args.phase == "score-test":
        return _score_test(args)

    manifest = Path(args.manifest).resolve()
    checkpoint_path = Path(args.edl_checkpoint).resolve()
    manifest_hash = sha256_file(manifest)
    candidates, payload = load_route_manifest(
        manifest,
        nsd_tolerance_mm=args.nsd_tolerance_mm,
        nsd_weight=args.nsd_weight,
        interaction_cost=args.interaction_cost,
        accept_margin=args.accept_margin,
    )
    route_contract = validate_route_contract(
        candidates,
        candidate_routes=payload.get("candidate_routes"),
        claim_external_validation=args.claim_external_validation,
        minimum_test_patients=args.minimum_test_patients,
    )
    candidate_route_menu = tuple(route_contract["candidate_routes"])
    development_candidates = [
        candidate for candidate in candidates if candidate.split != "test"
    ]
    ridge_lambdas = tuple(
        float(value.strip()) for value in args.ridge_lambdas.split(",") if value.strip()
    )
    bandit, bandit_fit = fit_linear_contextual_bandit(
        development_candidates,
        ridge_lambdas=ridge_lambdas,
        max_harmful_study_rate=SAFETY_MAX_HARMFUL_STUDY_RATE,
        bootstrap_samples=SAFETY_BOOTSTRAP_SAMPLES,
        seed=SAFETY_BOOTSTRAP_SEED,
    )

    model, normalizer, checkpoint = load_checkpoint_bundle(
        checkpoint_path, device="cpu"
    )
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise ValueError(
            "EDL checkpoint/route manifest mismatch: checkpoint was not trained on this exact frozen manifest"
        )
    if checkpoint.get("threshold_source") != "policy_validation":
        raise ValueError(
            "strict route evaluation requires EDL thresholds frozen on policy_validation"
        )
    if bool(checkpoint.get("mechanics_smoke", False)):
        raise ValueError(
            "mechanics-smoke EDL checkpoints cannot be used for frozen route evaluation"
        )
    policy_validation_candidates = [
        candidate for candidate in candidates if candidate.split == "policy_validation"
    ]
    policy_validation_scores = score_edl_candidates(
        policy_validation_candidates,
        model=model,
        normalizer=normalizer,
        checkpoint=checkpoint,
    )
    grouped_edl_thresholds, grouped_edl_selection = select_grouped_edl_thresholds(
        policy_validation_candidates,
        policy_validation_scores,
        max_harmful_study_rate=SAFETY_MAX_HARMFUL_STUDY_RATE,
        bootstrap_samples=SAFETY_BOOTSTRAP_SAMPLES,
        seed=SAFETY_BOOTSTRAP_SEED,
    )
    test_candidates = [
        candidate for candidate in candidates if candidate.split == "test"
    ]
    # This is the sole test forward pass, after every threshold/model choice is frozen.
    edl_scores = score_edl_candidates(
        test_candidates,
        model=model,
        normalizer=normalizer,
        checkpoint=checkpoint,
        thresholds=grouped_edl_thresholds,
        force_keep_all=not bool(grouped_edl_selection["safety_deployed"]),
    )
    test_evaluation = evaluate_test_policies(
        candidates,
        edl_scores=edl_scores,
        bandit=bandit,
        candidate_routes=candidate_route_menu,
        edl_deploy_keep_all=not bool(grouped_edl_selection["safety_deployed"]),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if route_contract["prior_exposed_test_cases"]:
        report_status = "EXPLORATORY_INTERNAL_PRIOR_EXPOSED"
        claim_boundary = (
            "Internal prior-exposed frozen test only; no efficacy, external-validation, "
            "learned-selection, or clinical-generalization claim."
        )
    elif not route_contract["efficacy_claim_eligible"]:
        report_status = "EXPLORATORY_INSUFFICIENT_TEST_SAMPLE"
        claim_boundary = (
            "Exposure-independent frozen test is below the configured efficacy sample requirement; "
            "no clinical-generalization claim."
        )
    else:
        report_status = "COMPLETED_FROZEN_TEST"
        claim_boundary = (
            "External frozen validation under the configured exposure/sample contract; clinical validity "
            "remains out of scope."
            if args.claim_external_validation
            else "Frozen patient-disjoint test meeting the configured exposure/sample contract; clinical "
            "validity remains out of scope."
        )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": report_status,
        "claim_boundary": claim_boundary,
        "external_validation_eligible": route_contract["external_validation_eligible"],
        "efficacy_claim_eligible": route_contract["efficacy_claim_eligible"],
        "efficacy_ineligibility_reasons": route_contract[
            "efficacy_ineligibility_reasons"
        ],
        "minimum_test_patients": route_contract["minimum_test_patients"],
        "manifest": {
            "path": str(manifest),
            "sha256": manifest_hash,
            "source_status": payload.get("status", "unspecified"),
            "schema_version": payload.get("schema_version"),
            "candidate_routes": list(candidate_route_menu),
            "route_menu_source": route_contract["route_menu_source"],
        },
        "route_contract": route_contract,
        "utility_definition": {
            "delta_dice_weight": 1.0,
            "nsd_weight": float(args.nsd_weight),
            "interaction_cost": float(args.interaction_cost),
            "accept_margin": float(args.accept_margin),
            "hindsight_oracle": (
                "max realized utility among the declared proposal routes and KEEP=0"
            ),
            "candidate_proposal_count": len(route_contract["proposal_routes"]),
            "oracle_assistance_boundary": (
                "Candidate proposals are indirectly ground-truth-dependent because "
                "robot-user corrections are generated from ground truth. Deployable "
                "features exclude direct GT arrays, metrics, and scalars, but this "
                "remains offline oracle-assisted evidence."
            ),
        },
        "edl": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "manifest_sha256": checkpoint["manifest_sha256"],
            "config_sha256": checkpoint["config_sha256"],
            "upstream_flat_threshold_source": checkpoint["threshold_source"],
            "upstream_flat_thresholds_not_deployed": dict(checkpoint["thresholds"]),
            "deployed_grouped_menu_threshold_source": "policy_validation",
            "deployed_grouped_menu_thresholds": grouped_edl_thresholds,
            "deployment_decision": grouped_edl_selection["deployment_decision"],
            "fallback_reason": grouped_edl_selection["fallback_reason"],
            "grouped_menu_selection": grouped_edl_selection,
            "calibration": dict(checkpoint["calibration"]),
            "selection_rule": "highest predicted utility among ACCEPT candidates; otherwise KEEP ResEnc",
        },
        "linear_contextual_bandit": bandit_fit,
        "policy_safety_contract": {
            **_safety_gate_report(
                max_harmful_study_rate=SAFETY_MAX_HARMFUL_STUDY_RATE,
                bootstrap_samples=SAFETY_BOOTSTRAP_SAMPLES,
                seed=SAFETY_BOOTSTRAP_SEED,
            ),
            "selection_split": "policy_validation",
            "applies_to": [
                "edl_accept_gate",
                "full_information_linear_ridge_comparator",
            ],
            "fallback": "KEEP_ALL",
            "edl_grid": {
                "accept_probability": list(EDL_ACCEPT_PROBABILITY_GRID),
                "max_accept_vacuity": list(EDL_MAX_VACUITY_GRID),
                "min_predicted_utility": list(EDL_MIN_UTILITY_GRID),
                "candidate_grid_size": (
                    len(EDL_ACCEPT_PROBABILITY_GRID)
                    * len(EDL_MAX_VACUITY_GRID)
                    * len(EDL_MIN_UTILITY_GRID)
                ),
            },
            "ridge_lambdas": list(ridge_lambdas),
            "ridge_threshold_grid": (
                "min_score-epsilon, each unique policy-validation score, "
                "max_score+epsilon; epsilon=max(1,max_abs_score)*1e-9"
            ),
        },
        "no_test_tuning_audit": {
            "train": "normalizer and ridge coefficients only",
            "calibration": "bandit affine utility calibration; EDL temperature supplied by checkpoint",
            "policy_validation": (
                "grouped-menu ACCEPT/KEEP thresholds for EDL and bandit, plus bandit ridge selection"
            ),
            "test": (
                "one frozen EDL forward pass, one frozen ridge prediction pass, then metrics, "
                "W/T/L, harmful-action rate, bootstrap CI, and descriptive risk/coverage only"
            ),
            "test_label_evaluation_passes": 1,
            "test_used_for_model_or_threshold_selection": False,
        },
        "bootstrap": {
            "unit": "patient (cluster bootstrap over patient-level study means)",
            "samples": int(args.bootstrap_samples),
            "seed": int(args.seed),
            "interval": "percentile 95%",
        },
        "test_evaluation": test_evaluation,
    }
    paths = write_evaluation_outputs(report, output_dir=args.out_dir)
    print(json.dumps({"status": report["status"], **paths}, indent=2))
    return 0


def _manifest_metadata(manifest: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("route manifest schema_version must be 1")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("route manifest records must be a non-empty list")
    return payload


def _require_exact_manifest_splits(
    payload: Mapping[str, Any], expected: set[str], *, phase: str
) -> None:
    records = list(payload.get("records", []))
    splits = {str(record.get("split", "")) for record in records}
    if splits != expected:
        raise ValueError(
            f"{phase} requires manifest records from exactly {sorted(expected)}, "
            f"got {sorted(splits)}"
        )


def _policy_safety_contract(ridge_lambdas: Sequence[float]) -> dict[str, Any]:
    return {
        **_safety_gate_report(
            max_harmful_study_rate=SAFETY_MAX_HARMFUL_STUDY_RATE,
            bootstrap_samples=SAFETY_BOOTSTRAP_SAMPLES,
            seed=SAFETY_BOOTSTRAP_SEED,
        ),
        "applies_to": [
            "edl_accept_gate",
            "full_information_linear_ridge_comparator",
        ],
        "fallback": "KEEP_ALL",
        "edl_grid": {
            "accept_probability": list(EDL_ACCEPT_PROBABILITY_GRID),
            "max_accept_vacuity": list(EDL_MAX_VACUITY_GRID),
            "min_predicted_utility": list(EDL_MIN_UTILITY_GRID),
            "candidate_grid_size": (
                len(EDL_ACCEPT_PROBABILITY_GRID)
                * len(EDL_MAX_VACUITY_GRID)
                * len(EDL_MIN_UTILITY_GRID)
            ),
        },
        "ridge_lambdas": [float(value) for value in ridge_lambdas],
        "ridge_threshold_grid": (
            "min_score-epsilon, each unique policy-validation score, "
            "max_score+epsilon; epsilon=max(1,max_abs_score)*1e-9"
        ),
        "tie_break_order": [
            "patient_mean_realized_utility",
            "patient_cluster_bootstrap_95_ci_lower",
            "study_mean_realized_utility",
            "lower_harmful_study_rate",
            "coverage_then_deterministic_model_fields",
        ],
    }


def _freeze_development(args: argparse.Namespace) -> int:
    """Phase A: freeze/hash deployment without any test record or test path."""

    if args.deployment_plan or args.deployment_sha256:
        raise ValueError("freeze-development does not accept deployment inputs")
    manifest = Path(args.manifest).resolve()
    payload_preflight = _manifest_metadata(manifest)
    _require_exact_manifest_splits(
        payload_preflight,
        {"train", "calibration", "policy_validation"},
        phase="freeze-development",
    )
    manifest_hash = sha256_file(manifest)
    checkpoint_path = Path(args.edl_checkpoint).resolve()
    candidates, payload = load_route_manifest(
        manifest,
        nsd_tolerance_mm=args.nsd_tolerance_mm,
        nsd_weight=args.nsd_weight,
        interaction_cost=args.interaction_cost,
        accept_margin=args.accept_margin,
        exact_splits=("train", "calibration", "policy_validation"),
    )
    route_contract = validate_route_contract(
        candidates,
        candidate_routes=payload.get("candidate_routes"),
        required_splits=("train", "calibration", "policy_validation"),
        minimum_test_patients=args.minimum_test_patients,
    )
    ridge_lambdas = tuple(
        float(value.strip()) for value in args.ridge_lambdas.split(",") if value.strip()
    )
    bandit, bandit_fit = fit_linear_contextual_bandit(
        candidates,
        ridge_lambdas=ridge_lambdas,
        max_harmful_study_rate=SAFETY_MAX_HARMFUL_STUDY_RATE,
        bootstrap_samples=SAFETY_BOOTSTRAP_SAMPLES,
        seed=SAFETY_BOOTSTRAP_SEED,
    )
    model, normalizer, checkpoint = load_checkpoint_bundle(
        checkpoint_path, device="cpu"
    )
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise ValueError(
            "freeze-development requires an EDL checkpoint trained on the exact "
            "development-only manifest"
        )
    if checkpoint.get("threshold_source") != "policy_validation":
        raise ValueError("EDL checkpoint thresholds must come from policy_validation")
    checkpoint_counts = checkpoint.get("split_contract", {}).get("split_counts", {})
    development_checkpoint = (
        bool(checkpoint.get("development_freeze", False))
        and not bool(checkpoint.get("mechanics_smoke", False))
        and checkpoint.get("fit_mode") == "development_freeze"
        and checkpoint.get("status") == "DEVELOPMENT_FROZEN_NO_TEST"
        and int(checkpoint_counts.get("train", 0)) > 0
        and int(checkpoint_counts.get("calibration", 0)) > 0
        and int(checkpoint_counts.get("policy_validation", 0)) > 0
        and int(checkpoint_counts.get("test", 0)) == 0
        and int(checkpoint.get("test_metrics", {}).get("n", -1)) == 0
        and checkpoint.get("threshold_source") == "policy_validation"
        and checkpoint.get("threshold_role")
        == "upstream_flat_candidate_diagnostic_only"
        and not bool(checkpoint.get("external_validation_eligible", True))
        and not bool(checkpoint.get("efficacy_claim_eligible", True))
    )
    if not development_checkpoint:
        raise ValueError(
            "freeze-development requires an explicit DEVELOPMENT_FROZEN_NO_TEST "
            "EDL checkpoint with train/calibration/policy_validation and zero test"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("code_sha256", ""))):
        raise ValueError("development EDL checkpoint is missing a valid code_sha256")
    policy_validation = [
        candidate for candidate in candidates if candidate.split == "policy_validation"
    ]
    validation_scores = score_edl_candidates(
        policy_validation,
        model=model,
        normalizer=normalizer,
        checkpoint=checkpoint,
    )
    edl_thresholds, edl_selection = select_grouped_edl_thresholds(
        policy_validation,
        validation_scores,
        max_harmful_study_rate=SAFETY_MAX_HARMFUL_STUDY_RATE,
        bootstrap_samples=SAFETY_BOOTSTRAP_SAMPLES,
        seed=SAFETY_BOOTSTRAP_SEED,
    )
    utility_definition = {
        "delta_dice_weight": 1.0,
        "nsd_tolerance_mm": float(args.nsd_tolerance_mm),
        "nsd_weight": float(args.nsd_weight),
        "interaction_cost": float(args.interaction_cost),
        "accept_margin": float(args.accept_margin),
        "oracle_assistance_boundary": (
            "Candidate proposals are indirectly ground-truth-dependent because "
            "robot-user corrections are generated from ground truth. Deployable "
            "features exclude direct GT arrays, metrics, and scalars, but this "
            "remains offline oracle-assisted evidence."
        ),
    }
    destination = Path(args.out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    deployment = {
        "schema_version": 1,
        "artifact_type": "frozen_route_policy_deployment",
        "status": "FROZEN_DEVELOPMENT",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "development_manifest": {
            "path": str(manifest),
            "sha256": manifest_hash,
            "record_count": len(candidates),
            "candidate_routes": list(route_contract["candidate_routes"]),
            "splits_opened": ["train", "calibration", "policy_validation"],
            "test_records_opened": 0,
        },
        "development_patient_ids": sorted(
            {candidate.patient_id for candidate in candidates}
        ),
        "development_case_ids": sorted({candidate.case_id for candidate in candidates}),
        "route_contract": route_contract,
        "utility_definition": utility_definition,
        "policy_safety_contract": _policy_safety_contract(ridge_lambdas),
        "edl": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "manifest_sha256": checkpoint["manifest_sha256"],
            "config_sha256": checkpoint["config_sha256"],
            "code_sha256": checkpoint["code_sha256"],
            "calibration": dict(checkpoint["calibration"]),
            "deployed_thresholds": edl_thresholds,
            "selection": edl_selection,
            "deploy_keep_all": not bool(edl_selection["safety_deployed"]),
            "development_freeze": True,
        },
        "full_information_linear_ridge": {
            "model": bandit.report(),
            "fit_report": bandit_fit,
        },
        "selected_policies": {
            "edl_accept_gate": {
                "deployment_decision": edl_selection["deployment_decision"],
                "deploy_keep_all": not bool(edl_selection["safety_deployed"]),
                "thresholds": edl_thresholds,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "config_sha256": checkpoint["config_sha256"],
                "code_sha256": checkpoint["code_sha256"],
            },
            "full_information_linear_ridge_comparator": {
                "deployment_decision": bandit.report()["deployment_decision"],
                "deploy_keep_all": bool(bandit.deploy_keep_all),
                "ridge_lambda": float(bandit.ridge_lambda),
                "accept_threshold": float(bandit.threshold),
                "coefficients_sha256": bandit.report()["coefficients_sha256"],
            },
        },
        "test_open_control": {
            "pass_limit": 1,
            "attempt_receipt_path": str(destination / "test_open_attempt_receipt.json"),
            "completion_receipt_path": str(
                destination / "test_open_completion_receipt.json"
            ),
            "failure_receipt_path": str(destination / "test_open_failure_receipt.json"),
            "receipt_location_frozen_in_deployment": True,
        },
        "seal_audit": {
            "phase": "freeze-development",
            "test_manifest_opened": False,
            "test_paths_opened": False,
            "test_labels_available_to_fit_or_selection": False,
        },
    }
    deployment_path = destination / "route_policy_deployment.json"
    deployment_path.write_text(
        json.dumps(deployment, indent=2, sort_keys=True), encoding="utf-8"
    )
    deployment_hash = sha256_file(deployment_path)
    sidecar_path = destination / "route_policy_deployment.sha256"
    sidecar_path.write_text(
        f"{deployment_hash}  {deployment_path.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": deployment["status"],
                "deployment_plan": str(deployment_path),
                "deployment_sha256": deployment_hash,
                "deployment_sha256_file": str(sidecar_path),
            },
            indent=2,
        )
    )
    return 0


def _restore_bandit(deployment: Mapping[str, Any]) -> LinearContextualBandit:
    model = deployment["full_information_linear_ridge"]["model"]
    return LinearContextualBandit(
        ridge_lambda=float(model["ridge_lambda"]),
        threshold=float(model["accept_threshold"]),
        feature_mean=np.asarray(model["feature_mean"], dtype=np.float64),
        feature_std=np.asarray(model["feature_std"], dtype=np.float64),
        coefficients=np.asarray(model["coefficients"], dtype=np.float64),
        deploy_keep_all=str(model["deployment_decision"]) == "KEEP_ALL",
    )


def _score_test(args: argparse.Namespace) -> int:
    """Phase B: verify frozen deployment, then open/score test exactly once."""

    if not args.deployment_plan or not args.deployment_sha256:
        raise ValueError(
            "score-test requires --deployment-plan and --deployment-sha256"
        )
    deployment_path = Path(args.deployment_plan).resolve()
    expected_hash = str(args.deployment_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("--deployment-sha256 must be exactly 64 lowercase hex digits")
    actual_hash = sha256_file(deployment_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"deployment SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    if (
        deployment.get("schema_version") != 1
        or deployment.get("artifact_type") != "frozen_route_policy_deployment"
        or deployment.get("status") != "FROZEN_DEVELOPMENT"
    ):
        raise ValueError(
            "deployment plan is not a frozen schema-v1 development artifact"
        )
    checkpoint_path = Path(args.edl_checkpoint).resolve()
    if sha256_file(checkpoint_path) != deployment["edl"]["checkpoint_sha256"]:
        raise ValueError(
            "score-test EDL checkpoint hash differs from frozen deployment"
        )
    model, normalizer, checkpoint = load_checkpoint_bundle(
        checkpoint_path, device="cpu"
    )
    if (
        checkpoint.get("manifest_sha256")
        != deployment["development_manifest"]["sha256"]
    ):
        raise ValueError("EDL checkpoint is not bound to frozen development manifest")
    manifest = Path(args.manifest).resolve()
    test_manifest_hash = sha256_file(manifest)
    destination = Path(args.out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    control = deployment.get("test_open_control")
    if not isinstance(control, Mapping) or int(control.get("pass_limit", 0)) != 1:
        raise ValueError("deployment plan lacks a frozen one-shot test-open control")
    attempt_path = Path(str(control["attempt_receipt_path"])).resolve()
    completion_path = Path(str(control["completion_receipt_path"])).resolve()
    failure_path = Path(str(control["failure_receipt_path"])).resolve()
    if any(
        path.parent != deployment_path.parent
        for path in (attempt_path, completion_path, failure_path)
    ):
        raise ValueError(
            "frozen test-open receipts must be adjacent to deployment plan"
        )
    attempt = {
        "schema_version": 1,
        "artifact_type": "test_open_attempt_receipt",
        "status": "ATTEMPT_CONSUMED",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "deployment_path": str(deployment_path),
        "deployment_sha256": actual_hash,
        "test_manifest_path": str(manifest),
        "test_manifest_sha256": test_manifest_hash,
        "pass_limit": 1,
    }
    _write_exclusive_json(attempt_path, attempt)
    attempt_hash = sha256_file(attempt_path)
    try:
        report, paths = _execute_test_open(
            args=args,
            deployment_path=deployment_path,
            deployment_hash=actual_hash,
            deployment=deployment,
            manifest=manifest,
            test_manifest_hash=test_manifest_hash,
            checkpoint_path=checkpoint_path,
            model=model,
            normalizer=normalizer,
            checkpoint=checkpoint,
            attempt_path=attempt_path,
            attempt_hash=attempt_hash,
        )
    except BaseException as exc:
        _write_exclusive_json(
            failure_path,
            {
                "schema_version": 1,
                "artifact_type": "test_open_failure_receipt",
                "status": "FAILED_ATTEMPT_CONSUMED",
                "finished_at": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "attempt_receipt_sha256": attempt_hash,
                "deployment_sha256": actual_hash,
                "test_manifest_sha256": test_manifest_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    _write_exclusive_json(
        completion_path,
        {
            "schema_version": 1,
            "artifact_type": "test_open_completion_receipt",
            "status": "COMPLETED",
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "attempt_receipt_sha256": attempt_hash,
            "deployment_sha256": actual_hash,
            "test_manifest_sha256": test_manifest_hash,
            "report_sha256": sha256_file(paths["report"]),
        },
    )
    print(json.dumps({"status": report["status"], **paths}, indent=2))
    return 0


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"one-shot receipt already exists; audited test rerun is blocked: {path}"
        ) from exc


def _execute_test_open(
    *,
    args: argparse.Namespace,
    deployment_path: Path,
    deployment_hash: str,
    deployment: Mapping[str, Any],
    manifest: Path,
    test_manifest_hash: str,
    checkpoint_path: Path,
    model: Any,
    normalizer: Any,
    checkpoint: Mapping[str, Any],
    attempt_path: Path,
    attempt_hash: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload_preflight = _manifest_metadata(manifest)
    _require_exact_manifest_splits(payload_preflight, {"test"}, phase="score-test")
    if (
        payload_preflight.get("development_manifest_sha256")
        != deployment["development_manifest"]["sha256"]
    ):
        raise ValueError(
            "test manifest is not bound to the frozen development manifest"
        )
    if payload_preflight.get("deployment_sha256") != deployment_hash:
        raise ValueError("test manifest is not bound to this frozen deployment SHA-256")
    declared_routes, _source = _normalize_candidate_routes(
        payload_preflight.get("candidate_routes")
    )
    if list(declared_routes) != deployment["development_manifest"]["candidate_routes"]:
        raise ValueError("test candidate_routes differ from frozen development menu")

    utility = deployment["utility_definition"]
    candidates, payload = load_route_manifest(
        manifest,
        nsd_tolerance_mm=float(utility["nsd_tolerance_mm"]),
        nsd_weight=float(utility["nsd_weight"]),
        interaction_cost=float(utility["interaction_cost"]),
        accept_margin=float(utility["accept_margin"]),
        exact_splits=("test",),
    )
    route_contract = validate_route_contract(
        candidates,
        candidate_routes=declared_routes,
        required_splits=("test",),
        claim_external_validation=args.claim_external_validation,
        minimum_test_patients=args.minimum_test_patients,
    )
    overlap = sorted(
        set(deployment["development_patient_ids"])
        & {candidate.patient_id for candidate in candidates}
    )
    if overlap:
        raise ValueError(f"test patient leakage against frozen development: {overlap}")

    edl_scores = score_edl_candidates(
        candidates,
        model=model,
        normalizer=normalizer,
        checkpoint=checkpoint,
        thresholds=deployment["edl"]["deployed_thresholds"],
        force_keep_all=bool(deployment["edl"]["deploy_keep_all"]),
    )
    bandit = _restore_bandit(deployment)
    test_evaluation = evaluate_test_policies(
        candidates,
        edl_scores=edl_scores,
        bandit=bandit,
        candidate_routes=declared_routes,
        edl_deploy_keep_all=bool(deployment["edl"]["deploy_keep_all"]),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if route_contract["prior_exposed_test_cases"]:
        report_status = "EXPLORATORY_INTERNAL_PRIOR_EXPOSED"
        claim_boundary = (
            "Internal prior-exposed frozen test only; no efficacy, external-validation, "
            "learned-selection, or clinical-generalization claim."
        )
    elif not route_contract["efficacy_claim_eligible"]:
        report_status = "EXPLORATORY_INSUFFICIENT_TEST_SAMPLE"
        claim_boundary = (
            "Exposure-independent frozen test is below the configured efficacy sample "
            "requirement; no clinical-generalization claim."
        )
    else:
        report_status = "COMPLETED_FROZEN_TEST"
        claim_boundary = (
            "External frozen validation under the configured exposure/sample contract; "
            "clinical validity remains out of scope."
            if args.claim_external_validation
            else "Frozen patient-disjoint test meeting the configured exposure/sample "
            "contract; clinical validity remains out of scope."
        )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": report_status,
        "claim_boundary": claim_boundary,
        "external_validation_eligible": route_contract["external_validation_eligible"],
        "efficacy_claim_eligible": route_contract["efficacy_claim_eligible"],
        "efficacy_ineligibility_reasons": route_contract[
            "efficacy_ineligibility_reasons"
        ],
        "minimum_test_patients": route_contract["minimum_test_patients"],
        "deployment_artifact": {
            "path": str(deployment_path),
            "sha256": deployment_hash,
            "frozen_before_test_manifest_open": True,
        },
        "test_open_receipt": {
            "path": str(attempt_path),
            "sha256": attempt_hash,
            "test_manifest_sha256": test_manifest_hash,
            "pass_limit": 1,
        },
        "manifest": {
            "path": str(manifest),
            "sha256": test_manifest_hash,
            "source_status": payload.get("status", "unspecified"),
            "schema_version": payload.get("schema_version"),
            "candidate_routes": list(declared_routes),
            "development_manifest_sha256": payload.get("development_manifest_sha256"),
        },
        "route_contract": route_contract,
        "utility_definition": utility,
        "edl": {
            **deployment["edl"],
            "checkpoint": str(checkpoint_path),
            "selection_rule": (
                "highest predicted utility among ACCEPT candidates; otherwise KEEP ResEnc"
            ),
        },
        "linear_contextual_bandit": deployment["full_information_linear_ridge"][
            "fit_report"
        ],
        "policy_safety_contract": deployment["policy_safety_contract"],
        "no_test_tuning_audit": {
            "freeze_phase": (
                "development-only manifest opened; deployment JSON written and SHA-256 "
                "frozen before test manifest API access"
            ),
            "test_phase": (
                "verified deployment SHA-256, then one test loader/evaluation pass with "
                "no fit, calibration, threshold, or model selection"
            ),
            "test_label_evaluation_passes": 1,
            "test_used_for_model_or_threshold_selection": False,
        },
        "bootstrap": {
            "unit": "patient (cluster bootstrap over patient-level study means)",
            "samples": int(args.bootstrap_samples),
            "seed": int(args.seed),
            "interval": "percentile 95%",
        },
        "test_evaluation": test_evaluation,
    }
    paths = write_evaluation_outputs(report, output_dir=args.out_dir)
    return report, paths


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_binary(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    array = np.asarray(image.dataobj)
    if array.ndim != 3:
        raise ValueError(f"binary mask must be 3D: {path}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"binary mask contains non-finite values: {path}")
    return image, array > 0.5


def _require_same_grid(
    case_id: str,
    role: str,
    image: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
) -> None:
    if image.shape != reference.shape or not np.allclose(
        image.affine, reference.affine, atol=1e-4
    ):
        raise ValueError(f"{role} is not on the ground-truth grid for {case_id}")


def _group_by_case(
    candidates: Sequence[RouteCandidate],
) -> dict[str, list[RouteCandidate]]:
    groups: dict[str, list[RouteCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.case_id].append(candidate)
    return dict(groups)


def _route_lookup(group: Sequence[RouteCandidate]) -> dict[str, RouteCandidate]:
    return {candidate.route_id: candidate for candidate in group}


def _design_matrix(
    candidates: Sequence[RouteCandidate],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    features = np.stack([candidate.features for candidate in candidates]).astype(
        np.float64
    )
    normalized = (features - mean) / std
    routes = np.zeros((len(candidates), len(ROUTE_IDS)), dtype=np.float64)
    route_index = {route_id: index for index, route_id in enumerate(ROUTE_IDS)}
    for row, candidate in enumerate(candidates):
        routes[row, route_index[candidate.route_id]] = 1.0
    intercept = np.ones((len(candidates), 1), dtype=np.float64)
    return np.concatenate([normalized, routes, intercept], axis=1)


def _calibrate_bandit_threshold(
    candidates: Sequence[RouteCandidate],
    scores: np.ndarray,
    *,
    max_harmful_study_rate: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    finite = np.asarray(scores, dtype=np.float64)
    if not bool(np.isfinite(finite).all()):
        raise ValueError("bandit predictions are non-finite")
    groups = _ordered_case_groups(candidates)
    patient_ids, bootstrap_indices = _patient_bootstrap_plan(
        groups, samples=bootstrap_samples, seed=seed
    )
    span = max(1.0, float(np.max(np.abs(finite)))) * 1e-9
    thresholds = [
        float(finite.min() - span),
        *sorted({float(value) for value in finite}),
        float(finite.max() + span),
    ]
    ranked: list[tuple[tuple[float, ...], float, dict[str, Any]]] = []
    for threshold in thresholds:
        objective = _score_threshold_policy(
            candidates,
            finite,
            threshold,
            groups=groups,
            patient_ids=patient_ids,
            bootstrap_indices=bootstrap_indices,
            max_harmful_study_rate=max_harmful_study_rate,
        )
        key = (
            float(objective["patient_mean_realized_utility"]),
            float(
                objective["patient_cluster_bootstrap_95_ci_mean_realized_utility"][
                    "lower"
                ]
            ),
            float(objective["mean_realized_utility"]),
            -float(objective["harmful_action_rate_all_studies"]),
            float(objective["coverage"]),
            float(threshold),
        )
        ranked.append((key, float(threshold), objective))
    ranked.sort(key=lambda item: item[0], reverse=True)
    eligible = [
        item for item in ranked if bool(item[2]["safety_constraints_satisfied"])
    ]
    if eligible:
        _key, selected_threshold, selected_objective = eligible[0]
        safety_deployed = True
        deployment_decision = "SELECT_ROUTE_OR_KEEP"
        fallback_reason = None
    else:
        selected_threshold = float(finite.max() + span)
        selected_objective = _score_threshold_policy(
            candidates,
            finite,
            selected_threshold,
            groups=groups,
            patient_ids=patient_ids,
            bootstrap_indices=bootstrap_indices,
            max_harmful_study_rate=max_harmful_study_rate,
        )
        safety_deployed = False
        deployment_decision = "KEEP_ALL"
        fallback_reason = (
            "no policy-validation threshold satisfied both the harmful-study "
            "rate cap and strictly positive patient-cluster bootstrap lower bound"
        )
    return selected_threshold, {
        **selected_objective,
        "candidate_grid_size": len(thresholds),
        "eligible_grid_points": len(eligible),
        "independent_patient_count": len(patient_ids),
        "study_count": len(groups),
        "safety_deployed": safety_deployed,
        "deployment_decision": deployment_decision,
        "fallback_reason": fallback_reason,
        "safety_gate": _safety_gate_report(
            max_harmful_study_rate=max_harmful_study_rate,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }


def _score_threshold_policy(
    candidates: Sequence[RouteCandidate],
    scores: Sequence[float],
    threshold: float,
    *,
    groups: Sequence[Sequence[RouteCandidate]],
    patient_ids: Sequence[str],
    bootstrap_indices: np.ndarray,
    max_harmful_study_rate: float,
) -> dict[str, Any]:
    score_map = {
        candidate.uid: float(score)
        for candidate, score in zip(candidates, scores, strict=True)
    }
    choices = [
        _positive_or_keep(group, score_map, threshold=threshold) for group in groups
    ]
    return _deployment_objective(
        groups,
        choices,
        patient_ids=patient_ids,
        bootstrap_indices=bootstrap_indices,
        max_harmful_study_rate=max_harmful_study_rate,
    )


def _ordered_case_groups(
    candidates: Sequence[RouteCandidate],
) -> list[list[RouteCandidate]]:
    grouped = _group_by_case(candidates)
    return [grouped[case_id] for case_id in sorted(grouped)]


def _validate_safety_parameters(
    max_harmful_study_rate: float, bootstrap_samples: int
) -> None:
    if not math.isfinite(max_harmful_study_rate) or not (
        0.0 <= max_harmful_study_rate <= 1.0
    ):
        raise ValueError("max_harmful_study_rate must be finite and in [0, 1]")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")


def _patient_bootstrap_plan(
    groups: Sequence[Sequence[RouteCandidate]],
    *,
    samples: int,
    seed: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    if not groups:
        raise ValueError("patient-cluster bootstrap requires at least one study")
    patient_ids = tuple(sorted({group[0].patient_id for group in groups}))
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(patient_ids), size=(samples, len(patient_ids)), dtype=np.int64
    )
    return patient_ids, indices


def _deployment_objective(
    groups: Sequence[Sequence[RouteCandidate]],
    choices: Sequence[RouteCandidate | None],
    *,
    patient_ids: Sequence[str],
    bootstrap_indices: np.ndarray,
    max_harmful_study_rate: float,
) -> dict[str, Any]:
    if len(groups) != len(choices):
        raise ValueError("one grouped deployment choice is required per study")
    objective = _choice_objective(choices)
    utilities_by_patient: dict[str, list[float]] = defaultdict(list)
    for group, choice in zip(groups, choices, strict=True):
        if not group:
            raise ValueError("candidate study menu cannot be empty")
        patient = group[0].patient_id
        if any(candidate.patient_id != patient for candidate in group):
            raise ValueError("candidate study menu mixes patient IDs")
        utilities_by_patient[patient].append(
            0.0 if choice is None else float(choice.utility)
        )
    ordered_patient_ids = tuple(patient_ids)
    if set(ordered_patient_ids) != set(utilities_by_patient):
        raise ValueError("patient bootstrap plan does not match grouped study choices")
    patient_utilities = np.asarray(
        [np.mean(utilities_by_patient[patient]) for patient in ordered_patient_ids],
        dtype=np.float64,
    )
    if bootstrap_indices.ndim != 2 or bootstrap_indices.shape[1] != len(
        patient_utilities
    ):
        raise ValueError("patient bootstrap index matrix has the wrong shape")
    bootstrap_means = patient_utilities[bootstrap_indices].mean(axis=1)
    interval = {
        "lower": float(np.quantile(bootstrap_means, 0.025)),
        "upper": float(np.quantile(bootstrap_means, 0.975)),
    }
    constraints_satisfied = bool(
        objective["harmful_action_rate_all_studies"]
        <= max_harmful_study_rate + HARM_TOLERANCE
        and interval["lower"] > 0.0
    )
    return {
        **objective,
        "patient_mean_realized_utility": float(patient_utilities.mean()),
        "patient_cluster_bootstrap_95_ci_mean_realized_utility": interval,
        "independent_patient_count": len(patient_utilities),
        "safety_constraints_satisfied": constraints_satisfied,
    }


def _safety_gate_report(
    *,
    max_harmful_study_rate: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "selection_split": "policy_validation",
        "patient_unit": "unweighted mean realized utility across each patient's studies",
        "harmful_study_definition": f"selected proposal delta_dice < {-HARM_TOLERANCE}",
        "max_harmful_study_rate": float(max_harmful_study_rate),
        "bootstrap": {
            "method": "patient-cluster percentile bootstrap",
            "samples": int(bootstrap_samples),
            "seed": int(seed),
            "interval": "95% percentile [0.025, 0.975]",
        },
        "positive_utility_criterion": (
            "patient-cluster bootstrap 95% lower bound is strictly > 0"
        ),
        "fallback": "explicit KEEP_ALL when no candidate satisfies both constraints",
    }


def _choice_objective(choices: Sequence[RouteCandidate | None]) -> dict[str, float]:
    if not choices:
        raise ValueError("grouped-menu objective requires at least one study")
    realized = np.asarray(
        [0.0 if choice is None else choice.utility for choice in choices], dtype=float
    )
    covered = np.asarray([choice is not None for choice in choices], dtype=bool)
    harmful = np.asarray(
        [
            False if choice is None else choice.delta_dice < -HARM_TOLERANCE
            for choice in choices
        ],
        dtype=bool,
    )
    return {
        "mean_realized_utility": float(realized.mean()),
        "coverage": float(covered.mean()),
        "harmful_action_rate_all_studies": float(harmful.mean()),
        "harmful_action_rate_when_covered": float(
            harmful.sum() / max(int(covered.sum()), 1)
        ),
    }


def _positive_or_keep(
    candidates: Sequence[RouteCandidate],
    score_map: Mapping[str, float],
    *,
    threshold: float = 0.0,
) -> RouteCandidate | None:
    eligible = [
        candidate
        for candidate in candidates
        if float(score_map[candidate.uid]) >= threshold
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: (float(score_map[candidate.uid]), candidate.route_id),
    )


def _oracle_or_keep(candidates: Sequence[RouteCandidate]) -> RouteCandidate | None:
    improving = [candidate for candidate in candidates if candidate.utility > 0.0]
    if not improving:
        return None
    return max(
        improving,
        key=lambda candidate: (
            candidate.utility,
            candidate.delta_dice,
            candidate.route_id,
        ),
    )


def _choice_row(
    policy: str,
    exemplar: RouteCandidate,
    choice: RouteCandidate | None,
    *,
    confidence: float | None,
    p_accept: float | None,
    vacuity: float | None,
    predicted_utility: float | None,
) -> dict[str, Any]:
    final_dice = exemplar.baseline_dice if choice is None else choice.candidate_dice
    final_nsd = exemplar.baseline_nsd if choice is None else choice.candidate_nsd
    return {
        "policy": policy,
        "case_id": exemplar.case_id,
        "patient_id": exemplar.patient_id,
        "split": exemplar.split,
        "selected_route": "KEEP" if choice is None else choice.route_id,
        "selected_action": "KEEP" if choice is None else choice.action,
        "selected_round": None if choice is None else choice.round_index,
        "baseline_dice": float(exemplar.baseline_dice),
        "final_dice": float(final_dice),
        "delta_dice": float(final_dice - exemplar.baseline_dice),
        "baseline_nsd_2mm": float(exemplar.baseline_nsd),
        "final_nsd_2mm": float(final_nsd),
        "delta_nsd_2mm": float(final_nsd - exemplar.baseline_nsd),
        "realized_utility": 0.0 if choice is None else float(choice.utility),
        "covered": choice is not None,
        "harmful": bool(choice is not None and choice.delta_dice < -HARM_TOLERANCE),
        "confidence": confidence,
        "p_accept": p_accept,
        "vacuity": vacuity,
        "predicted_utility": predicted_utility,
    }


def _patient_rows(
    policy: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["patient_id"])].append(row)
    output: list[dict[str, Any]] = []
    for patient_id in sorted(grouped):
        items = grouped[patient_id]
        output.append(
            {
                "policy": policy,
                "patient_id": patient_id,
                "study_count": len(items),
                "mean_baseline_dice": _mean(items, "baseline_dice"),
                "mean_final_dice": _mean(items, "final_dice"),
                "mean_delta_dice": _mean(items, "delta_dice"),
                "mean_baseline_nsd_2mm": _mean(items, "baseline_nsd_2mm"),
                "mean_final_nsd_2mm": _mean(items, "final_nsd_2mm"),
                "mean_delta_nsd_2mm": _mean(items, "delta_nsd_2mm"),
                "mean_realized_utility": _mean(items, "realized_utility"),
                "coverage": float(np.mean([bool(item["covered"]) for item in items])),
                "harmful_action_rate_all_studies": float(
                    np.mean([bool(item["harmful"]) for item in items])
                ),
            }
        )
    return output


def _policy_summary(
    rows: Sequence[Mapping[str, Any]],
    patient_rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    covered = int(sum(bool(row["covered"]) for row in rows))
    harmful = int(sum(bool(row["harmful"]) for row in rows))
    study_delta_dice = [float(row["delta_dice"]) for row in rows]
    study_delta_nsd = [float(row["delta_nsd_2mm"]) for row in rows]
    patient_delta_dice = [float(row["mean_delta_dice"]) for row in patient_rows]
    patient_delta_nsd = [float(row["mean_delta_nsd_2mm"]) for row in patient_rows]
    return {
        "study_estimand": {
            "n": len(rows),
            "mean_baseline_dice": _mean(rows, "baseline_dice"),
            "mean_final_dice": _mean(rows, "final_dice"),
            "mean_delta_dice": float(np.mean(study_delta_dice)),
            "mean_baseline_nsd_2mm": _mean(rows, "baseline_nsd_2mm"),
            "mean_final_nsd_2mm": _mean(rows, "final_nsd_2mm"),
            "mean_delta_nsd_2mm": float(np.mean(study_delta_nsd)),
            "mean_realized_utility": _mean(rows, "realized_utility"),
            "dice_win_tie_loss_vs_keep": _win_tie_loss(study_delta_dice),
            "nsd_win_tie_loss_vs_keep": _win_tie_loss(study_delta_nsd),
        },
        "patient_estimand": {
            "n": len(patient_rows),
            "mean_baseline_dice": _mean(patient_rows, "mean_baseline_dice"),
            "mean_final_dice": _mean(patient_rows, "mean_final_dice"),
            "mean_delta_dice": float(np.mean(patient_delta_dice)),
            "mean_baseline_nsd_2mm": _mean(patient_rows, "mean_baseline_nsd_2mm"),
            "mean_final_nsd_2mm": _mean(patient_rows, "mean_final_nsd_2mm"),
            "mean_delta_nsd_2mm": float(np.mean(patient_delta_nsd)),
            "dice_win_tie_loss_vs_keep": _win_tie_loss(patient_delta_dice),
            "nsd_win_tie_loss_vs_keep": _win_tie_loss(patient_delta_nsd),
            "paired_bootstrap_95_ci_delta_dice": _bootstrap_ci(
                patient_delta_dice, bootstrap_samples, seed
            ),
            "paired_bootstrap_95_ci_delta_nsd_2mm": _bootstrap_ci(
                patient_delta_nsd, bootstrap_samples, seed + 1
            ),
        },
        "coverage": float(covered / len(rows)),
        "harmful_actions": harmful,
        "harmful_action_rate_all_studies": float(harmful / len(rows)),
        "harmful_action_rate_when_covered": float(harmful / max(covered, 1)),
    }


def _risk_coverage(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int | None]]:
    selected = [
        row for row in rows if bool(row["covered"]) and row["confidence"] is not None
    ]
    selected.sort(
        key=lambda row: (float(row["confidence"]), str(row["case_id"])), reverse=True
    )
    total = len(rows)
    points: list[dict[str, float | int | None]] = [
        {
            "selected_studies": 0,
            "coverage": 0.0,
            "harmful_action_risk": None,
            "mean_realized_utility_all_studies": 0.0,
            "minimum_confidence": None,
        }
    ]
    cumulative_harm = 0
    cumulative_utility = 0.0
    for index, row in enumerate(selected, start=1):
        cumulative_harm += int(bool(row["harmful"]))
        cumulative_utility += float(row["realized_utility"])
        points.append(
            {
                "selected_studies": index,
                "coverage": float(index / total),
                "harmful_action_risk": float(cumulative_harm / index),
                "mean_realized_utility_all_studies": float(cumulative_utility / total),
                "minimum_confidence": float(row["confidence"]),
            }
        )
    return points


def _win_tie_loss(values: Iterable[float]) -> dict[str, int]:
    values = [float(value) for value in values]
    return {
        "wins": sum(value > HARM_TOLERANCE for value in values),
        "ties": sum(abs(value) <= HARM_TOLERANCE for value in values),
        "losses": sum(value < -HARM_TOLERANCE for value in values),
    }


def _bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("bootstrap requires at least one patient")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
