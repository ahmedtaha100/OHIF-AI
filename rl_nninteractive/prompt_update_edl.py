"""Evidential accept/utility/STOP head for prompt-driven mask updates.

The segmentation model stays frozen.  This module scores a proposed
``current_mask -> proposed_mask`` update using inference-time information only:
PET, CT, optional TotalSegmentator labels, both masks, and prompt metadata.
Ground truth is accepted only by the offline label builder.

Two execution contracts are intentionally separate:

* strict: patient-disjoint train/calibration/policy-validation/test splits;
* mechanics smoke: patient-disjoint train/calibration only, with no efficacy or
  external-validation claim.

The latter exists because the exploratory AutoPET ``cohort4`` has only two
patients and therefore cannot populate four patient-disjoint splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception as exc:  # pragma: no cover - exercised only without optional ML deps
    raise ImportError("rl_nninteractive.prompt_update_edl requires PyTorch") from exc


SPLITS = ("train", "calibration", "policy_validation", "test")
CHECKPOINT_SCHEMA_VERSION = 1

GLOBAL_FEATURES = (
    "current_volume_fraction",
    "proposed_volume_fraction",
    "signed_delta_volume_fraction",
    "changed_volume_fraction",
    "union_volume_fraction",
    "xor_over_union",
    "current_component_count_log1p",
    "proposed_component_count_log1p",
    "current_pet_robust_mean",
    "proposed_pet_robust_mean",
    "current_ct_robust_mean",
    "proposed_ct_robust_mean",
    "current_totseg_nonzero_fraction",
    "proposed_totseg_nonzero_fraction",
)

REGION_FEATURES = (
    "voxel_fraction",
    "component_count_log1p",
    "largest_component_fraction",
    "mean_component_fraction",
    "centroid_axis0",
    "centroid_axis1",
    "centroid_axis2",
    "bbox_extent_axis0",
    "bbox_extent_axis1",
    "bbox_extent_axis2",
    "pet_robust_mean",
    "pet_robust_std",
    "pet_robust_p90",
    "pet_robust_max",
    "ct_robust_mean",
    "ct_robust_std",
    "ct_robust_p10",
    "ct_robust_p90",
    "totseg_nonzero_fraction",
    "totseg_unique_labels_log1p",
    "totseg_dominant_label_fraction",
    "prompt_distance_normalized",
)

PROMPT_FEATURES = (
    "prompt_round_log1p",
    "prompt_foreground_count_log1p",
    "prompt_background_count_log1p",
    "prompt_new_foreground_count_log1p",
    "prompt_new_background_count_log1p",
    "prompt_axis0",
    "prompt_axis1",
    "prompt_axis2",
    "prompt_positive_fraction",
    "change_has_added",
    "change_has_removed",
    "change_is_mixed",
)

FEATURE_ORDER = tuple(
    list(GLOBAL_FEATURES)
    + [f"added_{name}" for name in REGION_FEATURES]
    + [f"removed_{name}" for name in REGION_FEATURES]
    + list(PROMPT_FEATURES)
)


@dataclass(frozen=True)
class PromptMetadata:
    round_index: int = 0
    foreground_count: int = 0
    background_count: int = 0
    new_foreground_count: int = 0
    new_background_count: int = 0
    coordinates: tuple[tuple[float, float, float], ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PromptMetadata":
        if not value:
            return cls()
        coordinates: list[tuple[float, float, float]] = []
        raw_coordinates = value.get("coordinates", ())
        for coord in raw_coordinates:
            if len(coord) == 3:
                coordinates.append(tuple(float(x) for x in coord))
        for key in ("foreground_xyz", "background_xyz", "coord"):
            coord = value.get(key)
            if coord is not None and len(coord) == 3:
                coordinates.append(tuple(float(x) for x in coord))
        fg = int(
            value.get("foreground_count", value.get("cumulative_foreground_clicks", 0))
        )
        bg = int(
            value.get("background_count", value.get("cumulative_background_clicks", 0))
        )
        new_fg = int(
            value.get(
                "new_foreground_count", int(value.get("foreground_xyz") is not None)
            )
        )
        new_bg = int(
            value.get(
                "new_background_count", int(value.get("background_xyz") is not None)
            )
        )
        return cls(
            round_index=int(value.get("round_index", value.get("round", 0))),
            foreground_count=fg,
            background_count=bg,
            new_foreground_count=new_fg,
            new_background_count=new_bg,
            coordinates=tuple(coordinates),
        )


@dataclass(frozen=True)
class UpdateExample:
    case_id: str
    patient_id: str
    transition_id: str
    split: str
    prior_exposure: bool
    features: np.ndarray
    accept_label: int
    utility: float
    delta_dice: float
    delta_nsd: float | None = None


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureNormalizer":
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_ORDER):
            raise ValueError(f"features must have shape (N, {len(FEATURE_ORDER)})")
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = x.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)


class EvidentialUtilityHead(nn.Module):
    """Small MLP producing binary Dirichlet evidence and signed utility."""

    def __init__(self, in_dim: int = len(FEATURE_ORDER), hidden: int = 48) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
        )
        self.evidence_head = nn.Linear(self.hidden, 2)
        self.utility_head = nn.Linear(self.hidden, 1)

    def forward(
        self, features: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        hidden = self.backbone(features)
        evidence = F.softplus(self.evidence_head(hidden))
        alpha = evidence + 1.0
        utility = self.utility_head(hidden).squeeze(-1)
        return alpha, utility


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _as_bool(mask: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(mask).astype(bool, copy=False)
    if result.shape != shape:
        raise ValueError(
            f"{name} shape {result.shape} does not match image shape {shape}"
        )
    return result


def _finite_float(volume: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(volume, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(
            f"{name} shape {result.shape} does not match PET shape {shape}"
        )
    if not np.isfinite(result).all():
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def _robust_location_scale(volume: np.ndarray) -> tuple[float, float]:
    flat = volume.reshape(-1)
    stride = max(1, flat.size // 200_000)
    sample = flat[::stride]
    q25, median, q75 = np.percentile(sample, [25.0, 50.0, 75.0])
    scale = max(float((q75 - q25) / 1.349), float(np.std(sample)), 1e-6)
    return float(median), scale


def _component_summary(mask: np.ndarray) -> tuple[int, float, float]:
    count = int(mask.sum())
    if count == 0:
        return 0, 0.0, 0.0
    labels, n_components = ndimage.label(
        mask, structure=ndimage.generate_binary_structure(mask.ndim, 1)
    )
    sizes = np.bincount(labels.reshape(-1))[1:]
    largest = float(sizes.max() / count) if sizes.size else 0.0
    mean_fraction = float(sizes.mean() / mask.size) if sizes.size else 0.0
    return int(n_components), largest, mean_fraction


def _selected_robust_stats(
    volume: np.ndarray,
    mask: np.ndarray,
    location: float,
    scale: float,
    percentiles: tuple[float, float],
) -> tuple[float, float, float, float]:
    if not bool(mask.any()):
        return 0.0, 0.0, 0.0, 0.0
    values = (volume[mask].astype(np.float64) - location) / scale
    low, high = np.percentile(values, percentiles)
    return float(values.mean()), float(values.std()), float(low), float(high)


def _prompt_center(
    metadata: PromptMetadata, shape: tuple[int, ...]
) -> tuple[float, ...] | None:
    if not metadata.coordinates:
        return None
    center = np.mean(np.asarray(metadata.coordinates, dtype=np.float64), axis=0)
    if center.shape != (len(shape),):
        return None
    return tuple(
        float(np.clip(center[i], 0.0, max(shape[i] - 1, 0))) for i in range(len(shape))
    )


def _region_features(
    mask: np.ndarray,
    *,
    pet: np.ndarray,
    ct: np.ndarray,
    totseg: np.ndarray | None,
    pet_location_scale: tuple[float, float],
    ct_location_scale: tuple[float, float],
    prompt_center: tuple[float, ...] | None,
) -> dict[str, float]:
    count = int(mask.sum())
    n_components, largest_fraction, mean_component_fraction = _component_summary(mask)
    if count:
        centroid = tuple(float(x) for x in ndimage.center_of_mass(mask))
        bounds = ndimage.find_objects(mask.astype(np.uint8), max_label=1)[0]
        extents = tuple(
            float((sl.stop - sl.start) / max(mask.shape[i], 1))
            for i, sl in enumerate(bounds)
        )
    else:
        centroid = (0.0,) * mask.ndim
        extents = (0.0,) * mask.ndim
    centroid_normalized = tuple(
        float(centroid[i] / max(mask.shape[i] - 1, 1)) for i in range(mask.ndim)
    )
    pet_mean, pet_std, _pet_p10, pet_p90 = _selected_robust_stats(
        pet, mask, *pet_location_scale, (10.0, 90.0)
    )
    ct_mean, ct_std, ct_p10, ct_p90 = _selected_robust_stats(
        ct, mask, *ct_location_scale, (10.0, 90.0)
    )
    if count:
        pet_values = (
            pet[mask].astype(np.float64) - pet_location_scale[0]
        ) / pet_location_scale[1]
        pet_max = float(pet_values.max())
    else:
        pet_max = 0.0

    if totseg is not None and count:
        labels = np.asarray(totseg[mask], dtype=np.int64)
        nonzero = labels[labels > 0]
        nonzero_fraction = float(nonzero.size / labels.size)
        if nonzero.size:
            _, label_counts = np.unique(nonzero, return_counts=True)
            unique_labels = int(label_counts.size)
            dominant_fraction = float(label_counts.max() / nonzero.size)
        else:
            unique_labels = 0
            dominant_fraction = 0.0
    else:
        nonzero_fraction = 0.0
        unique_labels = 0
        dominant_fraction = 0.0

    if prompt_center is not None and count:
        scaled = [
            (centroid[i] - prompt_center[i]) / max(mask.shape[i] - 1, 1)
            for i in range(mask.ndim)
        ]
        prompt_distance = float(np.linalg.norm(scaled) / math.sqrt(mask.ndim))
    else:
        prompt_distance = 0.0

    return {
        "voxel_fraction": float(count / mask.size),
        "component_count_log1p": float(math.log1p(n_components)),
        "largest_component_fraction": largest_fraction,
        "mean_component_fraction": mean_component_fraction,
        "centroid_axis0": centroid_normalized[0],
        "centroid_axis1": centroid_normalized[1],
        "centroid_axis2": centroid_normalized[2],
        "bbox_extent_axis0": extents[0],
        "bbox_extent_axis1": extents[1],
        "bbox_extent_axis2": extents[2],
        "pet_robust_mean": pet_mean,
        "pet_robust_std": pet_std,
        "pet_robust_p90": pet_p90,
        "pet_robust_max": pet_max,
        "ct_robust_mean": ct_mean,
        "ct_robust_std": ct_std,
        "ct_robust_p10": ct_p10,
        "ct_robust_p90": ct_p90,
        "totseg_nonzero_fraction": nonzero_fraction,
        "totseg_unique_labels_log1p": float(math.log1p(unique_labels)),
        "totseg_dominant_label_fraction": dominant_fraction,
        "prompt_distance_normalized": prompt_distance,
    }


def extract_update_features(
    pet: Any,
    ct: Any,
    current_mask: Any,
    proposed_mask: Any,
    *,
    totseg: Any | None = None,
    prompt_metadata: Mapping[str, Any] | PromptMetadata | None = None,
) -> np.ndarray:
    """Return fixed-order, GT-free features for a proposed mask update."""

    pet_array = np.asarray(pet, dtype=np.float32)
    if pet_array.ndim != 3:
        raise ValueError("PET must be a 3D array")
    shape = pet_array.shape
    pet_array = _finite_float(pet_array, shape, "PET")
    ct_array = _finite_float(ct, shape, "CT")
    current = _as_bool(current_mask, shape, "current_mask")
    proposed = _as_bool(proposed_mask, shape, "proposed_mask")
    totseg_array = None if totseg is None else np.asarray(totseg)
    if totseg_array is not None and totseg_array.shape != shape:
        raise ValueError(
            f"TotSeg shape {totseg_array.shape} does not match PET shape {shape}"
        )
    metadata = (
        prompt_metadata
        if isinstance(prompt_metadata, PromptMetadata)
        else PromptMetadata.from_mapping(prompt_metadata)
    )

    added = proposed & ~current
    removed = current & ~proposed
    changed = added | removed
    union = current | proposed
    pet_location_scale = _robust_location_scale(pet_array)
    ct_location_scale = _robust_location_scale(ct_array)
    prompt_center = _prompt_center(metadata, shape)

    current_components, _, _ = _component_summary(current)
    proposed_components, _, _ = _component_summary(proposed)
    current_pet = _selected_robust_stats(
        pet_array, current, *pet_location_scale, (10.0, 90.0)
    )[0]
    proposed_pet = _selected_robust_stats(
        pet_array, proposed, *pet_location_scale, (10.0, 90.0)
    )[0]
    current_ct = _selected_robust_stats(
        ct_array, current, *ct_location_scale, (10.0, 90.0)
    )[0]
    proposed_ct = _selected_robust_stats(
        ct_array, proposed, *ct_location_scale, (10.0, 90.0)
    )[0]
    if totseg_array is None:
        current_totseg = proposed_totseg = 0.0
    else:
        current_totseg = (
            float(np.mean(totseg_array[current] > 0)) if bool(current.any()) else 0.0
        )
        proposed_totseg = (
            float(np.mean(totseg_array[proposed] > 0)) if bool(proposed.any()) else 0.0
        )

    values: dict[str, float] = {
        "current_volume_fraction": float(current.mean()),
        "proposed_volume_fraction": float(proposed.mean()),
        "signed_delta_volume_fraction": float(
            (int(proposed.sum()) - int(current.sum())) / current.size
        ),
        "changed_volume_fraction": float(changed.mean()),
        "union_volume_fraction": float(union.mean()),
        "xor_over_union": float(changed.sum() / max(int(union.sum()), 1)),
        "current_component_count_log1p": float(math.log1p(current_components)),
        "proposed_component_count_log1p": float(math.log1p(proposed_components)),
        "current_pet_robust_mean": current_pet,
        "proposed_pet_robust_mean": proposed_pet,
        "current_ct_robust_mean": current_ct,
        "proposed_ct_robust_mean": proposed_ct,
        "current_totseg_nonzero_fraction": current_totseg,
        "proposed_totseg_nonzero_fraction": proposed_totseg,
    }
    for prefix, mask in (("added", added), ("removed", removed)):
        region = _region_features(
            mask,
            pet=pet_array,
            ct=ct_array,
            totseg=totseg_array,
            pet_location_scale=pet_location_scale,
            ct_location_scale=ct_location_scale,
            prompt_center=prompt_center,
        )
        values.update({f"{prefix}_{name}": value for name, value in region.items()})

    if prompt_center is None:
        prompt_axes = (0.0, 0.0, 0.0)
    else:
        prompt_axes = tuple(prompt_center[i] / max(shape[i] - 1, 1) for i in range(3))
    prompt_total = metadata.new_foreground_count + metadata.new_background_count
    values.update(
        {
            "prompt_round_log1p": float(math.log1p(max(metadata.round_index, 0))),
            "prompt_foreground_count_log1p": float(
                math.log1p(max(metadata.foreground_count, 0))
            ),
            "prompt_background_count_log1p": float(
                math.log1p(max(metadata.background_count, 0))
            ),
            "prompt_new_foreground_count_log1p": float(
                math.log1p(max(metadata.new_foreground_count, 0))
            ),
            "prompt_new_background_count_log1p": float(
                math.log1p(max(metadata.new_background_count, 0))
            ),
            "prompt_axis0": float(prompt_axes[0]),
            "prompt_axis1": float(prompt_axes[1]),
            "prompt_axis2": float(prompt_axes[2]),
            "prompt_positive_fraction": float(
                metadata.new_foreground_count / max(prompt_total, 1)
            ),
            "change_has_added": float(bool(added.any())),
            "change_has_removed": float(bool(removed.any())),
            "change_is_mixed": float(bool(added.any()) and bool(removed.any())),
        }
    )
    result = np.asarray([values[name] for name in FEATURE_ORDER], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("feature extraction produced a non-finite value")
    return result


def dice_score(mask: Any, ground_truth: Any) -> float:
    a = np.asarray(mask).astype(bool)
    b = np.asarray(ground_truth).astype(bool)
    if a.shape != b.shape:
        raise ValueError("mask and ground truth shapes differ")
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denominator)


def make_update_example(
    *,
    case_id: str,
    patient_id: str,
    transition_id: str,
    split: str,
    prior_exposure: bool,
    pet: Any,
    ct: Any,
    current_mask: Any,
    proposed_mask: Any,
    ground_truth: Any,
    totseg: Any | None = None,
    prompt_metadata: Mapping[str, Any] | PromptMetadata | None = None,
    delta_nsd: float | None = None,
    nsd_weight: float = 0.0,
    interaction_cost: float = 0.0,
    accept_margin: float = 0.0,
) -> UpdateExample:
    """Build an offline label after GT-free feature extraction is complete."""

    features = extract_update_features(
        pet,
        ct,
        current_mask,
        proposed_mask,
        totseg=totseg,
        prompt_metadata=prompt_metadata,
    )
    delta_dice = dice_score(proposed_mask, ground_truth) - dice_score(
        current_mask, ground_truth
    )
    utility = float(delta_dice + nsd_weight * (delta_nsd or 0.0) - interaction_cost)
    return UpdateExample(
        case_id=str(case_id),
        patient_id=str(patient_id),
        transition_id=str(transition_id),
        split=str(split),
        prior_exposure=bool(prior_exposure),
        features=features,
        accept_label=int(utility > accept_margin),
        utility=utility,
        delta_dice=float(delta_dice),
        delta_nsd=None if delta_nsd is None else float(delta_nsd),
    )


def validate_split_contract(
    examples: Sequence[UpdateExample],
    *,
    require_all_splits: bool = True,
    claim_external_validation: bool = False,
    minimum_test_patients: int = 20,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("at least one update example is required")
    if int(minimum_test_patients) < 1:
        raise ValueError("minimum_test_patients must be at least 1")
    unknown = sorted({example.split for example in examples} - set(SPLITS))
    if unknown:
        raise ValueError(f"unknown split names: {unknown}; expected {list(SPLITS)}")
    patient_splits: dict[str, set[str]] = {}
    case_patients: dict[str, set[str]] = {}
    case_splits: dict[str, set[str]] = {}
    transition_keys: set[tuple[str, str]] = set()
    for example in examples:
        if not example.case_id:
            raise ValueError(
                f"missing case_id for {example.patient_id}/{example.transition_id}"
            )
        if not example.patient_id:
            raise ValueError(
                f"missing patient_id for {example.case_id}/{example.transition_id}"
            )
        if not example.transition_id:
            raise ValueError(f"missing transition_id for {example.case_id}")
        transition_key = (example.case_id, example.transition_id)
        if transition_key in transition_keys:
            raise ValueError(f"duplicate case/transition record: {transition_key}")
        transition_keys.add(transition_key)
        patient_splits.setdefault(example.patient_id, set()).add(example.split)
        case_patients.setdefault(example.case_id, set()).add(example.patient_id)
        case_splits.setdefault(example.case_id, set()).add(example.split)
    leakage = {
        patient: sorted(splits)
        for patient, splits in patient_splits.items()
        if len(splits) > 1
    }
    if leakage:
        raise ValueError(f"patient leakage across splits: {leakage}")
    case_identity_conflicts = {
        case_id: {
            "patients": sorted(case_patients[case_id]),
            "splits": sorted(case_splits[case_id]),
        }
        for case_id in case_patients
        if len(case_patients[case_id]) > 1 or len(case_splits[case_id]) > 1
    }
    if case_identity_conflicts:
        raise ValueError(
            f"case_id mapped across patients/splits: {case_identity_conflicts}"
        )
    split_counts = {
        split: sum(example.split == split for example in examples) for split in SPLITS
    }
    if require_all_splits:
        missing = [split for split, count in split_counts.items() if count == 0]
        if missing:
            raise ValueError(
                f"strict mode requires non-empty patient-disjoint splits: {missing}"
            )
    test_examples = [example for example in examples if example.split == "test"]
    prior_exposed_test = sorted(
        {example.case_id for example in test_examples if example.prior_exposure}
    )
    external_validation_eligible = bool(test_examples) and not prior_exposed_test
    test_patient_count = len({example.patient_id for example in test_examples})
    efficacy_ineligibility_reasons: list[str] = []
    if not test_examples:
        efficacy_ineligibility_reasons.append("test split is absent")
    if prior_exposed_test:
        efficacy_ineligibility_reasons.append(
            f"test contains prior-exposed cases: {prior_exposed_test}"
        )
    if test_patient_count < int(minimum_test_patients):
        efficacy_ineligibility_reasons.append(
            f"test patient count {test_patient_count} is below minimum {int(minimum_test_patients)}"
        )
    efficacy_claim_eligible = not efficacy_ineligibility_reasons
    if claim_external_validation and not external_validation_eligible:
        raise ValueError(
            "external validation rejected: test is absent or contains prior-exposed cases "
            f"({prior_exposed_test})"
        )
    return {
        "split_counts": split_counts,
        "patient_counts": {
            split: len(
                {example.patient_id for example in examples if example.split == split}
            )
            for split in SPLITS
        },
        "prior_exposed_test_cases": prior_exposed_test,
        "external_validation_eligible": external_validation_eligible,
        "minimum_test_patients": int(minimum_test_patients),
        "efficacy_claim_eligible": efficacy_claim_eligible,
        "efficacy_ineligibility_reasons": efficacy_ineligibility_reasons,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dirichlet_kl_to_uniform(alpha: "torch.Tensor") -> "torch.Tensor":
    k = alpha.shape[-1]
    sum_alpha = alpha.sum(dim=-1)
    log_beta_alpha = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(sum_alpha)
    log_beta_uniform = -torch.lgamma(
        torch.tensor(float(k), device=alpha.device, dtype=alpha.dtype)
    )
    digamma_term = (
        (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(sum_alpha.unsqueeze(-1)))
    ).sum(dim=-1)
    return log_beta_uniform - log_beta_alpha + digamma_term


def evidential_utility_loss(
    alpha: "torch.Tensor",
    utility_prediction: "torch.Tensor",
    labels: "torch.Tensor",
    utility_target: "torch.Tensor",
    *,
    anneal: float,
    utility_weight: float = 2.0,
) -> "torch.Tensor":
    one_hot = F.one_hot(labels, num_classes=2).to(alpha.dtype)
    strength = alpha.sum(dim=-1, keepdim=True)
    data_loss = (one_hot * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=-1)
    adjusted_alpha = one_hot + (1.0 - one_hot) * alpha
    kl = _dirichlet_kl_to_uniform(adjusted_alpha)
    utility_loss = F.smooth_l1_loss(
        utility_prediction, utility_target, reduction="none"
    )
    return (
        data_loss + float(anneal) * 0.05 * kl + utility_weight * utility_loss
    ).mean()


def _head_outputs(
    model: EvidentialUtilityHead,
    normalizer: FeatureNormalizer,
    examples: Sequence[UpdateExample],
    *,
    device: str,
) -> dict[str, np.ndarray]:
    if not examples:
        empty = np.asarray([], dtype=np.float64)
        return {"p_accept": empty, "vacuity": empty, "utility": empty}
    features = normalizer.transform(
        np.stack([example.features for example in examples])
    )
    with torch.no_grad():
        alpha, utility = model(torch.from_numpy(features).to(device))
        probability = alpha[:, 1] / alpha.sum(dim=-1)
        vacuity = 2.0 / alpha.sum(dim=-1)
    return {
        "p_accept": probability.cpu().numpy().astype(np.float64),
        "vacuity": vacuity.cpu().numpy().astype(np.float64),
        "utility": utility.cpu().numpy().astype(np.float64),
    }


def _temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p)) / float(temperature)
    return 1.0 / (1.0 + np.exp(-logits))


def calibrate_temperature(probability: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.size == 0 or np.unique(y).size < 2:
        return 1.0
    candidates = np.exp(np.linspace(math.log(0.25), math.log(4.0), 121))
    losses = []
    for temperature in candidates:
        calibrated = np.clip(
            _temperature_scale(p, float(temperature)), 1e-8, 1.0 - 1e-8
        )
        losses.append(
            float(
                -(y * np.log(calibrated) + (1.0 - y) * np.log(1.0 - calibrated)).mean()
            )
        )
    return float(candidates[int(np.argmin(losses))])


DEFAULT_THRESHOLDS = {
    "accept_probability": 0.5,
    "max_accept_vacuity": 0.6,
    "min_predicted_utility": 0.0,
    "stop_probability": 0.25,
    "max_stop_vacuity": 0.35,
    "stop_utility": 0.0,
}


def select_policy_thresholds(
    probability: np.ndarray,
    vacuity: np.ndarray,
    predicted_utility: np.ndarray,
    realized_utility: np.ndarray,
) -> dict[str, float]:
    if len(probability) == 0:
        return dict(DEFAULT_THRESHOLDS)
    best: tuple[float, float, float, float, float] | None = None
    for p_threshold in np.linspace(0.35, 0.8, 10):
        for max_vacuity in np.linspace(0.3, 0.9, 7):
            for min_utility in (-0.01, 0.0, 0.01, 0.02):
                accepted = (
                    (probability >= p_threshold)
                    & (vacuity <= max_vacuity)
                    & (predicted_utility >= min_utility)
                )
                score = float(np.where(accepted, realized_utility, 0.0).mean())
                coverage = float(accepted.mean())
                candidate = (
                    score,
                    -coverage,
                    float(p_threshold),
                    float(max_vacuity),
                    float(min_utility),
                )
                if best is None or candidate > best:
                    best = candidate
    assert best is not None
    return {
        "accept_probability": best[2],
        "max_accept_vacuity": best[3],
        "min_predicted_utility": best[4],
        "stop_probability": max(0.05, 1.0 - best[2]),
        "max_stop_vacuity": min(best[3], 0.5),
        "stop_utility": 0.0,
    }


def _evaluate_examples(
    examples: Sequence[UpdateExample],
    outputs: Mapping[str, np.ndarray],
    *,
    temperature: float,
    thresholds: Mapping[str, float],
    scope: str,
) -> dict[str, Any]:
    if not examples:
        return {"scope": scope, "n": 0}
    labels = np.asarray([example.accept_label for example in examples], dtype=np.int64)
    utility = np.asarray([example.utility for example in examples], dtype=np.float64)
    probability = _temperature_scale(outputs["p_accept"], temperature)
    accepted = (
        (probability >= thresholds["accept_probability"])
        & (outputs["vacuity"] <= thresholds["max_accept_vacuity"])
        & (outputs["utility"] >= thresholds["min_predicted_utility"])
    )
    return {
        "scope": scope,
        "n": int(len(examples)),
        "patients": int(len({example.patient_id for example in examples})),
        "accuracy_at_0_5": float(np.mean((probability >= 0.5) == labels)),
        "brier": float(np.mean((probability - labels) ** 2)),
        "utility_mae": float(np.mean(np.abs(outputs["utility"] - utility))),
        "mean_vacuity": float(np.mean(outputs["vacuity"])),
        "acceptance_rate": float(np.mean(accepted)),
        "realized_policy_utility": float(np.mean(np.where(accepted, utility, 0.0))),
    }


def fit_head(
    examples: Sequence[UpdateExample],
    *,
    manifest_sha256: str,
    config: Mapping[str, Any],
    seed: int = 20260715,
    epochs: int = 300,
    lr: float = 3e-3,
    hidden: int = 48,
    device: str = "cpu",
    mechanics_smoke: bool = False,
    development_freeze: bool = False,
    claim_external_validation: bool = False,
    minimum_test_patients: int = 20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit, calibrate, freeze thresholds, and evaluate without split reuse."""

    if mechanics_smoke and development_freeze:
        raise ValueError(
            "mechanics_smoke and development_freeze are mutually exclusive"
        )
    if development_freeze and claim_external_validation:
        raise ValueError(
            "development_freeze has no test split and cannot claim external validation"
        )
    contract = validate_split_contract(
        examples,
        require_all_splits=not (mechanics_smoke or development_freeze),
        claim_external_validation=claim_external_validation,
        minimum_test_patients=minimum_test_patients,
    )
    train_examples = [example for example in examples if example.split == "train"]
    calibration_examples = [
        example for example in examples if example.split == "calibration"
    ]
    policy_examples = [
        example for example in examples if example.split == "policy_validation"
    ]
    test_examples = [example for example in examples if example.split == "test"]
    if not train_examples or not calibration_examples:
        raise ValueError("training and calibration splits must both be non-empty")
    if development_freeze:
        if test_examples:
            raise ValueError(
                "development_freeze requires zero test records; test labels must "
                "remain sealed"
            )
        if not policy_examples:
            raise ValueError(
                "development_freeze requires a non-empty policy_validation split"
            )

    _set_seed(seed)
    normalizer = FeatureNormalizer.fit(
        np.stack([example.features for example in train_examples])
    )
    train_x = torch.from_numpy(
        normalizer.transform(np.stack([example.features for example in train_examples]))
    ).to(device)
    train_y = torch.tensor(
        [example.accept_label for example in train_examples],
        dtype=torch.long,
        device=device,
    )
    train_u = torch.tensor(
        [example.utility for example in train_examples],
        dtype=torch.float32,
        device=device,
    )
    model = EvidentialUtilityHead(hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    final_loss = float("nan")
    for epoch in range(int(epochs)):
        optimizer.zero_grad()
        alpha, utility_prediction = model(train_x)
        loss = evidential_utility_loss(
            alpha,
            utility_prediction,
            train_y,
            train_u,
            anneal=min(1.0, (epoch + 1) / max(epochs // 3, 1)),
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()

    calibration_outputs = _head_outputs(
        model, normalizer, calibration_examples, device=device
    )
    calibration_labels = np.asarray(
        [example.accept_label for example in calibration_examples], dtype=np.int64
    )
    temperature = calibrate_temperature(
        calibration_outputs["p_accept"], calibration_labels
    )
    policy_outputs = _head_outputs(model, normalizer, policy_examples, device=device)
    if policy_examples:
        thresholds = select_policy_thresholds(
            _temperature_scale(policy_outputs["p_accept"], temperature),
            policy_outputs["vacuity"],
            policy_outputs["utility"],
            np.asarray(
                [example.utility for example in policy_examples], dtype=np.float64
            ),
        )
        threshold_source = "policy_validation"
    else:
        thresholds = dict(DEFAULT_THRESHOLDS)
        threshold_source = "default_mechanics_smoke_no_policy_validation"

    test_outputs = (
        {}
        if development_freeze
        else _head_outputs(model, normalizer, test_examples, device=device)
    )
    calibration_metrics = _evaluate_examples(
        calibration_examples,
        calibration_outputs,
        temperature=temperature,
        thresholds=thresholds,
        scope=(
            "calibration_development_freeze"
            if development_freeze
            else (
                "calibration_mechanics_only"
                if mechanics_smoke
                else "calibration_diagnostic"
            )
        ),
    )
    test_metrics = (
        {"scope": "sealed_not_opened_development_freeze", "n": 0}
        if development_freeze
        else _evaluate_examples(
            test_examples,
            test_outputs,
            temperature=temperature,
            thresholds=thresholds,
            scope=(
                "internal_prior_exposed_test"
                if contract["prior_exposed_test_cases"]
                else "frozen_test"
            ),
        )
    )
    config_dict = dict(config)
    config_hash = sha256_json(config_dict)
    code_hash = sha256_file(Path(__file__).resolve())
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": "EvidentialUtilityHead",
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "feature_order": list(FEATURE_ORDER),
        "normalizer": {
            "mean": normalizer.mean.tolist(),
            "std": normalizer.std.tolist(),
        },
        "manifest_sha256": str(manifest_sha256),
        "config_sha256": config_hash,
        "code_sha256": code_hash,
        "config": config_dict,
        "seed": int(seed),
        "model_config": {"in_dim": len(FEATURE_ORDER), "hidden": int(hidden)},
        "calibration": {
            "method": "temperature_grid_nll",
            "temperature": float(temperature),
        },
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "split_contract": contract,
        "mechanics_smoke": bool(mechanics_smoke),
        "development_freeze": bool(development_freeze),
        "fit_mode": (
            "development_freeze"
            if development_freeze
            else ("mechanics_smoke" if mechanics_smoke else "strict_four_split")
        ),
        "external_validation_eligible": bool(
            contract["external_validation_eligible"]
            and not mechanics_smoke
            and not development_freeze
        ),
        "efficacy_claim_eligible": bool(
            contract["efficacy_claim_eligible"]
            and not mechanics_smoke
            and not development_freeze
        ),
    }
    if development_freeze:
        status = "DEVELOPMENT_FROZEN_NO_TEST"
        claim_boundary = (
            "Development-only train/calibration/policy-validation freeze; test "
            "records were absent and no efficacy, external-validation, or test "
            "performance claim is permitted."
        )
    elif mechanics_smoke:
        status = "EXPLORATORY_MECHANICS_ONLY"
        claim_boundary = "End-to-end mechanics only; cohort/splits are insufficient for calibrated efficacy."
    elif contract["prior_exposed_test_cases"]:
        status = "EXPLORATORY_INTERNAL_PRIOR_EXPOSED"
        claim_boundary = (
            "Internal prior-exposed test only; no efficacy or external-validation claim. "
            "Test metrics use frozen train/calibration/policy-validation decisions."
        )
    elif not contract["efficacy_claim_eligible"]:
        status = "EXPLORATORY_INSUFFICIENT_TEST_SAMPLE"
        claim_boundary = (
            "Exposure-independent test is below the configured efficacy sample requirement; "
            "test metrics use frozen train/calibration/policy-validation decisions."
        )
    else:
        status = "COMPLETED"
        claim_boundary = (
            "Test metrics use frozen train/calibration/policy-validation decisions and meet the configured "
            "exposure/sample contract; clinical validity remains out of scope."
        )
    threshold_role = (
        "upstream_flat_candidate_diagnostic_only"
        if development_freeze
        else "deployed_by_upstream_head"
    )
    checkpoint["status"] = status
    checkpoint["claim_boundary"] = claim_boundary
    checkpoint["threshold_role"] = threshold_role
    checkpoint["test_metrics"] = test_metrics
    report = {
        "schema_version": 1,
        "status": status,
        "claim_boundary": claim_boundary,
        "training": {
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "final_loss": final_loss,
            "train_examples": len(train_examples),
        },
        "split_contract": contract,
        "calibration": checkpoint["calibration"],
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "threshold_role": threshold_role,
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "manifest_sha256": str(manifest_sha256),
        "config_sha256": config_hash,
        "code_sha256": code_hash,
        "seed": int(seed),
        "development_freeze": bool(development_freeze),
        "fit_mode": checkpoint["fit_mode"],
        "external_validation_eligible": checkpoint["external_validation_eligible"],
        "efficacy_claim_eligible": checkpoint["efficacy_claim_eligible"],
        "efficacy_ineligibility_reasons": contract["efficacy_ineligibility_reasons"],
    }
    return checkpoint, report


def save_checkpoint_bundle(
    checkpoint: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "prompt_update_edl.pt"
    report_path = destination / "prompt_update_edl_report.json"
    torch.save(dict(checkpoint), checkpoint_path)
    report_payload = dict(report)
    report_payload["checkpoint"] = {
        "path": str(checkpoint_path.resolve()),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
    }
    report_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "report": str(report_path.resolve()),
        "checkpoint_sha256": report_payload["checkpoint"]["sha256"],
    }


def load_checkpoint_bundle(
    path: str | Path, *, device: str = "cpu"
) -> tuple[EvidentialUtilityHead, FeatureNormalizer, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    required = {
        "state_dict",
        "feature_order",
        "normalizer",
        "manifest_sha256",
        "config_sha256",
        "seed",
        "calibration",
        "thresholds",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {missing}")
    if tuple(payload["feature_order"]) != FEATURE_ORDER:
        raise ValueError("checkpoint feature order is incompatible with this code")
    model = EvidentialUtilityHead(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    normalizer = FeatureNormalizer(
        mean=np.asarray(payload["normalizer"]["mean"], dtype=np.float32),
        std=np.asarray(payload["normalizer"]["std"], dtype=np.float32),
    )
    return model, normalizer, payload


def decide_update(
    model: EvidentialUtilityHead,
    normalizer: FeatureNormalizer,
    checkpoint: Mapping[str, Any],
    *,
    pet: Any,
    ct: Any,
    current_mask: Any,
    proposed_mask: Any,
    totseg: Any | None = None,
    prompt_metadata: Mapping[str, Any] | PromptMetadata | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    features = extract_update_features(
        pet,
        ct,
        current_mask,
        proposed_mask,
        totseg=totseg,
        prompt_metadata=prompt_metadata,
    )
    x = torch.from_numpy(normalizer.transform(features)).to(device).unsqueeze(0)
    with torch.no_grad():
        alpha, utility = model(x)
    raw_probability = float((alpha[0, 1] / alpha[0].sum()).cpu())
    vacuity = float((2.0 / alpha[0].sum()).cpu())
    predicted_utility = float(utility[0].cpu())
    temperature = float(checkpoint["calibration"]["temperature"])
    probability = float(
        _temperature_scale(np.asarray([raw_probability]), temperature)[0]
    )
    thresholds = checkpoint["thresholds"]
    changed = float(features[FEATURE_ORDER.index("changed_volume_fraction")]) > 0.0
    accept = (
        changed
        and probability >= thresholds["accept_probability"]
        and vacuity <= thresholds["max_accept_vacuity"]
        and predicted_utility >= thresholds["min_predicted_utility"]
    )
    confident_stop = not changed or (
        probability <= thresholds["stop_probability"]
        and vacuity <= thresholds["max_stop_vacuity"]
        and predicted_utility <= thresholds["stop_utility"]
    )
    decision = "ACCEPT" if accept else ("STOP" if confident_stop else "REJECT_CONTINUE")
    return {
        "decision": decision,
        "p_accept": probability,
        "vacuity": vacuity,
        "predicted_utility": predicted_utility,
        "thresholds": dict(thresholds),
        "manifest_sha256": checkpoint["manifest_sha256"],
        "config_sha256": checkpoint["config_sha256"],
    }


def _load_nifti(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    try:
        import nibabel as nib
    except Exception as exc:  # pragma: no cover
        raise ImportError("NIfTI trajectory loading requires nibabel") from exc
    image = nib.as_closest_canonical(nib.load(str(path)))
    return (
        np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32),
        np.asarray(image.affine, dtype=np.float64),
        tuple(float(x) for x in image.header.get_zooms()[:3]),
    )


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def examples_from_manifest(
    manifest_path: str | Path,
    *,
    nsd_weight: float = 0.0,
    interaction_cost: float = 0.0,
    accept_margin: float = 0.0,
    exact_splits: Sequence[str] | None = None,
) -> tuple[list[UpdateExample], dict[str, Any]]:
    """Load a frozen trajectory manifest without resampling any volume."""

    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("trajectory manifest schema_version must be 1")
    records = payload.get("records", [])
    if not records:
        raise ValueError("trajectory manifest has no records")
    if exact_splits is not None:
        expected_splits = {str(split) for split in exact_splits}
        if not expected_splits:
            raise ValueError("exact_splits must contain at least one split")
        observed_splits = {str(record.get("split", "")) for record in records}
        if observed_splits != expected_splits:
            raise ValueError(
                "trajectory manifest changed or violates the exact split seal: "
                f"expected {sorted(expected_splits)}, got {sorted(observed_splits)}"
            )
    examples: list[UpdateExample] = []
    for record in records:
        loaded: dict[
            str, tuple[np.ndarray, np.ndarray, tuple[float, float, float]]
        ] = {}
        for role in ("pet", "ct", "current_mask", "proposed_mask", "ground_truth"):
            role_path = _resolve_path(path.parent, record[f"{role}_path"])
            expected_hash = record.get(f"{role}_sha256")
            if expected_hash and sha256_file(role_path) != expected_hash:
                raise ValueError(f"{role} SHA-256 mismatch for {record['case_id']}")
            loaded[role] = _load_nifti(role_path)
        reference_shape = loaded["pet"][0].shape
        reference_affine = loaded["pet"][1]
        for role, (array, affine, _spacing) in loaded.items():
            if array.shape != reference_shape or not np.allclose(
                affine, reference_affine, atol=1e-4
            ):
                raise ValueError(
                    f"{role} is not on the PET grid for {record['case_id']}"
                )
        totseg = None
        if record.get("totseg_path"):
            totseg_path = _resolve_path(path.parent, record["totseg_path"])
            expected_hash = record.get("totseg_sha256")
            if expected_hash and sha256_file(totseg_path) != expected_hash:
                raise ValueError(f"totseg SHA-256 mismatch for {record['case_id']}")
            totseg, totseg_affine, _ = _load_nifti(totseg_path)
            if totseg.shape != reference_shape or not np.allclose(
                totseg_affine, reference_affine, atol=1e-4
            ):
                raise ValueError(
                    f"TotSeg is not on the PET grid for {record['case_id']}"
                )
        examples.append(
            make_update_example(
                case_id=record["case_id"],
                patient_id=record["patient_id"],
                transition_id=record["transition_id"],
                split=record["split"],
                prior_exposure=bool(record.get("prior_exposure", False)),
                pet=loaded["pet"][0],
                ct=loaded["ct"][0],
                current_mask=loaded["current_mask"][0] > 0.5,
                proposed_mask=loaded["proposed_mask"][0] > 0.5,
                ground_truth=loaded["ground_truth"][0] > 0.5,
                totseg=totseg,
                prompt_metadata=record.get("prompt_metadata"),
                delta_nsd=record.get("delta_nsd"),
                nsd_weight=nsd_weight,
                interaction_cost=interaction_cost,
                accept_margin=accept_margin,
            )
        )
    return examples, payload


def cohort4_smoke_examples(
    cohort4_root: str | Path,
) -> tuple[list[UpdateExample], dict[str, Any]]:
    """Extract 8 real transitions with a patient-blocked 2-way smoke split."""

    root = Path(cohort4_root).resolve()
    aggregate_path = root / "cohort4_aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    by_case = {case["case"]: case for case in aggregate["cases"]}
    cases = sorted(by_case)
    examples: list[UpdateExample] = []
    for case_id in cases:
        patient_id = case_id.rsplit("_", 1)[0]
        split = "train" if patient_id == "train_0001" else "calibration"
        if case_id == "train_0001_FDG":
            case_root = root.parent / case_id
            predictions = [
                case_root / "autopetv_zero_scribble" / f"{case_id}.nii.gz",
                case_root / "one_error_click" / "prediction" / f"{case_id}.nii.gz",
                case_root / "two_error_clicks" / "prediction" / f"{case_id}.nii.gz",
            ]
            click_paths = [
                case_root / "one_error_click" / "simulated_error_clicks.json",
                case_root / "two_error_clicks" / "simulated_error_clicks.json",
            ]
        else:
            case_root = root / case_id
            predictions = [
                root / "batches" / "zero" / "prediction" / f"{case_id}.nii.gz",
                root / "batches" / "round1" / "prediction" / f"{case_id}.nii.gz",
                root / "batches" / "round2" / "prediction" / f"{case_id}.nii.gz",
            ]
            click_paths = [
                case_root / "round1" / "simulated_error_clicks.json",
                case_root / "round2" / "simulated_error_clicks.json",
            ]
        pet, pet_affine, _ = _load_nifti(case_root / "input" / f"{case_id}_0001.nii.gz")
        ct, ct_affine, _ = _load_nifti(case_root / "input" / f"{case_id}_0000.nii.gz")
        ground_truth, gt_affine, _ = _load_nifti(
            case_root / "labels" / f"{case_id}.nii.gz"
        )
        masks = [_load_nifti(path) for path in predictions]
        for role, array, affine in (
            ("CT", ct, ct_affine),
            ("ground_truth", ground_truth, gt_affine),
            *[(f"prediction_{i}", item[0], item[1]) for i, item in enumerate(masks)],
        ):
            if array.shape != pet.shape or not np.allclose(
                affine, pet_affine, atol=1e-4
            ):
                raise ValueError(f"cohort4 {role} is not on the PET grid for {case_id}")
        case_metrics = by_case[case_id]
        delta_nsd = [case_metrics["delta_nsd_round1"], case_metrics["delta_nsd_round2"]]
        for transition in range(2):
            metadata = json.loads(click_paths[transition].read_text(encoding="utf-8"))
            metadata["round_index"] = transition + 1
            examples.append(
                make_update_example(
                    case_id=case_id,
                    patient_id=patient_id,
                    transition_id=f"round{transition}_to_round{transition + 1}",
                    split=split,
                    prior_exposure=True,
                    pet=pet,
                    ct=ct,
                    current_mask=masks[transition][0] > 0.5,
                    proposed_mask=masks[transition + 1][0] > 0.5,
                    ground_truth=ground_truth > 0.5,
                    prompt_metadata=metadata,
                    delta_nsd=float(delta_nsd[transition]),
                )
            )
    return examples, aggregate


def _preflight_development_manifest(path: str | Path) -> dict[str, Any]:
    manifest = Path(path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("development manifest schema_version must be 1")
    if payload.get("status") != "FROZEN_DEVELOPMENT":
        raise ValueError(
            "--development-freeze requires manifest status FROZEN_DEVELOPMENT"
        )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("development manifest records must be a non-empty list")
    splits = {str(record.get("split", "")) for record in records}
    expected = {"train", "calibration", "policy_validation"}
    if "test" in splits:
        raise ValueError(
            "--development-freeze rejects test records before any volume is loaded"
        )
    if splits != expected:
        raise ValueError(
            "--development-freeze requires exactly train/calibration/"
            f"policy_validation records, got {sorted(splits)}"
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest", help="strict schema-v1 frozen trajectory manifest"
    )
    source.add_argument(
        "--cohort4-root", help="explicit mechanics-only smoke on the 2-patient cohort4"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nsd-weight", type=float, default=0.0)
    parser.add_argument("--interaction-cost", type=float, default=0.0)
    parser.add_argument("--accept-margin", type=float, default=0.0)
    parser.add_argument("--claim-external-validation", action="store_true")
    parser.add_argument("--minimum-test-patients", type=int, default=20)
    parser.add_argument(
        "--development-freeze",
        action="store_true",
        help=(
            "fit/calibrate/freeze on train/calibration/policy_validation only; "
            "requires FROZEN_DEVELOPMENT manifest with zero test records"
        ),
    )
    args = parser.parse_args(argv)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    if args.cohort4_root:
        if args.development_freeze:
            raise ValueError(
                "--development-freeze requires --manifest, not --cohort4-root"
            )
        if args.claim_external_validation:
            raise ValueError(
                "cohort4 is prior-exposed mechanics data and cannot support external validation"
            )
        examples, source_payload = cohort4_smoke_examples(args.cohort4_root)
        source_manifest = Path(args.cohort4_root) / "cohort4_aggregate.json"
        mechanics_smoke = True
    else:
        if args.development_freeze:
            if args.claim_external_validation:
                raise ValueError(
                    "--development-freeze cannot claim external validation"
                )
            _preflight_development_manifest(args.manifest)
        examples, source_payload = examples_from_manifest(
            args.manifest,
            nsd_weight=args.nsd_weight,
            interaction_cost=args.interaction_cost,
            accept_margin=args.accept_margin,
            exact_splits=(
                ("train", "calibration", "policy_validation")
                if args.development_freeze
                else None
            ),
        )
        source_manifest = Path(args.manifest)
        mechanics_smoke = False
    config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "hidden": args.hidden,
        "seed": args.seed,
        "device": args.device,
        "nsd_weight": args.nsd_weight,
        "interaction_cost": args.interaction_cost,
        "accept_margin": args.accept_margin,
        "source_status": source_payload.get("status", "unspecified"),
        "minimum_test_patients": args.minimum_test_patients,
        "development_freeze": bool(args.development_freeze),
    }
    checkpoint, report = fit_head(
        examples,
        manifest_sha256=sha256_file(source_manifest),
        config=config,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        device=args.device,
        mechanics_smoke=mechanics_smoke,
        development_freeze=bool(args.development_freeze),
        claim_external_validation=args.claim_external_validation,
        minimum_test_patients=args.minimum_test_patients,
    )
    report["started_at"] = started_at
    report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["source_manifest"] = str(source_manifest.resolve())
    report["examples"] = [
        {
            "case_id": example.case_id,
            "patient_id": example.patient_id,
            "transition_id": example.transition_id,
            "split": example.split,
            "prior_exposure": example.prior_exposure,
            "accept_label": example.accept_label,
            "utility": example.utility,
            "delta_dice": example.delta_dice,
            "delta_nsd": example.delta_nsd,
        }
        for example in examples
    ]
    paths = save_checkpoint_bundle(checkpoint, report, output_dir=args.out_dir)
    print(json.dumps({"status": report["status"], **paths}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
