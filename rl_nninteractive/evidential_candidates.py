"""Ground-truth-free next-prompt candidates from the evidential error model.

This is the module that closes the Phase-2 gap flagged on 2026-07-05:

    "current candidates still depend on ground truth in the runner, so the
     recommender path is not deployable until GT-free candidate generation is
     wired."

``robot_user.largest_component_robot_action(mask, ground_truth)`` and
``multitool.multi_tool_candidates(mask, ground_truth)`` locate the next
correction by comparing the mask to ground truth -- an oracle that does not
exist at annotation time. The functions here are their deployable twins: they
locate the next correction from the evidential model's predicted error map
(``predict_error_maps``), and they derive a stopping decision from the model's
own predicted residual error -- no ground truth anywhere on the inference path.

The action / decision types mirror ``robot_user`` and ``recommender`` so the
GT-free policy is a drop-in wherever the oracle policy was used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .adapter import VoxelCoord
from .deterministic_geometry import (
    ComponentGeometry,
    GeometryTool,
    Polarity,
    build_component_geometry,
)
from .env import POINT_NEGATIVE, POINT_POSITIVE, STOP
from .evidential import ErrorMaps, EvidentialErrorNet3D, predict_error_maps
from .multitool import MultiToolAction
from .recommender import PromptSuggestion


_STRUCT = np.ones((3, 3, 3), dtype=bool)


@dataclass(frozen=True)
class EvidentialCandidate:
    """One GT-free correction candidate located by the evidential model."""

    polarity: Polarity                 # "positive" (fix a missed region) / "negative" (remove leakage)
    action_type: int                   # POINT_POSITIVE or POINT_NEGATIVE
    coord: VoxelCoord                  # representative voxel (centroid-nearest, deterministic)
    component_mask: np.ndarray         # single connected component (for geometry builders)
    component_size: int
    predicted_error_mass: float        # sum of class-probability over the component
    mean_confidence: float             # mean predicted error-probability over the component (belief it is wrong)
    mean_vacuity: float                # mean epistemic vacuity over the component (ambiguity / defer-to-human flag)

    def geometry(self, tool: GeometryTool) -> ComponentGeometry:
        return build_component_geometry(self.component_mask, tool=tool, polarity=self.polarity)


@dataclass(frozen=True)
class EvidentialStopDecision:
    should_stop: bool
    predicted_error_voxels: int
    predicted_error_mass: float
    mean_vacuity: float
    largest_component_size: int
    reason: str


def _representative_coord(component: np.ndarray) -> VoxelCoord:
    coords = np.argwhere(component)
    centroid = coords.mean(axis=0)
    distances = np.square(coords - centroid).sum(axis=1)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], distances))
    chosen = coords[int(order[0])]
    return (int(chosen[0]), int(chosen[1]), int(chosen[2]))


def _polarity_error_field(
    error_maps: ErrorMaps,
    current_mask: np.ndarray,
    *,
    polarity: Polarity,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (boolean error field, per-voxel class prob) for one polarity.

    positive -> false-negative voxels the model thinks should be filled in
                (probability of FN above threshold AND currently background).
    negative -> false-positive voxels the model thinks are leakage
                (probability of FP above threshold AND currently foreground).
    """

    mask = np.asarray(current_mask).astype(bool)
    if polarity == "positive":
        prob = error_maps.p_false_negative
        field = (prob >= threshold) & (~mask)
    else:
        prob = error_maps.p_false_positive
        field = (prob >= threshold) & (mask)
    return field, prob


def _largest_component(field: np.ndarray, *, min_size: int) -> tuple[np.ndarray | None, int]:
    if not field.any():
        return None, 0
    labels, count = ndimage.label(field, structure=_STRUCT)
    counts = np.bincount(labels.ravel(), minlength=count + 1)
    counts[0] = 0
    size = int(counts.max())
    if size < min_size:
        return None, size
    label_id = int(np.flatnonzero(counts == size)[0])
    return labels == label_id, size


def evidential_candidate(
    error_maps: ErrorMaps,
    current_mask: Any,
    *,
    polarity: Polarity,
    threshold: float = 0.30,
    min_size: int = 3,
) -> EvidentialCandidate | None:
    """Largest predicted-error component for one polarity, or None."""

    mask = np.asarray(current_mask).astype(bool)
    field, prob = _polarity_error_field(error_maps, mask, polarity=polarity, threshold=threshold)
    component, size = _largest_component(field, min_size=min_size)
    if component is None:
        return None
    error_mass = float(prob[component].sum())
    mean_conf = float(prob[component].mean())
    mean_vac = float(error_maps.vacuity[component].mean())
    action_type = POINT_POSITIVE if polarity == "positive" else POINT_NEGATIVE
    return EvidentialCandidate(
        polarity=polarity,
        action_type=action_type,
        coord=_representative_coord(component),
        component_mask=component,
        component_size=size,
        predicted_error_mass=error_mass,
        mean_confidence=mean_conf,
        mean_vacuity=mean_vac,
    )


def _candidate_from_component(error_maps: ErrorMaps, component: np.ndarray, *, polarity: Polarity) -> EvidentialCandidate:
    prob = error_maps.p_false_negative if polarity == "positive" else error_maps.p_false_positive
    action_type = POINT_POSITIVE if polarity == "positive" else POINT_NEGATIVE
    return EvidentialCandidate(
        polarity=polarity,
        action_type=action_type,
        coord=_representative_coord(component),
        component_mask=component,
        component_size=int(component.sum()),
        predicted_error_mass=float(prob[component].sum()),
        mean_confidence=float(prob[component].mean()),
        mean_vacuity=float(error_maps.vacuity[component].mean()),
    )


def evidential_candidates_topk(
    error_maps: ErrorMaps,
    current_mask: Any,
    *,
    k: int = 3,
    threshold: float = 0.30,
    min_size: int = 3,
) -> list[EvidentialCandidate]:
    """Top-k predicted-error components per polarity (GT-free), largest first.

    This is the discrete action menu for the RL policy: a handful of candidate
    corrections the evidential model proposes, from which the policy chooses one
    (or STOP).
    """

    mask = np.asarray(current_mask).astype(bool)
    out: list[EvidentialCandidate] = []
    for polarity in ("positive", "negative"):
        field, _ = _polarity_error_field(error_maps, mask, polarity=polarity, threshold=threshold)
        if not field.any():
            continue
        labels, count = ndimage.label(field, structure=_STRUCT)
        counts = np.bincount(labels.ravel(), minlength=count + 1)
        counts[0] = 0
        order = np.argsort(counts)[::-1]
        taken = 0
        for label_id in order:
            if label_id == 0 or counts[label_id] < min_size or taken >= k:
                continue
            out.append(_candidate_from_component(error_maps, labels == label_id, polarity=polarity))
            taken += 1
    return out


def evidential_next_action(
    error_maps: ErrorMaps,
    current_mask: Any,
    *,
    threshold: float = 0.30,
    min_size: int = 3,
) -> EvidentialCandidate | None:
    """GT-free analogue of ``largest_component_robot_action``.

    Returns the larger of the best false-negative and best false-positive
    candidates (ties prefer false-negative = fill missed target first), or
    ``None`` to signal STOP when neither polarity has a qualifying component.
    """

    fn = evidential_candidate(error_maps, current_mask, polarity="positive", threshold=threshold, min_size=min_size)
    fp = evidential_candidate(error_maps, current_mask, polarity="negative", threshold=threshold, min_size=min_size)
    if fn is None and fp is None:
        return None
    if fp is None:
        return fn
    if fn is None:
        return fp
    # Prefer the larger predicted-error mass; tie -> false negative.
    if fn.predicted_error_mass >= fp.predicted_error_mass:
        return fn
    return fp


def evidential_stop_decision(
    error_maps: ErrorMaps,
    current_mask: Any,
    *,
    threshold: float = 0.30,
    min_size: int = 3,
    stop_error_voxels: int = 8,
) -> EvidentialStopDecision:
    """GT-free stop rule.

    Stops when the model predicts little residual correctable error: no
    qualifying error component of at least ``min_size`` remains, or the total
    predicted-error voxel count drops below ``stop_error_voxels``. This replaces
    the prostate-paper habit of taking the *best-Dice-over-trajectory using GT*;
    here the stop signal is the model's own predicted residual error, so the
    reported Dice is Dice-at-the-policy-chosen-stop.
    """

    mask = np.asarray(current_mask).astype(bool)
    fn_field, _ = _polarity_error_field(error_maps, mask, polarity="positive", threshold=threshold)
    fp_field, _ = _polarity_error_field(error_maps, mask, polarity="negative", threshold=threshold)
    error_field = fn_field | fp_field
    predicted_error_voxels = int(error_field.sum())
    error_mass = float(error_maps.p_error.sum())
    mean_vacuity = float(error_maps.vacuity.mean())
    _, fn_size = _largest_component(fn_field, min_size=min_size)
    _, fp_size = _largest_component(fp_field, min_size=min_size)
    largest = max(fn_size, fp_size)

    if largest < min_size:
        return EvidentialStopDecision(True, predicted_error_voxels, error_mass, mean_vacuity, largest,
                                      "no_component_above_min_size")
    if predicted_error_voxels < stop_error_voxels:
        return EvidentialStopDecision(True, predicted_error_voxels, error_mass, mean_vacuity, largest,
                                      "predicted_error_below_threshold")
    return EvidentialStopDecision(False, predicted_error_voxels, error_mass, mean_vacuity, largest, "continue")


def evidential_uncertainty_channel(
    model: EvidentialErrorNet3D,
    image: Any,
    current_mask: Any,
    *,
    channel: str = "p_error",
    device: "Any | None" = None,
) -> np.ndarray:
    """Return a (D,H,W) float32 evidential map to append to the RL state tensor.

    This is the "trained surrogate" uncertainty source in the Plan's ranked list
    (nnInteractive logits > TTA disagreement > trained surrogate), upgraded to an
    *evidential* surrogate. Feed the result to
    ``uncertainty.append_uncertainty_channel``. ``channel`` selects which map:
    ``p_error`` (default, best error localizer), ``vacuity`` (epistemic
    ambiguity), ``p_false_negative`` or ``p_false_positive``.
    """

    maps = predict_error_maps(model, image, current_mask, device=device)
    lookup = {
        "p_error": maps.p_error,
        "vacuity": maps.vacuity,
        "p_false_negative": maps.p_false_negative,
        "p_false_positive": maps.p_false_positive,
    }
    if channel not in lookup:
        raise ValueError(f"unknown channel {channel!r}; choose from {sorted(lookup)}")
    return lookup[channel].astype(np.float32)


def evidential_recommend_next_prompt(
    *,
    case_id: str,
    image: Any,
    current_mask: Any,
    model: EvidentialErrorNet3D,
    tool: GeometryTool = "point",
    threshold: float = 0.30,
    min_size: int = 3,
    stop_error_voxels: int = 8,
    device: "Any | None" = None,
) -> PromptSuggestion:
    """Deployable, GT-free replacement for ``recommender.recommend_next_prompt``.

    Runs the evidential model on (image, current_mask) and returns the next ghost
    prompt for the OHIF recommender -- no ground truth involved. When the stop
    rule fires, returns a ``stop`` suggestion.
    """

    error_maps = predict_error_maps(model, image, current_mask, device=device)
    stop = evidential_stop_decision(
        error_maps, current_mask, threshold=threshold, min_size=min_size, stop_error_voxels=stop_error_voxels
    )
    candidate = evidential_next_action(error_maps, current_mask, threshold=threshold, min_size=min_size)
    if stop.should_stop or candidate is None:
        action = MultiToolAction(tool="stop", polarity=None, geometry=None)
        return PromptSuggestion(
            case_id=case_id,
            action=action,
            confidence=float(max(0.0, min(1.0, 1.0 - stop.mean_vacuity))),
            reason=f"evidential_stop:{stop.reason}",
            requires_review=True,
        )
    geometry = candidate.geometry(tool)
    action = MultiToolAction(tool=tool, polarity=candidate.polarity, geometry=geometry, confidence=candidate.mean_confidence)
    return PromptSuggestion(
        case_id=case_id,
        action=action,
        confidence=float(max(0.0, min(1.0, candidate.mean_confidence))),
        reason=f"evidential_largest_{candidate.polarity}_error_component",
        requires_review=True,
    )
