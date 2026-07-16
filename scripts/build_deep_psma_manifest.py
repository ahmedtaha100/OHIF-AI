#!/usr/bin/env python3
"""Build a patient-grouped, provenance-bound DEEP-PSMA StudyManifest v2."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from rl_nninteractive.medical_geometry import physical_alignment_report
from rl_nninteractive.provenance import sha256_file, sha256_json


SPLIT_RATIOS = (
    ("train", 0.60),
    ("policy_validation", 0.15),
    ("calibration", 0.10),
    ("test", 0.15),
)
TRACERS = ("FDG", "PSMA")
MODALITIES = ("CT", "PET")


def main() -> None:
    args = _parse_args()
    data_root = args.data_root.resolve()
    patient_dirs = sorted(path for path in data_root.glob("train_[0-9][0-9][0-9][0-9]") if path.is_dir())
    if not patient_dirs:
        raise ValueError(f"no DEEP-PSMA patient directories found under {data_root}")

    patient_splits = _patient_splits([path.name for path in patient_dirs], args.split_seed)
    preprocessing = {
        "orientation": "RAS",
        "reference_modality": "PET",
        "image_interpolation": "trilinear",
        "label_interpolation": "nearest",
        "minimum_physical_overlap_fraction": args.minimum_overlap,
        "array_convention": "CZYX",
    }
    cases: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        for tracer in TRACERS:
            study_dir = patient_dir / tracer
            paths = {
                "CT": study_dir / "CT.nii.gz",
                "PET": study_dir / "PET.nii.gz",
                "ground_truth": study_dir / "TTB.nii.gz",
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing required files for {patient_id}/{tracer}: {missing}")

            geometry = {role: _geometry_record(path) for role, path in paths.items()}
            reference = nib.as_closest_canonical(nib.load(str(paths["PET"])))
            overlap: dict[str, float] = {}
            center_distance_mm: dict[str, float] = {}
            for role in ("CT", "ground_truth"):
                source = nib.as_closest_canonical(nib.load(str(paths[role])))
                overlap[role], center_distance_mm[role] = physical_alignment_report(source, reference)
                if overlap[role] < args.minimum_overlap:
                    raise ValueError(
                        f"{patient_id}/{tracer}/{role} overlap {overlap[role]:.6f} "
                        f"is below {args.minimum_overlap:.6f}"
                    )

            image_hashes = {modality: sha256_file(paths[modality]) for modality in MODALITIES}
            ground_truth_hash = sha256_file(paths["ground_truth"])
            split = patient_splits[patient_id]
            inclusion_record = {
                "patient_id": patient_id,
                "tracer": tracer,
                "split": split,
                "required_files": ["CT.nii.gz", "PET.nii.gz", "TTB.nii.gz"],
                "minimum_overlap_fraction": args.minimum_overlap,
                "included": True,
            }
            cases.append(
                {
                    "case_id": f"{patient_id}_{tracer}",
                    "patient_id": patient_id,
                    "site": "DEEP-PSMA-public",
                    "tracer": tracer,
                    "target_label": "TTB",
                    "split": split,
                    "prior_exposure": True,
                    "images": {modality: str(paths[modality].resolve()) for modality in MODALITIES},
                    "ground_truth": str(paths["ground_truth"].resolve()),
                    "image_sha256": image_hashes,
                    "ground_truth_sha256": ground_truth_hash,
                    "inclusion_hash": sha256_json(inclusion_record),
                    "geometry": geometry,
                }
            )
            qa_rows.append(
                {
                    "case_id": f"{patient_id}_{tracer}",
                    "split": split,
                    "overlap_fraction": overlap,
                    "center_distance_mm": center_distance_mm,
                }
            )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "version": 2,
        "generated_at": generated_at,
        "split_seed": args.split_seed,
        "split_provenance": "retrospectively_frozen_contaminated",
        "preprocessing": preprocessing,
        "preprocessing_hash": sha256_json(preprocessing),
        "datasets": [
            {
                "name": "DEEP-PSMA",
                "version": "v1-2025-11-21-doi-10.5281/zenodo.15281784",
                "annotation_version": "v1-post-challenge-updated-annotations",
                "source_url": "https://zenodo.org/records/15281784",
                "license": "CC-BY-NC-4.0",
                "modalities": list(MODALITIES),
                "reference_modality": "PET",
                "cases": cases,
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest_sha256 = sha256_file(args.output)

    report = _qa_report(
        generated_at=generated_at,
        data_root=data_root,
        manifest_path=args.output.resolve(),
        manifest_sha256=manifest_sha256,
        patient_splits=patient_splits,
        qa_rows=qa_rows,
        minimum_overlap=args.minimum_overlap,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "patients": len(patient_dirs),
                "studies": len(cases),
                "manifest": str(args.output.resolve()),
                "manifest_sha256": manifest_sha256,
                "qa_report": str(args.report.resolve()),
            }
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260715)
    parser.add_argument("--minimum-overlap", type=float, default=0.5)
    return parser.parse_args()


def _patient_splits(patient_ids: list[str], seed: int) -> dict[str, str]:
    shuffled = list(patient_ids)
    random.Random(seed).shuffle(shuffled)
    n_patients = len(shuffled)
    boundaries: list[int] = []
    cumulative = 0.0
    for _, ratio in SPLIT_RATIOS[:-1]:
        cumulative += ratio
        boundaries.append(round(n_patients * cumulative))
    split_names: list[str] = []
    start = 0
    for (name, _), end in zip(SPLIT_RATIOS[:-1], boundaries):
        split_names.extend([name] * (end - start))
        start = end
    split_names.extend([SPLIT_RATIOS[-1][0]] * (n_patients - start))
    return dict(zip(shuffled, split_names))


def _geometry_record(path: Path) -> dict[str, Any]:
    image = nib.load(str(path))
    if len(image.shape) != 3 or any(int(value) <= 1 for value in image.shape):
        raise ValueError(f"required 3D NIfTI has invalid shape {image.shape}: {path}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if not bool(np.isfinite(affine).all()):
        raise ValueError(f"non-finite affine: {path}")
    return {
        "affine": affine.tolist(),
        "orientation": "".join(str(value) for value in nib.aff2axcodes(affine)),
        "spacing": [float(value) for value in image.header.get_zooms()[:3]],
        "shape_xyz": [int(value) for value in image.shape],
    }


def _qa_report(
    *,
    generated_at: str,
    data_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    patient_splits: dict[str, str],
    qa_rows: list[dict[str, Any]],
    minimum_overlap: float,
) -> dict[str, Any]:
    overlaps = {
        role: [row["overlap_fraction"][role] for row in qa_rows]
        for role in ("CT", "ground_truth")
    }
    distances = {
        role: [row["center_distance_mm"][role] for row in qa_rows]
        for role in ("CT", "ground_truth")
    }
    return {
        "generated_at": generated_at,
        "status": "PASS",
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "patients": len(patient_splits),
        "studies": len(qa_rows),
        "patient_split_counts": dict(sorted(Counter(patient_splits.values()).items())),
        "study_split_counts": dict(sorted(Counter(row["split"] for row in qa_rows).items())),
        "minimum_required_overlap_fraction": minimum_overlap,
        "overlap_summary": {
            role: {"minimum": min(values), "median": float(np.median(values))}
            for role, values in overlaps.items()
        },
        "center_distance_mm_summary": {
            role: {"maximum": max(values), "median": float(np.median(values))}
            for role, values in distances.items()
        },
        "cases": qa_rows,
    }


if __name__ == "__main__":
    main()
