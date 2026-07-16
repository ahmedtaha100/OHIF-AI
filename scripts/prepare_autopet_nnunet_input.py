"""Prepare a physically aligned AutoPET CT/PET pair for nnU-Net inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np

from rl_nninteractive.medical_geometry import load_nifti_on_reference_grid


def main() -> None:
    args = _parse_args()
    source = args.case_dir.resolve() / args.tracer
    ct_path = source / "CT.nii.gz"
    pet_path = source / "PET.nii.gz"
    label_path = source / "TTB.nii.gz"
    for path in (ct_path, pet_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    input_dir = output_dir / "input"
    label_dir = output_dir / "labels"
    input_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    ct = load_nifti_on_reference_grid(ct_path, reference_path=pet_path)
    pet = load_nifti_on_reference_grid(pet_path, reference_path=pet_path)
    label = load_nifti_on_reference_grid(
        label_path,
        reference_path=pet_path,
        is_label=True,
    )

    stem = args.case_name or f"{args.case_dir.name}_{args.tracer}"
    outputs = {
        "ct": input_dir / f"{stem}_0000.nii.gz",
        "pet": input_dir / f"{stem}_0001.nii.gz",
        "label": label_dir / f"{stem}.nii.gz",
    }
    _save(ct.data_zyx, ct.geometry.output_affine_xyz, outputs["ct"], np.float32)
    _save(pet.data_zyx, pet.geometry.output_affine_xyz, outputs["pet"], np.float32)
    _save(label.data_zyx, label.geometry.output_affine_xyz, outputs["label"], np.uint8)

    report = {
        "schema_version": 1,
        "case": stem,
        "reference": str(pet_path),
        "shape_zyx": list(pet.data_zyx.shape),
        "geometry": {
            "ct": asdict(ct.geometry),
            "pet": asdict(pet.geometry),
            "label": asdict(label.geometry),
        },
        "outputs": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for key, path in outputs.items()
        },
    }
    report_path = output_dir / "input_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(report_path), "shape_zyx": report["shape_zyx"]}))


def _save(data_zyx: np.ndarray, affine: tuple[tuple[float, ...], ...], path: Path, dtype) -> None:
    data_xyz = np.transpose(data_zyx, (2, 1, 0)).astype(dtype, copy=False)
    nib.save(nib.Nifti1Image(data_xyz, np.asarray(affine, dtype=np.float64)), str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--tracer", choices=("FDG", "PSMA"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-name")
    return parser.parse_args()


if __name__ == "__main__":
    main()
