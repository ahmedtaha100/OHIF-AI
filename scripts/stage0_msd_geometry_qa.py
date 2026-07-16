#!/usr/bin/env python3
"""Generate native-grid geometry QA overlays for MSD lung and pancreas loaders."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_nninteractive.medical_geometry import load_nifti_on_reference_grid
from rl_nninteractive.provenance import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specifications = (
        ("Task06_Lung", "lung_001", 1),
        ("Task07_Pancreas", "pancreas_001", 2),
    )
    cases = []
    for dataset, case_id, tumor_label in specifications:
        image_path = args.data_root / dataset / "imagesTr" / f"{case_id}.nii.gz"
        label_path = args.data_root / dataset / "labelsTr" / f"{case_id}.nii.gz"
        image = load_nifti_on_reference_grid(
            image_path,
            reference_path=image_path,
            channel_index=0,
            reference_channel_index=0,
        )
        label = load_nifti_on_reference_grid(
            label_path,
            reference_path=image_path,
            is_label=True,
            reference_channel_index=0,
        )
        ground_truth = np.isclose(label.data_zyx, tumor_label)
        overlay_path = args.out_dir / f"{case_id}_native_overlay.png"
        _render_overlay(image.data_zyx, ground_truth, case_id, overlay_path)
        cases.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "tumor_label": tumor_label,
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "label_sha256": sha256_file(label_path),
                "shape_zyx": list(image.data_zyx.shape),
                "tumor_voxels": int(ground_truth.sum()),
                "label_physical_overlap_fraction": label.geometry.physical_overlap_fraction,
                "label_center_distance_mm": label.geometry.center_distance_mm,
                "image_source_orientation": image.geometry.source_orientation,
                "image_output_orientation": image.geometry.output_orientation,
                "image_source_spacing_xyz": list(image.geometry.source_spacing_xyz),
                "image_output_spacing_xyz": list(image.geometry.output_spacing_xyz),
                "label_interpolation_order": label.geometry.interpolation_order,
                "overlay": str(overlay_path.resolve()),
            }
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": (
            "PASS"
            if all(
                case["label_physical_overlap_fraction"] >= 0.5
                and case["label_interpolation_order"] == 0
                and case["tumor_voxels"] > 0
                for case in cases
            )
            else "FAIL"
        ),
        "cases": cases,
    }
    report_path = args.out_dir / "stage0_msd_geometry_qa.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path.resolve()), "status": report["status"]}))


def _render_overlay(image, ground_truth, case_id: str, output_path: Path) -> None:
    coords = np.argwhere(ground_truth)
    center = (
        tuple(int(value) for value in np.rint(coords.mean(axis=0)))
        if len(coords)
        else tuple(int(value // 2) for value in ground_truth.shape)
    )
    views = (
        ("Axial", image[center[0]], ground_truth[center[0]]),
        ("Coronal", image[:, center[1], :], ground_truth[:, center[1], :]),
        ("Sagittal", image[:, :, center[2]], ground_truth[:, :, center[2]]),
    )
    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=150)
    for axis, (title, image_slice, label_slice) in zip(axes, views):
        axis.imshow(image_slice, cmap="gray", vmin=low, vmax=high)
        if bool(label_slice.any()):
            axis.contour(
                label_slice.astype(np.uint8),
                levels=[0.5],
                colors=["#39FF14"],
                linewidths=0.9,
            )
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{case_id} — native-grid image/tumor-label overlay")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
