"""Score and visualize an AutoPET pre/post scribble refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from rl_nninteractive.metrics import dice_score, hd95, normalized_surface_dice


def main() -> None:
    args = _parse_args()
    gt_img = nib.load(str(args.ground_truth))
    initial_img = nib.load(str(args.initial))
    refined_img = nib.load(str(args.refined))
    gt = np.asarray(gt_img.dataobj) > 0
    initial = np.asarray(initial_img.dataobj) > 0
    refined = np.asarray(refined_img.dataobj) > 0
    if initial.shape != gt.shape or refined.shape != gt.shape:
        raise ValueError(f"shape mismatch: gt={gt.shape}, initial={initial.shape}, refined={refined.shape}")
    spacing = tuple(float(v) for v in gt_img.header.get_zooms()[:3])
    voxel_ml = float(np.prod(spacing) / 1000.0)

    initial_metrics = _metrics(initial, gt, spacing, voxel_ml)
    refined_metrics = _metrics(refined, gt, spacing, voxel_ml)
    payload = {
        "schema_version": 1,
        "case": args.case,
        "status": "exploratory-contaminated-local-cohort",
        "ground_truth_use": "robot-user simulation and scoring only; not a model input",
        "spacing_xyz_mm": list(spacing),
        "initial": initial_metrics,
        "refined": refined_metrics,
        "delta": {
            key: float(refined_metrics[key] - initial_metrics[key])
            for key in ("dice", "nsd_2mm", "fpv_ml", "fnv_ml")
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _render(gt, initial, refined, args.output_png)
    print(json.dumps(payload))


def _metrics(prediction: np.ndarray, ground_truth: np.ndarray, spacing, voxel_ml: float) -> dict:
    fp = prediction & ~ground_truth
    fn = ground_truth & ~prediction
    return {
        "dice": float(dice_score(prediction, ground_truth)),
        "nsd_2mm": float(normalized_surface_dice(prediction, ground_truth, tolerance=2.0, spacing=spacing)),
        "hd95_mm": float(hd95(prediction, ground_truth, spacing=spacing)),
        "positive_voxels": int(prediction.sum()),
        "fpv_ml": float(fp.sum() * voxel_ml),
        "fnv_ml": float(fn.sum() * voxel_ml),
    }


def _render(gt: np.ndarray, initial: np.ndarray, refined: np.ndarray, output: Path) -> None:
    coords = np.argwhere(gt)
    z = int(np.median(coords[:, 2])) if coords.size else gt.shape[2] // 2
    panels = ((initial, "Before scribble"), (refined, "After one scribble"), (gt, "Ground truth"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    base = gt[:, :, z].T
    for axis, (mask, title) in zip(axes, panels, strict=True):
        axis.imshow(base, cmap="gray", origin="lower", alpha=0.35)
        axis.contour(gt[:, :, z].T, levels=[0.5], colors=["lime"], linewidths=1.5)
        axis.contour(mask[:, :, z].T, levels=[0.5], colors=["magenta"], linewidths=1.2)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("AutoPET V prompt baseline: GT=green, prediction=magenta")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
