"""Render a PI-ready multifocal AutoPET prompt-refinement trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

from rl_nninteractive.metrics import dice_score, normalized_surface_dice


def main() -> None:
    args = _parse_args()
    pet_img = nib.load(str(args.pet))
    pet = np.asarray(pet_img.dataobj, dtype=np.float32)
    gt = _mask(args.ground_truth)
    predictions = [_mask(path) for path in args.prediction]
    if any(item.shape != gt.shape for item in [pet, *predictions]):
        raise ValueError("PET, ground truth, and prediction shapes must match")

    spacing = tuple(float(v) for v in pet_img.header.get_zooms()[:3])
    metrics = [
        {
            "dice": float(dice_score(prediction, gt)),
            "nsd_2mm": float(
                normalized_surface_dice(prediction, gt, tolerance=2.0, spacing=spacing)
            ),
        }
        for prediction in predictions
    ]
    components, count = ndimage.label(gt, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(components.ravel())
    component_ids = sorted(range(1, count + 1), key=lambda idx: int(sizes[idx]), reverse=True)
    slices = [int(np.median(np.argwhere(components == idx)[:, 2])) for idx in component_ids[:3]]
    if not slices:
        slices = [gt.shape[2] // 2]

    click_sets = [_read_clicks(path) for path in args.click_manifest]
    vmin, vmax = np.percentile(pet[np.isfinite(pet)], [2.0, 99.7])
    fig, axes = plt.subplots(
        len(slices),
        len(predictions) + 1,
        figsize=(4.2 * (len(predictions) + 1), 4.2 * len(slices)),
        squeeze=False,
        constrained_layout=True,
        facecolor="white",
    )
    titles = ["PET + ground truth"] + [
        f"{label}\nDice {score['dice']:.3f} · NSD {score['nsd_2mm']:.3f}"
        for label, score in zip(args.label, metrics, strict=True)
    ]
    for row, z in enumerate(slices):
        for column, axis in enumerate(axes[row]):
            axis.imshow(pet[:, :, z].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            _contour(axis, gt[:, :, z].T, "#38f26b", 2.0)
            if column:
                _contour(axis, predictions[column - 1][:, :, z].T, "#ff2bd6", 1.5)
                for clicks in click_sets[: column - 1]:
                    for click in clicks:
                        if abs(click[2] - z) <= 1:
                            axis.scatter(click[0], click[1], marker="*", s=90, c="#00e5ff", edgecolors="black")
            if row == 0:
                axis.set_title(titles[column], fontsize=12, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"Lesion {row + 1} · axial z={z}", fontsize=11)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "Official AutoPET V four-channel prompt model — multifocal correction on RTX 4080\n"
        "green: ground truth · magenta: prediction · cyan star: supplied positive correction",
        fontsize=15,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "lesion_slices": slices, "metrics": metrics}))


def _mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj) > 0


def _contour(axis, mask: np.ndarray, color: str, width: float) -> None:
    if bool(mask.any()):
        axis.contour(mask, levels=[0.5], colors=[color], linewidths=width)


def _read_clicks(path: Path) -> list[tuple[int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    click = payload.get("foreground_xyz")
    return [tuple(int(v) for v in click)] if click is not None else []


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pet", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--click-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.prediction) != len(args.label):
        parser.error("provide one --label for each --prediction")
    return args


if __name__ == "__main__":
    main()
