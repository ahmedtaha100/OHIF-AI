"""Create one deterministic foreground/background correction from segmentation errors.

Ground truth is used only as a robot-user simulator, matching an interactive
evaluation harness. It is never an input channel to the segmentation model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def main() -> None:
    args = _parse_args()
    source_dir = args.source_input.resolve()
    output_dir = args.output_input.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_path = _one(source_dir, "*_0000.nii.gz")
    pet_path = _one(source_dir, "*_0001.nii.gz")
    stem = ct_path.name[: -len("_0000.nii.gz")]
    ct_out = output_dir / f"{stem}_0000.nii.gz"
    pet_out = output_dir / f"{stem}_0001.nii.gz"
    shutil.copy2(ct_path, ct_out)
    shutil.copy2(pet_path, pet_out)

    reference = nib.load(str(pet_path))
    prediction = np.asarray(nib.load(str(args.prediction)).dataobj) > 0
    ground_truth = np.asarray(nib.load(str(args.ground_truth)).dataobj) > 0
    if prediction.shape != reference.shape or ground_truth.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: reference={reference.shape}, prediction={prediction.shape}, "
            f"ground_truth={ground_truth.shape}"
        )

    false_negative = ground_truth & ~prediction
    false_positive = prediction & ~ground_truth
    foreground = _deepest_largest_component(false_negative)
    background = _deepest_largest_component(false_positive)

    fg_map = _prior_heatmap(source_dir, "*_0002.nii.gz", reference.shape)
    bg_map = _prior_heatmap(source_dir, "*_0003.nii.gz", reference.shape)
    previous_foreground_clicks = int(np.count_nonzero(fg_map))
    previous_background_clicks = int(np.count_nonzero(bg_map))
    if foreground is not None:
        fg_map[foreground] = 1.0
    if background is not None:
        bg_map[background] = 1.0

    fg_path = output_dir / f"{stem}_0002.nii.gz"
    bg_path = output_dir / f"{stem}_0003.nii.gz"
    nib.save(nib.Nifti1Image(fg_map, reference.affine, reference.header), str(fg_path))
    nib.save(nib.Nifti1Image(bg_map, reference.affine, reference.header), str(bg_path))

    payload = {
        "schema_version": 1,
        "case": stem,
        "simulator": "deepest-voxel-in-largest-current-error-component",
        "ground_truth_boundary": "robot-user-simulation-only",
        "foreground_xyz": list(foreground) if foreground is not None else None,
        "background_xyz": list(background) if background is not None else None,
        "previous_foreground_clicks": previous_foreground_clicks,
        "previous_background_clicks": previous_background_clicks,
        "cumulative_foreground_clicks": int(np.count_nonzero(fg_map)),
        "cumulative_background_clicks": int(np.count_nonzero(bg_map)),
        "false_negative_voxels": int(false_negative.sum()),
        "false_positive_voxels": int(false_positive.sum()),
        "sources": {
            "prediction_sha256": _sha256(args.prediction),
            "ground_truth_sha256": _sha256(args.ground_truth),
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (ct_out, pet_out, fg_path, bg_path)
        },
    }
    manifest = output_dir.parent / "simulated_error_clicks.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), **{k: payload[k] for k in ("foreground_xyz", "background_xyz", "false_negative_voxels", "false_positive_voxels")}}))


def _deepest_largest_component(mask: np.ndarray) -> tuple[int, int, int] | None:
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    component = components == int(np.argmax(sizes))
    distances = ndimage.distance_transform_edt(component)
    return tuple(int(value) for value in np.unravel_index(int(np.argmax(distances)), mask.shape))


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def _prior_heatmap(root: Path, pattern: str, shape: tuple[int, ...]) -> np.ndarray:
    matches = sorted(root.glob(pattern))
    if not matches:
        return np.zeros(shape, dtype=np.float32)
    if len(matches) != 1:
        raise ValueError(f"expected at most one prior {pattern} under {root}, found {len(matches)}")
    heatmap = np.asarray(nib.load(str(matches[0])).dataobj, dtype=np.float32)
    if heatmap.shape != shape:
        raise ValueError(f"prior heatmap shape mismatch: {heatmap.shape} != {shape}")
    return heatmap.copy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-input", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
