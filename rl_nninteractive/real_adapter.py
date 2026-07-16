"""Real nnInteractive smoke-test helpers.

This module keeps heavyweight imports inside functions so the default scaffold
can run without torch, nibabel, huggingface_hub, or nnInteractive installed.
The smoke path verifies wiring only; it is not a benchmark or clinical result.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import Box3D, VoxelCoord, as_voxel_coord

DEFAULT_HF_REPO_ID = "MIC-DKFZ/nnInteractive"
DEFAULT_MODEL_NAME = "nnInteractive_v1.0"
DEFAULT_CHECKPOINT_NAME = "checkpoint_final.pth"
CHECKPOINT_LICENSE = "CC BY-NC-SA 4.0"


def center_point_for_shape(shape: tuple[int, int, int]) -> VoxelCoord:
    """Return the integer center point for a 3D array shape."""
    return as_voxel_coord(tuple(dim // 2 for dim in shape))


def parse_point(text: str) -> VoxelCoord:
    """Parse a `z,y,x` point string."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("point must be formatted as z,y,x")
    return as_voxel_coord(tuple(int(part) for part in parts))


def normalize_changed_bbox(changed_bbox: Any) -> list[list[int]] | None:
    """Normalize nnInteractive's optional changed bbox to JSON-friendly lists."""
    if changed_bbox is None:
        return None
    if len(changed_bbox) != 3:
        raise ValueError("changed_bbox must have three axis ranges")
    normalized: list[list[int]] = []
    for axis_range in changed_bbox:
        if len(axis_range) != 2:
            raise ValueError("changed_bbox axis ranges must have two values")
        normalized.append([int(axis_range[0]), int(axis_range[1])])
    return normalized


def box3d_from_changed_bbox(changed_bbox: Any) -> Box3D | None:
    """Convert a changed bbox from the real package to the local Box3D type."""
    normalized = normalize_changed_bbox(changed_bbox)
    if normalized is None:
        return None
    return (
        (normalized[0][0], normalized[0][1]),
        (normalized[1][0], normalized[1][1]),
        (normalized[2][0], normalized[2][1]),
    )


def find_nibabel_test_image() -> Path:
    """Return the public Nibabel anatomical test NIfTI path."""
    try:
        import nibabel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nibabel is required for --use-nibabel-test-image; run `make setup-real`."
        ) from exc

    image_path = Path(nibabel.__file__).resolve().parent / "tests" / "data" / "anatomical.nii"
    if not image_path.exists():
        raise FileNotFoundError(f"Nibabel test image not found: {image_path}")
    return image_path


def load_nifti_image(path: Path) -> np.ndarray:
    """Load a 3D NIfTI as float32 with no resampling or normalization."""
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise RuntimeError("nibabel is required to load NIfTI images; run `make setup-real`.") from exc

    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"real smoke image must be a 3D volume, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("real smoke image contains NaN or inf values")
    return data


def resolve_model_dir(
    checkpoint_root: Path,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    download_model: bool = False,
) -> Path:
    """Resolve or download the nnInteractive model folder."""
    model_dir = checkpoint_root / model_name
    checkpoint = model_dir / "fold_0" / DEFAULT_CHECKPOINT_NAME
    if download_model or not checkpoint.exists():
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download the checkpoint; run `make setup-real`."
            ) from exc
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{model_name}/*"],
            local_dir=str(checkpoint_root),
        )
    if not checkpoint.exists():
        raise FileNotFoundError(f"nnInteractive checkpoint not found: {checkpoint}")
    return model_dir


def _require_real_modules() -> tuple[Any, Any]:
    try:
        import torch
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Real nnInteractive smoke requires optional dependencies. "
            "Run `make setup-real` first."
        ) from exc
    return torch, nnInteractiveInferenceSession


def _package_version(name: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def run_real_point_smoke(
    *,
    image_path: Path,
    checkpoint_root: Path,
    output_dir: Path,
    point: VoxelCoord | None = None,
    repo_id: str = DEFAULT_HF_REPO_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    device_name: str = "auto",
    download_model: bool = False,
    allow_empty_mask: bool = False,
) -> dict[str, Any]:
    """Run one positive point through a real nnInteractive checkpoint."""
    torch, session_cls = _require_real_modules()
    image = load_nifti_image(image_path)
    model_dir = resolve_model_dir(
        checkpoint_root,
        repo_id=repo_id,
        model_name=model_name,
        download_model=download_model,
    )
    resolved_device = "cuda:0" if device_name == "auto" and torch.cuda.is_available() else device_name
    if resolved_device == "auto":
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    point = center_point_for_shape(image.shape) if point is None else point

    output_dir.mkdir(parents=True, exist_ok=True)
    image_4d = image[None]

    started = time.time()
    session = session_cls(
        device=device,
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=max(1, min(8, os.cpu_count() or 1)),
        do_autozoom=True,
    )
    session.initialize_from_trained_model_folder(str(model_dir))
    initialized_sec = time.time() - started

    interaction_started = time.time()
    session.set_image(image_4d)
    session.set_target_buffer(torch.zeros(image.shape, dtype=torch.uint8))
    changed_bbox = session.add_point_interaction(point, include_interaction=True)
    mask = session.target_buffer.clone().cpu().numpy()
    interaction_sec = time.time() - interaction_started

    if mask.shape != image.shape:
        raise RuntimeError(f"mask shape {mask.shape} does not match image shape {image.shape}")
    mask_sum = int(mask.sum())
    if mask_sum <= 0 and not allow_empty_mask:
        raise RuntimeError("real smoke produced an empty mask")

    summary: dict[str, Any] = {
        "status": "real nnInteractive smoke complete",
        "claim": "wiring smoke only; not a benchmark, training run, or clinical result",
        "repo_id": repo_id,
        "model_name": model_name,
        "checkpoint_license": CHECKPOINT_LICENSE,
        "model_dir": str(model_dir),
        "image_source": str(image_path),
        "image_shape": list(image_4d.shape),
        "array_order": "numpy index order treated as z,y,x for smoke wiring",
        "device": str(device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "package_versions": {
            "nninteractive": _package_version("nninteractive"),
            "torch": _package_version("torch"),
            "nibabel": _package_version("nibabel"),
            "huggingface_hub": _package_version("huggingface_hub"),
        },
        "point_zyx": list(point),
        "changed_bbox": normalize_changed_bbox(changed_bbox),
        "changed_box3d": box3d_from_changed_bbox(changed_bbox),
        "mask_shape": list(mask.shape),
        "mask_sum": mask_sum,
        "initialized_sec": round(initialized_sec, 3),
        "interaction_sec": round(interaction_sec, 3),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "mask.npz", mask=mask)
    return summary
