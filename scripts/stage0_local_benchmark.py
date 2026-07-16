#!/usr/bin/env python3
"""Stage-0 native-volume geometry, overlay, and RTX inference benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from rl_nninteractive.autopet_pipeline import (
    AutoPetVolume,
    _representative_coord,
    find_autopet_cases,
    lesion_components,
    load_autopet_volume,
)
from rl_nninteractive.provenance import sha256_file
from rl_nninteractive.real_rollout import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_ROOT,
    _add_point,
    _roi_bounds,
    make_session,
)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = tuple(value.strip() for value in args.case_ids.split(",") if value.strip())
    discovered = {case.case_id: case for case in find_autopet_cases(args.data_root, tracer="ANY")}
    missing = sorted(set(selected_ids) - set(discovered))
    if missing:
        raise ValueError(f"requested case IDs were not discovered: {missing}")

    checkpoint = (
        Path(args.model_root) / args.model_name / "fold_0" / "checkpoint_final.pth"
    ).resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is required unless --allow-cpu is set")

    session_started = time.perf_counter()
    session = make_session(
        device=device,
        model_root=str(Path(args.model_root).resolve()),
        model_name=args.model_name,
    )
    session_init_seconds = time.perf_counter() - session_started

    case_reports = []
    all_prompt_seconds: list[float] = []
    for case_id in selected_ids:
        case = discovered[case_id]
        pet_header = nib.load(str(case.pet))
        native_shape_zyx = tuple(int(value) for value in reversed(pet_header.shape[:3]))
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        volume = load_autopet_volume(case, target=native_shape_zyx)
        preprocessing_seconds = time.perf_counter() - wall_started
        preprocessing_cpu_seconds = time.process_time() - cpu_started

        overlay_path = args.out_dir / f"{case_id}_native_overlay.png"
        _render_overlay(volume, overlay_path)
        prompt_report = _benchmark_prompt(
            session,
            volume,
            repeats=args.prompt_repeats,
            roi_size=args.roi_size,
            device=device,
        )
        all_prompt_seconds.extend(prompt_report["prompt_seconds"])
        geometry = {
            role: {
                "source_shape_xyz": list(metadata.source_shape_xyz),
                "output_shape_xyz": list(metadata.output_shape_xyz),
                "source_orientation": metadata.source_orientation,
                "output_orientation": metadata.output_orientation,
                "source_spacing_xyz": list(metadata.source_spacing_xyz),
                "output_spacing_xyz": list(metadata.output_spacing_xyz),
                "physical_overlap_fraction": metadata.physical_overlap_fraction,
                "center_distance_mm": metadata.center_distance_mm,
                "transform_history": list(metadata.transform_history),
            }
            for role, metadata in volume.geometry.items()
        }
        case_reports.append(
            {
                "case_id": case_id,
                "tracer": case.pet.parent.name,
                "native_shape_zyx": list(native_shape_zyx),
                "lesions": volume.n_lesions,
                "preprocessing_wall_seconds": preprocessing_seconds,
                "preprocessing_cpu_seconds": preprocessing_cpu_seconds,
                "preprocessing_equivalent_cpu_cores": (
                    preprocessing_cpu_seconds / max(preprocessing_seconds, 1e-9)
                ),
                "geometry": geometry,
                "overlay": str(overlay_path.resolve()),
                "prompt_benchmark": prompt_report,
            }
        )

    prompt_array = np.asarray(all_prompt_seconds, dtype=np.float64)
    minimum_overlap = min(
        role["physical_overlap_fraction"]
        for case in case_reports
        for role in case["geometry"].values()
    )
    p95_prompt_seconds = float(np.percentile(prompt_array, 95))
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS" if minimum_overlap >= 0.5 and p95_prompt_seconds < 1.0 else "FAIL",
        "protocol": {
            "case_ids": list(selected_ids),
            "native_pet_reference_grid": True,
            "prompt_repeats_per_case": args.prompt_repeats,
            "roi_size_zyx": [args.roi_size] * 3,
            "prompt_seed": "largest-label-component representative point; latency benchmark only",
            "latency_includes": "reset, set-image, target-buffer initialization, first point prediction",
        },
        "hardware": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "gpu_total_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None
            ),
        },
        "model": {
            "name": args.model_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "session_init_seconds": session_init_seconds,
        },
        "aggregate": {
            "minimum_physical_overlap_fraction": minimum_overlap,
            "prompt_seconds_median": float(median(all_prompt_seconds)),
            "prompt_seconds_p95": p95_prompt_seconds,
            "prompt_seconds_maximum": float(prompt_array.max()),
            "p95_below_one_second": p95_prompt_seconds < 1.0,
            "peak_gpu_memory_bytes": max(
                case["prompt_benchmark"]["peak_gpu_memory_bytes"] for case in case_reports
            ),
        },
        "cases": case_reports,
    }
    report_path = args.out_dir / "stage0_local_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path.resolve()), **report["aggregate"]}))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-ids", default="train_0001_FDG,train_0001_PSMA")
    parser.add_argument("--prompt-repeats", type=int, default=10)
    parser.add_argument("--roi-size", type=int, default=36)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.prompt_repeats < 2:
        parser.error("--prompt-repeats must be at least 2")
    if args.roi_size < 8:
        parser.error("--roi-size must be at least 8")
    return args


def _benchmark_prompt(session, volume: AutoPetVolume, *, repeats: int, roi_size: int, device: str):
    components = lesion_components(volume.gt, min_size=1)
    if components:
        component = max(components, key=lambda value: int(value.sum()))
        coord = _representative_coord(component)
    else:
        coord = tuple(int(value) for value in np.unravel_index(np.argmax(volume.pet), volume.pet.shape))
    bounds = _roi_bounds(volume.pet.shape, coord, (roi_size, roi_size, roi_size))
    slices = tuple(slice(lower, upper) for lower, upper in bounds)
    roi = volume.pet[slices]
    local_coord = tuple(coord[axis] - bounds[axis][0] for axis in range(3))

    prompt_seconds = []
    peak_gpu_memory_bytes = 0
    for repeat in range(repeats + 1):
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        session.reset_interactions()
        session.set_image(roi[None])
        session.set_target_buffer(torch.zeros(roi.shape, dtype=torch.uint8))
        _add_point(session, local_coord, positive=True)
        if device == "cuda":
            torch.cuda.synchronize()
            peak_gpu_memory_bytes = max(
                peak_gpu_memory_bytes, int(torch.cuda.max_memory_allocated())
            )
        elapsed = time.perf_counter() - started
        if repeat > 0:
            prompt_seconds.append(elapsed)
    return {
        "roi_shape_zyx": list(roi.shape),
        "global_seed_zyx": list(coord),
        "prompt_seconds": prompt_seconds,
        "prompt_seconds_median": float(np.median(prompt_seconds)),
        "prompt_seconds_p95": float(np.percentile(prompt_seconds, 95)),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
    }


def _render_overlay(volume: AutoPetVolume, output_path: Path) -> None:
    coords = np.argwhere(volume.gt)
    center = (
        tuple(int(value) for value in np.rint(coords.mean(axis=0)))
        if len(coords)
        else tuple(int(value // 2) for value in volume.gt.shape)
    )
    views = (
        ("Axial", volume.ct[center[0]], volume.pet[center[0]], volume.gt[center[0]]),
        ("Coronal", volume.ct[:, center[1], :], volume.pet[:, center[1], :], volume.gt[:, center[1], :]),
        ("Sagittal", volume.ct[:, :, center[2]], volume.pet[:, :, center[2]], volume.gt[:, :, center[2]]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=150)
    for axis, (title, ct_slice, pet_slice, gt_slice) in zip(axes, views):
        axis.imshow(ct_slice, cmap="gray", vmin=-1.0, vmax=1.0)
        axis.imshow(pet_slice, cmap="magma", vmin=-1.0, vmax=1.0, alpha=0.38)
        if bool(gt_slice.any()):
            axis.contour(gt_slice.astype(np.uint8), levels=[0.5], colors=["#39FF14"], linewidths=0.8)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{volume.case_id} — native PET grid CT/PET/TTB overlay")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
