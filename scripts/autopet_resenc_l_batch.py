#!/usr/bin/env python
"""Run one provenance-rich AutoPET III ResEnc-L fold over a prepared cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def image_fingerprint(path: Path, *, include_hash: bool = True) -> dict:
    image = sitk.ReadImage(str(path))
    result = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }
    if include_hash:
        result["sha256"] = sha256(path)
    return result


def score_binary(prediction_path: Path, label_path: Path) -> dict:
    prediction_image = sitk.ReadImage(str(prediction_path))
    label_image = sitk.ReadImage(str(label_path))
    geometry_equal = {
        "size": prediction_image.GetSize() == label_image.GetSize(),
        "spacing": np.allclose(prediction_image.GetSpacing(), label_image.GetSpacing(), atol=1e-6),
        "origin": np.allclose(prediction_image.GetOrigin(), label_image.GetOrigin(), atol=1e-5),
        "direction": np.allclose(prediction_image.GetDirection(), label_image.GetDirection(), atol=1e-6),
    }
    if not all(geometry_equal.values()):
        raise RuntimeError(f"Prediction/label geometry mismatch: {geometry_equal}")
    pred = sitk.GetArrayViewFromImage(prediction_image) > 0
    target = sitk.GetArrayViewFromImage(label_image) > 0
    pred_voxels = int(pred.sum())
    target_voxels = int(target.sum())
    intersection = int(np.logical_and(pred, target).sum())
    union = pred_voxels + target_voxels - intersection
    denominator = pred_voxels + target_voxels
    return {
        "geometry_equal": geometry_equal,
        "prediction_voxels": pred_voxels,
        "target_voxels": target_voxels,
        "intersection_voxels": intersection,
        "dice": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
        "iou": 1.0 if union == 0 else intersection / union,
        "precision": 1.0 if pred_voxels == 0 and target_voxels == 0 else (
            0.0 if pred_voxels == 0 else intersection / pred_voxels
        ),
        "recall": 1.0 if target_voxels == 0 else intersection / target_voxels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    channel0 = sorted(args.input_dir.glob("*_0000.nii.gz"))
    cases = [path.name[:-12] for path in channel0]
    report: dict = {
        "schema_version": 1,
        "started_at_utc": iso_now(),
        "status": "RUNNING",
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "execution": {
            "folds": [args.fold],
            "checkpoint": args.checkpoint,
            "tta": False,
            "perform_everything_on_device": False,
            "preprocessing_processes": 1,
            "export_processes": 1,
            "overwrite": True,
        },
        "cases_requested": cases,
    }
    started = time.perf_counter()
    try:
        if not cases:
            raise ValueError(f"no *_0000.nii.gz cases found in {args.input_dir}")
        for case in cases:
            for path in (
                args.input_dir / f"{case}_0000.nii.gz",
                args.input_dir / f"{case}_0001.nii.gz",
                args.label_dir / f"{case}.nii.gz",
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        from nnunetv2 import __file__ as nnunet_module_file
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        report["nnunet_module"] = str(Path(nnunet_module_file).resolve())
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
        checkpoint_path = args.model_dir / f"fold_{args.fold}" / args.checkpoint
        report["checkpoint"] = {
            "path": str(checkpoint_path.resolve()),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
        }

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device("cuda"),
            verbose=False,
            allow_tqdm=False,
            verbose_preprocessing=False,
        )
        init_started = time.perf_counter()
        predictor.initialize_from_trained_model_folder(
            str(args.model_dir), [int(args.fold) if args.fold != "all" else "all"], args.checkpoint
        )
        report["model_initialization_seconds"] = time.perf_counter() - init_started
        inference_started = time.perf_counter()
        predictor.predict_from_files(
            str(args.input_dir),
            str(args.output_dir),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        torch.cuda.synchronize()
        report["inference_seconds"] = time.perf_counter() - inference_started
        report["gpu"].update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )

        case_reports = []
        for case in cases:
            prediction_path = args.output_dir / f"{case}.nii.gz"
            label_path = args.label_dir / f"{case}.nii.gz"
            case_reports.append(
                {
                    "case_id": case,
                    "inputs": [
                        image_fingerprint(args.input_dir / f"{case}_0000.nii.gz"),
                        image_fingerprint(args.input_dir / f"{case}_0001.nii.gz"),
                    ],
                    "label": image_fingerprint(label_path),
                    "prediction": image_fingerprint(prediction_path),
                    "metrics": score_binary(prediction_path, label_path),
                }
            )
        report["cases"] = case_reports
        report["status"] = "PASS"
        exit_code = 0
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    finally:
        report["finished_at_utc"] = iso_now()
        report["elapsed_seconds"] = time.perf_counter() - started
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "cases": len(report.get("cases", [])),
            "report": str(args.report.resolve()),
            "elapsed_seconds": report["elapsed_seconds"],
        }))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
