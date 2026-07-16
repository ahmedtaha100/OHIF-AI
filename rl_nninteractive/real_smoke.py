"""CLI for a gated real nnInteractive checkpoint smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .real_adapter import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_MODEL_NAME,
    find_nibabel_test_image,
    parse_point,
    run_real_point_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real nnInteractive checkpoint point-prompt smoke test."
    )
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Acknowledge that this downloads/uses real nnInteractive dependencies and checkpoint files.",
    )
    parser.add_argument("--image", type=Path, help="Path to a 3D public/de-identified NIfTI volume.")
    parser.add_argument(
        "--use-nibabel-test-image",
        action="store_true",
        help="Use Nibabel's bundled public anatomical.nii test fixture as the smoke input.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("artifacts/rl_nninteractive/checkpoints"),
        help="Directory containing or receiving the nnInteractive model folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl_nninteractive/real_smoke"),
        help="Directory for summary.json and mask.npz outputs.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--point", help="Optional z,y,x positive point. Defaults to image center.")
    parser.add_argument("--allow-empty-mask", action="store_true")
    return parser


def _resolve_image(args: argparse.Namespace) -> Path:
    if args.image and args.use_nibabel_test_image:
        raise ValueError("use either --image or --use-nibabel-test-image, not both")
    if args.use_nibabel_test_image:
        return find_nibabel_test_image()
    if args.image:
        if not args.image.exists():
            raise FileNotFoundError(f"image does not exist: {args.image}")
        return args.image
    raise ValueError("real smoke requires --image or --use-nibabel-test-image")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.require_real and os.environ.get("RL_NNINTERACTIVE_REQUIRE_REAL") != "1":
        parser.error("pass --require-real or set RL_NNINTERACTIVE_REQUIRE_REAL=1")

    summary = run_real_point_smoke(
        image_path=_resolve_image(args),
        checkpoint_root=args.checkpoint_root,
        output_dir=args.output_dir,
        point=parse_point(args.point) if args.point else None,
        repo_id=args.repo_id,
        model_name=args.model_name,
        device_name=args.device,
        download_model=args.download_model,
        allow_empty_mask=args.allow_empty_mask,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
