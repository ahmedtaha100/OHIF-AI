"""Command-line entrypoint for the RL nnInteractive scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize an RL-over-nnInteractive runtime config."
    )
    parser.add_argument(
        "--config",
        default="configs/rl_nninteractive_skeleton.json",
        help="Path to a JSON runtime config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    summary = {
        "cuda_visible_devices": config.cuda_visible_devices,
        "dataset_manifest": config.dataset_manifest,
        "max_interactions": config.max_interactions,
        "mock_mode": config.mock_mode,
        "nninteractive_endpoint": config.nninteractive_endpoint,
        "output_dir": config.output_dir,
        "seed": config.seed,
        "status": "mock scaffold" if config.mock_mode else "real backend requested",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not config.mock_mode and not config.nninteractive_endpoint:
        print(
            "Real nnInteractive mode requires nninteractive_endpoint in the config.",
            file=sys.stderr,
        )
        return 2
    return 0
