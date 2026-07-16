"""Runtime configuration loading for RL-over-nnInteractive experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    cuda_visible_devices: str
    max_interactions: int
    mock_mode: bool
    nninteractive_endpoint: str | None
    dataset_manifest: str
    output_dir: str

    @property
    def environment(self) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": self.cuda_visible_devices}


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    seed = _require_int(raw, "seed", minimum=0)
    cuda_visible_devices = _require_str(raw, "cuda_visible_devices")
    max_interactions = _require_int(raw, "max_interactions", minimum=1)
    mock_mode = _require_bool(raw, "mock_mode")
    nninteractive_endpoint = _optional_str(raw, "nninteractive_endpoint")
    output_dir = _require_str(raw, "output_dir")
    dataset_manifest = _require_existing_path(raw, "dataset_manifest", config_path)
    return RuntimeConfig(
        seed=seed,
        cuda_visible_devices=cuda_visible_devices,
        max_interactions=max_interactions,
        mock_mode=mock_mode,
        nninteractive_endpoint=nninteractive_endpoint,
        dataset_manifest=str(dataset_manifest),
        output_dir=output_dir,
    )


def _require_int(raw: dict[str, Any], key: str, minimum: int) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _require_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _require_existing_path(raw: dict[str, Any], key: str, config_path: Path) -> Path:
    value = _require_str(raw, key)
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                config_path.parent / path,
                config_path.parent.parent / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{key} does not exist: {value}")


def _require_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value
