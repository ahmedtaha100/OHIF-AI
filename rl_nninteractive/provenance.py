"""Deterministic provenance and cache-identity contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
CACHE_SCHEMA_VERSION = 1


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: str, *, field: str) -> str:
    normalized = str(value).lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class CacheIdentity:
    namespace: str
    case_ids: tuple[str, ...]
    target_label: str
    checkpoint_sha256: str
    config_sha256: str
    dataset_sha256: str

    def __post_init__(self) -> None:
        if not self.namespace or not self.target_label:
            raise ValueError("cache namespace and target_label must be non-empty")
        if not self.case_ids or any(not case_id for case_id in self.case_ids):
            raise ValueError("cache identity requires non-empty case_ids")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("cache identity case_ids must be unique")
        object.__setattr__(self, "checkpoint_sha256", require_sha256(self.checkpoint_sha256, field="checkpoint_sha256"))
        object.__setattr__(self, "config_sha256", require_sha256(self.config_sha256, field="config_sha256"))
        object.__setattr__(self, "dataset_sha256", require_sha256(self.dataset_sha256, field="dataset_sha256"))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["case_ids"] = list(self.case_ids)
        return payload


def make_cache_envelope(identity: CacheIdentity, payload: Any) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "identity": identity.to_dict(),
        "payload": payload,
    }


def unwrap_cache_envelope(envelope: Any, expected: CacheIdentity) -> Any:
    if not isinstance(envelope, Mapping) or envelope.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("cache is legacy or invalid; regenerate it with the provenance-bound schema")
    observed = envelope.get("identity")
    expected_dict = expected.to_dict()
    if not isinstance(observed, Mapping):
        raise ValueError("cache identity is missing")
    mismatches = [key for key, value in expected_dict.items() if observed.get(key) != value]
    if mismatches:
        raise ValueError(f"cache identity mismatch: {', '.join(mismatches)}")
    if "payload" not in envelope:
        raise ValueError("cache payload is missing")
    return envelope["payload"]
