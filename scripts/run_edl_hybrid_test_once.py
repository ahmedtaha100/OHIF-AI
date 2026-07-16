#!/usr/bin/env python
"""Fail-closed one-shot executor for the frozen EDL fusion hybrid test.

This entry point is deliberately the only operation authorized by the hybrid
test clearance.  It validates *non-test* frozen inputs first, atomically claims
the receipt paths already frozen by the original route-policy deployment, and
only then delegates the label-derived prompt/test workflow.  The module is
also dependency-injectable so its ordering and failure semantics can be tested
without resolving a real test identifier, path, image, or ground truth.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import importlib
import json
import os
import re
import secrets
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


OPERATION = "execute-edl-hybrid-test-once"
SCHEMA_VERSION = 6
PROTOCOL_REVISION = 7
FREEZE_SCHEMA_VERSION = 1
CLEARANCE_NAMES = ("test_opening_clearance.json", "test_open_clearance.json")
ATTEMPT_NAME = "test_open_attempt_receipt.json"
COMPLETION_NAME = "test_open_completion_receipt.json"
FAILURE_NAME = "test_open_failure_receipt.json"
TEST_MANIFEST_NAME = "fusion_only_test_open_manifest.json"
SCORE_DIRECTORY_NAME = "edl_hybrid_test_score"
REPORT_NAME = "edl_hybrid_test_report.json"
HEX64 = re.compile(r"[0-9a-f]{64}")
REQUIRED_ARTIFACT_ROLES = {
    "hybrid_protocol_contract",
    "base_contract",
    "base_safety_amendment",
    "development_manifest",
    "failed_deployment",
    "failed_selector_bundle",
    "reuse_provenance",
    "original_test_seal",
    "amended_test_seal",
    "hybrid_development_features",
    "hybrid_development_report",
    "pure_screen_policy",
    "hybrid_policy",
    "hybrid_edl_checkpoint",
    "hybrid_code_inventory",
    "hybrid_deployment",
}
REQUIRED_CODE_ROLES = {
    "hybrid_policy_module",
    "hybrid_freeze_cli",
    "hybrid_policy_tests",
    "hybrid_test_orchestrator",
    "hybrid_orchestrator_tests",
    "fusion_only_runner",
    "fusion_only_finalizer",
    "route_policy_eval",
    "prompt_update_edl",
}


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hex(value: object, field: str) -> str:
    text = str(value)
    if not HEX64.fullmatch(text):
        raise RuntimeError(f"{field} must be 64 lowercase hexadecimal characters")
    return text


def _read_json(path: Path, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{field} is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{field} must contain a JSON object")
    return payload


def _read_hashed_clearance(root: Path) -> tuple[dict[str, Any], Path, str]:
    paths = [root / name for name in CLEARANCE_NAMES]
    observed: list[str] = []
    for path in paths:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise RuntimeError(
                f"hybrid test remains sealed: clearance or sidecar absent: {path}"
            )
        digest = sha256_file(path)
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        if digest != expected:
            raise RuntimeError(f"hybrid clearance hash mismatch: {path}")
        observed.append(digest)
    if paths[0].read_bytes() != paths[1].read_bytes() or observed[0] != observed[1]:
        raise RuntimeError("canonical and short hybrid clearances differ")
    return _read_json(paths[0], "hybrid clearance"), paths[0], observed[0]


def _resolve_fingerprint(
    payload: Mapping[str, Any], field: str, *, require_sidecar: bool
) -> tuple[Path, str]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{field} fingerprint is absent")
    declared_path = Path(str(payload.get("path", "")))
    if not declared_path.is_absolute():
        raise RuntimeError(f"{field}.path must be absolute")
    path = declared_path.resolve()
    expected = _require_hex(payload.get("sha256"), f"{field}.sha256")
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"{field} artifact/hash mismatch")
    if require_sidecar:
        declared_sidecar = Path(str(payload.get("sidecar_path", "")))
        if not declared_sidecar.is_absolute():
            raise RuntimeError(f"{field}.sidecar_path must be absolute")
        sidecar = declared_sidecar.resolve()
        sidecar_hash = _require_hex(
            payload.get("sidecar_sha256"), f"{field}.sidecar_sha256"
        )
        if not sidecar.is_file() or sha256_file(sidecar) != sidecar_hash:
            raise RuntimeError(f"{field} sidecar/hash mismatch")
        tokens = sidecar.read_text(encoding="utf-8").split()
        if not tokens or tokens[0] != expected:
            raise RuntimeError(f"{field} sidecar does not bind artifact hash")
    return path, expected


def _contract_roles(contract: Mapping[str, Any], key: str) -> set[str]:
    execution = contract.get("execution_contract")
    if not isinstance(execution, Mapping):
        raise RuntimeError("hybrid protocol execution_contract is absent")
    values = execution.get(key)
    if not isinstance(values, list) or not values or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise RuntimeError(f"hybrid protocol {key} is invalid")
    return set(values)


def _expand_root_template(value: object, root: Path, field: str) -> Path:
    """Expand only the literal ``<root>/`` prefix and reject every escape."""

    text = str(value)
    prefix = "<root>/"
    if not text.startswith(prefix):
        raise RuntimeError(f"{field} must start with the exact {prefix!r} template")
    suffix = text[len(prefix) :]
    if not suffix or "\\" in suffix or "<" in suffix or ">" in suffix:
        raise RuntimeError(f"{field} contains a forbidden template or separator")
    relative = Path(suffix)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"{field} contains traversal or an absolute override")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"{field} escapes the frozen experiment root") from exc
    return resolved


def _expected_artifact_paths(root: Path) -> dict[str, Path]:
    development = root / "edl_hybrid_development_freeze"
    return {
        "hybrid_protocol_contract": root
        / f"edl_hybrid_post_failure_amendment_v{PROTOCOL_REVISION}.json",
        "base_contract": root / "fusion_only_v2_contract.json",
        "base_safety_amendment": root
        / "fusion_only_v2_safety_amendment_v1.json",
        "development_manifest": root / "fusion_only_development_manifest.json",
        "failed_deployment": root
        / "route_policy_development_freeze"
        / "route_policy_deployment.json",
        "failed_selector_bundle": root / "fusion_only_selector_bundle.json",
        "reuse_provenance": root / "reuse_provenance_binding_v1.json",
        "original_test_seal": root / "test_seal.json",
        "amended_test_seal": root / "test_seal_amendment_v1.json",
        "hybrid_development_features": development
        / "edl_hybrid_development_features.json",
        "hybrid_development_report": development
        / "edl_hybrid_development_report.json",
        "pure_screen_policy": development / "pure_consensus_uptake_policy.json",
        "hybrid_policy": development / "edl_fixed_route_hybrid_policy.json",
        "hybrid_edl_checkpoint": development / "edl_fixed_route_hybrid.pt",
        "hybrid_code_inventory": development / "edl_hybrid_code_inventory.json",
        "hybrid_deployment": development / "edl_hybrid_deployment.json",
    }


def _expected_code_paths() -> dict[str, Path]:
    repo = Path(__file__).resolve().parents[1]
    return {
        "hybrid_policy_module": repo / "rl_nninteractive" / "edl_fusion_hybrid.py",
        "hybrid_freeze_cli": repo / "scripts" / "freeze_edl_fusion_hybrid.py",
        "hybrid_policy_tests": repo / "tests" / "test_edl_fusion_hybrid.py",
        "hybrid_test_orchestrator": Path(__file__).resolve(),
        "hybrid_orchestrator_tests": repo
        / "tests"
        / "test_run_edl_hybrid_test_once.py",
        "fusion_only_runner": repo / "scripts" / "run_fusion_only_cohort_v2.py",
        "fusion_only_finalizer": repo
        / "scripts"
        / "finalize_fusion_only_cohort_v2.py",
        "route_policy_eval": repo / "rl_nninteractive" / "route_policy_eval.py",
        "prompt_update_edl": repo / "rl_nninteractive" / "prompt_update_edl.py",
    }


@dataclass(frozen=True)
class ReceiptPaths:
    attempt: Path
    completion: Path
    failure: Path


@dataclass(frozen=True)
class Preflight:
    root: Path
    clearance: Mapping[str, Any]
    clearance_path: Path
    clearance_sha256: str
    protocol: Mapping[str, Any]
    artifacts: Mapping[str, Path]
    artifact_hashes: Mapping[str, str]
    code_paths: Mapping[str, Path]
    code_hashes: Mapping[str, str]
    receipts: ReceiptPaths


class _TransactionAuthority:
    """Process-local capability; the 256-bit secret is never serialized or logged."""

    __slots__ = ("_attempt_sha256", "_preflight", "_proof", "_secret", "_token")

    def __init__(self, preflight: Preflight, attempt_sha256: str) -> None:
        self._preflight = preflight
        self._attempt_sha256 = attempt_sha256
        self._secret = secrets.token_bytes(32)
        self._token = object()
        self._proof = hmac.new(
            self._secret, attempt_sha256.encode("ascii"), hashlib.sha256
        ).digest()

    @property
    def token(self) -> object:
        return self._token

    def verify(self, candidate: object) -> str:
        if candidate is not self._token or len(self._secret) != 32:
            raise RuntimeError("invalid in-process hybrid transaction capability")
        observed = _require_live_attempt(self._preflight, self._attempt_sha256)
        if observed.get("clearance_sha256") != self._preflight.clearance_sha256:
            raise RuntimeError("attempt receipt/clearance binding drifted")
        expected = hmac.new(
            self._secret, self._attempt_sha256.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(self._proof, expected):
            raise RuntimeError("hybrid transaction capability proof mismatch")
        return self._attempt_sha256

    def close(self) -> None:
        self._secret = b""
        self._proof = b""
        self._token = None


@dataclass
class _GroundTruthLoadAudit:
    physical_loads: dict[Path, int]

    def require_exactly_once(self, expected_studies: int) -> None:
        if len(self.physical_loads) != expected_studies or any(
            count != 1 for count in self.physical_loads.values()
        ):
            raise RuntimeError(
                "one-ground-truth-volume-load-per-study invariant failed: "
                f"{len(self.physical_loads)} paths / {self.physical_loads}"
            )


@contextlib.contextmanager
def _single_ground_truth_volume_loads(root: Path):
    """Cache sealed GT NIfTI objects across both prompt rounds and scoring."""

    nib = importlib.import_module("nibabel")
    original_load = nib.load
    sealed_label_root = (root / "sealed" / "test_labels").resolve()
    cache: dict[Path, Any] = {}
    audit = _GroundTruthLoadAudit(physical_loads={})

    def guarded_load(filename: object, *args: Any, **kwargs: Any):
        try:
            path = Path(os.fspath(filename)).resolve()
            path.relative_to(sealed_label_root)
        except (TypeError, ValueError):
            return original_load(filename, *args, **kwargs)
        if path not in cache:
            cache[path] = original_load(filename, *args, **kwargs)
            audit.physical_loads[path] = audit.physical_loads.get(path, 0) + 1
        return cache[path]

    nib.load = guarded_load
    try:
        yield audit
    finally:
        nib.load = original_load
        cache.clear()


def validate_non_test_preflight(root: Path) -> Preflight:
    """Validate frozen metadata only; never inspect a test ID/path/manifest/image."""

    root = root.resolve()
    clearance, clearance_path, clearance_hash = _read_hashed_clearance(root)
    if clearance.get("artifact_type") != "edl_hybrid_test_opening_clearance":
        raise RuntimeError("unexpected hybrid clearance artifact_type")
    if clearance.get("status") != "AUDIT_CLEARED":
        raise RuntimeError("hybrid clearance status is not AUDIT_CLEARED")
    if clearance.get("allowed_operations") != [OPERATION]:
        raise RuntimeError("clearance must authorize only the one-shot hybrid operation")

    audit = clearance.get("independent_audit")
    if not isinstance(audit, Mapping) or audit.get("status") != "PASS":
        raise RuntimeError("independent hybrid audit is not PASS")
    if not audit.get("reviewer") or not audit.get("reviewed_at"):
        raise RuntimeError("independent hybrid audit identity/timestamp is absent")
    audit_path = Path(str(audit.get("report_path", ""))).resolve()
    audit_hash = _require_hex(audit.get("report_sha256"), "audit.report_sha256")
    if not audit_path.is_file() or sha256_file(audit_path) != audit_hash:
        raise RuntimeError("independent hybrid audit report/hash mismatch")
    audit_payload = _read_json(audit_path, "independent hybrid audit report")
    if audit_payload.get("status") != "PASS":
        raise RuntimeError("independent hybrid audit report is not PASS")
    if audit_payload.get("reviewer") != audit.get("reviewer") or audit_payload.get(
        "reviewed_at"
    ) != audit.get("reviewed_at"):
        raise RuntimeError("clearance/audit identity or timestamp mismatch")

    raw_artifacts = clearance.get("artifact_bindings")
    if not isinstance(raw_artifacts, Mapping):
        raise RuntimeError("hybrid clearance artifact_bindings are absent")
    protocol_binding = raw_artifacts.get("hybrid_protocol_contract")
    protocol_path, protocol_hash = _resolve_fingerprint(
        protocol_binding, "artifact_bindings.hybrid_protocol_contract", require_sidecar=True
    )
    protocol = _read_json(protocol_path, "hybrid protocol contract")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"hybrid protocol schema_version must be {SCHEMA_VERSION}"
        )
    if protocol.get("protocol_revision") != PROTOCOL_REVISION:
        raise RuntimeError(
            f"hybrid protocol_revision must be {PROTOCOL_REVISION}"
        )
    if (
        protocol.get("artifact_type")
        != "post_failure_exploratory_edl_fixed_route_hybrid_protocol"
    ):
        raise RuntimeError("unexpected hybrid protocol artifact_type")
    if protocol.get("status") != "POST_FAILURE_EXPLORATORY_PROTOCOL_FROZEN":
        raise RuntimeError("hybrid protocol is not frozen")
    execution = protocol.get("execution_contract")
    if not isinstance(execution, Mapping):
        raise RuntimeError("hybrid protocol execution contract is absent")
    expected_execution = {
        "operation": OPERATION,
        "entrypoint": (
            "scripts/run_edl_hybrid_test_once.py --phase "
            "execute-edl-hybrid-test-once --root <frozen-root>"
        ),
        "pass_limit": 1,
        "failed_attempt_consumes_pass": True,
        "failure_receipt_on_any_post_claim_exception": True,
        "attempt_before_any_test_resolution": True,
        "both_policies_same_gt_load": True,
        "one_ground_truth_volume_load_per_study": True,
        "in_process_capability": (
            "random 256-bit process-secret closure token plus exact attempt-receipt "
            "SHA; never persisted/logged"
        ),
        "direct_legacy_test_entrypoints_forbidden": True,
        "alternate_output_directory_forbidden": True,
        "alternate_receipt_paths_forbidden": True,
        "preclaim_frozen_metadata_exception": (
            "Before atomic receipt claim, code may stat, read, or opaque-SHA-256-hash "
            "already-frozen seal and clearance metadata files and compare literal "
            "canonical output strings; it must not semantically parse or enumerate "
            "test identifiers, construct or resolve an underlying test-data path, "
            "parse a test manifest, or access ground truth, prompts, images, or outcomes."
        ),
        "postclaim_test_semantics": (
            "All semantic test-identifier, test-path, and test-manifest parsing and "
            "every ground-truth, prompt, image, or outcome access remains strictly "
            "after atomic receipt creation."
        ),
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise RuntimeError(f"hybrid protocol execution semantic mismatch: {key}")
    clearance_schema = execution.get("clearance_schema_version")
    receipt_schema = execution.get("receipt_schema_version")
    if not isinstance(clearance_schema, int) or clearance.get(
        "schema_version"
    ) != clearance_schema:
        raise RuntimeError("clearance schema does not match the frozen protocol")
    if not isinstance(receipt_schema, int) or receipt_schema < 1:
        raise RuntimeError("frozen receipt schema_version is invalid")

    required_artifacts = _contract_roles(protocol, "required_artifact_roles")
    if required_artifacts != REQUIRED_ARTIFACT_ROLES:
        raise RuntimeError("required artifact role inventory drift")
    if set(raw_artifacts) != required_artifacts:
        raise RuntimeError("hybrid clearance artifact binding inventory is not exact")
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for role, fingerprint in raw_artifacts.items():
        path, digest = _resolve_fingerprint(
            fingerprint,
            f"artifact_bindings.{role}",
            require_sidecar=True,
        )
        artifact_paths[str(role)] = path
        artifact_hashes[str(role)] = digest
    expected_artifacts = {
        role: path.resolve() for role, path in _expected_artifact_paths(root).items()
    }
    if artifact_paths != expected_artifacts:
        raise RuntimeError("hybrid artifact paths differ from canonical locations")
    if artifact_paths["hybrid_protocol_contract"] != protocol_path:
        raise RuntimeError("hybrid protocol binding drift")
    if artifact_hashes["hybrid_protocol_contract"] != protocol_hash:
        raise RuntimeError("hybrid protocol hash binding drift")

    raw_code = clearance.get("code_hashes")
    required_code = _contract_roles(protocol, "required_code_roles")
    if required_code != REQUIRED_CODE_ROLES:
        raise RuntimeError("required code role inventory drift")
    if not isinstance(raw_code, Mapping) or set(raw_code) != required_code:
        raise RuntimeError("hybrid clearance code hash inventory is not exact")
    code_paths: dict[str, Path] = {}
    code_hashes: dict[str, str] = {}
    for role, fingerprint in raw_code.items():
        path, digest = _resolve_fingerprint(
            fingerprint, f"code_hashes.{role}", require_sidecar=False
        )
        code_paths[str(role)] = path
        code_hashes[str(role)] = digest
    expected_code = {
        role: path.resolve() for role, path in _expected_code_paths().items()
    }
    if code_paths != expected_code:
        raise RuntimeError("hybrid code paths differ from canonical locations")
    expected_self = Path(__file__).resolve()
    if code_paths.get("hybrid_test_orchestrator") != expected_self:
        raise RuntimeError("clearance is not bound to this hybrid orchestrator path")
    if "hybrid_policy_module" in code_paths:
        hybrid = importlib.import_module("rl_nninteractive.edl_fusion_hybrid")
        if Path(str(getattr(hybrid, "__file__", ""))).resolve() != code_paths[
            "hybrid_policy_module"
        ]:
            raise RuntimeError("imported hybrid module differs from audited code path")
        if not callable(
            getattr(hybrid, "build_label_free_test_rows", None)
        ) or not callable(
            getattr(hybrid, "select_frozen_policy_routes", None)
        ):
            raise RuntimeError("audited hybrid test-facing interfaces are absent")

    parent_bindings = protocol.get("hash_bindings")
    if not isinstance(parent_bindings, Mapping) or set(parent_bindings) != {
        "base_contract",
        "base_safety_amendment",
        "development_manifest",
        "failed_deployment",
        "original_test_seal",
        "amended_test_seal",
        "prompt_update_edl_code",
        "route_policy_eval_code",
    }:
        raise RuntimeError("parent hash binding inventory drift")
    for role in (
        "base_contract",
        "base_safety_amendment",
        "development_manifest",
        "failed_deployment",
        "original_test_seal",
        "amended_test_seal",
    ):
        binding = parent_bindings[role]
        if not isinstance(binding, Mapping) or binding.get("sha256") != artifact_hashes[
            role
        ]:
            raise RuntimeError(f"parent artifact cross-binding mismatch: {role}")
        expected_relative = str(
            artifact_paths[role].relative_to(root).as_posix()
        )
        if binding.get("path") != expected_relative:
            raise RuntimeError(f"parent artifact path drift: {role}")
    for contract_role, code_role in (
        ("prompt_update_edl_code", "prompt_update_edl"),
        ("route_policy_eval_code", "route_policy_eval"),
    ):
        binding = parent_bindings[contract_role]
        if not isinstance(binding, Mapping) or binding.get("sha256") != code_hashes[
            code_role
        ]:
            raise RuntimeError(f"parent code cross-binding mismatch: {contract_role}")
        expected_relative = str(
            code_paths[code_role]
            .relative_to(Path(__file__).resolve().parents[1])
            .as_posix()
        )
        if binding.get("path") != expected_relative:
            raise RuntimeError(f"parent code path drift: {contract_role}")

    inventory = _read_json(
        artifact_paths["hybrid_code_inventory"], "hybrid code inventory"
    )
    if (
        inventory.get("schema_version") != FREEZE_SCHEMA_VERSION
        or inventory.get("code_hashes") != raw_code
    ):
        raise RuntimeError("hybrid code inventory/clearance cross-binding mismatch")

    deployment = _read_json(
        artifact_paths["hybrid_deployment"], "hybrid frozen deployment"
    )
    if (
        deployment.get("schema_version") != SCHEMA_VERSION
        or deployment.get("artifact_type") != "edl_hybrid_frozen_deployment"
        or deployment.get("status") != "FROZEN_BEFORE_TEST_OPENING"
        or deployment.get("test_outcomes_opened") is not False
    ):
        raise RuntimeError("hybrid deployment schema/status/seal mismatch")
    if deployment.get("protocol") != {
        "path": str(protocol_path),
        "sha256": protocol_hash,
    }:
        raise RuntimeError("hybrid deployment/protocol cross-binding mismatch")
    development_roles = {
        "hybrid_development_features",
        "hybrid_development_report",
        "pure_screen_policy",
        "hybrid_policy",
        "hybrid_edl_checkpoint",
        "hybrid_code_inventory",
    }
    expected_development_bindings = {
        role: raw_artifacts[role] for role in development_roles
    }
    if deployment.get("artifact_bindings") != expected_development_bindings:
        raise RuntimeError("hybrid deployment/development artifact binding mismatch")
    if deployment.get("parent_hash_bindings") != parent_bindings:
        raise RuntimeError("hybrid deployment/parent hash binding mismatch")
    if deployment.get("code_hashes") != raw_code:
        raise RuntimeError("hybrid deployment/code inventory mismatch")

    if audit_payload.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("independent audit/protocol hash mismatch")
    if audit_payload.get("artifact_bindings") != raw_artifacts:
        raise RuntimeError("independent audit/artifact inventory mismatch")
    if audit_payload.get("code_hashes") != raw_code:
        raise RuntimeError("independent audit/code inventory mismatch")
    if audit_payload.get("deployment_sha256") != artifact_hashes[
        "hybrid_deployment"
    ]:
        raise RuntimeError("independent audit/deployment hash mismatch")

    failed_deployment = artifact_paths.get("failed_deployment")
    if failed_deployment is None:
        raise RuntimeError("failed_deployment binding is required for canonical receipts")
    deployment = _read_json(failed_deployment, "failed frozen deployment")
    control = deployment.get("test_open_control")
    if not isinstance(control, Mapping) or control.get("pass_limit") != 1:
        raise RuntimeError("failed deployment lacks the frozen one-shot receipt control")
    receipts = ReceiptPaths(
        attempt=Path(str(control.get("attempt_receipt_path", ""))).resolve(),
        completion=Path(str(control.get("completion_receipt_path", ""))).resolve(),
        failure=Path(str(control.get("failure_receipt_path", ""))).resolve(),
    )
    expected_names = (ATTEMPT_NAME, COMPLETION_NAME, FAILURE_NAME)
    for path, name in zip(
        (receipts.attempt, receipts.completion, receipts.failure),
        expected_names,
        strict=True,
    ):
        if path.name != name or path.parent != failed_deployment.parent:
            raise RuntimeError("canonical receipt path differs from frozen deployment")
    if len({receipts.attempt, receipts.completion, receipts.failure}) != 3:
        raise RuntimeError("canonical receipt paths are not distinct")
    if any(path.exists() for path in (receipts.attempt, receipts.completion, receipts.failure)):
        raise RuntimeError("canonical one-shot receipt already exists; test pass is unavailable")

    canonical_outputs = execution.get("canonical_outputs")
    if not isinstance(canonical_outputs, Mapping):
        raise RuntimeError("hybrid protocol canonical_outputs are absent")
    if canonical_outputs.get("test_manifest_path") != (
        f"<root>/{TEST_MANIFEST_NAME}"
    ):
        raise RuntimeError("alternate test manifest path is forbidden")
    if canonical_outputs.get("score_directory") != (
        f"<root>/{SCORE_DIRECTORY_NAME}"
    ):
        raise RuntimeError("alternate score directory is forbidden")

    return Preflight(
        root=root,
        clearance=clearance,
        clearance_path=clearance_path,
        clearance_sha256=clearance_hash,
        protocol=protocol,
        artifacts=artifact_paths,
        artifact_hashes=artifact_hashes,
        code_paths=code_paths,
        code_hashes=code_hashes,
        receipts=receipts,
    )


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Claim a path with O_EXCL and durably persist the receipt."""

    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def claim_attempt(preflight: Preflight) -> str:
    receipt_schema = int(
        preflight.protocol["execution_contract"]["receipt_schema_version"]
    )
    payload = {
        "schema_version": receipt_schema,
        "artifact_type": "edl_hybrid_test_open_attempt_receipt",
        "status": "ATTEMPT_CONSUMED",
        "started_at": iso_now(),
        "operation": OPERATION,
        "pass_limit": 1,
        "clearance_path": str(preflight.clearance_path),
        "clearance_sha256": preflight.clearance_sha256,
        "hybrid_protocol_sha256": preflight.artifact_hashes[
            "hybrid_protocol_contract"
        ],
        "failed_deployment_sha256": preflight.artifact_hashes["failed_deployment"],
        "test_access_started": False,
    }
    _write_exclusive_json(preflight.receipts.attempt, payload)
    return sha256_file(preflight.receipts.attempt)


@dataclass(frozen=True)
class ExecutionHooks:
    stage_round1: Callable[[Preflight, object], Mapping[str, Any]]
    infer_round1: Callable[[Preflight, object], Mapping[str, Any]]
    stage_round2: Callable[[Preflight, object], Mapping[str, Any]]
    infer_round2: Callable[[Preflight, object], Mapping[str, Any]]
    freeze_manifest: Callable[[Preflight, object], Mapping[str, Any]]
    score_both_once: Callable[[Preflight, object, Path], Mapping[str, Any]]


def _require_live_attempt(preflight: Preflight, attempt_sha256: str) -> dict[str, Any]:
    if not preflight.receipts.attempt.is_file():
        raise RuntimeError("canonical attempt receipt disappeared")
    if sha256_file(preflight.receipts.attempt) != attempt_sha256:
        raise RuntimeError("canonical attempt receipt drifted")
    payload = _read_json(preflight.receipts.attempt, "canonical attempt receipt")
    if payload.get("status") != "ATTEMPT_CONSUMED":
        raise RuntimeError("canonical attempt receipt is not consumed")
    return payload


def _canonical_output_paths(
    preflight: Preflight,
    authority: _TransactionAuthority,
    capability: object,
) -> tuple[Path, Path]:
    """Construct canonical test paths only after receipt capability verification."""

    authority.verify(capability)
    outputs = preflight.protocol["execution_contract"]["canonical_outputs"]
    manifest = _expand_root_template(
        outputs["test_manifest_path"],
        preflight.root,
        "execution_contract.canonical_outputs.test_manifest_path",
    )
    score_dir = _expand_root_template(
        outputs["score_directory"],
        preflight.root,
        "execution_contract.canonical_outputs.score_directory",
    )
    return manifest, score_dir


@contextlib.contextmanager
def _legacy_transaction_guard(
    preflight: Preflight,
    authority: _TransactionAuthority,
    capability: object,
):
    """Authorize legacy helpers only inside this already-claimed transaction."""

    repo = Path(__file__).resolve().parents[1]
    scripts = repo / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    runner = importlib.import_module("run_fusion_only_cohort_v2")
    finalizer = importlib.import_module("finalize_fusion_only_cohort_v2")
    original_runner = runner.require_test_opening_clearance
    original_finalizer = finalizer.require_test_opening_clearance

    allowed_internal = {
        "stage-round1-test",
        "infer-round1-test",
        "stage-round2-test",
        "infer-round2-test",
        "freeze-test-manifest",
    }

    def transaction_guard(root: Path, operation: str) -> dict[str, Any]:
        if Path(root).resolve() != preflight.root or operation not in allowed_internal:
            raise RuntimeError("operation is outside the claimed hybrid transaction")
        attempt_sha256 = authority.verify(capability)
        return {
            "clearance_path": str(preflight.clearance_path),
            "clearance_sha256": preflight.clearance_sha256,
            "operation": OPERATION,
            "attempt_receipt_path": str(preflight.receipts.attempt),
            "attempt_receipt_sha256": attempt_sha256,
            "dev_manifest_path": str(preflight.artifacts["development_manifest"]),
            "dev_manifest_sha256": preflight.artifact_hashes["development_manifest"],
            "selector_bundle_path": str(preflight.artifacts["failed_selector_bundle"]),
        }

    runner.require_test_opening_clearance = transaction_guard
    finalizer.require_test_opening_clearance = transaction_guard
    try:
        yield runner, finalizer
    finally:
        runner.require_test_opening_clearance = original_runner
        finalizer.require_test_opening_clearance = original_finalizer


def _default_hooks(authority: _TransactionAuthority) -> ExecutionHooks:
    def stage(preflight: Preflight, capability: object, round_index: int):
        authority.verify(capability)
        with _legacy_transaction_guard(
            preflight, authority, capability
        ) as (runner, _finalizer):
            return runner.stage_prompt_round(preflight.root, round_index, "test")

    def infer(preflight: Preflight, capability: object, round_index: int):
        authority.verify(capability)
        with _legacy_transaction_guard(
            preflight, authority, capability
        ) as (runner, _finalizer):
            return runner.run_prompt_no_score(
                preflight.root,
                round_index,
                runner.DEFAULT_NNUNET_RESULTS.resolve(),
                "test",
            )

    def freeze_manifest(preflight: Preflight, capability: object):
        authority.verify(capability)
        with _legacy_transaction_guard(
            preflight, authority, capability
        ) as (_runner, finalizer):
            result = finalizer.freeze_test_open(
                preflight.root, preflight.artifacts["failed_deployment"]
            )
        if result.get("records") != 48:
            raise RuntimeError("hybrid transaction did not freeze exactly 48 proposals")
        path = Path(str(result.get("path", ""))).resolve()
        expected_manifest, _score_dir = _canonical_output_paths(
            preflight, authority, capability
        )
        if path != expected_manifest:
            raise RuntimeError("hybrid transaction produced an alternate test manifest")
        return result

    return ExecutionHooks(
        stage_round1=lambda p, a: stage(p, a, 1),
        infer_round1=lambda p, a: infer(p, a, 1),
        stage_round2=lambda p, a: stage(p, a, 2),
        infer_round2=lambda p, a: infer(p, a, 2),
        freeze_manifest=freeze_manifest,
        score_both_once=lambda p, c, m: _score_both_frozen_policies_once(
            p, c, m, authority=authority
        ),
    )


def _score_both_frozen_policies_once(
    preflight: Preflight,
    capability: object,
    manifest: Path,
    *,
    authority: _TransactionAuthority,
) -> Mapping[str, Any]:
    """Select label-free routes, then load GT once and score both fixed policies."""

    attempt_sha256 = authority.verify(capability)
    hybrid = importlib.import_module("rl_nninteractive.edl_fusion_hybrid")
    builder = getattr(hybrid, "build_label_free_test_rows", None)
    selector = getattr(hybrid, "select_frozen_policy_routes", None)
    if not callable(builder) or not callable(selector):
        raise RuntimeError("audited hybrid label-free builder/selector interface is absent")
    manifest_sha256 = sha256_file(manifest)
    rows = builder(manifest, expected_manifest_sha256=manifest_sha256)
    selections = selector(
        rows,
        pure_policy=preflight.artifacts["pure_screen_policy"],
        edl_checkpoint=preflight.artifacts["hybrid_edl_checkpoint"],
    )
    if not isinstance(selections, Mapping):
        raise RuntimeError("hybrid selector returned an invalid route map")

    evaluator = importlib.import_module("rl_nninteractive.route_policy_eval")
    candidates, manifest_payload = evaluator.load_route_manifest(
        manifest,
        nsd_tolerance_mm=2.0,
        nsd_weight=0.0,
        interaction_cost=0.0,
        accept_margin=0.0,
        exact_splits=("test",),
    )
    groups = evaluator._group_by_case(candidates)
    if len(groups) != 12 or set(groups) != set(selections):
        raise RuntimeError("hybrid route map/test manifest case set mismatch")

    policies = (
        ("edl_fixed_route_hybrid_v1", "edl_hybrid_route"),
        ("pure_zero_harm_consensus_plus_uptake_v1", "pure_screen_route"),
    )
    per_study: list[dict[str, Any]] = []
    per_patient: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for policy_name, selection_key in policies:
        study_rows: list[dict[str, Any]] = []
        for case_id in sorted(groups):
            selection = selections[case_id]
            if not isinstance(selection, Mapping):
                raise RuntimeError(f"invalid frozen selection payload for {case_id}")
            route = str(selection.get(selection_key, ""))
            lookup = evaluator._route_lookup(groups[case_id])
            if route == "KEEP":
                choice = None
            elif route in lookup:
                choice = lookup[route]
            else:
                raise RuntimeError(f"invalid frozen route for {case_id}: {route}")
            score = selection
            study_rows.append(
                evaluator._choice_row(
                    policy_name,
                    groups[case_id][0],
                    choice,
                    confidence=score.get("confidence"),
                    p_accept=score.get("p_accept"),
                    vacuity=score.get("vacuity"),
                    predicted_utility=score.get("predicted_utility"),
                )
            )
        patient_rows = evaluator._patient_rows(policy_name, study_rows)
        summaries[policy_name] = evaluator._policy_summary(
            study_rows, patient_rows, 10_000, 20260715
        )
        per_study.extend(study_rows)
        per_patient.extend(patient_rows)

    def safety_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
        patient = summary["patient_estimand"]
        interval = patient["paired_bootstrap_95_ci_delta_dice"]
        harmful = int(summary["harmful_actions"])
        lower = float(interval["lower"])
        return {
            "maximum_harmful_studies": 0,
            "observed_harmful_studies": harmful,
            "harmful_study_rate_max": 0.05,
            "observed_harmful_study_rate": float(
                summary["harmful_action_rate_all_studies"]
            ),
            "patient_bootstrap_95_ci_lower": lower,
            "requires_strictly_positive_lower_bound": True,
            "passed": harmful == 0 and lower > 0.0,
        }

    primary_name = "edl_fixed_route_hybrid_v1"
    secondary_name = "pure_zero_harm_consensus_plus_uptake_v1"
    gate_results = {
        primary_name: safety_gate(summaries[primary_name]),
        secondary_name: safety_gate(summaries[secondary_name]),
    }
    for policy_name, gate in gate_results.items():
        summaries[policy_name]["passes_frozen_gate"] = gate["passed"]
        summaries[policy_name]["maximum_allowed_harm_count"] = 0
        summaries[policy_name]["endpoint_role"] = (
            "primary_decision_endpoint"
            if policy_name == primary_name
            else "secondary_descriptive_only"
        )

    _manifest_path, score_dir = _canonical_output_paths(
        preflight, authority, capability
    )
    score_dir.mkdir(parents=False, exist_ok=False)
    report = {
        "schema_version": int(
            preflight.protocol["execution_contract"].get("report_schema_version", 1)
        ),
        "artifact_type": "edl_hybrid_frozen_test_report",
        "generated_at": iso_now(),
        "status": "EXPLORATORY_INTERNAL_PRIOR_EXPOSED",
        "claim_boundary": "Sealed prior-exposed exploratory evidence; no external, clinical, or confirmatory efficacy claim.",
        "test_label_evaluation_passes": 1,
        "both_frozen_policies_same_gt_load": True,
        "one_ground_truth_volume_load_per_study_enforced_by_outer_transaction": True,
        "attempt_receipt_sha256": attempt_sha256,
        "clearance_sha256": preflight.clearance_sha256,
        "protocol_sha256": preflight.artifact_hashes["hybrid_protocol_contract"],
        "manifest": {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "record_count": len(candidates),
            "source_status": manifest_payload.get("status"),
        },
        "policies": summaries,
        "safety_gate_results": gate_results,
        "fixed_sequence_interpretation": {
            "primary_policy": primary_name,
            "secondary_policy": secondary_name,
            "primary_passed": gate_results[primary_name]["passed"],
            "secondary_is_descriptive_only": True,
            "secondary_cannot_rescue_failed_primary": True,
        },
        "frozen_route_choices": selections,
        "per_study": per_study,
        "per_patient": per_patient,
    }
    report_path = score_dir / REPORT_NAME
    _write_exclusive_json(report_path, report)
    return {
        "status": report["status"],
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "test_label_evaluation_passes": 1,
    }


def execute_once(root: Path, hooks: ExecutionHooks | None = None) -> Mapping[str, Any]:
    """Execute the single transaction; preflight failures never claim/open test."""

    preflight = validate_non_test_preflight(root)
    attempt_sha = claim_attempt(preflight)
    authority: _TransactionAuthority | None = None
    try:
        authority = _TransactionAuthority(preflight, attempt_sha)
        capability = authority.token
        expected_manifest, expected_score_dir = _canonical_output_paths(
            preflight, authority, capability
        )
        if expected_manifest.exists() or expected_score_dir.exists():
            raise RuntimeError(
                "stale canonical test output exists after pass consumption; refusing overwrite"
            )
        production_hooks = hooks is None
        active_hooks = hooks or _default_hooks(authority)
        load_context = (
            _single_ground_truth_volume_loads(preflight.root)
            if production_hooks
            else contextlib.nullcontext(None)
        )
        with load_context as ground_truth_audit:
            authority.verify(capability)
            active_hooks.stage_round1(preflight, capability)
            authority.verify(capability)
            active_hooks.infer_round1(preflight, capability)
            authority.verify(capability)
            active_hooks.stage_round2(preflight, capability)
            authority.verify(capability)
            active_hooks.infer_round2(preflight, capability)
            authority.verify(capability)
            frozen = active_hooks.freeze_manifest(preflight, capability)
            manifest = Path(str(frozen.get("path", ""))).resolve()
            if manifest != expected_manifest:
                raise RuntimeError("alternate test manifest path is forbidden")
            if frozen.get("records") != 48:
                raise RuntimeError("test manifest must contain exactly 48 proposals")
            authority.verify(capability)
            scored = active_hooks.score_both_once(preflight, capability, manifest)
            if scored.get("test_label_evaluation_passes") != 1:
                raise RuntimeError(
                    "dual-policy scorer did not certify one GT evaluation pass"
                )
            if ground_truth_audit is not None:
                ground_truth_audit.require_exactly_once(12)
        report_path = Path(str(scored.get("report_path", ""))).resolve()
        if report_path.parent != expected_score_dir:
            raise RuntimeError("dual-policy scorer used an alternate output directory")
        receipt_schema = int(
            preflight.protocol["execution_contract"]["receipt_schema_version"]
        )
        completion = {
            "schema_version": receipt_schema,
            "artifact_type": "edl_hybrid_test_open_completion_receipt",
            "status": "COMPLETED",
            "finished_at": iso_now(),
            "attempt_receipt_sha256": attempt_sha,
            "clearance_sha256": preflight.clearance_sha256,
            "test_manifest_path": str(manifest),
            "test_manifest_sha256": sha256_file(manifest),
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "test_label_evaluation_passes": 1,
            "policies_scored": [
                "edl_fixed_route_hybrid_v1",
                "pure_zero_harm_consensus_plus_uptake_v1",
            ],
        }
        _write_exclusive_json(preflight.receipts.completion, completion)
        result = {
            "status": "COMPLETED",
            "attempt_receipt": str(preflight.receipts.attempt),
            "attempt_receipt_sha256": attempt_sha,
            "completion_receipt": str(preflight.receipts.completion),
            "completion_receipt_sha256": sha256_file(
                preflight.receipts.completion
            ),
            "test_manifest": str(manifest),
            "test_manifest_sha256": sha256_file(manifest),
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
        }
    except BaseException as exc:
        receipt_schema = int(
            preflight.protocol["execution_contract"]["receipt_schema_version"]
        )
        failure = {
            "schema_version": receipt_schema,
            "artifact_type": "edl_hybrid_test_open_failure_receipt",
            "status": "FAILED_ATTEMPT_CONSUMED",
            "finished_at": iso_now(),
            "attempt_receipt_sha256": attempt_sha,
            "clearance_sha256": preflight.clearance_sha256,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_exclusive_json(preflight.receipts.failure, failure)
        raise
    finally:
        if authority is not None:
            authority.close()
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(OPERATION,), required=True)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = execute_once(args.root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
