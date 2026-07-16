from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_edl_hybrid_test_once as subject  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_sidecar(path: Path) -> Path:
    sidecar = path.parent / f"{path.name}.sha256"
    sidecar.write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.deployment_dir = root / "route_policy_development_freeze"
        self.attempt = self.deployment_dir / subject.ATTEMPT_NAME
        self.completion = self.deployment_dir / subject.COMPLETION_NAME
        self.failure = self.deployment_dir / subject.FAILURE_NAME
        self.artifacts: dict[str, Path] = {}
        self.audit_path = root / "independent_hybrid_audit.json"
        self._build()

    @staticmethod
    def _binding(path: Path) -> dict:
        sidecar = path.parent / f"{path.name}.sha256"
        return {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "sidecar_path": str(sidecar.resolve()),
            "sidecar_sha256": sha256(sidecar),
        }

    def _build(self) -> None:
        expected = subject._expected_artifact_paths(self.root)
        for role, path in expected.items():
            if role in {
                "hybrid_protocol_contract",
                "failed_deployment",
                "hybrid_code_inventory",
                "hybrid_deployment",
                "hybrid_edl_checkpoint",
            }:
                continue
            write_json(path, {"role": role, "status": "FROZEN"})
            add_sidecar(path)
            self.artifacts[role] = path

        failed = expected["failed_deployment"]
        write_json(
            failed,
            {
                "schema_version": 1,
                "status": "FROZEN_DEVELOPMENT",
                "test_open_control": {
                    "pass_limit": 1,
                    "attempt_receipt_path": str(self.attempt.resolve()),
                    "completion_receipt_path": str(self.completion.resolve()),
                    "failure_receipt_path": str(self.failure.resolve()),
                },
            },
        )
        add_sidecar(failed)
        self.artifacts["failed_deployment"] = failed

        checkpoint = expected["hybrid_edl_checkpoint"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"frozen-hybrid-edl-checkpoint")
        add_sidecar(checkpoint)
        self.artifacts["hybrid_edl_checkpoint"] = checkpoint

        code_hashes = {
            role: {"path": str(path.resolve()), "sha256": sha256(path)}
            for role, path in subject._expected_code_paths().items()
        }
        inventory = expected["hybrid_code_inventory"]
        write_json(
            inventory,
            {
                "schema_version": subject.FREEZE_SCHEMA_VERSION,
                "code_hashes": code_hashes,
            },
        )
        add_sidecar(inventory)
        self.artifacts["hybrid_code_inventory"] = inventory

        parent_hash_bindings = {
            role: {
                "path": str(self.artifacts[role].relative_to(self.root).as_posix()),
                "sha256": sha256(self.artifacts[role]),
            }
            for role in (
                "base_contract",
                "base_safety_amendment",
                "development_manifest",
                "failed_deployment",
                "original_test_seal",
                "amended_test_seal",
            )
        }
        parent_hash_bindings["prompt_update_edl_code"] = {
            "path": "rl_nninteractive/prompt_update_edl.py",
            "sha256": code_hashes["prompt_update_edl"]["sha256"],
        }
        parent_hash_bindings["route_policy_eval_code"] = {
            "path": "rl_nninteractive/route_policy_eval.py",
            "sha256": code_hashes["route_policy_eval"]["sha256"],
        }

        protocol = expected["hybrid_protocol_contract"]
        write_json(
            protocol,
            {
                "schema_version": subject.SCHEMA_VERSION,
                "protocol_revision": subject.PROTOCOL_REVISION,
                "artifact_type": "post_failure_exploratory_edl_fixed_route_hybrid_protocol",
                "status": "POST_FAILURE_EXPLORATORY_PROTOCOL_FROZEN",
                "hash_bindings": parent_hash_bindings,
                "execution_contract": {
                    "operation": subject.OPERATION,
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
                        "random 256-bit process-secret closure token plus exact "
                        "attempt-receipt SHA; never persisted/logged"
                    ),
                    "direct_legacy_test_entrypoints_forbidden": True,
                    "alternate_output_directory_forbidden": True,
                    "alternate_receipt_paths_forbidden": True,
                    "preclaim_frozen_metadata_exception": (
                        "Before atomic receipt claim, code may stat, read, or opaque-"
                        "SHA-256-hash already-frozen seal and clearance metadata files "
                        "and compare literal canonical output strings; it must not "
                        "semantically parse or enumerate test identifiers, construct or "
                        "resolve an underlying test-data path, parse a test manifest, or "
                        "access ground truth, prompts, images, or outcomes."
                    ),
                    "postclaim_test_semantics": (
                        "All semantic test-identifier, test-path, and test-manifest "
                        "parsing and every ground-truth, prompt, image, or outcome access "
                        "remains strictly after atomic receipt creation."
                    ),
                    "clearance_schema_version": subject.SCHEMA_VERSION,
                    "receipt_schema_version": subject.SCHEMA_VERSION,
                    "report_schema_version": subject.SCHEMA_VERSION,
                    "required_artifact_roles": sorted(subject.REQUIRED_ARTIFACT_ROLES),
                    "required_code_roles": sorted(subject.REQUIRED_CODE_ROLES),
                    "canonical_outputs": {
                        "test_manifest_path": f"<root>/{subject.TEST_MANIFEST_NAME}",
                        "score_directory": f"<root>/{subject.SCORE_DIRECTORY_NAME}",
                    },
                },
            },
        )
        add_sidecar(protocol)
        self.artifacts["hybrid_protocol_contract"] = protocol

        development_roles = {
            "hybrid_development_features",
            "hybrid_development_report",
            "pure_screen_policy",
            "hybrid_policy",
            "hybrid_edl_checkpoint",
            "hybrid_code_inventory",
        }
        development_bindings = {
            role: self._binding(self.artifacts[role]) for role in development_roles
        }
        deployment = expected["hybrid_deployment"]
        write_json(
            deployment,
            {
                "schema_version": subject.SCHEMA_VERSION,
                "artifact_type": "edl_hybrid_frozen_deployment",
                "status": "FROZEN_BEFORE_TEST_OPENING",
                "test_outcomes_opened": False,
                "protocol": {"path": str(protocol), "sha256": sha256(protocol)},
                "artifact_bindings": development_bindings,
                "parent_hash_bindings": parent_hash_bindings,
                "code_hashes": code_hashes,
            },
        )
        add_sidecar(deployment)
        self.artifacts["hybrid_deployment"] = deployment

        bindings = {
            role: self._binding(path) for role, path in self.artifacts.items()
        }
        if set(bindings) != subject.REQUIRED_ARTIFACT_ROLES:
            raise AssertionError("fixture artifact inventory drift")
        reviewer = "independent-test-role"
        reviewed_at = "2026-07-15T20:30:00-04:00"
        audit = self.audit_path
        write_json(
            audit,
            {
                "status": "PASS",
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "protocol_sha256": sha256(protocol),
                "artifact_bindings": bindings,
                "code_hashes": code_hashes,
                "deployment_sha256": sha256(deployment),
            },
        )
        clearance = {
            "schema_version": subject.SCHEMA_VERSION,
            "artifact_type": "edl_hybrid_test_opening_clearance",
            "status": "AUDIT_CLEARED",
            "allowed_operations": [subject.OPERATION],
            "independent_audit": {
                "status": "PASS",
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "report_path": str(audit.resolve()),
                "report_sha256": sha256(audit),
            },
            "artifact_bindings": bindings,
            "code_hashes": code_hashes,
        }
        encoded = json.dumps(clearance, indent=2, sort_keys=True) + "\n"
        for name in subject.CLEARANCE_NAMES:
            path = self.root / name
            path.write_text(encoded, encoding="utf-8")
            add_sidecar(path)

    def read_clearance(self) -> dict:
        return json.loads(
            (self.root / subject.CLEARANCE_NAMES[0]).read_text(encoding="utf-8")
        )

    def write_clearance(self, payload: dict) -> None:
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        for name in subject.CLEARANCE_NAMES:
            path = self.root / name
            path.write_text(encoded, encoding="utf-8")
            add_sidecar(path)

    def refresh_audit_hash(self, clearance: dict) -> None:
        clearance["independent_audit"]["report_sha256"] = sha256(self.audit_path)
        self.write_clearance(clearance)

    def hooks(self, calls: list[str], *, fail_at: str | None = None):
        def accessed(name: str):
            self.assert_attempt_exists()
            calls.append(name)
            if name == fail_at:
                raise RuntimeError(f"simulated {name} failure")
            return {"status": "PASS"}

        def freeze(_preflight, _attempt):
            accessed("freeze_manifest")
            manifest = self.root / subject.TEST_MANIFEST_NAME
            write_json(manifest, {"status": "FROZEN_TEST_OPEN", "records": 48})
            return {"path": str(manifest.resolve()), "records": 48}

        def score(_preflight, _attempt, _manifest):
            accessed("score_both_once")
            score_dir = self.root / subject.SCORE_DIRECTORY_NAME
            score_dir.mkdir()
            report = score_dir / subject.REPORT_NAME
            write_json(report, {"status": "PASS", "test_label_evaluation_passes": 1})
            return {
                "status": "PASS",
                "report_path": str(report.resolve()),
                "report_sha256": sha256(report),
                "test_label_evaluation_passes": 1,
            }

        return subject.ExecutionHooks(
            stage_round1=lambda _p, _a: accessed("stage_round1"),
            infer_round1=lambda _p, _a: accessed("infer_round1"),
            stage_round2=lambda _p, _a: accessed("stage_round2"),
            infer_round2=lambda _p, _a: accessed("infer_round2"),
            freeze_manifest=freeze,
            score_both_once=score,
        )

    def assert_attempt_exists(self) -> None:
        if not self.attempt.is_file():
            raise AssertionError("simulated test access occurred before attempt receipt")
        payload = json.loads(self.attempt.read_text(encoding="utf-8"))
        if payload.get("status") != "ATTEMPT_CONSUMED":
            raise AssertionError("attempt was not durably consumed before test access")


class HybridOneShotGuardTests(unittest.TestCase):
    def test_missing_clearance_has_zero_test_or_gt_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            (root / subject.CLEARANCE_NAMES[0]).unlink()
            calls: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "remains sealed"):
                subject.execute_once(root, fixture.hooks(calls))
            self.assertEqual(calls, [])
            self.assertFalse(fixture.attempt.exists())
            self.assertFalse(fixture.failure.exists())

    def test_mismatched_clearance_has_zero_test_or_gt_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            canonical = root / subject.CLEARANCE_NAMES[0]
            canonical.write_text(canonical.read_text(encoding="utf-8") + " ", encoding="utf-8")
            calls: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "clearance hash mismatch"):
                subject.execute_once(root, fixture.hooks(calls))
            self.assertEqual(calls, [])
            self.assertFalse(fixture.attempt.exists())
            self.assertFalse(fixture.failure.exists())

    def test_preexisting_receipt_blocks_before_test_or_gt_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            write_json(fixture.attempt, {"status": "ATTEMPT_CONSUMED"})
            calls: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "receipt already exists"):
                subject.execute_once(root, fixture.hooks(calls))
            self.assertEqual(calls, [])
            self.assertFalse(fixture.failure.exists())

    def test_attempt_is_written_before_first_simulated_test_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            calls: list[str] = []
            result = subject.execute_once(root, fixture.hooks(calls))
            self.assertEqual(
                calls,
                [
                    "stage_round1",
                    "infer_round1",
                    "stage_round2",
                    "infer_round2",
                    "freeze_manifest",
                    "score_both_once",
                ],
            )
            self.assertEqual(result["status"], "COMPLETED")
            self.assertTrue(fixture.attempt.is_file())
            self.assertTrue(fixture.completion.is_file())
            self.assertFalse(fixture.failure.exists())

    def test_post_claim_exception_writes_failure_and_consumes_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            calls: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "simulated infer_round1 failure"):
                subject.execute_once(root, fixture.hooks(calls, fail_at="infer_round1"))
            self.assertEqual(calls, ["stage_round1", "infer_round1"])
            self.assertTrue(fixture.attempt.is_file())
            self.assertTrue(fixture.failure.is_file())
            self.assertFalse(fixture.completion.exists())
            failure = json.loads(fixture.failure.read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "FAILED_ATTEMPT_CONSUMED")

    def test_phase_is_required_and_exact(self):
        with self.assertRaises(SystemExit):
            subject.parse_args(["--root", "unused"])
        with self.assertRaises(SystemExit):
            subject.parse_args(["--phase", "score-test", "--root", "unused"])
        parsed = subject.parse_args(
            ["--phase", subject.OPERATION, "--root", "unused"]
        )
        self.assertEqual(parsed.phase, subject.OPERATION)

    def test_forged_stale_and_closed_capabilities_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            preflight = subject.validate_non_test_preflight(fixture.root)
            attempt_sha = subject.claim_attempt(preflight)
            authority = subject._TransactionAuthority(preflight, attempt_sha)
            with self.assertRaisesRegex(RuntimeError, "invalid in-process"):
                authority.verify(object())
            token = authority.token
            self.assertEqual(authority.verify(token), attempt_sha)
            fixture.attempt.write_text(
                fixture.attempt.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "receipt drifted"):
                authority.verify(token)
            authority.close()
            with self.assertRaisesRegex(RuntimeError, "invalid in-process"):
                authority.verify(token)

    def test_audit_inventory_mismatch_blocks_with_zero_test_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            audit = json.loads(fixture.audit_path.read_text(encoding="utf-8"))
            audit["artifact_bindings"].pop("hybrid_policy")
            write_json(fixture.audit_path, audit)
            clearance = fixture.read_clearance()
            fixture.refresh_audit_hash(clearance)
            calls: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "audit/artifact inventory"):
                subject.execute_once(fixture.root, fixture.hooks(calls))
            self.assertEqual(calls, [])
            self.assertFalse(fixture.attempt.exists())

    def test_deployment_parent_mismatch_blocks_with_zero_test_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            deployment = fixture.artifacts["hybrid_deployment"]
            payload = json.loads(deployment.read_text(encoding="utf-8"))
            payload["parent_hash_bindings"]["base_contract"]["sha256"] = "0" * 64
            write_json(deployment, payload)
            add_sidecar(deployment)
            clearance = fixture.read_clearance()
            clearance["artifact_bindings"]["hybrid_deployment"] = fixture._binding(
                deployment
            )
            audit = json.loads(fixture.audit_path.read_text(encoding="utf-8"))
            audit["artifact_bindings"] = clearance["artifact_bindings"]
            audit["deployment_sha256"] = sha256(deployment)
            write_json(fixture.audit_path, audit)
            fixture.refresh_audit_hash(clearance)
            calls: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "deployment/parent"):
                subject.execute_once(fixture.root, fixture.hooks(calls))
            self.assertEqual(calls, [])
            self.assertFalse(fixture.attempt.exists())

    def test_schema_revision_and_protocol_path_are_exact(self):
        self.assertTrue(
            subject._expected_artifact_paths(Path("C:/frozen"))[
                "hybrid_protocol_contract"
            ].name.endswith("_v7.json")
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            clearance = fixture.read_clearance()
            clearance["schema_version"] = subject.SCHEMA_VERSION - 1
            fixture.write_clearance(clearance)
            with self.assertRaisesRegex(RuntimeError, "clearance schema"):
                subject.validate_non_test_preflight(fixture.root)

    def test_global_execution_schema_is_rejected_for_local_code_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            inventory = fixture.artifacts["hybrid_code_inventory"]
            payload = json.loads(inventory.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], subject.FREEZE_SCHEMA_VERSION)
            self.assertNotEqual(subject.FREEZE_SCHEMA_VERSION, subject.SCHEMA_VERSION)
            payload["schema_version"] = subject.SCHEMA_VERSION
            write_json(inventory, payload)
            add_sidecar(inventory)

            deployment = fixture.artifacts["hybrid_deployment"]
            deployment_payload = json.loads(deployment.read_text(encoding="utf-8"))
            deployment_payload["artifact_bindings"]["hybrid_code_inventory"] = (
                fixture._binding(inventory)
            )
            write_json(deployment, deployment_payload)
            add_sidecar(deployment)

            clearance = fixture.read_clearance()
            clearance["artifact_bindings"]["hybrid_code_inventory"] = fixture._binding(
                inventory
            )
            clearance["artifact_bindings"]["hybrid_deployment"] = fixture._binding(
                deployment
            )
            audit = json.loads(fixture.audit_path.read_text(encoding="utf-8"))
            audit["artifact_bindings"] = clearance["artifact_bindings"]
            audit["deployment_sha256"] = sha256(deployment)
            write_json(fixture.audit_path, audit)
            fixture.refresh_audit_hash(clearance)

            with self.assertRaisesRegex(RuntimeError, "code inventory/clearance"):
                subject.validate_non_test_preflight(fixture.root)
            self.assertFalse(fixture.attempt.exists())

    def test_root_template_rejects_alternate_and_traversal_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = subject._expand_root_template(
                f"<root>/{subject.TEST_MANIFEST_NAME}", root, "manifest"
            )
            self.assertEqual(expected, (root / subject.TEST_MANIFEST_NAME).resolve())
            for value in (
                "C:/alternate/test.json",
                "<frozen-root>/test.json",
                "<root>/../escape.json",
                "<root>/nested\\escape.json",
                "<root>/<other>/escape.json",
            ):
                with self.subTest(value=value), self.assertRaises(RuntimeError):
                    subject._expand_root_template(value, root, "manifest")

    def test_test_path_template_is_not_expanded_before_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            original = subject._expand_root_template
            observations: list[bool] = []

            def guarded(value, root, field):
                observations.append(fixture.attempt.is_file())
                return original(value, root, field)

            with mock.patch.object(subject, "_expand_root_template", side_effect=guarded):
                subject.execute_once(fixture.root, fixture.hooks([]))
            self.assertTrue(observations)
            self.assertTrue(all(observations))

    def test_direct_legacy_stage_is_blocked_before_gt_or_image_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            import run_fusion_only_cohort_v2 as legacy

            calls = {"gt": 0, "image": 0}

            def gt_probe(*_args, **_kwargs):
                calls["gt"] += 1
                raise AssertionError("GT resolver reached")

            def image_probe(*_args, **_kwargs):
                calls["image"] += 1
                raise AssertionError("image loader reached")

            with mock.patch.object(legacy, "ground_truth_path", side_effect=gt_probe), mock.patch.object(
                legacy.nib, "load", side_effect=image_probe
            ):
                with self.assertRaises(RuntimeError):
                    legacy.stage_prompt_round(fixture.root, 1, "test")
            self.assertEqual(calls, {"gt": 0, "image": 0})

    def test_ground_truth_cache_physically_loads_each_study_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            label_root = root / "sealed" / "test_labels"
            label_root.mkdir(parents=True)
            first = label_root / "first.nii.gz"
            second = label_root / "second.nii.gz"
            other = root / "not-ground-truth.nii.gz"
            physical: list[str] = []

            def fake_load(path, *_args, **_kwargs):
                physical.append(str(path))
                return object()

            import nibabel as nib

            with mock.patch.object(nib, "load", side_effect=fake_load):
                with subject._single_ground_truth_volume_loads(root) as audit:
                    nib.load(str(first))
                    nib.load(str(first))
                    nib.load(str(second))
                    nib.load(str(second))
                    nib.load(str(other))
                    audit.require_exactly_once(2)
            self.assertEqual(physical.count(str(first)), 1)
            self.assertEqual(physical.count(str(second)), 1)
            self.assertEqual(physical.count(str(other)), 1)

    def test_completion_write_failure_emits_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            calls: list[str] = []
            base = fixture.hooks(calls)

            def score_then_block_completion(preflight, capability, manifest):
                result = base.score_both_once(preflight, capability, manifest)
                write_json(fixture.completion, {"status": "RACE"})
                return result

            hooks = replace(base, score_both_once=score_then_block_completion)
            with self.assertRaisesRegex(FileExistsError, "File exists"):
                subject.execute_once(fixture.root, hooks)
            self.assertTrue(fixture.attempt.exists())
            self.assertTrue(fixture.failure.exists())

    def test_no_second_attempt_after_failure_or_completion(self):
        for failure_mode in (False, True):
            with self.subTest(failure_mode=failure_mode), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                calls: list[str] = []
                hooks = fixture.hooks(
                    calls, fail_at="infer_round1" if failure_mode else None
                )
                if failure_mode:
                    with self.assertRaises(RuntimeError):
                        subject.execute_once(fixture.root, hooks)
                else:
                    subject.execute_once(fixture.root, hooks)
                prior_calls = list(calls)
                with self.assertRaisesRegex(RuntimeError, "receipt already exists"):
                    subject.execute_once(fixture.root, fixture.hooks(calls))
                self.assertEqual(calls, prior_calls)


if __name__ == "__main__":
    unittest.main()
