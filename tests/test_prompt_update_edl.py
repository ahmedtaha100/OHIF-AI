import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import rl_nninteractive.prompt_update_edl as prompt_update_edl_module
from rl_nninteractive.prompt_update_edl import (
    FEATURE_ORDER,
    EvidentialUtilityHead,
    UpdateExample,
    decide_update,
    extract_update_features,
    fit_head,
    load_checkpoint_bundle,
    main,
    save_checkpoint_bundle,
    validate_split_contract,
)


def _example(
    patient, split, value, *, prior=False, case_id=None, transition_id="r0-r1"
):
    features = np.linspace(-0.2, 0.2, len(FEATURE_ORDER), dtype=np.float32) + value
    return UpdateExample(
        case_id=case_id or f"{patient}-case",
        patient_id=patient,
        transition_id=transition_id,
        split=split,
        prior_exposure=prior,
        features=features,
        accept_label=int(value > 0),
        utility=float(value),
        delta_dice=float(value),
    )


class FeatureContractTests(unittest.TestCase):
    def test_features_capture_signed_added_and_removed_regions(self):
        shape = (12, 10, 8)
        pet = np.zeros(shape, dtype=np.float32)
        ct = np.zeros(shape, dtype=np.float32)
        current = np.zeros(shape, dtype=bool)
        current[2:5, 2:5, 2:5] = True
        proposed = current.copy()
        proposed[2:3, 2:5, 2:5] = False
        proposed[7:10, 6:9, 4:7] = True
        pet[7:10, 6:9, 4:7] = 5.0
        totseg = np.zeros(shape, dtype=np.int16)
        totseg[7:10, 6:9, 4:7] = 4

        features = extract_update_features(
            pet,
            ct,
            current,
            proposed,
            totseg=totseg,
            prompt_metadata={"round_index": 1, "foreground_xyz": [8, 7, 5]},
        )

        self.assertEqual(features.shape, (len(FEATURE_ORDER),))
        self.assertGreater(features[FEATURE_ORDER.index("added_voxel_fraction")], 0.0)
        self.assertGreater(features[FEATURE_ORDER.index("removed_voxel_fraction")], 0.0)
        self.assertEqual(features[FEATURE_ORDER.index("change_is_mixed")], 1.0)
        self.assertGreater(
            features[FEATURE_ORDER.index("added_totseg_nonzero_fraction")], 0.0
        )

    def test_inference_feature_api_cannot_receive_ground_truth(self):
        self.assertNotIn(
            "ground_truth", inspect.signature(extract_update_features).parameters
        )

    def test_model_produces_valid_dirichlet_and_utility(self):
        model = EvidentialUtilityHead(hidden=8)
        alpha, utility = model(torch.zeros(3, len(FEATURE_ORDER)))
        self.assertEqual(alpha.shape, (3, 2))
        self.assertEqual(utility.shape, (3,))
        self.assertTrue(bool(torch.all(alpha >= 1.0)))
        vacuity = 2.0 / alpha.sum(dim=-1)
        self.assertTrue(bool(torch.all((vacuity > 0.0) & (vacuity <= 1.0))))


class ManifestLoadingTests(unittest.TestCase):
    def test_rejects_totseg_sha256_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = {"case_id": "case-1"}
            for role in (
                "pet",
                "ct",
                "current_mask",
                "proposed_mask",
                "ground_truth",
            ):
                source = root / f"{role}.nii.gz"
                source.write_bytes(role.encode("utf-8"))
                record[f"{role}_path"] = source.name
                record[f"{role}_sha256"] = prompt_update_edl_module.sha256_file(source)
            totseg = root / "totseg.nii.gz"
            totseg.write_bytes(b"totseg")
            record["totseg_path"] = totseg.name
            record["totseg_sha256"] = "0" * 64
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": 1, "records": [record]}),
                encoding="utf-8",
            )
            loaded_volume = (
                np.zeros((2, 2, 2), dtype=np.float32),
                np.eye(4),
                (1.0, 1.0, 1.0),
            )
            with mock.patch.object(
                prompt_update_edl_module,
                "_load_nifti",
                return_value=loaded_volume,
            ) as volume_loader:
                with self.assertRaisesRegex(
                    ValueError, "totseg SHA-256 mismatch for case-1"
                ):
                    prompt_update_edl_module.examples_from_manifest(manifest)
            self.assertEqual(volume_loader.call_count, 5)


class SplitContractTests(unittest.TestCase):
    def test_rejects_patient_leakage(self):
        examples = [
            _example("p1", "train", 0.1),
            _example("p1", "test", -0.1, transition_id="r1-r2"),
        ]
        with self.assertRaisesRegex(ValueError, "patient leakage"):
            validate_split_contract(examples, require_all_splits=False)

    def test_rejects_external_claim_for_prior_exposed_test(self):
        examples = [
            _example("p1", "train", 0.1),
            _example("p2", "calibration", -0.1),
            _example("p3", "policy_validation", 0.2),
            _example("p4", "test", -0.2, prior=True),
        ]
        with self.assertRaisesRegex(ValueError, "external validation rejected"):
            validate_split_contract(examples, claim_external_validation=True)

    def test_strict_mode_requires_four_nonempty_splits(self):
        with self.assertRaisesRegex(ValueError, "strict mode"):
            validate_split_contract(
                [_example("p1", "train", 0.1), _example("p2", "calibration", -0.1)]
            )

    def test_rejects_case_id_mapped_across_patients_or_splits(self):
        examples = [
            _example("p1", "train", 0.1, case_id="same-case"),
            _example("p2", "test", -0.1, case_id="same-case", transition_id="r1-r2"),
        ]
        with self.assertRaisesRegex(
            ValueError, "case_id mapped across patients/splits"
        ):
            validate_split_contract(examples, require_all_splits=False)

    def test_rejects_duplicate_case_transition_record(self):
        examples = [
            _example("p1", "train", 0.1),
            _example("p1", "train", -0.1),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate case/transition record"):
            validate_split_contract(examples, require_all_splits=False)

    def test_separates_exposure_from_efficacy_sample_eligibility(self):
        examples = [
            _example("p1", "train", 0.1),
            _example("p2", "calibration", -0.1),
            _example("p3", "policy_validation", 0.2),
            _example("p4", "test", -0.2),
        ]
        contract = validate_split_contract(examples, minimum_test_patients=20)
        self.assertTrue(contract["external_validation_eligible"])
        self.assertFalse(contract["efficacy_claim_eligible"])
        self.assertEqual(contract["minimum_test_patients"], 20)
        self.assertIn(
            "below minimum 20", " ".join(contract["efficacy_ineligibility_reasons"])
        )


class CheckpointContractTests(unittest.TestCase):
    @staticmethod
    def _development_examples():
        return [
            _example("p1", "train", 0.2),
            _example("p1", "train", -0.2, transition_id="r1-r2"),
            _example("p2", "calibration", 0.15),
            _example("p2", "calibration", -0.15, transition_id="r1-r2"),
            _example("p3", "policy_validation", 0.1),
            _example("p3", "policy_validation", -0.1, transition_id="r1-r2"),
        ]

    def test_development_freeze_has_no_test_and_hashes_provenance(self):
        checkpoint, report = fit_head(
            self._development_examples(),
            manifest_sha256="c" * 64,
            config={"development_freeze_test": True},
            seed=11,
            epochs=2,
            hidden=8,
            development_freeze=True,
        )
        self.assertEqual(checkpoint["status"], "DEVELOPMENT_FROZEN_NO_TEST")
        self.assertEqual(report["status"], "DEVELOPMENT_FROZEN_NO_TEST")
        self.assertTrue(checkpoint["development_freeze"])
        self.assertFalse(checkpoint["mechanics_smoke"])
        self.assertEqual(checkpoint["fit_mode"], "development_freeze")
        self.assertEqual(checkpoint["threshold_source"], "policy_validation")
        self.assertEqual(
            report["threshold_role"], "upstream_flat_candidate_diagnostic_only"
        )
        self.assertEqual(report["test_metrics"]["n"], 0)
        self.assertEqual(checkpoint["test_metrics"]["n"], 0)
        self.assertEqual(
            report["test_metrics"]["scope"],
            "sealed_not_opened_development_freeze",
        )
        self.assertEqual(checkpoint["split_contract"]["split_counts"]["test"], 0)
        self.assertFalse(checkpoint["external_validation_eligible"])
        self.assertFalse(checkpoint["efficacy_claim_eligible"])
        self.assertEqual(len(checkpoint["manifest_sha256"]), 64)
        self.assertEqual(len(checkpoint["config_sha256"]), 64)
        self.assertEqual(len(checkpoint["code_sha256"]), 64)
        self.assertEqual(report["code_sha256"], checkpoint["code_sha256"])
        self.assertIn("no efficacy", checkpoint["claim_boundary"])

    def test_development_freeze_rejects_missing_policy_test_and_mode_overlap(self):
        without_policy = [
            example
            for example in self._development_examples()
            if example.split != "policy_validation"
        ]
        with self.assertRaisesRegex(ValueError, "non-empty policy_validation"):
            fit_head(
                without_policy,
                manifest_sha256="d" * 64,
                config={},
                epochs=1,
                development_freeze=True,
            )
        with_test = self._development_examples() + [_example("p4", "test", 0.1)]
        with self.assertRaisesRegex(ValueError, "zero test records"):
            fit_head(
                with_test,
                manifest_sha256="d" * 64,
                config={},
                epochs=1,
                development_freeze=True,
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            fit_head(
                self._development_examples(),
                manifest_sha256="d" * 64,
                config={},
                epochs=1,
                mechanics_smoke=True,
                development_freeze=True,
            )

    def test_development_cli_rejects_test_before_manifest_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "bad_development.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "FROZEN_DEVELOPMENT",
                        "records": [
                            {"split": "train"},
                            {"split": "calibration"},
                            {"split": "policy_validation"},
                            {"split": "test", "ground_truth_path": "sealed.nii.gz"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                prompt_update_edl_module, "examples_from_manifest"
            ) as loader:
                with self.assertRaisesRegex(ValueError, "rejects test records"):
                    main(
                        [
                            "--manifest",
                            str(manifest),
                            "--development-freeze",
                            "--out-dir",
                            str(root / "out"),
                            "--epochs",
                            "1",
                        ]
                    )
                loader.assert_not_called()

    def test_development_cli_swap_after_preflight_never_loads_test_volume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "development_swap.json"
            development_payload = {
                "schema_version": 1,
                "status": "FROZEN_DEVELOPMENT",
                "records": [
                    {"split": "train"},
                    {"split": "calibration"},
                    {"split": "policy_validation"},
                ],
            }
            manifest.write_text(json.dumps(development_payload), encoding="utf-8")
            swapped_payload = {
                "schema_version": 1,
                "status": "FROZEN_DEVELOPMENT",
                "records": [
                    {
                        "split": "test",
                        "pet_path": "TEST_PET.nii.gz",
                        "ct_path": "TEST_CT.nii.gz",
                        "current_mask_path": "TEST_CURRENT.nii.gz",
                        "proposed_mask_path": "TEST_PROPOSED.nii.gz",
                        "ground_truth_path": "TEST_GT.nii.gz",
                    }
                ],
            }
            original_preflight = (
                prompt_update_edl_module._preflight_development_manifest
            )

            def preflight_then_swap(path):
                result = original_preflight(path)
                manifest.write_text(json.dumps(swapped_payload), encoding="utf-8")
                return result

            with (
                mock.patch.object(
                    prompt_update_edl_module,
                    "_preflight_development_manifest",
                    side_effect=preflight_then_swap,
                ),
                mock.patch.object(
                    prompt_update_edl_module, "_load_nifti"
                ) as volume_loader,
            ):
                with self.assertRaisesRegex(ValueError, "exact split seal"):
                    main(
                        [
                            "--manifest",
                            str(manifest),
                            "--development-freeze",
                            "--out-dir",
                            str(root / "out"),
                            "--epochs",
                            "1",
                        ]
                    )
                volume_loader.assert_not_called()

    def test_development_cli_serializes_explicit_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "development.json"
            payload = {
                "schema_version": 1,
                "status": "FROZEN_DEVELOPMENT",
                "records": [
                    {"split": "train"},
                    {"split": "calibration"},
                    {"split": "policy_validation"},
                ],
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "out"
            with mock.patch.object(
                prompt_update_edl_module,
                "examples_from_manifest",
                return_value=(self._development_examples(), payload),
            ):
                exit_code = main(
                    [
                        "--manifest",
                        str(manifest),
                        "--development-freeze",
                        "--out-dir",
                        str(output),
                        "--epochs",
                        "1",
                        "--hidden",
                        "8",
                    ]
                )
            self.assertEqual(exit_code, 0)
            _model, _normalizer, checkpoint = load_checkpoint_bundle(
                output / "prompt_update_edl.pt"
            )
            self.assertEqual(checkpoint["status"], "DEVELOPMENT_FROZEN_NO_TEST")
            self.assertTrue(checkpoint["development_freeze"])
            saved_report = json.loads(
                (output / "prompt_update_edl_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_report["test_metrics"]["n"], 0)
            self.assertEqual(saved_report["code_sha256"], checkpoint["code_sha256"])

    def test_fit_save_load_and_stop_inference(self):
        examples = [
            _example("p1", "train", 0.2),
            _example("p1", "train", -0.2, transition_id="r1-r2"),
            _example("p2", "calibration", 0.15),
            _example("p2", "calibration", -0.15, transition_id="r1-r2"),
        ]
        checkpoint, report = fit_head(
            examples,
            manifest_sha256="a" * 64,
            config={"test": True},
            seed=7,
            epochs=8,
            hidden=8,
            mechanics_smoke=True,
        )
        for key in (
            "state_dict",
            "feature_order",
            "normalizer",
            "manifest_sha256",
            "config_sha256",
            "seed",
            "calibration",
            "thresholds",
        ):
            self.assertIn(key, checkpoint)
        self.assertFalse(checkpoint["external_validation_eligible"])
        self.assertEqual(report["status"], "EXPLORATORY_MECHANICS_ONLY")

        with tempfile.TemporaryDirectory() as tmp:
            paths = save_checkpoint_bundle(checkpoint, report, output_dir=tmp)
            model, normalizer, loaded = load_checkpoint_bundle(paths["checkpoint"])
            shape = (6, 6, 6)
            result = decide_update(
                model,
                normalizer,
                loaded,
                pet=np.zeros(shape, dtype=np.float32),
                ct=np.zeros(shape, dtype=np.float32),
                current_mask=np.zeros(shape, dtype=bool),
                proposed_mask=np.zeros(shape, dtype=bool),
            )
            self.assertEqual(result["decision"], "STOP")
            self.assertTrue(Path(paths["report"]).is_file())
            self.assertEqual(len(paths["checkpoint_sha256"]), 64)

    def test_prior_exposed_strict_test_is_explicitly_exploratory(self):
        examples = [
            _example("p1", "train", 0.2),
            _example("p2", "calibration", -0.15),
            _example("p3", "policy_validation", 0.1),
            _example("p4", "test", -0.1, prior=True),
        ]
        checkpoint, report = fit_head(
            examples,
            manifest_sha256="b" * 64,
            config={"test": True},
            seed=7,
            epochs=1,
            hidden=8,
        )
        self.assertEqual(report["status"], "EXPLORATORY_INTERNAL_PRIOR_EXPOSED")
        self.assertFalse(checkpoint["efficacy_claim_eligible"])
        self.assertIn(
            "prior-exposed", " ".join(report["efficacy_ineligibility_reasons"])
        )


if __name__ == "__main__":
    unittest.main()
