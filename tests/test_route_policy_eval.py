import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

import rl_nninteractive.prompt_update_edl as prompt_update_edl_module
import rl_nninteractive.route_policy_eval as route_policy_eval_module
from rl_nninteractive.prompt_update_edl import (
    FEATURE_ORDER,
    examples_from_manifest,
    fit_head,
    save_checkpoint_bundle,
    sha256_file,
)
from rl_nninteractive.route_policy_eval import (
    DEFAULT_CANDIDATE_ROUTES,
    ROUTE_IDS,
    EvidentialScore,
    RouteCandidate,
    evaluate_test_policies,
    fit_linear_contextual_bandit,
    main,
    parse_route_metadata,
    select_grouped_edl_thresholds,
    validate_route_contract,
)


FUSION_CANDIDATE_ROUTES = (
    "KEEP",
    "r1_intersection",
    "r2_intersection",
    "r1_union",
    "r2_union",
)


def _case(
    patient,
    split,
    *,
    utilities=None,
    feature_shift=0.0,
    route_ids=ROUTE_IDS,
):
    utilities = utilities or {
        "r1_replace": -0.10,
        "r1_intersection": 0.20,
        "r1_union": 0.10,
        "r2_replace": 0.30,
        "r2_intersection": -0.05,
        "r2_union": 0.25,
    }
    output = []
    for index, route_id in enumerate(route_ids):
        round_index = int(route_id[1])
        action = route_id.split("_", 1)[1]
        utility = float(utilities[route_id])
        features = np.linspace(-0.2, 0.2, len(FEATURE_ORDER), dtype=np.float32)
        features = features + feature_shift + index * 0.01
        features[FEATURE_ORDER.index("changed_volume_fraction")] = 0.1
        output.append(
            RouteCandidate(
                case_id=f"{patient}_study",
                patient_id=patient,
                transition_id=f"{route_id}_{patient}",
                split=split,
                prior_exposure=False,
                action=action,
                round_index=round_index,
                features=features,
                utility=utility,
                baseline_dice=0.50,
                candidate_dice=0.50 + utility,
                baseline_nsd=0.40,
                candidate_nsd=0.40 + utility / 2.0,
            )
        )
    return output


def _strict_candidates():
    return (
        _case("p_train", "train", feature_shift=-0.20)
        + _case("p_cal", "calibration", feature_shift=-0.05)
        + _case("p_policy", "policy_validation", feature_shift=0.05)
        + _case("p_test", "test", feature_shift=0.20)
    )


class RouteManifestContractTests(unittest.TestCase):
    def test_parses_additive_fields_and_aliases(self):
        self.assertEqual(
            parse_route_metadata(
                {"transition_id": "x", "composition": "intersect", "prompt_round": 2}
            ),
            ("intersection", 2),
        )
        self.assertEqual(
            parse_route_metadata({"transition_id": "r1-union-candidate"}),
            ("union", 1),
        )

    def test_requires_all_six_routes(self):
        candidates = _strict_candidates()
        contract = validate_route_contract(candidates)
        self.assertEqual(contract["candidate_routes"], list(DEFAULT_CANDIDATE_ROUTES))
        self.assertEqual(contract["route_menu_source"], "legacy_six_route_default")
        missing = [
            candidate
            for candidate in candidates
            if candidate.uid != "p_test_study::r2_union"
        ]
        with self.assertRaisesRegex(ValueError, "all six routes"):
            validate_route_contract(missing)

    def test_declared_fusion_menu_is_exact_for_every_study(self):
        proposal_routes = FUSION_CANDIDATE_ROUTES[1:]
        candidates = []
        for patient, split in (
            ("p_train", "train"),
            ("p_cal", "calibration"),
            ("p_policy", "policy_validation"),
            ("p_test", "test"),
        ):
            candidates.extend(_case(patient, split, route_ids=proposal_routes))
        contract = validate_route_contract(
            candidates, candidate_routes=FUSION_CANDIDATE_ROUTES
        )
        self.assertEqual(contract["proposal_routes"], list(proposal_routes))
        self.assertEqual(contract["route_menu_source"], "manifest_candidate_routes")

        missing = [
            candidate
            for candidate in candidates
            if candidate.uid != "p_test_study::r2_union"
        ]
        with self.assertRaisesRegex(ValueError, "exactly the declared"):
            validate_route_contract(missing, candidate_routes=FUSION_CANDIDATE_ROUTES)

        undeclared = list(candidates) + [
            candidate
            for candidate in _case("p_test", "test")
            if candidate.route_id == "r1_replace"
        ]
        with self.assertRaisesRegex(ValueError, "undeclared routes"):
            validate_route_contract(
                undeclared, candidate_routes=FUSION_CANDIDATE_ROUTES
            )

    def test_rejects_invalid_candidate_route_declarations(self):
        candidates = _strict_candidates()
        with self.assertRaisesRegex(ValueError, "baseline route"):
            validate_route_contract(candidates, candidate_routes=ROUTE_IDS)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_route_contract(
                candidates,
                candidate_routes=("KEEP", "r1_union", "r1_union"),
            )
        with self.assertRaisesRegex(ValueError, "unknown route IDs"):
            validate_route_contract(
                candidates,
                candidate_routes=("KEEP", "r3_union"),
            )

    def test_rejects_patient_leakage(self):
        candidates = _strict_candidates()
        leaked = list(candidates)
        leaked[0] = dataclasses.replace(leaked[0], patient_id="p_test")
        with self.assertRaisesRegex(ValueError, "patient leakage"):
            validate_route_contract(leaked)


class NoTestTuningTests(unittest.TestCase):
    def test_ridge_fit_rejects_test_labels_at_api_boundary(self):
        candidates = _strict_candidates()
        with self.assertRaisesRegex(ValueError, "test labels must remain inaccessible"):
            fit_linear_contextual_bandit(candidates, ridge_lambdas=(0.1, 1.0))

    def test_ridge_fit_is_invariant_after_test_is_withheld(self):
        candidates = _strict_candidates()
        development = [
            candidate for candidate in candidates if candidate.split != "test"
        ]
        first, first_report = fit_linear_contextual_bandit(
            development,
            ridge_lambdas=(0.1, 1.0),
            bootstrap_samples=200,
        )
        changed = [
            dataclasses.replace(
                candidate,
                utility=(-100.0 if candidate.utility > 0 else 100.0),
                candidate_dice=(0.0 if candidate.utility > 0 else 1.0),
            )
            if candidate.split == "test"
            else candidate
            for candidate in candidates
        ]
        changed_development = [
            candidate for candidate in changed if candidate.split != "test"
        ]
        second, second_report = fit_linear_contextual_bandit(
            changed_development,
            ridge_lambdas=(0.1, 1.0),
            bootstrap_samples=200,
        )
        self.assertEqual(first.ridge_lambda, second.ridge_lambda)
        self.assertEqual(first.threshold, second.threshold)
        np.testing.assert_allclose(
            first.coefficients, second.coefficients, atol=0.0, rtol=0.0
        )
        self.assertEqual(
            first_report["candidate_models"], second_report["candidate_models"]
        )

    def test_edl_thresholds_are_selected_on_grouped_policy_menus(self):
        candidates = _case("p_policy", "policy_validation")
        scores = {}
        for candidate in candidates:
            harmful = candidate.route_id == "r1_replace"
            scores[candidate.uid] = EvidentialScore(
                p_accept=0.9 if harmful else 0.7,
                vacuity=0.8 if harmful else 0.2,
                predicted_utility=1.0 if harmful else candidate.utility,
                accepted=False,
                confidence=0.1,
            )
        thresholds, report = select_grouped_edl_thresholds(candidates, scores)
        self.assertEqual(
            report["selection_unit"], "grouped per-study six-candidate menu"
        )
        self.assertEqual(report["selection_split"], "policy_validation")
        self.assertLess(thresholds["max_accept_vacuity"], 0.8)
        self.assertGreater(report["selected_objective"]["mean_realized_utility"], 0.0)
        self.assertTrue(report["safety_deployed"])
        self.assertEqual(report["candidate_grid_size"], 280)
        self.assertLessEqual(
            report["selected_objective"]["harmful_action_rate_all_studies"], 0.05
        )
        self.assertGreater(
            report["selected_objective"][
                "patient_cluster_bootstrap_95_ci_mean_realized_utility"
            ]["lower"],
            0.0,
        )

    def test_edl_safety_gate_falls_back_to_keep_all(self):
        utilities = {route_id: -0.1 for route_id in ROUTE_IDS}
        candidates = _case(
            "p_policy", "policy_validation", utilities=utilities
        ) + _case("p_policy_2", "policy_validation", utilities=utilities)
        scores = {
            candidate.uid: EvidentialScore(
                p_accept=0.9,
                vacuity=0.1,
                predicted_utility=1.0,
                accepted=True,
                confidence=0.81,
            )
            for candidate in candidates
        }
        _thresholds, report = select_grouped_edl_thresholds(
            candidates, scores, bootstrap_samples=200
        )
        self.assertFalse(report["safety_deployed"])
        self.assertEqual(report["deployment_decision"], "KEEP_ALL")
        self.assertIsNotNone(report["fallback_reason"])
        self.assertEqual(report["selected_objective"]["coverage"], 0.0)

    def test_ridge_safety_gate_falls_back_to_keep_all(self):
        candidates = [
            candidate for candidate in _strict_candidates() if candidate.split != "test"
        ]
        unsafe = [
            dataclasses.replace(
                candidate,
                utility=-0.1,
                candidate_dice=candidate.baseline_dice - 0.1,
            )
            if candidate.split == "policy_validation"
            else candidate
            for candidate in candidates
        ]
        model, report = fit_linear_contextual_bandit(
            unsafe, ridge_lambdas=(0.1, 1.0), bootstrap_samples=200
        )
        self.assertTrue(model.deploy_keep_all)
        self.assertEqual(report["deployment_decision"], "KEEP_ALL")
        self.assertEqual(report["eligible_model_count"], 0)
        self.assertIsNotNone(report["fallback_reason"])


class FrozenEvaluationTests(unittest.TestCase):
    def test_fixed_oracle_edl_bandit_and_statistics(self):
        candidates = _strict_candidates()
        validate_route_contract(candidates)
        bandit, _ = fit_linear_contextual_bandit(
            [candidate for candidate in candidates if candidate.split != "test"],
            ridge_lambdas=(0.1, 1.0),
            bootstrap_samples=200,
        )
        scores = {}
        for candidate in candidates:
            accepted = candidate.route_id in {"r1_intersection", "r2_union"}
            predicted = 0.5 if candidate.route_id == "r2_union" else 0.4
            scores[candidate.uid] = EvidentialScore(
                p_accept=0.9 if accepted else 0.1,
                vacuity=0.1,
                predicted_utility=predicted,
                accepted=accepted,
                confidence=0.81 if accepted else 0.09,
            )
        result = evaluate_test_policies(
            candidates,
            edl_scores=scores,
            bandit=bandit,
            bootstrap_samples=200,
            seed=7,
        )
        self.assertEqual(result["study_count"], 1)
        policies = result["policies"]
        self.assertAlmostEqual(
            policies["hindsight_oracle"]["study_estimand"]["mean_delta_dice"], 0.30
        )
        self.assertEqual(
            policies["fixed_r1_replace"]["study_estimand"]["dice_win_tie_loss_vs_keep"],
            {"wins": 0, "ties": 0, "losses": 1},
        )
        edl_row = next(
            row
            for row in result["per_study"]
            if row["policy"] == "edl_accept_best_utility"
        )
        self.assertEqual(edl_row["selected_route"], "r2_union")
        self.assertTrue(policies["edl_accept_best_utility"]["risk_coverage"])
        self.assertIn(
            "paired_bootstrap_95_ci_delta_dice",
            policies["hindsight_oracle"]["patient_estimand"],
        )

    def test_declared_fusion_menu_controls_fixed_policy_report(self):
        proposal_routes = FUSION_CANDIDATE_ROUTES[1:]
        candidates = []
        for patient, split in (
            ("p_train", "train"),
            ("p_cal", "calibration"),
            ("p_policy", "policy_validation"),
            ("p_test", "test"),
        ):
            candidates.extend(_case(patient, split, route_ids=proposal_routes))
        validate_route_contract(candidates, candidate_routes=FUSION_CANDIDATE_ROUTES)
        bandit, _ = fit_linear_contextual_bandit(
            [candidate for candidate in candidates if candidate.split != "test"],
            ridge_lambdas=(0.1,),
            bootstrap_samples=200,
        )
        scores = {
            candidate.uid: EvidentialScore(
                p_accept=0.9,
                vacuity=0.1,
                predicted_utility=candidate.utility,
                accepted=candidate.utility > 0,
                confidence=0.81,
            )
            for candidate in candidates
        }
        result = evaluate_test_policies(
            candidates,
            edl_scores=scores,
            bandit=bandit,
            candidate_routes=FUSION_CANDIDATE_ROUTES,
            bootstrap_samples=200,
            seed=7,
        )
        self.assertEqual(result["candidate_routes"], list(FUSION_CANDIDATE_ROUTES))
        self.assertEqual(result["test_label_evaluation_passes"], 1)
        self.assertNotIn("fixed_r1_replace", result["policies"])
        self.assertNotIn("fixed_r2_replace", result["policies"])
        self.assertEqual(len(result["policies"]), 8)

    def test_end_to_end_strict_manifest_checkpoint_and_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            splits = (
                ("patient_train", "train"),
                ("patient_calibration", "calibration"),
                ("patient_policy", "policy_validation"),
                ("patient_test", "test"),
            )
            affine = np.eye(4)
            shape = (6, 6, 6)
            for patient_index, (patient, split) in enumerate(splits):
                case_id = f"{patient}_FDG"
                case_dir = root / case_id
                case_dir.mkdir()
                pet = np.indices(shape).sum(axis=0).astype(np.float32) + patient_index
                ct = np.zeros(shape, dtype=np.float32)
                current = np.zeros(shape, dtype=np.uint8)
                current[1:3, 1:3, 1:3] = 1
                ground_truth = np.zeros(shape, dtype=np.uint8)
                ground_truth[2:5, 2:5, 2:5] = 1
                common = {
                    "pet": pet,
                    "ct": ct,
                    "current_mask": current,
                    "ground_truth": ground_truth,
                }
                common_paths = {}
                for role, array in common.items():
                    path = case_dir / f"{role}.nii.gz"
                    nib.save(nib.Nifti1Image(array, affine), path)
                    common_paths[role] = path
                for route_index, route_id in enumerate(ROUTE_IDS):
                    round_index = int(route_id[1])
                    action = route_id.split("_", 1)[1]
                    proposed = current.copy()
                    if route_index % 3 == 0:
                        proposed = ground_truth.copy()
                    elif route_index % 3 == 1:
                        proposed[2:4, 2:4, 2:4] = 1
                    else:
                        proposed[0, 0, 0] = 1
                    proposed_path = case_dir / f"{route_id}.nii.gz"
                    nib.save(nib.Nifti1Image(proposed, affine), proposed_path)
                    record = {
                        "case_id": case_id,
                        "patient_id": patient,
                        "transition_id": f"{case_id}_{route_id}",
                        "split": split,
                        "prior_exposure": False,
                        "action": action,
                        "round_index": round_index,
                        "prompt_metadata": {"round_index": round_index},
                        "proposed_mask_path": str(proposed_path),
                    }
                    for role, path in common_paths.items():
                        record[f"{role}_path"] = str(path)
                        record[f"{role}_sha256"] = sha256_file(path)
                    record["proposed_mask_sha256"] = sha256_file(proposed_path)
                    records.append(record)
            manifest = root / "route_manifest.json"
            manifest.write_text(
                json.dumps(
                    {"schema_version": 1, "status": "FROZEN", "records": records}
                ),
                encoding="utf-8",
            )
            examples, _ = examples_from_manifest(manifest)
            checkpoint, training_report = fit_head(
                examples,
                manifest_sha256=sha256_file(manifest),
                config={"integration_test": True},
                seed=3,
                epochs=2,
                hidden=4,
            )
            bundle = save_checkpoint_bundle(
                checkpoint, training_report, output_dir=root / "edl"
            )
            output = root / "evaluation"
            exit_code = main(
                [
                    "--manifest",
                    str(manifest),
                    "--edl-checkpoint",
                    bundle["checkpoint"],
                    "--out-dir",
                    str(output),
                    "--bootstrap-samples",
                    "20",
                    "--ridge-lambdas",
                    "0.1,1.0",
                    "--minimum-test-patients",
                    "1",
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads(
                (output / "route_policy_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "COMPLETED_FROZEN_TEST")
            self.assertFalse(
                report["no_test_tuning_audit"][
                    "test_used_for_model_or_threshold_selection"
                ]
            )
            self.assertEqual(report["test_evaluation"]["study_count"], 1)
            self.assertEqual(len(report["test_evaluation"]["policies"]), 10)
            self.assertTrue((output / "route_policy_per_study.csv").is_file())
            self.assertTrue((output / "route_policy_per_patient.csv").is_file())

            dev_manifest = root / "route_manifest_development.json"
            dev_records = [record for record in records if record["split"] != "test"]
            dev_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "FROZEN_DEVELOPMENT",
                        "candidate_routes": list(DEFAULT_CANDIDATE_ROUTES),
                        "records": dev_records,
                    }
                ),
                encoding="utf-8",
            )
            dev_examples, _ = examples_from_manifest(dev_manifest)
            dev_checkpoint, dev_training_report = fit_head(
                dev_examples,
                manifest_sha256=sha256_file(dev_manifest),
                config={"two_phase_integration_test": True},
                seed=5,
                epochs=2,
                hidden=4,
                development_freeze=True,
            )
            dev_bundle = save_checkpoint_bundle(
                dev_checkpoint,
                dev_training_report,
                output_dir=root / "edl_development",
            )
            freeze_output = root / "freeze"
            freeze_args = [
                "--phase",
                "freeze-development",
                "--manifest",
                str(dev_manifest),
                "--edl-checkpoint",
                dev_bundle["checkpoint"],
                "--out-dir",
                str(freeze_output),
                "--ridge-lambdas",
                "0.1,1.0",
                "--minimum-test-patients",
                "1",
            ]
            original_load_nifti = prompt_update_edl_module._load_nifti

            def reject_test_path(path):
                if "patient_test" in str(path):
                    raise AssertionError("phase A attempted to open sealed test data")
                return original_load_nifti(path)

            with mock.patch.object(
                prompt_update_edl_module,
                "_load_nifti",
                side_effect=reject_test_path,
            ):
                freeze_exit = main(freeze_args)
            self.assertEqual(freeze_exit, 0)
            deployment_path = freeze_output / "route_policy_deployment.json"
            deployment_hash = sha256_file(deployment_path)
            deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
            self.assertEqual(deployment["status"], "FROZEN_DEVELOPMENT")
            self.assertEqual(
                deployment["development_manifest"]["test_records_opened"], 0
            )
            self.assertFalse(deployment["seal_audit"]["test_manifest_opened"])

            test_manifest = root / "route_manifest_test_open.json"
            test_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "FROZEN_TEST_OPEN",
                        "candidate_routes": list(DEFAULT_CANDIDATE_ROUTES),
                        "development_manifest_sha256": sha256_file(dev_manifest),
                        "deployment_sha256": deployment_hash,
                        "records": [
                            record for record in records if record["split"] == "test"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            score_output = root / "score_test"
            score_args = [
                "--phase",
                "score-test",
                "--manifest",
                str(test_manifest),
                "--edl-checkpoint",
                dev_bundle["checkpoint"],
                "--deployment-plan",
                str(deployment_path),
                "--deployment-sha256",
                deployment_hash,
                "--out-dir",
                str(score_output),
                "--bootstrap-samples",
                "20",
                "--minimum-test-patients",
                "1",
            ]
            with mock.patch.object(
                route_policy_eval_module,
                "load_route_manifest",
                wraps=route_policy_eval_module.load_route_manifest,
            ) as test_loader:
                score_exit = main(score_args)
                retry_args = list(score_args)
                retry_args[retry_args.index(str(score_output))] = str(
                    root / "score_test_retry_output"
                )
                with self.assertRaisesRegex(
                    RuntimeError, "one-shot receipt already exists"
                ):
                    main(retry_args)
            self.assertEqual(score_exit, 0)
            self.assertEqual(test_loader.call_count, 1)
            two_phase_report = json.loads(
                (score_output / "route_policy_report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                two_phase_report["deployment_artifact"][
                    "frozen_before_test_manifest_open"
                ]
            )
            self.assertEqual(
                two_phase_report["no_test_tuning_audit"][
                    "test_label_evaluation_passes"
                ],
                1,
            )
            self.assertEqual(
                two_phase_report["test_evaluation"]["test_label_evaluation_passes"],
                1,
            )
            self.assertTrue(
                (freeze_output / "test_open_attempt_receipt.json").is_file()
            )
            self.assertTrue(
                (freeze_output / "test_open_completion_receipt.json").is_file()
            )

            failed_freeze_output = root / "freeze_failed_attempt"
            failed_freeze_args = list(freeze_args)
            failed_freeze_args[failed_freeze_args.index(str(freeze_output))] = str(
                failed_freeze_output
            )
            self.assertEqual(main(failed_freeze_args), 0)
            failed_deployment_path = (
                failed_freeze_output / "route_policy_deployment.json"
            )
            failed_deployment_hash = sha256_file(failed_deployment_path)
            bad_manifest = root / "route_manifest_test_bad.json"
            bad_payload = json.loads(test_manifest.read_text(encoding="utf-8"))
            bad_payload["deployment_sha256"] = failed_deployment_hash
            bad_payload["records"] = bad_payload["records"][:-1]
            bad_manifest.write_text(json.dumps(bad_payload), encoding="utf-8")
            failed_output = root / "score_test_failed"
            failed_args = list(score_args)
            failed_args[failed_args.index(str(test_manifest))] = str(bad_manifest)
            failed_args[failed_args.index(str(deployment_path))] = str(
                failed_deployment_path
            )
            failed_args[failed_args.index(deployment_hash)] = failed_deployment_hash
            failed_args[failed_args.index(str(score_output))] = str(failed_output)
            with mock.patch.object(
                route_policy_eval_module,
                "load_route_manifest",
                wraps=route_policy_eval_module.load_route_manifest,
            ) as failed_loader:
                with self.assertRaisesRegex(
                    ValueError, "exactly the declared candidate route menu"
                ):
                    main(failed_args)
                self.assertEqual(failed_loader.call_count, 1)
                with self.assertRaisesRegex(
                    RuntimeError, "one-shot receipt already exists"
                ):
                    retry_failed_args = list(failed_args)
                    retry_failed_args[retry_failed_args.index(str(failed_output))] = (
                        str(root / "score_test_failed_retry_output")
                    )
                    main(retry_failed_args)
                self.assertEqual(failed_loader.call_count, 1)
            self.assertTrue(
                (failed_freeze_output / "test_open_failure_receipt.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
