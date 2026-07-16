from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from rl_nninteractive import edl_fusion_hybrid as subject
from rl_nninteractive.prompt_update_edl import FEATURE_ORDER, calibrate_temperature


def _features(**values: float) -> np.ndarray:
    result = np.zeros(len(FEATURE_ORDER), dtype=np.float32)
    for name, value in values.items():
        result[FEATURE_ORDER.index(name)] = value
    return result


def _menu(
    case_id: str,
    patient_id: str,
    tracer: str,
    *,
    delta_scale: float | None = None,
) -> list[subject.HybridCandidate]:
    candidates: list[subject.HybridCandidate] = []
    for index, route in enumerate(subject.ROUTES):
        candidates.append(
            subject.HybridCandidate(
                case_id=case_id,
                patient_id=patient_id,
                split="train",
                tracer=tracer,
                route=route,
                features=_features(
                    changed_volume_fraction=0.1,
                    added_pet_robust_mean=2.0,
                    added_pet_robust_p90=3.0,
                    removed_pet_robust_mean=1.0,
                    removed_pet_robust_p90=1.5,
                    current_volume_fraction=0.02 * (index + 1),
                ),
                round_agreement_dice=0.9,
                delta_dice=(
                    None
                    if delta_scale is None
                    else delta_scale * (-1.0 if index % 2 else 1.0) * (index + 1)
                ),
            )
        )
    return candidates


class HybridPolicyTests(unittest.TestCase):
    def test_exact_rule_tuple_does_not_use_rounded_signature_identity(self) -> None:
        base = {
            "tracer": "FDG",
            "route": "r1_union",
            "conditions": [
                {"feature": "round_agreement_dice", "op": ">=", "threshold": 0.9},
                {
                    "feature": "added_pet_robust_mean",
                    "op": ">=",
                    "threshold": 1.0000001,
                },
            ],
        }
        other = json.loads(json.dumps(base))
        other["conditions"][1]["threshold"] = 1.0000002
        self.assertNotEqual(subject._rule_identity(base), subject._rule_identity(other))

    def test_temperature_grid_and_unique_label_fallback_are_exact(self) -> None:
        self.assertEqual(
            calibrate_temperature(np.asarray([0.2, 0.8]), np.asarray([1, 1])), 1.0
        )
        candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), 121))
        observed = calibrate_temperature(
            np.asarray([0.05, 0.35, 0.65, 0.95]), np.asarray([0, 0, 1, 1])
        )
        self.assertIn(observed, candidates)

    def test_stable_full_development_partition_matches_contract(self) -> None:
        fit, calibration = subject.stable_patient_partition(
            [f"train_{index:04d}" for index in range(1, 25)]
        )
        self.assertEqual(
            calibration,
            (
                "train_0019",
                "train_0006",
                "train_0018",
                "train_0020",
                "train_0023",
            ),
        )
        self.assertEqual(len(fit), 19)

    def test_selector_can_only_accept_or_veto_the_pure_route(self) -> None:
        candidates = _menu("case_FDG", "patient", "FDG")
        policy = {
            "rules": [
                {
                    "tracer": "FDG",
                    "route": "r1_union",
                    "conditions": [
                        {
                            "feature": "round_agreement_dice",
                            "op": ">=",
                            "threshold": 0.5,
                        },
                        {
                            "feature": "added_pet_robust_mean",
                            "op": ">=",
                            "threshold": 1.0,
                        },
                    ],
                },
                None,
            ]
        }
        scores = {
            candidate.uid: {
                "p_accept": 0.9,
                "vacuity": 0.1,
                "predicted_utility": 0.2,
                "temperature": 1.0,
            }
            for candidate in candidates
        }
        with mock.patch.object(subject, "load_edl_checkpoint", return_value=object()), mock.patch.object(
            subject, "score_edl", return_value=scores
        ):
            selected = subject.select_frozen_policy_routes(
                candidates, pure_policy=policy, edl_checkpoint={}
            )
        self.assertEqual(set(selected), {"case_FDG"})
        self.assertEqual(selected["case_FDG"]["pure_screen_route"], "r1_union")
        self.assertEqual(selected["case_FDG"]["edl_hybrid_route"], "r1_union")
        scores["case_FDG::r1_union"]["p_accept"] = 0.49
        with mock.patch.object(subject, "load_edl_checkpoint", return_value=object()), mock.patch.object(
            subject, "score_edl", return_value=scores
        ):
            vetoed = subject.select_frozen_policy_routes(
                candidates, pure_policy=policy, edl_checkpoint={}
            )
        self.assertEqual(vetoed["case_FDG"]["pure_screen_route"], "r1_union")
        self.assertEqual(vetoed["case_FDG"]["edl_hybrid_route"], "KEEP")

    def test_deterministic_cpu_edl_fit(self) -> None:
        fit = _menu("fit_FDG", "fit_patient", "FDG", delta_scale=0.01)
        calibration = _menu("cal_PSMA", "cal_patient", "PSMA", delta_scale=0.02)
        first = subject.train_edl(fit, calibration)
        second = subject.train_edl(fit, calibration)
        first_scores = subject.score_edl(first, fit + calibration)
        second_scores = subject.score_edl(second, fit + calibration)
        self.assertEqual(first.temperature, second.temperature)
        self.assertEqual(first_scores, second_scores)

    def test_label_free_test_builder_never_touches_ground_truth_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_paths: dict[str, Path] = {}
            for name in ("pet", "ct", "current", *subject.ROUTES):
                path = root / f"{name}.dat"
                path.write_bytes(name.encode("utf-8"))
                asset_paths[name] = path

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            records = []
            for patient_index in range(1, 7):
                for tracer in subject.TRACERS:
                    case_id = f"test_{patient_index:04d}_{tracer}"
                    for route in subject.ROUTES:
                        round_index = int(route[1])
                        action = "intersection" if "intersection" in route else "union"
                        records.append(
                            {
                                "case_id": case_id,
                                "patient_id": f"test_{patient_index:04d}",
                                "split": "test",
                                "tracer": tracer,
                                "route_id": route,
                                "round_index": round_index,
                                "action": action,
                                "pet_path": str(asset_paths["pet"]),
                                "pet_sha256": digest(asset_paths["pet"]),
                                "ct_path": str(asset_paths["ct"]),
                                "ct_sha256": digest(asset_paths["ct"]),
                                "current_mask_path": str(asset_paths["current"]),
                                "current_mask_sha256": digest(asset_paths["current"]),
                                "proposed_mask_path": str(asset_paths[route]),
                                "proposed_mask_sha256": digest(asset_paths[route]),
                                "ground_truth_path": str(root / "MUST_NOT_BE_TOUCHED.nii.gz"),
                                "ground_truth_sha256": "intentionally-invalid-and-ignored",
                            }
                        )
            manifest = root / "test_manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": 1, "records": records}), encoding="utf-8"
            )

            def loader(path: Path) -> tuple[np.ndarray, np.ndarray]:
                if path.name == "pet.dat":
                    array = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
                elif path.name == "ct.dat":
                    array = np.zeros((2, 2, 2), dtype=np.float32)
                elif path.name == "current.dat":
                    array = np.zeros((2, 2, 2), dtype=np.float32)
                    array[0, 0, 0] = 1
                else:
                    array = np.zeros((2, 2, 2), dtype=np.float32)
                    if "union" in path.name:
                        array[0, 0, 0] = 1
                        array[1, 1, 1] = 1
                return array, np.eye(4, dtype=np.float64)

            candidates = subject.build_label_free_test_rows(
                manifest,
                expected_manifest_sha256=digest(manifest),
                volume_loader=loader,
            )
            self.assertEqual(len(candidates), 48)
            self.assertTrue(all(candidate.delta_dice is None for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
