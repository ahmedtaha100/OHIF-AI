import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_nninteractive.dataset_manifest import load_study_manifest
from rl_nninteractive.autopet_rl_recovery import _paired_patient_statistics
from rl_nninteractive.provenance import (
    CacheIdentity,
    make_cache_envelope,
    sha256_file,
    sha256_json,
    unwrap_cache_envelope,
)


HAS_NIBABEL = importlib.util.find_spec("nibabel") is not None


class StudyManifestTests(unittest.TestCase):
    def test_v2_binds_patient_split_geometry_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_v2_manifest(root)
            study = load_study_manifest(manifest_path)

        self.assertEqual(study.version, 2)
        self.assertEqual(study.split_seed, 17)
        self.assertEqual(study.split_provenance, "retrospectively_frozen_contaminated")
        self.assertEqual(len(study.cases), 2)
        self.assertEqual(study.cases[0].patient_id, "patient-001")
        self.assertEqual(study.cases[1].split, "policy_validation")
        self.assertEqual(study.cases[0].modalities, ("CT",))
        self.assertEqual(study.cases[0].reference_modality, "CT")
        self.assertTrue(study.cases[0].prior_exposure)
        self.assertEqual(len(study.manifest_sha256), 64)

    def test_v2_rejects_patient_leakage_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_v2_manifest(root, second_patient="patient-001")
            with self.assertRaisesRegex(ValueError, "appears in multiple splits"):
                load_study_manifest(manifest_path)

    def test_v2_rejects_source_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_v2_manifest(root)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["datasets"][0]["cases"][0]["image_sha256"]["CT"] = "0" * 64
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CT SHA-256 mismatch"):
                load_study_manifest(manifest_path)


class CacheProvenanceTests(unittest.TestCase):
    def test_cache_rejects_checkpoint_or_case_identity_drift(self):
        identity = _cache_identity(checkpoint="1" * 64)
        envelope = make_cache_envelope(identity, {"episodes": [1, 2]})
        self.assertEqual(unwrap_cache_envelope(envelope, identity), {"episodes": [1, 2]})

        changed = _cache_identity(checkpoint="2" * 64)
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            unwrap_cache_envelope(envelope, changed)

    def test_cache_rejects_legacy_payload(self):
        with self.assertRaisesRegex(ValueError, "legacy or invalid"):
            unwrap_cache_envelope({"train": []}, _cache_identity(checkpoint="1" * 64))


class PairedStatisticsTests(unittest.TestCase):
    def test_statistics_cluster_paired_tracers_by_patient(self):
        case_ids = ["p1_FDG", "p1_PSMA", "p2_FDG", "p2_PSMA"]
        patient_by_case = {
            "p1_FDG": "p1",
            "p1_PSMA": "p1",
            "p2_FDG": "p2",
            "p2_PSMA": "p2",
        }
        result = _paired_patient_statistics(
            case_ids,
            patient_by_case,
            reference=[0.0, 0.0, 0.0, 0.0],
            candidate=[0.1, 0.2, -0.1, -0.1],
        )

        self.assertAlmostEqual(result["mean_delta"], 0.025)
        self.assertEqual(result["patients"], 2)
        self.assertEqual(result["patient_wins"], 1)
        self.assertEqual(result["patient_losses"], 1)


@unittest.skipUnless(HAS_NIBABEL, "nibabel is an optional real-data dependency")
class MedicalGeometryTests(unittest.TestCase):
    def test_resamples_modalities_on_reference_physical_grid(self):
        import nibabel as nib

        from rl_nninteractive.medical_geometry import load_nifti_on_reference_grid

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_path = root / "pet.nii.gz"
            moving_path = root / "ct.nii.gz"
            label_path = root / "label.nii.gz"
            reference_affine = np.diag([2.0, 2.0, 2.0, 1.0])
            moving_affine = np.diag([6 / 7, 8 / 9, 10 / 11, 1.0])
            nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), reference_affine), reference_path)
            nib.save(nib.Nifti1Image(np.ones((8, 10, 12), dtype=np.float32), moving_affine), moving_path)
            label = np.zeros((4, 5, 6), dtype=np.uint8)
            label[1:3, 2:4, 2:5] = 1
            nib.save(nib.Nifti1Image(label, reference_affine), label_path)

            moving = load_nifti_on_reference_grid(
                moving_path,
                reference_path=reference_path,
                target_shape_zyx=(6, 5, 4),
            )
            aligned_label = load_nifti_on_reference_grid(
                label_path,
                reference_path=reference_path,
                target_shape_zyx=(6, 5, 4),
                is_label=True,
            )

        self.assertEqual(moving.data_zyx.shape, (6, 5, 4))
        self.assertGreater(moving.geometry.physical_overlap_fraction, 0.99)
        self.assertEqual(moving.geometry.output_orientation, "RAS")
        self.assertEqual(set(np.unique(aligned_label.data_zyx)), {0.0, 1.0})
        self.assertEqual(aligned_label.geometry.interpolation_order, 0)

    def test_rejects_non_overlapping_physical_fields_of_view(self):
        import nibabel as nib

        from rl_nninteractive.medical_geometry import load_nifti_on_reference_grid

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_path = root / "reference.nii.gz"
            source_path = root / "source.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)), reference_path)
            translated = np.eye(4)
            translated[:3, 3] = 1000.0
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), translated), source_path)
            with self.assertRaisesRegex(ValueError, "do not overlap"):
                load_nifti_on_reference_grid(source_path, reference_path=reference_path)

    def test_msd_loader_selects_4d_channel_and_aligns_label_in_physical_space(self):
        import nibabel as nib

        from rl_nninteractive.evidential_dataset import load_msd_case
        from rl_nninteractive.real_rollout import load_case_zyx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.nii.gz"
            label_path = root / "label.nii.gz"
            image = np.zeros((4, 5, 6, 2), dtype=np.float32)
            image[..., 0] = 100.0
            reference_affine = np.diag([2.0, 2.0, 2.0, 1.0])
            labels = np.zeros((8, 10, 12), dtype=np.uint8)
            labels[2:6, 3:7, 4:9] = 2
            label_affine = np.diag([6 / 7, 8 / 9, 10 / 11, 1.0])
            nib.save(nib.Nifti1Image(image, reference_affine), image_path)
            nib.save(nib.Nifti1Image(labels, label_affine), label_path)

            loaded_image, ground_truth = load_msd_case(image_path, label_path, tumor_label=2)
            raw_image, windowed_image, rollout_ground_truth = load_case_zyx(
                image_path, label_path, tumor_label=2
            )

        self.assertEqual(loaded_image.shape, (6, 5, 4))
        self.assertEqual(ground_truth.shape, loaded_image.shape)
        self.assertTrue(bool(ground_truth.any()))
        self.assertEqual(raw_image.shape, loaded_image.shape)
        self.assertEqual(windowed_image.shape, loaded_image.shape)
        np.testing.assert_array_equal(rollout_ground_truth, ground_truth)

    def test_v2_nifti_manifest_verifies_recorded_geometry_and_aligns_label(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ct_path = root / "ct.nii.gz"
            pet_path = root / "pet.nii.gz"
            label_path = root / "label.nii.gz"
            affine = np.diag([2.0, 2.0, 2.0, 1.0])
            pet = np.zeros((4, 5, 6), dtype=np.float32)
            ct = np.ones((8, 10, 12), dtype=np.float32)
            ct_affine = np.diag([6 / 7, 8 / 9, 10 / 11, 1.0])
            label = np.zeros((4, 5, 6), dtype=np.uint8)
            label[1:3, 2:4, 2:5] = 1
            nib.save(nib.Nifti1Image(ct, ct_affine), ct_path)
            nib.save(nib.Nifti1Image(pet, affine), pet_path)
            nib.save(nib.Nifti1Image(label, affine), label_path)
            geometry = {
                "CT": {
                    "affine": ct_affine.tolist(),
                    "orientation": "RAS",
                    "spacing": [6 / 7, 8 / 9, 10 / 11],
                },
                "PET": {"affine": affine.tolist(), "orientation": "RAS", "spacing": [2, 2, 2]},
                "ground_truth": {
                    "affine": affine.tolist(),
                    "orientation": "RAS",
                    "spacing": [2, 2, 2],
                },
            }
            payload = {
                "version": 2,
                "split_seed": 3,
                "split_provenance": "retrospectively_frozen_contaminated",
                "preprocessing_hash": sha256_json({"canonical": "RAS"}),
                "datasets": [
                    {
                        "name": "nifti-fixture",
                        "version": "v1",
                        "annotation_version": "v1",
                        "modalities": ["CT", "PET"],
                        "reference_modality": "PET",
                        "cases": [
                            {
                                "case_id": "case-nifti",
                                "patient_id": "patient-nifti",
                                "site": "public-site",
                                "tracer": "none",
                                "target_label": "tumor",
                                "split": "test",
                                "prior_exposure": True,
                                "images": {"CT": ct_path.name, "PET": pet_path.name},
                                "ground_truth": label_path.name,
                                "image_sha256": {
                                    "CT": sha256_file(ct_path),
                                    "PET": sha256_file(pet_path),
                                },
                                "ground_truth_sha256": sha256_file(label_path),
                                "inclusion_hash": sha256_json({"included": True}),
                                "geometry": geometry,
                            }
                        ],
                    }
                ],
            }
            manifest_path = root / "study_manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            study = load_study_manifest(manifest_path)

        self.assertEqual(study.cases[0].image.shape, (2, 6, 5, 4))
        self.assertEqual(study.cases[0].ground_truth.shape, (6, 5, 4))
        self.assertTrue(bool(study.cases[0].ground_truth.any()))


def _cache_identity(*, checkpoint: str) -> CacheIdentity:
    return CacheIdentity(
        namespace="test",
        case_ids=("case-1", "case-2"),
        target_label="tumor",
        checkpoint_sha256=checkpoint,
        config_sha256=sha256_json({"grid": [8, 8, 8]}),
        dataset_sha256="3" * 64,
    )


def _write_v2_manifest(root: Path, *, second_patient: str = "patient-002") -> Path:
    image = np.zeros((3, 3, 3), dtype=np.float32)
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1
    image_path = root / "image.npy"
    ground_truth_path = root / "ground_truth.npy"
    np.save(image_path, image)
    np.save(ground_truth_path, ground_truth)
    geometry = {
        "CT": {"affine": np.eye(4).tolist(), "orientation": "RAS", "spacing": [1, 1, 1]},
        "ground_truth": {
            "affine": np.eye(4).tolist(),
            "orientation": "RAS",
            "spacing": [1, 1, 1],
        },
    }
    common = {
        "site": "public-site",
        "tracer": "none",
        "target_label": "tumor",
        "prior_exposure": True,
        "images": {"CT": image_path.name},
        "ground_truth": ground_truth_path.name,
        "image_sha256": {"CT": sha256_file(image_path)},
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "inclusion_hash": sha256_json({"included": True}),
        "geometry": geometry,
    }
    payload = {
        "version": 2,
        "split_seed": 17,
        "split_provenance": "retrospectively_frozen_contaminated",
        "preprocessing_hash": sha256_json({"orientation": "RAS", "spacing": [1, 1, 1]}),
        "datasets": [
            {
                "name": "public-fixture",
                "version": "2026-07-15",
                "annotation_version": "v1",
                "modalities": ["CT"],
                "reference_modality": "CT",
                "cases": [
                    {**common, "case_id": "case-001", "patient_id": "patient-001", "split": "train"},
                    {
                        **common,
                        "case_id": "case-002",
                        "patient_id": second_patient,
                        "split": "policy_validation",
                    },
                ],
            }
        ],
    }
    manifest_path = root / "study_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    unittest.main()
