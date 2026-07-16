import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_nninteractive.verify_baseline import (
    make_synthetic_tumor_cases,
    public_image_verification_case,
    run_baseline_verification,
    run_verification_case,
)


class BaselineVerificationTests(unittest.TestCase):
    def test_synthetic_verification_runs_three_passing_cases(self):
        summary = run_baseline_verification(include_public_nibabel=False)

        self.assertEqual(summary["case_count"], 3)
        self.assertTrue(summary["all_cases_passed"])
        self.assertEqual(
            [result["name"] for result in summary["results"]],
            [
                "synthetic_single_voxel_tumor",
                "synthetic_adjacent_two_voxel_tumor",
                "synthetic_missed_tumor_plus_initial_false_positive",
            ],
        )
        for result in summary["results"]:
            self.assertEqual(float(result["final_dice"]), 1.0)
            self.assertEqual(float(result["evaluation"]["final_dice"]), 1.0)
            self.assertIsNotNone(result["evaluation"]["noc_at_85"])
            self.assertIsNotNone(result["evaluation"]["noc_at_90"])
            self.assertIn("1", result["evaluation"]["dice_at"])
            self.assertTrue(result["terminated"])
            self.assertFalse(result["truncated"])

    def test_public_image_case_uses_real_image_shape_with_synthetic_gt(self):
        image = np.zeros((5, 7, 9), dtype=np.float32)
        case = public_image_verification_case(image, image_source="unit-test-public-image")

        self.assertEqual(case.name, "public_nibabel_anatomical_synthetic_gt")
        self.assertEqual(case.image.shape, (1, 5, 7, 9))
        self.assertEqual(case.ground_truth.shape, (5, 7, 9))
        self.assertEqual(int(case.ground_truth.sum()), 2)
        self.assertEqual(case.initial_point, (2, 3, 4))
        self.assertIn("synthetic", case.ground_truth_source)

        result = run_verification_case(case)
        self.assertTrue(result["passed"])
        self.assertEqual(float(result["final_dice"]), 1.0)
        self.assertEqual(result["evaluation"]["noc_at_85"], 1)
        self.assertEqual(result["evaluation"]["noc_at_90"], 1)

    def test_verification_writes_json_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = run_baseline_verification(output_dir=output_dir)

            summary_path = output_dir / "summary.json"
            self.assertTrue(summary_path.exists())
            self.assertIn("baseline verification complete", summary_path.read_text())
            self.assertTrue(summary["all_cases_passed"])

    def test_synthetic_cases_are_tiny_and_mock_only(self):
        cases = make_synthetic_tumor_cases()

        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertEqual(case.image_source, "synthetic")
            self.assertEqual(case.image.shape[0], 1)
            self.assertEqual(case.image.shape[1:], case.ground_truth.shape)

    def test_public_case_rejects_invalid_images(self):
        with self.assertRaisesRegex(ValueError, "3D"):
            public_image_verification_case(np.zeros((1, 2, 3, 4)), image_source="bad")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            public_image_verification_case(
                np.array([[[np.nan, 0.0], [0.0, 0.0]]], dtype=np.float32),
                image_source="bad",
            )


if __name__ == "__main__":
    unittest.main()
