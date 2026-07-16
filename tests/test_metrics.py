import math
import unittest

import numpy as np

from rl_nninteractive.metrics import (
    dice_at_steps,
    dice_score,
    hd95,
    noc_at_85,
    noc_at_90,
    noc_at_threshold,
    normalized_surface_dice,
)


class MetricTests(unittest.TestCase):
    def test_dice_handles_identical_disjoint_and_empty_masks(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        disjoint = np.zeros_like(mask)
        disjoint[0, 0, 0] = True

        self.assertEqual(dice_score(mask, mask), 1.0)
        self.assertEqual(dice_score(mask, disjoint), 0.0)
        self.assertEqual(dice_score(np.zeros_like(mask), np.zeros_like(mask)), 1.0)

    def test_surface_metrics_handle_identical_and_shifted_single_voxels(self):
        pred = np.zeros((4, 4, 4), dtype=bool)
        target = np.zeros_like(pred)
        pred[1, 1, 1] = True
        target[2, 1, 1] = True

        self.assertEqual(hd95(pred, pred), 0.0)
        self.assertEqual(normalized_surface_dice(pred, pred, tolerance=0.0), 1.0)
        self.assertAlmostEqual(hd95(pred, target), 1.0)
        self.assertEqual(normalized_surface_dice(pred, target, tolerance=0.5), 0.0)
        self.assertEqual(normalized_surface_dice(pred, target, tolerance=1.0), 1.0)

    def test_hd95_uses_max_of_directed_percentiles(self):
        pred = np.zeros((1, 100), dtype=bool)
        target = np.zeros_like(pred)
        pred[0, :] = True
        target[0, 0] = True

        expected_directed_p95 = np.percentile(np.arange(100, dtype=float), 95)
        self.assertAlmostEqual(hd95(pred, target), expected_directed_p95)
        self.assertAlmostEqual(hd95(pred, target), hd95(target, pred))
        self.assertAlmostEqual(
            normalized_surface_dice(pred, target, tolerance=0.5),
            2 / 101,
        )

    def test_surface_metrics_respect_anisotropic_spacing(self):
        pred = np.zeros((3, 3, 3), dtype=bool)
        target = np.zeros_like(pred)
        pred[0, 1, 1] = True
        target[1, 1, 1] = True

        self.assertAlmostEqual(hd95(pred, target, spacing=(2.0, 1.0, 1.0)), 2.0)
        self.assertEqual(
            normalized_surface_dice(
                pred,
                target,
                tolerance=1.99,
                spacing=(2.0, 1.0, 1.0),
            ),
            0.0,
        )
        self.assertEqual(
            normalized_surface_dice(
                pred,
                target,
                tolerance=2.0,
                spacing=(2.0, 1.0, 1.0),
            ),
            1.0,
        )

    def test_2d_and_3d_perfect_masks_have_surface_metric_parity(self):
        mask_2d = np.zeros((5, 5), dtype=bool)
        mask_2d[1:4, 1:4] = True
        mask_3d = np.zeros((5, 5, 5), dtype=bool)
        mask_3d[1:4, 1:4, 1:4] = True

        self.assertEqual(hd95(mask_2d, mask_2d), 0.0)
        self.assertEqual(hd95(mask_3d, mask_3d), 0.0)
        self.assertEqual(normalized_surface_dice(mask_2d, mask_2d, tolerance=0.0), 1.0)
        self.assertEqual(normalized_surface_dice(mask_3d, mask_3d, tolerance=0.0), 1.0)

    def test_surface_metrics_handle_empty_masks(self):
        empty = np.zeros((3, 3, 3), dtype=bool)
        one_voxel = np.zeros_like(empty)
        one_voxel[1, 1, 1] = True

        self.assertEqual(hd95(empty, empty), 0.0)
        self.assertEqual(normalized_surface_dice(empty, empty, tolerance=1.0), 1.0)
        self.assertTrue(math.isinf(hd95(empty, one_voxel)))
        self.assertEqual(normalized_surface_dice(empty, one_voxel, tolerance=1.0), 0.0)
        self.assertEqual(normalized_surface_dice(one_voxel, empty, tolerance=1.0), 0.0)

    def test_probability_masks_require_threshold_and_reject_nan(self):
        probabilities = np.array([[0.1, 0.8], [0.6, 0.2]], dtype=float)
        target = np.array([[False, True], [True, False]], dtype=bool)

        with self.assertRaisesRegex(ValueError, "threshold"):
            dice_score(probabilities, target)
        self.assertEqual(dice_score(probabilities, target, threshold=0.5), 1.0)
        self.assertEqual(hd95(probabilities, target, threshold=0.5), 0.0)
        self.assertEqual(
            normalized_surface_dice(probabilities, target, threshold=0.5, tolerance=0.0),
            1.0,
        )

        with self.assertRaisesRegex(ValueError, "NaN"):
            dice_score(np.array([[math.nan, 1.0]], dtype=float), target, threshold=0.5)
        with self.assertRaisesRegex(ValueError, "threshold"):
            dice_score(probabilities, target, threshold=math.nan)
        with self.assertRaisesRegex(ValueError, "multi-label"):
            dice_score(np.array([[0, 1], [2, 0]]), target, threshold=0.5)

    def test_interaction_metrics_report_noc_and_dice_at_steps(self):
        trajectory = [0.2, 0.84, 0.86, 0.91]

        self.assertEqual(noc_at_threshold(trajectory, 0.85), 3)
        self.assertEqual(noc_at_85(trajectory), 3)
        self.assertEqual(noc_at_90(trajectory), 4)
        self.assertIsNone(noc_at_threshold(trajectory, 0.95))
        self.assertEqual(dice_at_steps(trajectory), {1: 0.2, 3: 0.86, 5: None})

    def test_interaction_metrics_reject_non_finite_scores(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            noc_at_threshold([math.nan, 0.9], 0.85)
        with self.assertRaisesRegex(ValueError, "finite"):
            noc_at_85([0.8, math.inf])
        with self.assertRaisesRegex(ValueError, "finite"):
            dice_at_steps([0.1, math.nan])

    def test_invalid_inputs_are_rejected(self):
        mask = np.zeros((3, 3), dtype=bool)

        with self.assertRaisesRegex(ValueError, "mask shapes differ"):
            dice_score(mask, np.zeros((4, 4), dtype=bool))
        with self.assertRaisesRegex(ValueError, "tolerance"):
            normalized_surface_dice(mask, mask, tolerance=-1.0)
        with self.assertRaisesRegex(ValueError, "threshold"):
            noc_at_threshold([0.1], 1.5)
        with self.assertRaisesRegex(ValueError, "steps"):
            dice_at_steps([0.1], steps=(0,))
        with self.assertRaisesRegex(ValueError, "spacing"):
            hd95(mask, mask, spacing=(math.inf, 1.0))


if __name__ == "__main__":
    unittest.main()
