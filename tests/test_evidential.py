"""Unit tests for the evidential error model, GT-free candidates, and channel.

These run on CPU with tiny volumes so they are fast and deterministic. The
candidate/stop tests build ``ErrorMaps`` by hand (a planted error blob) so they
exercise the deployable GT-free logic without needing a trained checkpoint.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from rl_nninteractive.evidential import (
    ERR_CORRECT,
    ERR_FALSE_NEGATIVE,
    ERR_FALSE_POSITIVE,
    NUM_ERROR_CLASSES,
    ErrorMaps,
    EvidentialErrorNet3D,
    dirichlet_alpha,
    dirichlet_uncertainty,
    error_labels_from_masks,
    evidential_segmentation_loss,
    inverse_frequency_class_weights,
    predict_error_maps,
    set_seed,
)
from rl_nninteractive.env import POINT_NEGATIVE, POINT_POSITIVE
from rl_nninteractive.evidential_candidates import (
    evidential_next_action,
    evidential_stop_decision,
    evidential_candidate,
)


def _zero_maps(shape):
    zeros = np.zeros(shape, dtype=np.float32)
    prob = np.zeros((NUM_ERROR_CLASSES, *shape), dtype=np.float32)
    prob[ERR_CORRECT] = 1.0
    return ErrorMaps(
        prob=prob,
        p_error=zeros.copy(),
        p_false_negative=zeros.copy(),
        p_false_positive=zeros.copy(),
        vacuity=np.full(shape, 0.1, dtype=np.float32),
        strength=np.full(shape, 30.0, dtype=np.float32),
    )


class TestModel(unittest.TestCase):
    def test_forward_shape_and_nonneg_evidence_on_odd_sizes(self):
        set_seed(0)
        model = EvidentialErrorNet3D(in_channels=2, base_channels=8)
        x = torch.randn(2, 2, 26, 30, 22)
        ev = model(x)
        self.assertEqual(tuple(ev.shape), (2, NUM_ERROR_CLASSES, 26, 30, 22))
        self.assertGreaterEqual(float(ev.min()), 0.0)

    def test_dirichlet_uncertainty_ranges(self):
        set_seed(1)
        model = EvidentialErrorNet3D(in_channels=2, base_channels=8)
        ev = model(torch.randn(1, 2, 16, 16, 16))
        alpha = dirichlet_alpha(ev)
        u = dirichlet_uncertainty(alpha)
        probs = u["prob"].sum(dim=1)
        self.assertTrue(torch.allclose(probs, torch.ones_like(probs), atol=1e-4))
        vac = u["vacuity"]
        self.assertTrue(float(vac.min()) > 0.0 and float(vac.max()) <= 1.0 + 1e-5)

    def test_determinism(self):
        set_seed(7)
        m1 = EvidentialErrorNet3D(in_channels=2, base_channels=8)
        set_seed(7)
        m2 = EvidentialErrorNet3D(in_channels=2, base_channels=8)
        x = torch.randn(1, 2, 12, 12, 12)
        with torch.no_grad():
            self.assertTrue(torch.allclose(m1(x), m2(x)))


class TestLabelsAndLoss(unittest.TestCase):
    def test_error_labels(self):
        gt = np.zeros((4, 4, 4), dtype=bool)
        gt[1:3, 1:3, 1:3] = True
        current = np.zeros((4, 4, 4), dtype=bool)
        current[2, 2, 2] = True       # correct FG voxel
        current[0, 0, 0] = True       # false positive
        labels = error_labels_from_masks(current, gt)
        self.assertEqual(labels[2, 2, 2], ERR_CORRECT)
        self.assertEqual(labels[0, 0, 0], ERR_FALSE_POSITIVE)
        self.assertEqual(labels[1, 1, 1], ERR_FALSE_NEGATIVE)  # gt but not current
        self.assertEqual(labels[3, 3, 3], ERR_CORRECT)

    def test_loss_finite_and_positive(self):
        set_seed(2)
        model = EvidentialErrorNet3D(in_channels=2, base_channels=8)
        ev = model(torch.randn(2, 2, 12, 12, 12))
        labels = (torch.rand(2, 12, 12, 12) > 0.8).long()
        out = evidential_segmentation_loss(ev, labels, epoch=0)
        self.assertTrue(np.isfinite(float(out["loss"])))
        self.assertGreater(float(out["loss"]), 0.0)

    def test_loss_decreases_on_overfit(self):
        set_seed(3)
        model = EvidentialErrorNet3D(in_channels=2, base_channels=8)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        x = torch.randn(1, 2, 16, 16, 16)
        labels = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        labels[0, 4:8, 4:8, 4:8] = ERR_FALSE_NEGATIVE
        labels[0, 10:12, 10:12, 10:12] = ERR_FALSE_POSITIVE
        cw = inverse_frequency_class_weights(labels)
        first = None
        last = None
        for step in range(40):
            opt.zero_grad()
            out = evidential_segmentation_loss(model(x), labels, epoch=step, class_weights=cw)
            loss_value = float(out["loss"].detach())
            out["loss"].backward()
            opt.step()
            if first is None:
                first = loss_value
            last = loss_value
        self.assertLess(last, first * 0.7)

    def test_inverse_frequency_weights_favor_rare_classes(self):
        labels = torch.zeros(1000, dtype=torch.long)
        labels[:5] = ERR_FALSE_NEGATIVE   # rare
        w = inverse_frequency_class_weights(labels)
        self.assertGreater(float(w[ERR_FALSE_NEGATIVE]), float(w[ERR_CORRECT]))


class TestGtFreeCandidates(unittest.TestCase):
    def _maps_with_fn_blob(self, shape=(20, 20, 20), fn_center=(5, 5, 5), fp_center=None):
        prob = np.zeros((NUM_ERROR_CLASSES, *shape), dtype=np.float32)
        prob[ERR_CORRECT] = 1.0
        p_fn = np.zeros(shape, dtype=np.float32)
        p_fp = np.zeros(shape, dtype=np.float32)
        z, y, x = fn_center
        p_fn[z - 2 : z + 2, y - 2 : y + 2, x - 2 : x + 2] = 0.9
        if fp_center is not None:
            z2, y2, x2 = fp_center
            p_fp[z2 - 1 : z2 + 1, y2 - 1 : y2 + 1, x2 - 1 : x2 + 1] = 0.9
        prob[ERR_FALSE_NEGATIVE] = p_fn
        prob[ERR_FALSE_POSITIVE] = p_fp
        prob[ERR_CORRECT] = np.clip(1.0 - p_fn - p_fp, 0.0, 1.0)
        return ErrorMaps(
            prob=prob,
            p_error=(p_fn + p_fp).astype(np.float32),
            p_false_negative=p_fn,
            p_false_positive=p_fp,
            vacuity=np.full(shape, 0.1, dtype=np.float32),
            strength=np.full(shape, 30.0, dtype=np.float32),
        )

    def test_candidate_lands_in_planted_fn_blob(self):
        shape = (20, 20, 20)
        maps = self._maps_with_fn_blob(shape, fn_center=(5, 5, 5))
        current = np.zeros(shape, dtype=bool)  # blob is background -> valid FN
        cand = evidential_candidate(maps, current, polarity="positive")
        self.assertIsNotNone(cand)
        self.assertEqual(cand.action_type, POINT_POSITIVE)
        # chosen coord is inside the planted blob region
        z, y, x = cand.coord
        self.assertTrue(3 <= z < 7 and 3 <= y < 7 and 3 <= x < 7)

    def test_next_action_prefers_larger_error_mass(self):
        shape = (24, 24, 24)
        # large FN blob vs tiny FP blob -> should pick positive (FN)
        maps = self._maps_with_fn_blob(shape, fn_center=(6, 6, 6), fp_center=(18, 18, 18))
        current = np.zeros(shape, dtype=bool)
        current[17:20, 17:20, 17:20] = True  # make the FP region valid (currently foreground)
        action = evidential_next_action(maps, current)
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, POINT_POSITIVE)

    def test_stop_when_no_predicted_error(self):
        shape = (16, 16, 16)
        maps = _zero_maps(shape)
        current = np.zeros(shape, dtype=bool)
        decision = evidential_stop_decision(maps, current)
        self.assertTrue(decision.should_stop)
        self.assertIsNone(evidential_next_action(maps, current))

    def test_no_stop_with_large_error(self):
        shape = (20, 20, 20)
        maps = self._maps_with_fn_blob(shape, fn_center=(10, 10, 10))
        current = np.zeros(shape, dtype=bool)
        decision = evidential_stop_decision(maps, current)
        self.assertFalse(decision.should_stop)


class TestInferenceHelper(unittest.TestCase):
    def test_predict_error_maps_shapes(self):
        set_seed(5)
        model = EvidentialErrorNet3D(in_channels=2, base_channels=8)
        image = np.random.randn(18, 20, 16).astype(np.float32)
        mask = (np.random.rand(18, 20, 16) > 0.7)
        maps = predict_error_maps(model, image, mask, device="cpu")
        self.assertEqual(maps.p_error.shape, (18, 20, 16))
        self.assertEqual(maps.prob.shape, (NUM_ERROR_CLASSES, 18, 20, 16))
        self.assertTrue(np.isfinite(maps.vacuity).all())


if __name__ == "__main__":
    unittest.main()
