import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_nninteractive.env import POINT_NEGATIVE, POINT_POSITIVE, STOP
from rl_nninteractive.mock_adapter import MockNnInteractiveSession
from rl_nninteractive.phase1_small import run_phase1_small
from rl_nninteractive.point_policy import (
    FEATURE_DIM,
    PointPolicy,
    collect_behavior_cloning_samples,
    fine_tune_dqn,
    point_action_candidates,
    rollout_point_policy,
    rollout_robot_user,
    train_behavior_cloning,
)
from rl_nninteractive.toy_dataset import synthetic_toy_cases


class CapacityLimitedSession(MockNnInteractiveSession):
    active = 0
    peak = 0
    capacity = 1

    def __init__(self):
        if CapacityLimitedSession.active >= CapacityLimitedSession.capacity:
            raise RuntimeError("server is at capacity")
        CapacityLimitedSession.active += 1
        CapacityLimitedSession.peak = max(
            CapacityLimitedSession.peak,
            CapacityLimitedSession.active,
        )
        super().__init__()
        self.closed = False

    def close(self):
        if not self.closed:
            CapacityLimitedSession.active -= 1
            self.closed = True


def reset_capacity_limited_session() -> None:
    CapacityLimitedSession.active = 0
    CapacityLimitedSession.peak = 0


class Phase1PointPolicyTests(unittest.TestCase):
    def test_point_action_candidates_rank_false_negative_then_false_positive(self):
        target = np.zeros((4, 4, 4), dtype=bool)
        current = np.zeros_like(target)
        target[1, 1, 1] = True
        target[1, 1, 2] = True
        current[3, 3, 3] = True

        candidates = point_action_candidates(current, target, top_k=3)

        self.assertEqual(candidates[0].action_type, POINT_POSITIVE)
        self.assertEqual(candidates[0].component_size, 2)
        self.assertEqual(candidates[1].action_type, POINT_NEGATIVE)
        self.assertEqual(candidates[-1].action_type, STOP)

    def test_behavior_cloning_learns_robot_candidate_on_toy_split(self):
        cases = synthetic_toy_cases("train")
        samples = collect_behavior_cloning_samples(cases, max_interactions=5, top_k=3)
        policy = train_behavior_cloning(samples, epochs=8)

        correct = 0
        for sample in samples:
            if policy.choose_index(sample.candidate_features) == sample.label_index:
                correct += 1

        self.assertGreaterEqual(correct, len(samples) - 1)

    def test_robot_label_is_top_non_stop_candidate_on_toy_split(self):
        samples = collect_behavior_cloning_samples(
            synthetic_toy_cases("train"),
            max_interactions=5,
            top_k=3,
        )

        for sample in samples:
            chosen = sample.candidates[sample.label_index]
            if chosen.action_type != STOP:
                with self.subTest(case=sample.case_name, step=sample.step_index):
                    self.assertEqual(sample.label_index, 0)

    def test_dqn_rollout_reaches_non_empty_final_masks(self):
        cases = synthetic_toy_cases("train")
        samples = collect_behavior_cloning_samples(cases, max_interactions=5, top_k=3)
        policy = fine_tune_dqn(
            train_behavior_cloning(samples, epochs=8),
            cases,
            episodes=4,
            max_interactions=5,
            top_k=3,
            epsilon=0.0,
        )

        episode = rollout_point_policy(policy, synthetic_toy_cases("val")[0], max_interactions=5)

        self.assertGreater(episode.final_dice, 0.0)
        self.assertTrue(episode.decisions)

    def test_dqn_reuses_one_remote_session_lease_for_multi_case_training(self):
        cases = synthetic_toy_cases("train")[:3]
        samples = collect_behavior_cloning_samples(cases, max_interactions=2, top_k=2)
        reset_capacity_limited_session()

        fine_tune_dqn(
            train_behavior_cloning(samples, epochs=1),
            cases,
            episodes=6,
            max_interactions=2,
            top_k=2,
            epsilon=0.0,
            session_factory=CapacityLimitedSession,
        )

        self.assertEqual(CapacityLimitedSession.peak, 1)
        self.assertEqual(CapacityLimitedSession.active, 0)

    def test_bc_and_validation_rollouts_release_capacity_limited_sessions(self):
        cases = synthetic_toy_cases("train")[:3]
        reset_capacity_limited_session()

        samples = collect_behavior_cloning_samples(
            cases,
            max_interactions=2,
            top_k=2,
            session_factory=CapacityLimitedSession,
        )

        self.assertTrue(samples)
        self.assertEqual(CapacityLimitedSession.peak, 1)
        self.assertEqual(CapacityLimitedSession.active, 0)

        policy = train_behavior_cloning(samples, epochs=1)
        reset_capacity_limited_session()
        rollout_point_policy(
            policy,
            cases[0],
            max_interactions=2,
            top_k=2,
            session_factory=CapacityLimitedSession,
        )
        rollout_robot_user(
            cases[1],
            max_interactions=2,
            session_factory=CapacityLimitedSession,
        )

        self.assertEqual(CapacityLimitedSession.peak, 1)
        self.assertEqual(CapacityLimitedSession.active, 0)

    def test_policy_rollout_allows_immediate_stop(self):
        weights = np.zeros(FEATURE_DIM, dtype=np.float64)
        weights[1] = 1.0
        policy = PointPolicy(weights=weights)

        episode = rollout_point_policy(
            policy,
            synthetic_toy_cases("val")[0],
            max_interactions=5,
        )

        self.assertEqual(episode.decisions[0].action_type, STOP)
        self.assertEqual(episode.dice_by_step, ())

    def test_phase1_small_writes_summary_and_reports_pivot_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_phase1_small(
                output_dir=Path(tmp),
                max_interactions=5,
                top_k=3,
                bc_epochs=8,
                dqn_episodes=4,
            )

            self.assertEqual(result["status"], "phase1 small-scale proof complete")
            self.assertEqual(result["dataset"], "synthetic_toy_v1")
            self.assertEqual(result["state_encoder"]["channel_count"], 5)
            self.assertEqual(len(result["comparison_rows"]), 3)
            self.assertTrue(result["noc_comparison"]["noc_at_85"]["comparable"])
            self.assertIn("decision", result["go_no_go"])
            self.assertTrue((Path(tmp) / "phase1_small_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
