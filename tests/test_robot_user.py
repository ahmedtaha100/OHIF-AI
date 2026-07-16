import unittest

import numpy as np

from rl_nninteractive.env import POINT_NEGATIVE, POINT_POSITIVE, STOP, RlNnInteractiveEnv
from rl_nninteractive.robot_user import (
    largest_component_robot_action,
    run_largest_component_robot_user,
)


def _image(shape=(3, 3, 3)):
    return np.zeros((1, *shape), dtype=np.float32)


class RobotUserBaselineTests(unittest.TestCase):
    def test_selects_largest_false_negative_component_as_positive_point(self):
        current = np.zeros((3, 3, 3), dtype=bool)
        target = np.zeros_like(current)
        target[1, 1, 1] = True
        target[1, 1, 2] = True

        decision = largest_component_robot_action(current, target)

        self.assertEqual(decision.action_type, POINT_POSITIVE)
        self.assertEqual(decision.error_kind, "false_negative")
        self.assertEqual(decision.component_size, 2)
        self.assertEqual(decision.coord, (1, 1, 1))
        self.assertEqual(
            decision.to_env_action(),
            {"action_type": POINT_POSITIVE, "coord": (1, 1, 1)},
        )

    def test_selects_largest_false_positive_component_as_negative_point(self):
        current = np.zeros((3, 3, 3), dtype=bool)
        target = np.zeros_like(current)
        target[0, 0, 0] = True
        current[2, 2, 2] = True
        current[2, 2, 1] = True
        current[2, 1, 2] = True

        decision = largest_component_robot_action(current, target)

        self.assertEqual(decision.action_type, POINT_NEGATIVE)
        self.assertEqual(decision.error_kind, "false_positive")
        self.assertEqual(decision.component_size, 3)
        self.assertEqual(decision.coord, (2, 2, 2))

    def test_equal_size_error_components_prefer_false_negative(self):
        current = np.zeros((3, 3, 3), dtype=bool)
        target = np.zeros_like(current)
        current[0, 0, 0] = True
        target[2, 2, 2] = True

        decision = largest_component_robot_action(current, target)

        self.assertEqual(decision.action_type, POINT_POSITIVE)
        self.assertEqual(decision.error_kind, "false_negative")
        self.assertEqual(decision.coord, (2, 2, 2))

    def test_stop_when_current_mask_matches_ground_truth(self):
        target = np.zeros((3, 3, 3), dtype=bool)
        target[1, 1, 1] = True

        decision = largest_component_robot_action(target, target)

        self.assertEqual(decision.action_type, STOP)
        self.assertEqual(decision.error_kind, "none")
        self.assertEqual(decision.component_size, 0)
        self.assertEqual(decision.coord, (0, 0, 0))

    def test_rejects_invalid_masks(self):
        current = np.zeros((3, 3, 3), dtype=bool)
        target = np.zeros_like(current)
        fractional = np.zeros_like(current, dtype=np.float32)
        fractional[1, 1, 1] = 0.5

        with self.assertRaisesRegex(ValueError, "3D volume"):
            largest_component_robot_action(np.zeros((3, 3), dtype=bool), target)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            largest_component_robot_action(current, np.zeros((4, 4, 4), dtype=bool))
        with self.assertRaisesRegex(ValueError, r"\{0, 1\}"):
            largest_component_robot_action(fractional, target)

    def test_runner_reaches_synthetic_target_with_mock_env(self):
        target = np.zeros((3, 3, 3), dtype=bool)
        target[1, 1, 1] = True
        target[1, 1, 2] = True
        env = RlNnInteractiveEnv((3, 3, 3), max_interactions=5)

        episode = run_largest_component_robot_user(
            env,
            image=_image(),
            ground_truth=target,
        )

        self.assertEqual(
            [decision.action_type for decision in episode.decisions],
            [POINT_POSITIVE, POINT_POSITIVE, STOP],
        )
        self.assertEqual(len(episode.dice_by_step), 2)
        self.assertAlmostEqual(episode.dice_by_step[0], 2 / 3)
        self.assertAlmostEqual(episode.dice_by_step[1], 1.0)
        self.assertAlmostEqual(episode.total_reward, 1.0)
        self.assertTrue(episode.terminated)
        self.assertFalse(episode.truncated)
        self.assertEqual(float(episode.final_info["dice"]), 1.0)

    def test_runner_removes_false_positive_from_empty_target(self):
        env = RlNnInteractiveEnv((3, 3, 3), max_interactions=5)
        empty_target = np.zeros((3, 3, 3), dtype=bool)

        episode = run_largest_component_robot_user(
            env,
            image=_image(),
            ground_truth=empty_target,
            initial_point=(1, 1, 1),
        )

        self.assertEqual(
            [decision.action_type for decision in episode.decisions],
            [POINT_NEGATIVE, STOP],
        )
        self.assertEqual(episode.dice_by_step, (1.0,))
        self.assertAlmostEqual(episode.total_reward, 1.0)
        self.assertTrue(episode.terminated)
        self.assertFalse(episode.truncated)
        self.assertEqual(float(episode.final_info["dice"]), 1.0)

    def test_runner_validates_positive_step_limit(self):
        env = RlNnInteractiveEnv((3, 3, 3), max_interactions=5)
        target = np.zeros((3, 3, 3), dtype=bool)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_largest_component_robot_user(
                env,
                image=_image(),
                ground_truth=target,
                max_steps=1.5,
            )
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            env.step({"action_type": STOP, "coord": (0, 0, 0)})


if __name__ == "__main__":
    unittest.main()
