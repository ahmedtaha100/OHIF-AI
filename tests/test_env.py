import unittest

import numpy as np

from rl_nninteractive.env import (
    POINT_NEGATIVE,
    POINT_POSITIVE,
    STOP,
    RlNnInteractiveEnv,
)
from rl_nninteractive.mock_adapter import MockNnInteractiveSession


def _image(shape=(3, 3, 3)):
    return np.zeros((1, *shape), dtype=np.float32)


def _target_with_center(shape=(3, 3, 3)):
    target = np.zeros(shape, dtype=bool)
    target[1, 1, 1] = True
    return target


class RlNnInteractiveEnvTests(unittest.TestCase):
    def test_reset_accepts_initial_seed_and_stop_terminates(self):
        env = RlNnInteractiveEnv((3, 3, 3), max_interactions=2)
        obs, info = env.reset(
            options={
                "image": _image(),
                "ground_truth": _target_with_center(),
                "initial_point": (1, 1, 1),
            }
        )

        self.assertEqual(obs["image"].shape, (1, 3, 3, 3))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(int(obs["mask"].sum()), 1)
        self.assertEqual(float(info["dice"]), 1.0)
        self.assertTrue(info["initial_seed_used"])

        _, reward, terminated, truncated, stop_info = env.step(
            {"action_type": STOP, "coord": (0, 0, 0)}
        )
        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(stop_info["done_reason"], "stop")

    def test_point_actions_reward_delta_dice_against_ground_truth(self):
        env = RlNnInteractiveEnv((3, 3, 3), max_interactions=3)
        obs, info = env.reset(options={"image": _image(), "ground_truth": _target_with_center()})
        self.assertEqual(float(info["dice"]), 0.0)
        self.assertEqual(int(obs["mask"].sum()), 0)

        obs, reward, terminated, truncated, info = env.step(
            {"action_type": POINT_POSITIVE, "coord": (1, 1, 1)}
        )
        self.assertEqual(float(reward), 1.0)
        self.assertEqual(float(info["dice"]), 1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(int(obs["mask"].sum()), 1)

        _, reward, _, _, info = env.step(
            {"action_type": POINT_NEGATIVE, "coord": (1, 1, 1)}
        )
        self.assertEqual(float(reward), -1.0)
        self.assertEqual(float(info["dice"]), 0.0)

    def test_redundant_point_has_zero_delta_reward(self):
        env = RlNnInteractiveEnv((3, 3, 3), max_interactions=3)
        env.reset(
            options={
                "image": _image(),
                "ground_truth": _target_with_center(),
                "initial_point": (1, 1, 1),
            }
        )

        _, reward, _, _, info = env.step(
            {"action_type": POINT_POSITIVE, "coord": (1, 1, 1)}
        )

        self.assertEqual(float(reward), 0.0)
        self.assertEqual(float(info["dice"]), 1.0)

    def test_initial_include_false_is_allowed_but_not_positive(self):
        env = RlNnInteractiveEnv((3, 3, 3), max_interactions=2)
        obs, info = env.reset(
            options={
                "image": _image(),
                "ground_truth": _target_with_center(),
                "initial_point": (1, 1, 1),
                "initial_include": False,
            }
        )

        self.assertTrue(info["initial_seed_used"])
        self.assertEqual(int(obs["mask"].sum()), 0)
        self.assertEqual(float(info["dice"]), 0.0)

    def test_max_interactions_truncates_after_point_action(self):
        env = RlNnInteractiveEnv((3, 3, 3), max_interactions=1)
        env.reset(options={"image": _image(), "ground_truth": _target_with_center()})

        _, reward, terminated, truncated, info = env.step(
            {"action_type": POINT_POSITIVE, "coord": (1, 1, 1)}
        )

        self.assertEqual(float(reward), 1.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["done_reason"], "max_interactions")
        with self.assertRaisesRegex(RuntimeError, "episode ended"):
            env.step({"action_type": STOP, "coord": (0, 0, 0)})

    def test_reset_validates_shapes_and_required_options(self):
        env = RlNnInteractiveEnv((3, 3, 3))
        with self.assertRaisesRegex(ValueError, "requires options"):
            env.reset()
        with self.assertRaisesRegex(ValueError, "image shape"):
            env.reset(options={"image": np.zeros((3, 3, 3)), "ground_truth": _target_with_center()})
        with self.assertRaisesRegex(ValueError, "ground_truth shape"):
            env.reset(options={"image": _image(), "ground_truth": np.zeros((2, 2, 2))})

    def test_failed_reset_does_not_publish_partial_episode_state(self):
        env = RlNnInteractiveEnv((3, 3, 3), max_interactions=3)
        with self.assertRaisesRegex(ValueError, "inside volume_shape"):
            env.reset(
                options={
                    "image": _image(),
                    "ground_truth": _target_with_center(),
                    "initial_point": (3, 0, 0),
                }
            )
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            env.step({"action_type": STOP, "coord": (0, 0, 0)})

        env.reset(options={"image": _image(), "ground_truth": _target_with_center()})
        with self.assertRaisesRegex(ValueError, "inside volume_shape"):
            env.reset(
                options={
                    "image": _image(),
                    "ground_truth": np.ones((3, 3, 3), dtype=bool),
                    "initial_point": (3, 0, 0),
                }
            )

        _, reward, terminated, truncated, info = env.step(
            {"action_type": POINT_POSITIVE, "coord": (1, 1, 1)}
        )

        self.assertEqual(float(reward), 1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(float(info["dice"]), 1.0)

    def test_adapter_buffer_contract_failure_ends_episode(self):
        class BadShapeSession(MockNnInteractiveSession):
            def add_point_interaction(self, coord, *, include_interaction):
                result = super().add_point_interaction(
                    coord,
                    include_interaction=include_interaction,
                )
                self.target_buffer = np.zeros((1, 1, 1), dtype=np.uint8)
                return result

        env = RlNnInteractiveEnv(
            (3, 3, 3),
            max_interactions=3,
            session_factory=BadShapeSession,
        )
        env.reset(options={"image": _image(), "ground_truth": _target_with_center()})

        with self.assertRaisesRegex(RuntimeError, "target_buffer shape"):
            env.step({"action_type": POINT_POSITIVE, "coord": (1, 1, 1)})
        with self.assertRaisesRegex(RuntimeError, "episode ended"):
            env.step({"action_type": STOP, "coord": (0, 0, 0)})

    def test_reset_rejects_adapter_buffer_contract_failure_without_publishing_state(self):
        class BadResetBufferSession(MockNnInteractiveSession):
            def set_target_buffer(self, mask):
                self.target_buffer = np.zeros((1, 1, 1), dtype=np.uint8)

        env = RlNnInteractiveEnv(
            (3, 3, 3),
            max_interactions=3,
            session_factory=BadResetBufferSession,
        )

        with self.assertRaisesRegex(RuntimeError, "target_buffer shape"):
            env.reset(options={"image": _image(), "ground_truth": _target_with_center()})
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            env.step({"action_type": STOP, "coord": (0, 0, 0)})

    def test_adapter_buffer_must_be_binary(self):
        class NonBinarySession(MockNnInteractiveSession):
            def add_point_interaction(self, coord, *, include_interaction):
                result = super().add_point_interaction(
                    coord,
                    include_interaction=include_interaction,
                )
                self.target_buffer = self.target_buffer.astype(np.float32)
                self.target_buffer[coord] = 0.7
                return result

        env = RlNnInteractiveEnv(
            (3, 3, 3),
            max_interactions=3,
            session_factory=NonBinarySession,
        )
        env.reset(options={"image": _image(), "ground_truth": _target_with_center()})

        with self.assertRaisesRegex(RuntimeError, r"\{0, 1\}"):
            env.step({"action_type": POINT_POSITIVE, "coord": (1, 1, 1)})
        with self.assertRaisesRegex(RuntimeError, "episode ended"):
            env.step({"action_type": STOP, "coord": (0, 0, 0)})

    def test_invalid_action_and_coordinate_are_rejected(self):
        env = RlNnInteractiveEnv((3, 3, 3))
        env.reset(options={"image": _image(), "ground_truth": _target_with_center()})
        with self.assertRaisesRegex(ValueError, "inside volume_shape"):
            env.step({"action_type": POINT_POSITIVE, "coord": (3, 0, 0)})
        with self.assertRaisesRegex(ValueError, "action_type"):
            env.step({"action_type": 99, "coord": (0, 0, 0)})
        with self.assertRaisesRegex(ValueError, "requires action_type"):
            env.step({"coord": (0, 0, 0)})
        with self.assertRaisesRegex(ValueError, "integer"):
            env.step({"action_type": 1.7, "coord": (0, 0, 0)})
        with self.assertRaisesRegex(ValueError, "require coord"):
            env.step({"action_type": POINT_POSITIVE})
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            RlNnInteractiveEnv((3, 3, 3)).step({"action_type": STOP, "coord": (0, 0, 0)})

    def test_custom_session_factory_is_used_on_reset(self):
        sessions = []

        class RecordingSession(MockNnInteractiveSession):
            def __init__(self):
                super().__init__()
                self.set_image_count = 0
                self.reset_interactions_count = 0

            def set_image(self, image):
                super().set_image(image)
                self.set_image_count += 1

            def reset_interactions(self):
                super().reset_interactions()
                self.reset_interactions_count += 1

        def factory():
            session = RecordingSession()
            sessions.append(session)
            return session

        env = RlNnInteractiveEnv((3, 3, 3), session_factory=factory)
        env.reset(options={"image": _image(), "ground_truth": _target_with_center()})

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].set_image_count, 1)

        env.step({"action_type": STOP, "coord": (0, 0, 0)})
        env.reset(options={"image": _image(), "ground_truth": _target_with_center()})
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].set_image_count, 1)
        self.assertEqual(sessions[0].reset_interactions_count, 1)
        self.assertEqual(int(sessions[0].target_buffer.sum()), 0)

        changed_image = _image()
        changed_image[0, 0, 0, 0] = 1.0
        env.reset(options={"image": changed_image, "ground_truth": _target_with_center()})
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].set_image_count, 2)
        self.assertEqual(sessions[0].reset_interactions_count, 2)

    def test_spaces_match_volume_shape(self):
        env = RlNnInteractiveEnv((4, 5, 6), max_interactions=7)
        self.assertEqual(env.observation_space["image"].shape, (1, 4, 5, 6))
        self.assertEqual(env.observation_space["mask"].shape, (4, 5, 6))
        self.assertEqual(env.action_space["coord"].nvec.tolist(), [4, 5, 6])

    def test_constructor_requires_integer_shape_and_max_interactions(self):
        with self.assertRaisesRegex(ValueError, "volume_shape.*integers"):
            RlNnInteractiveEnv((3.5, 3, 3))
        with self.assertRaisesRegex(ValueError, "volume_shape.*integers"):
            RlNnInteractiveEnv((True, 3, 3))
        with self.assertRaisesRegex(ValueError, "max_interactions.*integers"):
            RlNnInteractiveEnv((3, 3, 3), max_interactions=1.5)

    def test_ground_truth_numeric_values_must_be_exactly_binary(self):
        env = RlNnInteractiveEnv((3, 3, 3))
        fractional = np.zeros((3, 3, 3), dtype=np.float32)
        fractional[1, 1, 1] = 0.5

        with self.assertRaisesRegex(ValueError, r"\{0, 1\}"):
            env.reset(options={"image": _image(), "ground_truth": fractional})

    def test_empty_ground_truth_behavior_is_explicit(self):
        env = RlNnInteractiveEnv((3, 3, 3), max_interactions=2)
        empty = np.zeros((3, 3, 3), dtype=bool)
        _, info = env.reset(options={"image": _image(), "ground_truth": empty})
        self.assertEqual(float(info["dice"]), 1.0)

        _, reward, _, _, info = env.step(
            {"action_type": POINT_POSITIVE, "coord": (1, 1, 1)}
        )
        self.assertEqual(float(reward), -1.0)
        self.assertEqual(float(info["dice"]), 0.0)


if __name__ == "__main__":
    unittest.main()
