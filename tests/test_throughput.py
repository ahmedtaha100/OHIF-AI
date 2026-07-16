import sys
import types
import unittest

import numpy as np

from rl_nninteractive.mock_adapter import MockNnInteractiveSession
from rl_nninteractive.throughput import (
    measure_point_throughput,
    parse_point,
    run_remote_point_throughput,
)


def _image(shape=(3, 3, 3)):
    return np.zeros((1, *shape), dtype=np.float32)


class ThroughputHarnessTests(unittest.TestCase):
    def test_measure_point_throughput_with_mock_session(self):
        session = MockNnInteractiveSession()

        result = measure_point_throughput(
            session,
            image=_image(),
            points=[(1, 1, 1), (1, 1, 2)],
            iterations=4,
            warmup_steps=1,
        )

        self.assertEqual(result.env_steps, 4)
        self.assertEqual(result.warmup_steps, 1)
        self.assertGreater(result.elapsed_sec, 0.0)
        self.assertGreater(result.env_steps_per_sec, 0.0)
        self.assertEqual(result.image_shape, (1, 3, 3, 3))
        self.assertEqual(result.target_shape, (3, 3, 3))
        self.assertEqual(result.mask_sum, 2)
        self.assertEqual(result.to_json_dict()["env_steps"], 4)
        self.assertFalse(result.to_json_dict()["set_image_timed"])
        self.assertEqual(result.to_json_dict()["timed_operation"], "add_point_interaction_only")

    def test_parse_point_requires_zyx(self):
        self.assertEqual(parse_point("1, 2, 3"), (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "z,y,x"):
            parse_point("1,2")

    def test_rejects_invalid_measurement_inputs(self):
        session = MockNnInteractiveSession()
        with self.assertRaisesRegex(ValueError, "shape"):
            measure_point_throughput(
                session,
                image=np.zeros((3, 3, 3), dtype=np.float32),
                points=[(1, 1, 1)],
                iterations=1,
            )
        with self.assertRaisesRegex(ValueError, "inside image"):
            measure_point_throughput(
                session,
                image=_image(),
                points=[(3, 0, 0)],
                iterations=1,
            )
        with self.assertRaisesRegex(ValueError, "iterations"):
            measure_point_throughput(
                session,
                image=_image(),
                points=[(1, 1, 1)],
                iterations=0,
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            measure_point_throughput(
                session,
                image=_image(),
                points=[(1, 1, 1)],
                iterations=1.5,
            )

    def test_remote_throughput_aggregates_parallel_sessions_with_fake_client(self):
        class FakeRemoteSession(MockNnInteractiveSession):
            def __init__(self, *, server_url, api_key=None):
                super().__init__()
                self.server_url = server_url
                self.api_key = api_key

        with _fake_remote_session_module(FakeRemoteSession):
            summary = run_remote_point_throughput(
                server_url="http://127.0.0.1:1527",
                image=_image(),
                points=[(1, 1, 1)],
                iterations=2,
                warmup_steps=0,
                parallel_sessions=2,
                server_max_sessions=2,
                image_label="fake",
            )

        self.assertEqual(summary["parallel_sessions_requested"], 2)
        self.assertEqual(summary["parallel_sessions_completed"], 2)
        self.assertEqual(summary["aggregate_env_steps"], 4)
        self.assertEqual(len(summary["session_results"]), 2)
        self.assertEqual(summary["image_label"], "fake")

    def test_remote_throughput_rejects_parallelism_above_server_cap(self):
        with self.assertRaisesRegex(ValueError, "server_max_sessions"):
            run_remote_point_throughput(
                server_url="http://127.0.0.1:1527",
                image=_image(),
                points=[(1, 1, 1)],
                iterations=1,
                warmup_steps=0,
                parallel_sessions=2,
                server_max_sessions=1,
            )


class _fake_remote_session_module:
    def __init__(self, session_cls):
        self.session_cls = session_cls
        self.originals = {}

    def __enter__(self):
        names = [
            "nnInteractive",
            "nnInteractive.inference",
            "nnInteractive.inference.remote",
            "nnInteractive.inference.remote.remote_session",
        ]
        for name in names:
            self.originals[name] = sys.modules.get(name)
            sys.modules[name] = types.ModuleType(name)
        sys.modules[
            "nnInteractive.inference.remote.remote_session"
        ].nnInteractiveRemoteInferenceSession = self.session_cls
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, original in self.originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        return False


if __name__ == "__main__":
    unittest.main()
