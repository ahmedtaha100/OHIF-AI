import json
import tempfile
import unittest
from pathlib import Path

from rl_nninteractive.config import RuntimeConfig, load_config


class RuntimeConfigTests(unittest.TestCase):
    def test_loads_skeleton_config(self):
        config = load_config("configs/rl_nninteractive_skeleton.json")

        self.assertIsInstance(config, RuntimeConfig)
        self.assertEqual(config.seed, 20260704)
        self.assertEqual(config.environment, {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertTrue(config.mock_mode)
        self.assertEqual(config.max_interactions, 5)
        self.assertTrue(config.dataset_manifest.endswith("data_manifest.example.json"))

    def test_rejects_invalid_max_interactions(self):
        payload = {
            "seed": 1,
            "cuda_visible_devices": "0",
            "max_interactions": 0,
            "mock_mode": True,
            "nninteractive_endpoint": None,
            "dataset_manifest": "data_manifest.md",
            "output_dir": "artifacts/rl_nninteractive",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "max_interactions"):
                load_config(path)

    def test_rejects_missing_dataset_manifest(self):
        payload = {
            "seed": 1,
            "cuda_visible_devices": "0",
            "max_interactions": 5,
            "mock_mode": True,
            "nninteractive_endpoint": None,
            "dataset_manifest": "missing-manifest.json",
            "output_dir": "artifacts/rl_nninteractive",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "dataset_manifest"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
