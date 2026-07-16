import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_nninteractive.dataset_manifest import load_manifest_cases
from rl_nninteractive.phase1_real import run_phase1_real


class DatasetManifestPhase1RealTests(unittest.TestCase):
    def test_loads_manifest_cases_and_defaults_initial_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_manifest(Path(tmp))

            train_cases = load_manifest_cases(manifest, split="train")
            val_cases = load_manifest_cases(manifest, split="val")

        self.assertEqual(len(train_cases), 1)
        self.assertEqual(len(val_cases), 1)
        self.assertEqual(train_cases[0].image.shape, (1, 3, 3, 3))
        self.assertEqual(train_cases[0].ground_truth.shape, (3, 3, 3))
        self.assertEqual(train_cases[0].initial_point, (1, 1, 1))

    def test_phase1_real_dry_run_validates_manifest_without_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_manifest(root)
            result = run_phase1_real(
                dataset_manifest=manifest,
                server_url="http://127.0.0.1:1527",
                output_dir=root / "out",
                dqn_episodes=3,
                dry_run_manifest=True,
            )

            summary_path = root / "out" / "phase1_real_summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "phase1 real manifest validated")
        self.assertEqual(payload["train_case_count"], 1)
        self.assertEqual(payload["validation_case_count"], 1)
        self.assertEqual(payload["dqn_episodes"], 3)
        self.assertIn("estimated_remote_env_steps_upper_bound", payload)

    def test_phase1_real_rejects_unbudgeted_remote_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_manifest(root)
            with self.assertRaisesRegex(ValueError, "estimated remote env steps"):
                run_phase1_real(
                    dataset_manifest=manifest,
                    server_url="http://127.0.0.1:1527",
                    output_dir=root / "out",
                    dqn_episodes=250000,
                    max_remote_env_steps=10000,
                    dry_run_manifest=False,
                )


def _write_manifest(root: Path) -> Path:
    image = np.zeros((3, 3, 3), dtype=np.float32)
    ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth[1, 1, 1] = 1
    np.save(root / "image.npy", image)
    np.save(root / "gt.npy", ground_truth)
    manifest = {
        "version": 1,
        "datasets": [
            {
                "name": "public_fixture",
                "cases": [
                    {
                        "case_id": "train_case",
                        "split": "train",
                        "image": "image.npy",
                        "ground_truth": "gt.npy",
                    },
                    {
                        "case_id": "val_case",
                        "split": "val",
                        "image": "image.npy",
                        "ground_truth": "gt.npy",
                        "initial_point": [1, 1, 1],
                    },
                ],
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
