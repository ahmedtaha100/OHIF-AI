from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class DataManifestTests(unittest.TestCase):
    def test_manifest_lists_all_phase0_inputs(self):
        text = (REPO_ROOT / "docs" / "data_manifest.md").read_text(encoding="utf-8")

        for required in (
            "synthetic_unit_masks",
            "sample_manifest_example",
            "nibabel_anatomical_fixture",
            "nibabel_center_synthetic_gt",
            "nninteractive_v1_checkpoint",
            "phase0_artifacts",
            "CC BY-NC-SA 4.0",
            "not a tumor benchmark",
        ):
            self.assertIn(required, text)

    def test_repro_readme_links_manifest(self):
        text = (REPO_ROOT / "docs" / "rl-nninteractive-repro.md").read_text(encoding="utf-8")

        self.assertIn("docs/data_manifest.md", text)

    def test_tracked_example_manifest_exists(self):
        self.assertTrue(
            (REPO_ROOT / "sample-data" / "rl_nninteractive" / "data_manifest.example.json").is_file()
        )


if __name__ == "__main__":
    unittest.main()
