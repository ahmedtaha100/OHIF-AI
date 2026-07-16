from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
finalizer = importlib.import_module("finalize_fusion_only_cohort_v2")


class FusionOnlySelectorBundleTests(unittest.TestCase):
    def test_evaluator_code_sha256_round_trips_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = finalizer.self_test_selector_integration(Path(temporary))

        self.assertTrue(result["actual_phase_a_schema_accepted"])
        self.assertTrue(result["selected_policies_exactly_mapped"])
        self.assertTrue(result["edl_code_sha256_preserved"])
        self.assertTrue(result["invalid_edl_code_sha256_rejected"])
        self.assertFalse(result["test_outcomes_opened"])


if __name__ == "__main__":
    unittest.main()
