import json
import tempfile
import unittest
from pathlib import Path

from rl_nninteractive.blackwell_handoff import (
    build_blackwell_handoff,
    write_handoff_files,
)


class BlackwellHandoffTests(unittest.TestCase):
    def test_builds_remaining_gpu_runbook(self):
        handoff = build_blackwell_handoff(
            server_url="http://127.0.0.1:1527",
            dataset_manifest=Path("manifests/blackwell_datasets.json"),
            output_dir=Path("artifacts/out"),
            max_sessions=6,
            env_count=6,
        )

        self.assertEqual(handoff["status"], "blackwell handoff ready")
        self.assertIn("--port 1527", handoff["server_start_command"])
        self.assertIn("--max-sessions 6", handoff["server_start_command"])
        self.assertIn("--interactions-storage blosc2", handoff["server_start_command"])
        self.assertNotIn("92-252", handoff["total_gpu_hours_estimate"])
        items = [item["plan_item"] for item in handoff["remaining_runs"]]
        self.assertIn("Phase 0 throughput harness vs the remote inference server", items)
        self.assertIn("Phase 4 multi-tumor evaluation vs all baselines", items)

    def test_writes_json_and_markdown(self):
        handoff = build_blackwell_handoff(
            server_url="http://127.0.0.1:1527",
            dataset_manifest=Path("manifest.json"),
            output_dir=Path("unused"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_handoff_files(handoff, out)

            payload = json.loads((out / "blackwell_handoff.json").read_text(encoding="utf-8"))
            markdown = (out / "blackwell_handoff.md").read_text(encoding="utf-8")

        self.assertEqual(payload["claim"], "runbook only; no large-scale result is claimed")
        self.assertIn("# Blackwell Handoff", markdown)
        self.assertIn("make throughput-remote", markdown)
        self.assertIn("not rental-ready", markdown)


if __name__ == "__main__":
    unittest.main()
