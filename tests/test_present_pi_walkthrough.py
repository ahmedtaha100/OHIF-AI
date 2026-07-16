from io import StringIO
from pathlib import Path
import tempfile
import unittest

from scripts.present_pi_walkthrough import (
    TourStop,
    check_tour_readiness,
    main,
    render_section,
)


class PiWalkthroughTests(unittest.TestCase):
    def test_list_reports_all_named_sections_without_ansi(self):
        output = StringIO()

        result = main(["--list", "--no-color"], stream=output)

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertNotIn("\x1b[", rendered)
        for key in (
            "files",
            "pipeline",
            "policy",
            "paper",
            "release",
            "code",
            "statistics",
            "compute",
            "decision",
        ):
            self.assertIn(key, rendered)

    def test_selected_policy_section_contains_sealed_primary_result(self):
        output = StringIO()

        result = main(
            ["--section", "policy", "--mode", "auto", "--delay", "0", "--no-color"],
            stream=output,
        )

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertIn("0.608227", rendered)
        self.assertIn("-0.003204", rendered)
        self.assertIn("[-0.009612, 0]", rendered)
        self.assertIn("1 / 12", rendered)
        self.assertIn("Keep ResEnc-L / KEEP", rendered)
        self.assertIn("train_0025_PSMA", rendered)
        self.assertIn("P(accept) 0.7377", rendered)
        self.assertIn("It was not an RL network", rendered)
        self.assertNotIn("1. PAPER CLAIMS", rendered)

    def test_full_walkthrough_exposes_each_audited_discrepancy(self):
        output = StringIO()

        result = main(
            ["--mode", "auto", "--delay", "0", "--no-color", "--width", "104"],
            stream=output,
        )

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        expected_fragments = (
            "0 / 154 keys match",
            "10.1007/s11548-026-03628-w",
            "FYP/logs_bk/run_20250211_003345/final_model.pth",
            "Ground-truth distance-transform",
            "center-region prompt",
            "Four channels: 3 MR + current",
            "Six channels: 3 MR + current",
            "Logical union / accumulation",
            "terminated=False; only",
            "maximum-step truncation",
            "BatchNorm / ReLU / residual",
            "70:15:15 plus materially",
            "different YAML and runtime",
            "0.0006419",
            "7.086e-22",
            "8.3435x",
            "p=0.14",
            "Would an H200 have changed the result?".upper(),
            "does not establish fabrication or deliberate deception",
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, rendered)
        self.assertNotIn("\x1b[", rendered)

    def test_files_section_gives_exact_ordered_paths_and_symbols(self):
        rendered = render_section("files", width=108, color=False)

        expected_in_order = (
            "docs/pi-presentation-guide.md",
            "scripts/prepare_autopet_nnunet_input.py",
            "scripts/run_fusion_only_cohort_v2.py",
            "scripts/finalize_fusion_only_cohort_v2.py",
            "rl_nninteractive/evidential.py",
            "rl_nninteractive/evidential_candidates.py",
            "rl_nninteractive/rl_policy.py",
            "rl_nninteractive/autopet_rl_recovery.py",
            "rl_nninteractive/prompt_update_edl.py",
            "rl_nninteractive/edl_fusion_hybrid.py",
            "scripts/run_edl_hybrid_test_once.py",
            "rl_nninteractive/route_policy_eval.py",
            "scripts/present_pi_walkthrough.py",
        )
        positions = [rendered.index(path) for path in expected_in_order]
        self.assertEqual(positions, sorted(positions))
        for fragment in (
            "prepare_autopet_nnunet_input.py",
            "stage_prompt_round",
            "evidential_stop_decision",
            "RealEdlEnv",
            "RecoveryPolicy",
            "EvidentialUtilityHead",
            "select_frozen_policy_routes",
            "_score_both_frozen_policies_once",
            "_bootstrap_ci",
            "_release_forensics",
            "oracle-assisted",
            "no learned STOP or value network",
            "not an RL network",
        ):
            self.assertIn(fragment, rendered)

    def test_readiness_flag_passes_for_checked_out_repository(self):
        output = StringIO()

        result = main(["--check-readiness", "--no-color"], stream=output)

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertIn("PI PRESENTATION READINESS: PASS", rendered)
        self.assertIn("Verified 13 files and 46 symbols/sections", rendered)
        self.assertNotIn("\x1b[", rendered)

    def test_readiness_reports_missing_file_and_symbol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "present.py").write_text(
                "def available():\n    return True\n", encoding="utf-8"
            )
            stops = (
                TourStop(
                    "Demo",
                    "present.py",
                    ("available", "missing"),
                    "show",
                    "why",
                ),
                TourStop("Absent", "absent.py", ("anything",), "show", "why"),
            )

            errors = check_tour_readiness(root, stops=stops)

        self.assertEqual(
            errors,
            (
                "MISSING ANCHOR: present.py :: missing",
                "MISSING FILE: absent.py",
            ),
        )

    def test_color_renderer_adds_ansi_without_changing_evidence(self):
        plain = render_section("release", width=100, color=False)
        colored = render_section("release", width=100, color=True)

        self.assertNotIn("\x1b[", plain)
        self.assertIn("\x1b[", colored)
        self.assertIn("0 / 154 keys match", plain)
        self.assertIn("0 / 154 keys match", colored)

    def test_default_interactive_order_starts_with_paper_discrepancies(self):
        output = StringIO()
        prompts = []

        def quit_after_first(prompt):
            prompts.append(prompt)
            return "q"

        result = main(
            ["--mode", "interactive", "--no-color"],
            stream=output,
            input_fn=quit_after_first,
        )

        self.assertEqual(result, 0)
        self.assertEqual(len(prompts), 1)
        rendered = output.getvalue()
        self.assertIn("1. PAPER CLAIMS VERSUS THE PUBLIC RELEASE", rendered)
        self.assertNotIn("2. WHY THE RELEASE", rendered)
        self.assertNotIn("5. LIVE REPOSITORY FILE TOUR", rendered)
        self.assertIn("Walkthrough stopped by presenter.", rendered)

    def test_auto_mode_waits_only_between_selected_sections(self):
        output = StringIO()
        delays = []

        result = main(
            [
                "--section",
                "policy",
                "--section",
                "decision",
                "--mode",
                "auto",
                "--delay",
                "0.2",
                "--no-color",
            ],
            stream=output,
            sleep_fn=delays.append,
        )

        self.assertEqual(result, 0)
        self.assertEqual(delays, [0.2])
        self.assertIn("7. SEALED SAFETY-SCREEN", output.getvalue())
        self.assertIn("9. FINAL PI DECISION", output.getvalue())

    def test_unknown_section_is_rejected_by_argparse(self):
        with self.assertRaises(SystemExit) as error:
            main(["--section", "not-a-section"], stream=StringIO())
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
