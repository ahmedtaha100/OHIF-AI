from io import StringIO
import unittest

from scripts.present_pi_walkthrough import main, render_section


class PiWalkthroughTests(unittest.TestCase):
    def test_list_reports_all_named_sections_without_ansi(self):
        output = StringIO()

        result = main(["--list", "--no-color"], stream=output)

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertNotIn("\x1b[", rendered)
        for key in (
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
        self.assertNotIn("3. PAPER CLAIMS", rendered)

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

    def test_color_renderer_adds_ansi_without_changing_evidence(self):
        plain = render_section("release", width=100, color=False)
        colored = render_section("release", width=100, color=True)

        self.assertNotIn("\x1b[", plain)
        self.assertIn("\x1b[", colored)
        self.assertIn("0 / 154 keys match", plain)
        self.assertIn("0 / 154 keys match", colored)

    def test_interactive_quit_stops_before_second_section(self):
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
        self.assertIn("1. FROM BASELINE", rendered)
        self.assertNotIn("2. SEALED RL", rendered)
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
        self.assertIn("2. SEALED RL", output.getvalue())
        self.assertIn("8. FINAL PI DECISION", output.getvalue())

    def test_unknown_section_is_rejected_by_argparse(self):
        with self.assertRaises(SystemExit) as error:
            main(["--section", "not-a-section"], stream=StringIO())
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
