import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.render_fusion_only_pi import FUSION_ROUTES, _verdict, render


def _policy(
    final_dice,
    delta_dice,
    ci,
    *,
    coverage,
    harm_rate,
    wins,
    ties,
    losses,
):
    return {
        "study_estimand": {
            "n": 12,
            "mean_final_dice": final_dice,
            "mean_delta_dice": delta_dice,
        },
        "patient_estimand": {
            "n": 6,
            "mean_final_dice": final_dice,
            "mean_delta_dice": delta_dice,
            "mean_delta_nsd_2mm": delta_dice * 0.8,
            "paired_bootstrap_95_ci_delta_dice": {
                "lower": ci[0],
                "upper": ci[1],
            },
            "dice_win_tie_loss_vs_keep": {
                "wins": wins,
                "ties": ties,
                "losses": losses,
            },
        },
        "coverage": coverage,
        "harmful_action_rate_all_studies": harm_rate,
    }


def _synthetic_report(trap_path):
    policies = {
        "keep_resenc": _policy(
            0.60,
            0.0,
            (0.0, 0.0),
            coverage=0.0,
            harm_rate=0.0,
            wins=0,
            ties=6,
            losses=0,
        ),
        "edl_accept_best_utility": _policy(
            0.62,
            0.02,
            (0.005, 0.035),
            coverage=0.5,
            harm_rate=0.0,
            wins=4,
            ties=2,
            losses=0,
        ),
        "linear_contextual_bandit": _policy(
            0.61,
            0.01,
            (-0.01, 0.03),
            coverage=0.25,
            harm_rate=0.0,
            wins=2,
            ties=3,
            losses=1,
        ),
        "hindsight_oracle": _policy(
            0.65,
            0.05,
            (0.03, 0.07),
            coverage=0.75,
            harm_rate=0.0,
            wins=5,
            ties=1,
            losses=0,
        ),
    }
    for route, final_dice, delta, wins, ties, losses in (
        ("r1_intersection", 0.615, 0.015, 3, 2, 1),
        ("r2_intersection", 0.612, 0.012, 3, 1, 2),
        ("r1_union", 0.605, 0.005, 2, 2, 2),
        ("r2_union", 0.608, 0.008, 2, 3, 1),
    ):
        policies[f"fixed_{route}"] = _policy(
            final_dice,
            delta,
            (delta - 0.02, delta + 0.02),
            coverage=1.0,
            harm_rate=losses / 12,
            wins=wins,
            ties=ties,
            losses=losses,
        )
    return {
        "schema_version": 1,
        "synthetic": True,
        "status": "EXPLORATORY_INTERNAL_PRIOR_EXPOSED",
        "claim_boundary": "Synthetic internal renderer validation only.",
        "efficacy_claim_eligible": False,
        "external_validation_eligible": False,
        "manifest": {"path": str(trap_path)},
        "deployment_artifact": {
            "path": str(trap_path),
            "frozen_before_test_manifest_open": True,
        },
        "edl": {
            "deploy_keep_all": False,
            "checkpoint": str(trap_path),
            "selection": {
                "safety_deployed": True,
                "deployment_decision": "DEPLOY_SELECTED_POLICY",
            },
        },
        "linear_contextual_bandit": {"deployment_decision": "DEPLOY_SELECTED_POLICY"},
        "no_test_tuning_audit": {
            "test_label_evaluation_passes": 1,
            "test_used_for_model_or_threshold_selection": False,
        },
        "test_evaluation": {
            "scope": "frozen_test",
            "candidate_routes": list(FUSION_ROUTES),
            "test_label_evaluation_passes": 1,
            "study_count": 12,
            "patient_count": 6,
            "policies": policies,
            "per_study": [{"ground_truth_path": str(trap_path)}],
            "per_patient": [{"patient_id": "synthetic-do-not-use"}],
        },
    }


class FusionOnlyPiRendererTests(unittest.TestCase):
    def test_synthetic_report_renders_without_dereferencing_report_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trap = root / "must_not_be_opened.nii.gz"
            report = root / "synthetic_report.json"
            output = root / "summary.png"
            markdown = root / "summary.md"
            report.write_text(json.dumps(_synthetic_report(trap)), encoding="utf-8")
            original_read_text = Path.read_text
            reads = []

            def guarded_read_text(path, *args, **kwargs):
                resolved = path.resolve()
                reads.append(resolved)
                if resolved == trap.resolve():
                    raise AssertionError("renderer dereferenced a report-owned path")
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", guarded_read_text):
                result = render(report, output, markdown)

            self.assertEqual(reads, [report.resolve()])
            self.assertEqual(result["status"], "SYNTHETIC_RENDER_VALIDATED")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 20_000)
            self.assertIn("SYNTHETIC SMOKE TEST", markdown.read_text(encoding="utf-8"))
            self.assertIn("not online RL", markdown.read_text(encoding="utf-8"))

    def test_replacement_route_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "synthetic_report.json"
            output = root / "summary.png"
            payload = _synthetic_report(root / "trap.nii.gz")
            payload["test_evaluation"]["candidate_routes"][1] = "r1_replace"
            report.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "fusion-only route menu"):
                render(report, output)
            self.assertFalse(output.exists())

    def test_internal_report_cannot_be_rendered_with_efficacy_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "synthetic_report.json"
            payload = _synthetic_report(root / "trap.nii.gz")
            payload["efficacy_claim_eligible"] = True
            report.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "efficacy/external claim"):
                render(report)

    def test_keep_all_verdict_never_claims_improvement(self):
        verdict, tone = _verdict(
            {
                "mean_delta_dice": 0.0,
                "ci_lower": 0.0,
                "harm_rate": 0.0,
            },
            deploy_keep_all=True,
        )
        self.assertIn("abstained", verdict)
        self.assertIn("no segmentation gain", verdict)
        self.assertEqual(tone, "warning")


if __name__ == "__main__":
    unittest.main()
