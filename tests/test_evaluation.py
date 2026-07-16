import math
import unittest

from rl_nninteractive.evaluation import (
    evaluate_interaction_trajectory,
    summarize_interaction_evaluations,
)


class InteractionEvaluationTests(unittest.TestCase):
    def test_evaluates_noc_and_dice_at_steps(self):
        evaluation = evaluate_interaction_trajectory(
            "case-a",
            [0.2, 0.84, 0.86, 0.91],
        )

        self.assertEqual(evaluation.point_interaction_count, 4)
        self.assertAlmostEqual(evaluation.final_dice, 0.91)
        self.assertEqual(evaluation.noc_at_85, 3)
        self.assertEqual(evaluation.noc_at_90, 4)
        self.assertAlmostEqual(evaluation.dice_at[1], 0.2)
        self.assertAlmostEqual(evaluation.dice_at[3], 0.86)
        self.assertIsNone(evaluation.dice_at[5])
        json_dice_at = evaluation.to_json_dict()["dice_at"]
        self.assertAlmostEqual(json_dice_at["1"], 0.2)
        self.assertAlmostEqual(json_dice_at["3"], 0.86)
        self.assertIsNone(json_dice_at["5"])

    def test_accepts_explicit_final_dice_for_empty_trajectory(self):
        evaluation = evaluate_interaction_trajectory(
            "already-perfect",
            [],
            final_dice=1.0,
        )

        self.assertEqual(evaluation.point_interaction_count, 0)
        self.assertEqual(evaluation.final_dice, 1.0)
        self.assertIsNone(evaluation.noc_at_85)
        self.assertIsNone(evaluation.noc_at_90)
        self.assertEqual(evaluation.dice_at, {1: None, 3: None, 5: None})

    def test_rejects_invalid_scores(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_interaction_trajectory("bad", [math.nan])
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            evaluate_interaction_trajectory("bad", [1.2])
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            evaluate_interaction_trajectory("bad", [], final_dice=-0.1)

    def test_summarizes_multiple_evaluations(self):
        rows = [
            evaluate_interaction_trajectory("a", [0.5, 0.9]),
            evaluate_interaction_trajectory("b", [0.2, 0.4], final_dice=0.4),
        ]

        summary = summarize_interaction_evaluations(rows)

        self.assertEqual(summary["case_count"], 2)
        self.assertAlmostEqual(summary["mean_final_dice"], 0.65)
        self.assertAlmostEqual(summary["mean_point_interactions"], 2.0)
        self.assertEqual(summary["reached_85_count"], 1)
        self.assertEqual(summary["reached_90_count"], 1)
        self.assertEqual(len(summary["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
