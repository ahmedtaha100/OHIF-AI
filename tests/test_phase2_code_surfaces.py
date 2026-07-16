import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_nninteractive.deterministic_geometry import (
    build_component_geometry,
    largest_error_component_mask,
)
from rl_nninteractive.eval_harness import (
    classify_failure,
    default_ablation_grid,
    summarize_ablation_results,
)
from rl_nninteractive.interaction_log import InteractionLogEvent
from rl_nninteractive.learning_loops import (
    calibrate_stop_threshold,
    dagger_samples_from_logs,
    preference_pairs_from_logs,
)
from rl_nninteractive.mock_adapter import MockNnInteractiveSession
from rl_nninteractive.multitool import dispatch_multi_tool_action, multi_tool_candidates
from rl_nninteractive.phase2_smoke import run_phase2_smoke
from rl_nninteractive.recommender import recommend_next_prompt
from rl_nninteractive.safety_reward import safety_shaped_reward
from rl_nninteractive.toy_dataset import synthetic_toy_cases
from rl_nninteractive.uncertainty import append_uncertainty_channel, tta_disagreement_channel


class Phase2CodeSurfaceTests(unittest.TestCase):
    def test_deterministic_geometry_builds_all_tools(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 1:4] = True

        geometry = build_component_geometry(mask, tool="lasso", polarity="positive")

        self.assertEqual(geometry.tool, "lasso")
        self.assertEqual(geometry.component_size, 3)
        self.assertEqual(geometry.bbox, ((2, 3), (2, 3), (1, 4)))
        self.assertGreater(int(geometry.scribble.sum()), 0)
        self.assertGreater(int(geometry.lasso.sum()), 0)

    def test_multi_tool_candidates_dispatch_to_mock_adapter(self):
        case = synthetic_toy_cases("val")[0]
        current = np.zeros_like(case.ground_truth)
        actions = multi_tool_candidates(current, case.ground_truth)
        session = MockNnInteractiveSession()
        session.set_image(case.image)
        session.set_target_buffer(np.zeros_like(case.ground_truth, dtype=np.uint8))

        result = dispatch_multi_tool_action(session, actions[0])

        self.assertIsNotNone(result)
        self.assertGreater(int(session.target_buffer.sum()), 0)
        self.assertEqual(actions[-1].tool, "stop")

    def test_uncertainty_channel_appends_disagreement(self):
        predictions = np.zeros((3, 2, 2, 2), dtype=np.uint8)
        predictions[1, 0, 0, 0] = 1
        predictions[2, 0, 0, 0] = 1
        disagreement = tta_disagreement_channel(predictions)

        channels = append_uncertainty_channel(np.zeros((5, 2, 2, 2), dtype=np.float32), disagreement)

        self.assertEqual(channels.shape, (6, 2, 2, 2))
        self.assertGreater(float(disagreement[0, 0, 0]), 0.0)

    def test_safety_reward_penalizes_leakage_and_rewards_good_stop(self):
        target = np.zeros((3, 3, 3), dtype=bool)
        target[1, 1, 1] = True
        current = target.copy()
        current[0, 0, 0] = True
        organ = np.zeros_like(target)
        organ[1, 1, 1] = True

        reward = safety_shaped_reward(
            previous_mask=np.zeros_like(target),
            current_mask=current,
            ground_truth=target,
            organ_mask=organ,
            is_stop=True,
            stop_threshold=0.50,
        )

        self.assertGreater(reward.stop_reward, 0.0)
        self.assertGreater(reward.leakage_penalty, 0.0)

    def test_recommender_and_logs_feed_learning_helpers(self):
        case = synthetic_toy_cases("val")[0]
        suggestion = recommend_next_prompt(
            case_id=case.name,
            current_mask=np.zeros_like(case.ground_truth),
            ground_truth_for_mock=case.ground_truth,
        )
        event = InteractionLogEvent(
            case_id=case.name,
            step_index=0,
            tool=suggestion.action.tool,
            decision="edited",
            proposed_prompt=suggestion.to_json_dict(),
            final_prompt={"tool": "box"},
            elapsed_ms=10,
        )

        self.assertTrue(suggestion.requires_review)
        self.assertEqual(len(dagger_samples_from_logs([event])), 1)
        self.assertEqual(preference_pairs_from_logs([event])[0].preferred_tool, "box")
        self.assertGreaterEqual(calibrate_stop_threshold([0.80, 0.95]), 0.90)

    def test_eval_harness_failure_and_ablation_summary(self):
        target = np.zeros((3, 3, 3), dtype=bool)
        target[1, 1, 1] = True
        prediction = np.zeros_like(target)
        prediction[0, 0, 0] = True
        organ = np.zeros_like(target)
        organ[1, 1, 1] = True
        failures = classify_failure(prediction, target, organ)
        grid = default_ablation_grid()
        summary = summarize_ablation_results(
            {"ablation": config.name, "case": "case", "failure": failures[0]}
            for config in grid
        )

        self.assertIn("missed_target", failures)
        self.assertIn("leakage_outside_organ", failures)
        self.assertEqual(summary["ablation_count"], len(grid))

    def test_phase2_smoke_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_phase2_smoke(output_dir=Path(tmp))

            self.assertEqual(result["status"], "phase2-4 code smoke complete")
            self.assertEqual(result["state_channel_count_with_uncertainty"], 6)
            self.assertGreaterEqual(result["candidate_count"], 5)
            self.assertTrue((Path(tmp) / "phase2_smoke_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
