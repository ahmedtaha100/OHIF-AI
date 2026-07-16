import inspect
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

from rl_nninteractive.adapter import (
    InteractionResult,
    NnInteractiveSession,
    as_box3d,
    as_voxel_coord,
)
from rl_nninteractive.mock_adapter import MockNnInteractiveSession
from rl_nninteractive.nninteractive_contract import (
    EXPECTED_NNINTERACTIVE_METHOD_PARAMS,
    NNINTERACTIVE_CHECKPOINT_LICENSE,
    NNINTERACTIVE_REQUIREMENT,
    NNINTERACTIVE_REQUIREMENT_SOURCE,
    NNINTERACTIVE_SOURCE_URL,
)


class FakeSession:
    def __init__(self):
        self.image = None
        self.target_buffer = None
        self.interactions = []

    def set_image(self, image):
        self.image = image

    def set_target_buffer(self, mask):
        self.target_buffer = mask.copy()

    def add_point_interaction(self, coord, *, include_interaction):
        self.interactions.append(("point", coord, include_interaction))
        return InteractionResult(changed_bbox=((0, 1), (0, 1), (0, 1)))

    def add_bbox_interaction(self, box, *, include_interaction):
        self.interactions.append(("bbox", box, include_interaction))
        return InteractionResult(changed_bbox=box)

    def add_scribble_interaction(
        self,
        scribble_image,
        *,
        include_interaction,
        interaction_bbox=None,
    ):
        self.interactions.append(
            ("scribble", scribble_image.shape, include_interaction, interaction_bbox)
        )
        return InteractionResult()

    def add_lasso_interaction(
        self,
        lasso_image,
        *,
        include_interaction,
        interaction_bbox=None,
    ):
        self.interactions.append(
            ("lasso", lasso_image.shape, include_interaction, interaction_bbox)
        )
        return InteractionResult()

    def reset_interactions(self):
        self.interactions.clear()


class AdapterInterfaceTests(unittest.TestCase):
    def test_fake_session_exercises_protocol_method_shapes(self):
        session = FakeSession()
        image = np.zeros((1, 4, 4, 4), dtype=np.float32)
        mask = np.zeros((4, 4, 4), dtype=bool)
        scribble = np.zeros_like(mask)
        bbox = ((0, 2), (0, 2), (0, 2))

        session.set_image(image)
        session.set_target_buffer(mask)
        point_result = session.add_point_interaction(
            (1, 2, 3),
            include_interaction=True,
        )
        box_result = session.add_bbox_interaction(bbox, include_interaction=True)
        session.add_scribble_interaction(
            scribble_image=scribble,
            include_interaction=True,
            interaction_bbox=bbox,
        )
        session.add_lasso_interaction(
            scribble,
            include_interaction=False,
            interaction_bbox=bbox,
        )

        self.assertEqual(session.target_buffer.shape, mask.shape)
        self.assertEqual(point_result.changed_bbox, ((0, 1), (0, 1), (0, 1)))
        self.assertEqual(box_result.changed_bbox, bbox)
        self.assertEqual(len(session.interactions), 4)
        session.reset_interactions()
        self.assertEqual(session.interactions, [])

    def test_protocol_signatures_match_basic_infer_call_contract(self):
        self.assertEqual(NNINTERACTIVE_REQUIREMENT, "nninteractive==2.5.0")
        self.assertIn("pypi.org/project/nninteractive/2.5.0", NNINTERACTIVE_SOURCE_URL)
        self.assertIn("CC-BY-NC-SA 4.0", NNINTERACTIVE_CHECKPOINT_LICENSE)
        self.assertIn("unverified pinned-stub target", NNINTERACTIVE_REQUIREMENT_SOURCE)
        for method_name, expected_params in EXPECTED_NNINTERACTIVE_METHOD_PARAMS.items():
            signature = inspect.signature(getattr(NnInteractiveSession, method_name))
            for param_name in expected_params:
                self.assertIn(param_name, signature.parameters)
        self.assertFalse(hasattr(NnInteractiveSession, "read_target_buffer"))

        basic_infer = Path("monai-label/monailabel/tasks/infer/basic_infer.py")
        if not basic_infer.exists():
            self.skipTest("basic_infer.py not present in this checkout")
        source = basic_infer.read_text(encoding="utf-8")
        expected_call_tokens = [
            "session.target_buffer.clone()",
            "session.add_bbox_interaction(",
            "scribble_image=img",
            "interaction_bbox=bbox",
        ]
        for token in expected_call_tokens:
            self.assertIn(token, source)

    def test_real_nninteractive_session_matches_pinned_contract_when_available(self):
        require_real = os.environ.get("RL_NNINTERACTIVE_REQUIRE_REAL") == "1"
        if importlib.util.find_spec("nnInteractive") is None:
            if require_real:
                self.fail("RL_NNINTERACTIVE_REQUIRE_REAL=1 but nnInteractive is absent")
            self.skipTest("nnInteractive package is not installed in the scaffold venv")

        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

        for method_name, expected_params in EXPECTED_NNINTERACTIVE_METHOD_PARAMS.items():
            signature = inspect.signature(getattr(nnInteractiveInferenceSession, method_name))
            for param_name in expected_params:
                self.assertIn(param_name, signature.parameters)

    def test_importable_mock_adapter_is_clearly_mock_and_mutates_target_buffer(self):
        session = MockNnInteractiveSession()
        target = np.zeros((3, 3, 3), dtype=np.uint8)
        session.set_target_buffer(target)

        point_result = session.add_point_interaction((1, 1, 1), include_interaction=True)
        self.assertEqual(point_result.changed_bbox, ((1, 2), (1, 2), (1, 2)))
        self.assertEqual(int(session.target_buffer[1, 1, 1]), 1)

        box = ((0, 2), (0, 2), (0, 2))
        box_result = session.add_bbox_interaction(box, include_interaction=False)
        self.assertEqual(box_result.changed_bbox, box)
        self.assertEqual(int(session.target_buffer[1, 1, 1]), 0)

        empty = np.zeros((3, 3, 3), dtype=bool)
        empty_result = session.add_lasso_interaction(
            empty,
            include_interaction=True,
            interaction_bbox=None,
        )
        self.assertIsNone(empty_result.changed_bbox)

    def test_coordinate_normalizers_require_3d_shapes(self):
        self.assertEqual(as_voxel_coord([1, 2, 3]), (1, 2, 3))
        self.assertEqual(as_box3d(([0, 3], [1, 4], [2, 5])), ((0, 3), (1, 4), (2, 5)))

        with self.assertRaisesRegex(ValueError, "3 values"):
            as_voxel_coord([1, 2])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            as_voxel_coord([1, -2, 3])
        with self.assertRaisesRegex(ValueError, "integers"):
            as_voxel_coord([1.5, 2, 3])
        with self.assertRaisesRegex(ValueError, "3 axis ranges"):
            as_box3d(((0, 1), (2, 3)))
        with self.assertRaisesRegex(ValueError, "2 values"):
            as_box3d(((0, 1), (2, 3), (4, 5, 6)))
        with self.assertRaisesRegex(ValueError, "greater than start"):
            as_box3d(((1, 1), (2, 3), (4, 5)))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            as_box3d(((-1, 1), (2, 3), (4, 5)))
        with self.assertRaisesRegex(ValueError, "integers"):
            as_box3d(((0, 1), (2.5, 3), (4, 5)))


if __name__ == "__main__":
    unittest.main()
