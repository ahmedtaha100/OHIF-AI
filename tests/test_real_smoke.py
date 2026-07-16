import unittest

from rl_nninteractive.real_adapter import (
    box3d_from_changed_bbox,
    center_point_for_shape,
    normalize_changed_bbox,
    parse_point,
)


class RealSmokeHelperTests(unittest.TestCase):
    def test_center_point_for_shape_uses_zyx_order(self):
        self.assertEqual(center_point_for_shape((33, 41, 25)), (16, 20, 12))

    def test_parse_point_requires_three_nonnegative_values(self):
        self.assertEqual(parse_point("1, 2, 3"), (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "z,y,x"):
            parse_point("1,2")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_point("1,-2,3")

    def test_changed_bbox_normalization_keeps_none_and_lists(self):
        self.assertIsNone(normalize_changed_bbox(None))
        changed_bbox = [[0, 33], [0, 41], [0, 25]]
        self.assertEqual(normalize_changed_bbox(changed_bbox), changed_bbox)
        self.assertEqual(box3d_from_changed_bbox(changed_bbox), ((0, 33), (0, 41), (0, 25)))

    def test_changed_bbox_rejects_bad_shape(self):
        with self.assertRaisesRegex(ValueError, "three axis ranges"):
            normalize_changed_bbox([[0, 1], [0, 1]])
        with self.assertRaisesRegex(ValueError, "two values"):
            normalize_changed_bbox([[0, 1], [0, 1], [0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
