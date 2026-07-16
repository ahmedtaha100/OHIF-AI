import unittest

import numpy as np

from rl_nninteractive.state_encoder import encode_state_channels


class StateEncoderTests(unittest.TestCase):
    def test_encode_state_channels_stacks_prompt_history_and_step(self):
        image = np.zeros((1, 3, 4, 5), dtype=np.float32)
        image[0, 1, 2, 3] = 2.0
        mask = np.zeros((3, 4, 5), dtype=np.uint8)
        positive = np.zeros_like(mask)
        negative = np.zeros_like(mask)
        mask[1, 2, 3] = 1
        positive[1, 2, 3] = 1
        negative[0, 0, 0] = 1

        encoded = encode_state_channels(
            image=image,
            current_mask=mask,
            positive_prompt_history=positive,
            negative_prompt_history=negative,
            step_index=2,
            max_steps=5,
        )

        self.assertEqual(encoded.channels.shape, (5, 3, 4, 5))
        self.assertEqual(encoded.channel_names[0], "image")
        self.assertAlmostEqual(encoded.step_fraction, 0.4)
        self.assertEqual(float(encoded.channels[1, 1, 2, 3]), 1.0)
        self.assertEqual(float(encoded.channels[2, 1, 2, 3]), 1.0)
        self.assertEqual(float(encoded.channels[3, 0, 0, 0]), 1.0)
        self.assertTrue(np.allclose(encoded.channels[4], 0.4))

    def test_encode_state_channels_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "current_mask shape"):
            encode_state_channels(
                image=np.zeros((1, 3, 3, 3), dtype=np.float32),
                current_mask=np.zeros((3, 3, 2), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
