"""Tests for OpticalFlowValidator in src/filters.py."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.filters import OpticalFlowValidator


def _textured_frame(shape=(240, 320, 3), seed: int = 0) -> np.ndarray:
    """Build a random-textured frame that Lucas-Kanade can actually track."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape, dtype=np.uint8)


def _shift_frame(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a frame by (dx, dy) pixels — fills exposed borders with zeros."""
    h, w = frame.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h))


class OpticalFlowValidatorTests(unittest.TestCase):
    def test_first_frame_passes_through_unchanged(self):
        validator = OpticalFlowValidator(max_residual_pixels=25)
        frame = _textured_frame(seed=1)
        lms = [(0.5, 0.5, 0.9)] + [None] * 32

        out = validator.validate(frame, lms, frame_shape=frame.shape[:2])
        self.assertEqual(out, lms)

    def test_smooth_motion_passes_through(self):
        """A landmark moving by the same amount as the whole frame is consistent."""
        validator = OpticalFlowValidator(max_residual_pixels=25)
        frame_a = _textured_frame(seed=2)
        dx, dy = 4, 3
        frame_b = _shift_frame(frame_a, dx, dy)
        h, w = frame_a.shape[:2]

        x0_px, y0_px = 160, 120
        lm_a = [(x0_px / w, y0_px / h, 0.9)] + [None] * 32
        lm_b = [((x0_px + dx) / w, (y0_px + dy) / h, 0.9)] + [None] * 32

        validator.validate(frame_a, lm_a, frame_shape=frame_a.shape[:2])
        out = validator.validate(frame_b, lm_b, frame_shape=frame_b.shape[:2])

        self.assertIsNotNone(out[0])
        x_out, y_out, vis_out = out[0]
        # Should NOT be downweighted (no big residual).
        self.assertAlmostEqual(vis_out, 0.9, places=3)
        self.assertAlmostEqual(x_out, lm_b[0][0], places=3)
        self.assertAlmostEqual(y_out, lm_b[0][1], places=3)

    def test_teleport_is_rejected(self):
        """A landmark that snaps across the frame should be replaced with flow prediction."""
        validator = OpticalFlowValidator(max_residual_pixels=25)
        frame_a = _textured_frame(seed=3)
        frame_b = _shift_frame(frame_a, 4, 3)  # frame actually shifted by ~5 px
        h, w = frame_a.shape[:2]

        # Previous position in middle of frame.
        x0_px, y0_px = 160, 120
        lm_a = [(x0_px / w, y0_px / h, 0.9)] + [None] * 32

        # Detector "teleports" the landmark 100 pixels away — huge residual vs. flow.
        teleport_x_px = x0_px + 100
        teleport_y_px = y0_px + 80
        lm_b = [(teleport_x_px / w, teleport_y_px / h, 0.9)] + [None] * 32

        validator.validate(frame_a, lm_a, frame_shape=frame_a.shape[:2])
        out = validator.validate(frame_b, lm_b, frame_shape=frame_b.shape[:2])

        self.assertIsNotNone(out[0])
        x_out, y_out, vis_out = out[0]
        # Confidence should be reduced (teleport rejected).
        self.assertAlmostEqual(vis_out, 0.9 * 0.5, places=3)
        # Result should be far from the teleported position.
        self.assertLess(abs(x_out * w - teleport_x_px), 100)
        self.assertLess(abs(y_out * h - teleport_y_px), 100)

    def test_none_landmarks_pass_through(self):
        validator = OpticalFlowValidator(max_residual_pixels=25)
        frame_a = _textured_frame(seed=4)
        frame_b = _shift_frame(frame_a, 2, 1)

        lms_a = [None] * 33
        lms_b = [None] * 33
        validator.validate(frame_a, lms_a, frame_shape=frame_a.shape[:2])
        out = validator.validate(frame_b, lms_b, frame_shape=frame_b.shape[:2])
        self.assertEqual(out, lms_b)

    def test_dict_landmarks_preserve_extra_fields(self):
        validator = OpticalFlowValidator(max_residual_pixels=25)
        frame_a = _textured_frame(seed=5)
        frame_b = _shift_frame(frame_a, 3, 2)
        h, w = frame_a.shape[:2]

        # Dict form with extra field "z".
        lm_a = [{"x": 160 / w, "y": 120 / h, "z": 0.1, "visibility": 0.8}] + [None] * 32
        lm_b = [{"x": (160 + 3) / w, "y": (120 + 2) / h, "z": 0.1, "visibility": 0.8}] + [None] * 32

        validator.validate(frame_a, lm_a, frame_shape=frame_a.shape[:2])
        out = validator.validate(frame_b, lm_b, frame_shape=frame_b.shape[:2])

        self.assertIsInstance(out[0], dict)
        self.assertIn("z", out[0])
        self.assertEqual(out[0]["z"], 0.1)  # preserved

    def test_reset_clears_state(self):
        validator = OpticalFlowValidator()
        frame = _textured_frame(seed=6)
        lms = [(0.5, 0.5, 0.9)] + [None] * 32
        validator.validate(frame, lms, frame_shape=frame.shape[:2])
        validator.reset()
        # Next call should behave like a first call (no rejection possible).
        frame2 = _textured_frame(seed=7)
        out = validator.validate(frame2, lms, frame_shape=frame2.shape[:2])
        self.assertEqual(out, lms)


if __name__ == "__main__":
    unittest.main()
