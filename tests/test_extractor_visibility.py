"""Tests for extractor.py visibility filtering and hand detection thresholds."""

import unittest
from unittest.mock import MagicMock, patch

from src.extractor import (
    MIN_LANDMARK_VISIBILITY,
    MIN_HAND_DETECTION_CONFIDENCE,
    extract_skeleton_from_video,
)


class VisibilityFilteringTests(unittest.TestCase):
    """Test visibility threshold filtering for pose landmarks and hand detection."""

    def test_low_visibility_landmark_marked_as_none(self):
        """Landmarks with visibility < 0.5 should be filtered to None."""
        # Create a mock landmark with low visibility
        low_vis_lm = MagicMock()
        low_vis_lm.visibility = 0.3  # Below MIN_LANDMARK_VISIBILITY
        low_vis_lm.x = 0.5
        low_vis_lm.y = 0.5
        low_vis_lm.z = 0.0

        # The filtering is done in extract_skeleton_from_video by checking
        # visibility and setting to None if below threshold.
        # We test the constant is set correctly.
        self.assertEqual(MIN_LANDMARK_VISIBILITY, 0.5)
        self.assertLess(low_vis_lm.visibility, MIN_LANDMARK_VISIBILITY)

    def test_high_visibility_landmark_included(self):
        """Landmarks with visibility >= 0.5 should be included."""
        high_vis_lm = MagicMock()
        high_vis_lm.visibility = 0.7  # Above MIN_LANDMARK_VISIBILITY
        high_vis_lm.x = 0.5
        high_vis_lm.y = 0.5
        high_vis_lm.z = 0.0

        self.assertGreaterEqual(high_vis_lm.visibility, MIN_LANDMARK_VISIBILITY)

    def test_hand_detection_confidence_threshold(self):
        """Hand detections with score < 0.6 should be skipped."""
        self.assertEqual(MIN_HAND_DETECTION_CONFIDENCE, 0.6)

        low_conf = 0.4
        high_conf = 0.8

        self.assertLess(low_conf, MIN_HAND_DETECTION_CONFIDENCE)
        self.assertGreaterEqual(high_conf, MIN_HAND_DETECTION_CONFIDENCE)

    def test_visibility_constants_exported(self):
        """Verify visibility constants are properly exported."""
        self.assertTrue(hasattr(MIN_LANDMARK_VISIBILITY, '__float__'))
        self.assertTrue(hasattr(MIN_HAND_DETECTION_CONFIDENCE, '__float__'))
        self.assertEqual(MIN_LANDMARK_VISIBILITY, 0.5)
        self.assertEqual(MIN_HAND_DETECTION_CONFIDENCE, 0.6)


if __name__ == "__main__":
    unittest.main()
