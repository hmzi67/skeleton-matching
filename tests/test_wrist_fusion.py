"""Tests for wrist reconciliation between pose and hand models."""

import unittest

from src.extractor import _reconcile_wrist, MIN_HAND_DETECTION_CONFIDENCE


class WristFusionTests(unittest.TestCase):
    """Test wrist reconciliation logic that prefers hand model when confident."""

    def test_hand_wrist_preferred_when_confident(self):
        """When hand_score >= 0.6, hand_wrist should be preferred."""
        pose_wrist = {"x": 0.5, "y": 0.4, "z": 0.0}
        hand_wrist = {"x": 0.51, "y": 0.41, "z": 0.0}
        hand_score = 0.8  # High confidence

        best, source = _reconcile_wrist(pose_wrist, hand_wrist, hand_score)

        self.assertEqual(best, hand_wrist)
        self.assertEqual(source, "hand")

    def test_pose_wrist_fallback_when_hand_unavailable(self):
        """When hand_wrist is None, pose_wrist should be used."""
        pose_wrist = {"x": 0.5, "y": 0.4, "z": 0.0}
        hand_wrist = None
        hand_score = 0.8

        best, source = _reconcile_wrist(pose_wrist, hand_wrist, hand_score)

        self.assertEqual(best, pose_wrist)
        self.assertEqual(source, "pose")

    def test_pose_wrist_fallback_when_hand_confidence_low(self):
        """When hand_score < 0.6, pose_wrist should be preferred."""
        pose_wrist = {"x": 0.5, "y": 0.4, "z": 0.0}
        hand_wrist = {"x": 0.51, "y": 0.41, "z": 0.0}
        hand_score = 0.4  # Low confidence, below MIN_HAND_DETECTION_CONFIDENCE

        best, source = _reconcile_wrist(pose_wrist, hand_wrist, hand_score)

        self.assertEqual(best, pose_wrist)
        self.assertEqual(source, "pose")

    def test_both_unavailable_returns_none(self):
        """When both wrists are None, result should be (None, None)."""
        pose_wrist = None
        hand_wrist = None
        hand_score = 0.8

        best, source = _reconcile_wrist(pose_wrist, hand_wrist, hand_score)

        self.assertIsNone(best)
        self.assertIsNone(source)

    def test_confidence_threshold_boundary(self):
        """Test reconciliation at the exact confidence boundary."""
        pose_wrist = {"x": 0.5, "y": 0.4, "z": 0.0}
        hand_wrist = {"x": 0.51, "y": 0.41, "z": 0.0}

        # Exactly at threshold (0.6) → should use hand
        best_at, source_at = _reconcile_wrist(
            pose_wrist, hand_wrist, MIN_HAND_DETECTION_CONFIDENCE
        )
        self.assertEqual(best_at, hand_wrist)
        self.assertEqual(source_at, "hand")

        # Just below threshold → should use pose
        best_below, source_below = _reconcile_wrist(
            pose_wrist, hand_wrist, MIN_HAND_DETECTION_CONFIDENCE - 0.01
        )
        self.assertEqual(best_below, pose_wrist)
        self.assertEqual(source_below, "pose")


if __name__ == "__main__":
    unittest.main()
