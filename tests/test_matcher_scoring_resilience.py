import unittest

from src.exercise_weights import get_weights
from src.matcher import match_single_frame


def _normalized_landmarks_with_joint_offset(joint_idx: int, offset_x: float) -> list[dict]:
    """Create a minimal normalized-landmark list with one displaced joint."""
    points = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0}
        for _ in range(33)
    ]
    points[joint_idx] = {
        "x": offset_x,
        "y": 0.0,
        "z": 0.0,
        "visibility": 1.0,
    }
    return points


class MatcherScoringResilienceTests(unittest.TestCase):
    def test_pose_deadzone_allows_small_style_variation(self) -> None:
        gt = {"angles": {"left_elbow": 90.0}}
        user = {"angles": {"left_elbow": 95.0}}  # 5° diff
        result = match_single_frame(
            gt,
            user,
            exercise_weights={"left_elbow": 1.0},
        )
        self.assertGreaterEqual(result["overall_score"], 99.0)

    def test_hand_deadzone_allows_small_finger_variation(self) -> None:
        gt = {"angles": {"left_hand_index_curl": 120.0}}
        user = {"angles": {"left_hand_index_curl": 127.0}}  # 7° diff
        result = match_single_frame(
            gt,
            user,
            exercise_weights={"left_hand_index_curl": 1.0},
        )
        self.assertGreaterEqual(result["overall_score"], 99.0)

    def test_missing_hand_landmarks_do_not_collapse_score(self) -> None:
        # Finger exercise expects mostly hand joints, but body-anchor joints
        # may be the only reliable ones in some frames.
        weights = get_weights("finger_exercise")

        gt_angles = {
            "left_elbow": 160.0,
            "right_elbow": 160.0,
            "left_shoulder": 70.0,
            "right_shoulder": 70.0,
            "left_hand_thumb_curl": 80.0,
            "left_hand_index_curl": 90.0,
            "left_hand_middle_curl": 85.0,
            "left_hand_ring_curl": 82.0,
            "left_hand_pinky_curl": 78.0,
            "right_hand_thumb_curl": 79.0,
            "right_hand_index_curl": 91.0,
            "right_hand_middle_curl": 84.0,
            "right_hand_ring_curl": 83.0,
            "right_hand_pinky_curl": 77.0,
        }
        user_angles = {
            # Simulate intermittent hand detection drop: only anchor joints remain.
            "left_elbow": 160.0,
            "right_elbow": 160.0,
            "left_shoulder": 70.0,
            "right_shoulder": 70.0,
        }

        result = match_single_frame(
            {"angles": gt_angles},
            {"angles": user_angles},
            exercise_weights=weights,
        )

        # finger_exercise now sets ALL pose joints to weight 0.0.
        # When only body joints are visible (hand landmarks dropped), no scored
        # joint is active → max_angle_penalty == 0 → overall_score == 0.0.
        # This is correct: we cannot evaluate hand form without hand data.
        self.assertEqual(result["overall_score"], 0.0)

    def test_pose_position_mismatch_penalized_even_if_angle_matches(self) -> None:
        gt = {
            "angles": {"left_elbow": 90.0},
            "normalized_landmarks": _normalized_landmarks_with_joint_offset(13, 0.0),
        }
        user = {
            "angles": {"left_elbow": 90.0},
            "normalized_landmarks": _normalized_landmarks_with_joint_offset(13, 0.35),
        }
        result = match_single_frame(
            gt,
            user,
            exercise_weights={"left_elbow": 1.0},
        )

        self.assertIn("spatial_diff", result["left_elbow"])
        self.assertLess(result["overall_score"], 95.0)


if __name__ == "__main__":
    unittest.main()
