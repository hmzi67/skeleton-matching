import unittest

from src.exercise_weights import get_weights
from src.matcher import match_single_frame


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

        # Should still be reduced for low coverage, but not collapse to ~30.
        self.assertGreaterEqual(result["overall_score"], 55.0)
        self.assertLessEqual(result["overall_score"], 65.0)


if __name__ == "__main__":
    unittest.main()
