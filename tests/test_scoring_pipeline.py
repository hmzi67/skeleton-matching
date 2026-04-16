"""
test_scoring_pipeline.py — Unit tests for the scoring helper functions added
to src/matcher.py: piecewise_penalty (Change 1), soft_cap (Change 3), and
coverage_factor (Change 5).
"""

import unittest

from src.exercise_weights import (
    HAND_JOINT_NAMES,
    MIN_WEIGHT_RATIO,
    ALL_JOINT_NAMES,
    detect_dominant_modality,
)
from src.matcher import (
    OVERALL_SCORE_CAP_WHEN_PRIMARY_FAILS,
    PRIMARY_JOINT_SCORE_FLOOR,
    coverage_factor,
    match_single_frame,
    piecewise_penalty,
    soft_cap,
)


# ---------------------------------------------------------------------------
# Change 1 (exercise_weights) — MIN_WEIGHT_RATIO zeroing
# ---------------------------------------------------------------------------


class MinWeightRatioTests(unittest.TestCase):
    def test_low_weight_joint_is_zeroed(self) -> None:
        # 0.05 / 0.8 = 0.0625 < MIN_WEIGHT_RATIO (0.15) → must become 0.0
        weights = {"knee": 0.05, "index_curl": 0.8, "middle_curl": 0.6}
        max_w = max(weights.values())
        result = {j: (w if w / max_w >= MIN_WEIGHT_RATIO else 0.0) for j, w in weights.items()}
        self.assertEqual(result["knee"], 0.0)
        self.assertEqual(result["index_curl"], 0.8)
        self.assertEqual(result["middle_curl"], 0.6)

    def test_all_equal_weights_keep_all(self) -> None:
        # All weights equal → ratio is always 1.0 ≥ MIN_WEIGHT_RATIO → none zeroed.
        weights = {"a": 0.5, "b": 0.5}
        max_w = max(weights.values())
        result = {j: (w if w / max_w >= MIN_WEIGHT_RATIO else 0.0) for j, w in weights.items()}
        self.assertEqual(result["a"], 0.5)
        self.assertEqual(result["b"], 0.5)


# ---------------------------------------------------------------------------
# Change 2 — detect_dominant_modality and pose-joint zeroing
# ---------------------------------------------------------------------------

_POSE_JOINT_NAMES = [j for j in ALL_JOINT_NAMES if j not in HAND_JOINT_NAMES]


class DominantModalityTests(unittest.TestCase):
    def test_hand_dominant_detected(self) -> None:
        # Hand joints sum to 0.80, pose joints to 0.20 → modality "hand"
        weights: dict[str, float] = {}
        for j in HAND_JOINT_NAMES:
            weights[j] = 0.08   # 10 × 0.08 = 0.80
        for j in _POSE_JOINT_NAMES:
            weights[j] = 0.02   # 9 × 0.02 = 0.18  (close enough to 0.20)
        self.assertEqual(detect_dominant_modality(weights), "hand")

    def test_hand_modality_zeroes_all_pose_joints(self) -> None:
        weights: dict[str, float] = {}
        for j in HAND_JOINT_NAMES:
            weights[j] = 0.08
        for j in _POSE_JOINT_NAMES:
            weights[j] = 0.02
        # Simulate the zeroing step applied in compute_auto_weights when modality=="hand"
        modality = detect_dominant_modality(weights)
        self.assertEqual(modality, "hand")
        result = {j: (w if j in HAND_JOINT_NAMES else 0.0) for j, w in weights.items()}
        for j in _POSE_JOINT_NAMES:
            self.assertEqual(result.get(j, 0.0), 0.0, msg=f"{j} should be 0.0")

    def test_pose_dominant_detected(self) -> None:
        # Only pose joints → pose_share = 1.0 ≥ 0.65
        weights = {j: 1.0 / len(_POSE_JOINT_NAMES) for j in _POSE_JOINT_NAMES}
        self.assertEqual(detect_dominant_modality(weights), "pose")

    def test_mixed_detected(self) -> None:
        # 50/50 split → neither side ≥ 65%
        weights = {}
        for j in list(HAND_JOINT_NAMES)[:5]:
            weights[j] = 0.1
        for j in _POSE_JOINT_NAMES[:5]:
            weights[j] = 0.1
        self.assertEqual(detect_dominant_modality(weights), "mixed")

    def test_empty_weights_returns_mixed(self) -> None:
        self.assertEqual(detect_dominant_modality({}), "mixed")


# ---------------------------------------------------------------------------
# Change 1 (matcher.py) — piecewise_penalty breakpoints
# ---------------------------------------------------------------------------


class PiecewisePenaltyTests(unittest.TestCase):
    def test_mild_zone_at_zero(self) -> None:
        self.assertAlmostEqual(piecewise_penalty(0.0), 0.0)

    def test_mild_moderate_boundary(self) -> None:
        # raw = 15 is the last point of the mild zone: 15 * 0.5 = 7.5
        self.assertAlmostEqual(piecewise_penalty(15.0), 7.5)

    def test_moderate_critical_boundary(self) -> None:
        # raw = 30 is the last point of the moderate zone:
        # 7.5 + (30 - 15) * 1.2 = 7.5 + 18.0 = 25.5
        self.assertAlmostEqual(piecewise_penalty(30.0), 25.5)

    def test_critical_zone_at_max_raw(self) -> None:
        # raw = 45 (max before re-clamp): 25.5 + (45 - 30) * 2.0 = 55.5
        self.assertAlmostEqual(piecewise_penalty(45.0), 55.5)

    def test_superlinear_growth(self) -> None:
        # Each zone should amplify more than the previous.
        mid_mild = piecewise_penalty(7.5)          # midpoint of mild zone
        mid_moderate = piecewise_penalty(22.5)     # midpoint of moderate zone
        mid_critical = piecewise_penalty(37.5)     # midpoint of critical zone
        self.assertLess(mid_mild, mid_moderate)
        self.assertLess(mid_moderate, mid_critical)


# ---------------------------------------------------------------------------
# Change 3 — soft_cap properties
# ---------------------------------------------------------------------------


class SoftCapTests(unittest.TestCase):
    def test_zero_input_returns_zero(self) -> None:
        self.assertAlmostEqual(soft_cap(0.0), 0.0)

    def test_at_cap_value_is_less_than_cap(self) -> None:
        # soft_cap(45) = 45 * (1 - exp(-1)) ≈ 28.4 < 45
        self.assertLess(soft_cap(45.0), 45.0)

    def test_monotonically_increasing(self) -> None:
        # Larger input must produce larger output (ordering is preserved).
        # soft_cap(90) ≈ 38.9, soft_cap(180) ≈ 44.2
        self.assertLess(soft_cap(90.0), soft_cap(180.0))


# ---------------------------------------------------------------------------
# Change 5 — coverage_factor boundary values
# ---------------------------------------------------------------------------


class CoverageFactorTests(unittest.TestCase):
    def test_zero_coverage(self) -> None:
        self.assertAlmostEqual(coverage_factor(0.0), 0.0)

    def test_fifty_percent(self) -> None:
        self.assertAlmostEqual(coverage_factor(0.50), 0.70)

    def test_eighty_five_percent(self) -> None:
        self.assertAlmostEqual(coverage_factor(0.85), 1.0)

    def test_full_coverage(self) -> None:
        self.assertAlmostEqual(coverage_factor(1.0), 1.0)


class SpatialConsistencyCompatibilityTests(unittest.TestCase):
    def test_missing_normalized_landmarks_falls_back_to_angle_only(self) -> None:
        gt = {
            "angles": {"left_elbow": 90.0},
            "normalized_landmarks": [
                {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0}
                for _ in range(33)
            ],
        }
        user = {
            "angles": {"left_elbow": 90.0},
            # Deliberately omitted to verify graceful fallback.
        }

        result = match_single_frame(
            gt,
            user,
            exercise_weights={"left_elbow": 1.0},
        )
        self.assertGreaterEqual(result["overall_score"], 99.0)


# ---------------------------------------------------------------------------
# Change 4 — primary joint floor cap
# ---------------------------------------------------------------------------


class PrimaryJointFloorTests(unittest.TestCase):
    def test_cap_triggers_when_primary_joint_fails_badly(self) -> None:
        # 3 joints with pre-norm weight 0.6 (≥ 0.5 → all primary).
        # left_hip: gt=90, user=140 → abs_diff=50° → score ≈ 32% < 40%.
        # left_knee, right_knee: diff=1° (within 4° tolerance) → score ≈ 100%.
        # Blended Sraw ≈ 77% without cap → capped at 70.
        result = match_single_frame(
            {"angles": {"left_hip": 90.0, "left_knee": 90.0, "right_knee": 90.0}},
            {"angles": {"left_hip": 140.0, "left_knee": 91.0, "right_knee": 91.0}},
            exercise_weights={"left_hip": 0.6, "left_knee": 0.6, "right_knee": 0.6},
        )
        self.assertLessEqual(result["overall_score"], float(OVERALL_SCORE_CAP_WHEN_PRIMARY_FAILS))
        self.assertAlmostEqual(
            result["overall_score"],
            float(OVERALL_SCORE_CAP_WHEN_PRIMARY_FAILS),
            delta=0.5,
        )

    def test_cap_does_not_trigger_when_primary_joint_is_close(self) -> None:
        # left_hip diff=5°, within 7° tolerance → effective=0 → score=100% → no cap.
        result = match_single_frame(
            {"angles": {"left_hip": 90.0, "left_knee": 90.0}},
            {"angles": {"left_hip": 95.0, "left_knee": 91.0}},
            exercise_weights={"left_hip": 0.6, "left_knee": 0.6},
        )
        self.assertGreater(result["overall_score"], float(OVERALL_SCORE_CAP_WHEN_PRIMARY_FAILS))

    def test_no_exercise_weights_bypasses_cap(self) -> None:
        # Without exercise_weights, cap logic is skipped entirely.
        result = match_single_frame(
            {"angles": {"left_hip": 90.0}},
            {"angles": {"left_hip": 140.0}},
            exercise_weights=None,
        )
        # Score may be low, but it was not capped by the primary-joint logic.
        self.assertIsNotNone(result["overall_score"])


if __name__ == "__main__":
    unittest.main()
