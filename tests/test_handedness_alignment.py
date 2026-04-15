import unittest

from src.normalizer import reconcile_hand_sides


def _make_pose_landmarks() -> list[dict]:
    """Build a minimal 33-point pose with stable torso and visible wrists."""
    landmarks = [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}
        for _ in range(33)
    ]

    # Front-facing orientation in unmirrored image space:
    # anatomical left appears on image-right.
    landmarks[11] = {"x": 0.70, "y": 0.30, "z": 0.0, "visibility": 1.0}  # left shoulder
    landmarks[12] = {"x": 0.30, "y": 0.30, "z": 0.0, "visibility": 1.0}  # right shoulder
    landmarks[23] = {"x": 0.65, "y": 0.60, "z": 0.0, "visibility": 1.0}  # left hip
    landmarks[24] = {"x": 0.35, "y": 0.60, "z": 0.0, "visibility": 1.0}  # right hip

    landmarks[15] = {"x": 0.75, "y": 0.45, "z": 0.0, "visibility": 1.0}  # left wrist
    landmarks[16] = {"x": 0.25, "y": 0.45, "z": 0.0, "visibility": 1.0}  # right wrist
    landmarks[13] = {"x": 0.72, "y": 0.40, "z": 0.0, "visibility": 1.0}  # left elbow
    landmarks[14] = {"x": 0.28, "y": 0.40, "z": 0.0, "visibility": 1.0}  # right elbow
    return landmarks


def _make_pose_landmarks_mirrored_view() -> list[dict]:
    """Build pose landmarks for a mirrored camera view.

    Landmark IDs stay anatomical, but their x-positions are mirrored in image
    space (left appears on image-left).
    """
    landmarks = _make_pose_landmarks()

    # Swap image-space positions for key left/right pairs while keeping IDs.
    pairs = [
        (11, 12),  # shoulders
        (13, 14),  # elbows
        (15, 16),  # wrists
        (23, 24),  # hips
    ]
    for left_idx, right_idx in pairs:
        lx, ly = landmarks[left_idx]["x"], landmarks[left_idx]["y"]
        rx, ry = landmarks[right_idx]["x"], landmarks[right_idx]["y"]
        landmarks[left_idx]["x"], landmarks[left_idx]["y"] = rx, ry
        landmarks[right_idx]["x"], landmarks[right_idx]["y"] = lx, ly
    return landmarks


def _make_hand(center_x: float, center_y: float = 0.45) -> list[dict]:
    hand: list[dict] = []
    for idx in range(21):
        hand.append(
            {
                "x": center_x + (idx * 0.001),
                "y": center_y + (idx * 0.001),
                "z": 0.0,
                "visibility": 1.0,
            }
        )
    return hand


class HandednessAlignmentTests(unittest.TestCase):
    def test_single_hand_reassigned_to_nearest_side(self) -> None:
        pose = _make_pose_landmarks()
        hand_landmarks = {
            "left": None,
            # Deliberately mislabeled as right, but physically near left wrist.
            "right": _make_hand(center_x=0.74),
        }

        reconciled = reconcile_hand_sides(pose, hand_landmarks)

        self.assertIsNotNone(reconciled["left"])
        self.assertIsNone(reconciled["right"])

    def test_two_hands_swap_when_detector_labels_are_inverted(self) -> None:
        pose = _make_pose_landmarks()
        hand_landmarks = {
            # Intentionally inverted labels.
            "left": _make_hand(center_x=0.24),
            "right": _make_hand(center_x=0.74),
        }

        reconciled = reconcile_hand_sides(pose, hand_landmarks)

        self.assertGreater(reconciled["left"][0]["x"], reconciled["right"][0]["x"])

    def test_single_hand_fallback_works_when_one_wrist_is_occluded(self) -> None:
        pose = _make_pose_landmarks()
        # Simulate missing right wrist visibility.
        pose[16]["visibility"] = 0.0

        hand_landmarks = {
            "left": None,
            # Wrong label from detector; hand is anatomically left.
            "right": _make_hand(center_x=0.73),
        }

        reconciled = reconcile_hand_sides(pose, hand_landmarks)

        self.assertIsNotNone(reconciled["left"])
        self.assertIsNone(reconciled["right"])

    def test_reconciliation_handles_mirrored_view(self) -> None:
        pose = _make_pose_landmarks_mirrored_view()

        hand_landmarks = {
            "left": None,
            # In mirrored view, anatomical-left wrist appears on image-left.
            "right": _make_hand(center_x=0.26),
        }

        reconciled = reconcile_hand_sides(pose, hand_landmarks)

        self.assertIsNotNone(reconciled["left"])
        self.assertIsNone(reconciled["right"])


if __name__ == "__main__":
    unittest.main()
