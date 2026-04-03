"""
exercise_weights.py — Per-exercise joint importance weights.

Different exercises rely on different joints. Squats depend heavily on knee
and hip angles; finger exercises require hand joint angles; shoulder press
demands elbow and shoulder tracking.  This module provides weight vectors
that the matcher uses to scale each joint's contribution to the overall score.

Design rules:
  - Pose joints:  9 total (elbow×2, knee×2, hip×2, shoulder×2, torso_lean)
  - Hand joints: 10 total (5 finger curls × 2 hands)
  - Weights must sum to ~1.0 per exercise (normalised at match time)
  - Joints with weight 0.0 are detected and shown in feedback but do NOT
    affect the score — this avoids penalising irrelevant body parts.
  - When ``exercise_weights`` is ``None``, the matcher uses equal weight for
    pose joints only.  Hand joints contribute only when a weight dict is
    explicitly supplied.

Provides:
  - EXERCISE_WEIGHTS        — registry of weight dicts
  - get_weights()           — look up weights by exercise name
  - compute_auto_weights()  — derive weights from reference video ROM
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Joint name lists
# ---------------------------------------------------------------------------

_JOINT_NAMES: list[str] = [
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
    "torso_lean",
]

_HAND_JOINT_NAMES: list[str] = [
    "left_hand_thumb_curl",  "left_hand_index_curl",  "left_hand_middle_curl",
    "left_hand_ring_curl",   "left_hand_pinky_curl",
    "right_hand_thumb_curl", "right_hand_index_curl", "right_hand_middle_curl",
    "right_hand_ring_curl",  "right_hand_pinky_curl",
]

# All joints the matcher can compare (pose + hands).
ALL_JOINT_NAMES: list[str] = _JOINT_NAMES + _HAND_JOINT_NAMES

_MIN_WEIGHT = 0.01


# ---------------------------------------------------------------------------
# Weight registry
# ---------------------------------------------------------------------------
# Joints absent from a dict get weight 0.0 when that dict is used.
# Hand joints default to 0.0 for all pose-dominant exercises so they are
# reported (visible in feedback) but not scored.

EXERCISE_WEIGHTS: dict[str, dict[str, float]] = {
    # --- Equal baseline ---
    "default": {j: 1.0 / len(_JOINT_NAMES) for j in _JOINT_NAMES},

    # --- Leg / lower-body dominant ---
    "squat": {
        "left_knee":      0.20,
        "right_knee":     0.20,
        "left_hip":       0.15,
        "right_hip":      0.15,
        "torso_lean":     0.20,
        "left_shoulder":  0.03,
        "right_shoulder": 0.03,
        "left_elbow":     0.02,
        "right_elbow":    0.02,
    },
    "lunge": {
        "left_knee":      0.20,
        "right_knee":     0.20,
        "left_hip":       0.18,
        "right_hip":      0.18,
        "torso_lean":     0.12,
        "left_shoulder":  0.03,
        "right_shoulder": 0.03,
        "left_elbow":     0.03,
        "right_elbow":    0.03,
    },
    "deadlift": {
        "left_hip":       0.20,
        "right_hip":      0.20,
        "left_knee":      0.15,
        "right_knee":     0.15,
        "torso_lean":     0.18,
        "left_shoulder":  0.05,
        "right_shoulder": 0.05,
        "left_elbow":     0.01,
        "right_elbow":    0.01,
    },

    # --- Upper-body dominant ---
    "pushup": {
        "left_elbow":     0.25,
        "right_elbow":    0.25,
        "left_shoulder":  0.12,
        "right_shoulder": 0.12,
        "torso_lean":     0.15,
        "left_hip":       0.04,
        "right_hip":      0.04,
        "left_knee":      0.015,
        "right_knee":     0.015,
    },
    "shoulder_press": {
        "left_shoulder":  0.22,
        "right_shoulder": 0.22,
        "left_elbow":     0.18,
        "right_elbow":    0.18,
        "torso_lean":     0.10,
        "left_hip":       0.025,
        "right_hip":      0.025,
        "left_knee":      0.025,
        "right_knee":     0.025,
    },

    # --- Arm / elbow dominant (with hands scored) ---
    "bicep_curl": {
        "left_elbow":            0.22,
        "right_elbow":           0.22,
        "left_shoulder":         0.08,
        "right_shoulder":        0.08,
        "torso_lean":            0.06,
        "left_hip":              0.02,
        "right_hip":             0.02,
        "left_knee":             0.01,
        "right_knee":            0.01,
        # hand joints contribute when detected
        "left_hand_index_curl":  0.04,
        "right_hand_index_curl": 0.04,
        "left_hand_middle_curl": 0.04,
        "right_hand_middle_curl":0.04,
        "left_hand_ring_curl":   0.02,
        "right_hand_ring_curl":  0.02,
    },

    # --- Wrist / forearm dominant ---
    "wrist_curl": {
        "left_elbow":             0.10,
        "right_elbow":            0.10,
        "left_shoulder":          0.05,
        "right_shoulder":         0.05,
        "torso_lean":             0.03,
        "left_hip":               0.01,
        "right_hip":              0.01,
        "left_knee":              0.005,
        "right_knee":             0.005,
        # hand joints dominate
        "left_hand_thumb_curl":   0.05,
        "left_hand_index_curl":   0.08,
        "left_hand_middle_curl":  0.08,
        "left_hand_ring_curl":    0.06,
        "left_hand_pinky_curl":   0.06,
        "right_hand_thumb_curl":  0.05,
        "right_hand_index_curl":  0.08,
        "right_hand_middle_curl": 0.08,
        "right_hand_ring_curl":   0.06,
        "right_hand_pinky_curl":  0.06,
    },

    # --- Hand / finger dominant (rehabilitation, music, typing) ---
    "finger_exercise": {
        "left_elbow":             0.04,
        "right_elbow":            0.04,
        "left_shoulder":          0.02,
        "right_shoulder":         0.02,
        "torso_lean":             0.01,
        "left_hip":               0.005,
        "right_hip":              0.005,
        "left_knee":              0.005,
        "right_knee":             0.005,
        # fingers are primary
        "left_hand_thumb_curl":   0.07,
        "left_hand_index_curl":   0.08,
        "left_hand_middle_curl":  0.08,
        "left_hand_ring_curl":    0.08,
        "left_hand_pinky_curl":   0.07,
        "right_hand_thumb_curl":  0.07,
        "right_hand_index_curl":  0.08,
        "right_hand_middle_curl": 0.08,
        "right_hand_ring_curl":   0.08,
        "right_hand_pinky_curl":  0.07,
    },

    # --- Shoulder rotation / overhead work (no meaningful hand involvement) ---
    "shoulder_rotation": {
        "left_shoulder":  0.28,
        "right_shoulder": 0.28,
        "left_elbow":     0.12,
        "right_elbow":    0.12,
        "torso_lean":     0.10,
        "left_hip":       0.04,
        "right_hip":      0.04,
        "left_knee":      0.01,
        "right_knee":     0.01,
    },
}


def get_weights(exercise: str = "default") -> dict[str, float]:
    """Return the joint weight vector for the given exercise.

    Parameters
    ----------
    exercise : str
        Exercise name (case-insensitive). Falls back to ``"default"``
        (equal pose-joint weights) if the name is not recognised.

    Returns
    -------
    dict[str, float]
        ``{joint_name: weight}`` — values sum to ~1.0.
    """
    return EXERCISE_WEIGHTS.get(exercise.lower(), EXERCISE_WEIGHTS["default"])


def compute_auto_weights(
    gt_norm: list[dict],
) -> dict[str, Any]:
    """Derive joint weights automatically from a normalised reference skeleton.

    Joints that move more in the reference video receive higher weights.
    Both pose joints and hand joints (when detected) are considered.
    The joint with the largest range of motion is designated the
    *primary joint* for rep/phase detection.

    Parameters
    ----------
    gt_norm : list[dict]
        Output of ``normalizer.normalize_skeleton``.

    Returns
    -------
    dict
        ``{"weights": {joint: float}, "primary_joint": str,
        "primary_min": float, "primary_max": float}``
    """
    # Collect all joint names that actually appear in the GT.
    observed_joints: set[str] = set()
    for frame in gt_norm:
        observed_joints.update(frame.get("angles", {}).keys())

    # Only consider joints in our known list (not symmetry_* ratios).
    candidate_joints = [j for j in ALL_JOINT_NAMES if j in observed_joints]
    if not candidate_joints:
        candidate_joints = list(_JOINT_NAMES)

    rom_min: dict[str, float] = {j: 360.0 for j in candidate_joints}
    rom_max: dict[str, float] = {j: 0.0 for j in candidate_joints}

    for frame in gt_norm:
        angles = frame.get("angles", {})
        for j in candidate_joints:
            val = angles.get(j)
            if val is None:
                continue
            rom_min[j] = min(rom_min[j], val)
            rom_max[j] = max(rom_max[j], val)

    ranges: dict[str, float] = {
        j: max(0.0, rom_max[j] - rom_min[j]) for j in candidate_joints
    }

    total_range = sum(ranges.values())

    if total_range < 1e-6:
        weights = {j: 1.0 / len(candidate_joints) for j in candidate_joints}
    else:
        raw = {j: ranges[j] / total_range for j in candidate_joints}
        weights = {j: max(w, _MIN_WEIGHT) for j, w in raw.items()}
        wsum = sum(weights.values())
        weights = {j: w / wsum for j, w in weights.items()}

    primary_joint = max(ranges, key=lambda k: ranges[k])  # type: ignore[arg-type]

    return {
        "weights": weights,
        "primary_joint": primary_joint,
        "primary_min": rom_min[primary_joint],
        "primary_max": rom_max[primary_joint],
    }
