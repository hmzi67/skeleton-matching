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

# Joints with weight below this threshold are excluded from feedback/reports.
FEEDBACK_WEIGHT_THRESHOLD = 0.05
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
        "left_elbow":             0.06,
        "right_elbow":            0.06,
        "left_shoulder":          0.04,
        "right_shoulder":         0.04,
        "torso_lean":             0.0,
        "left_hip":               0.0,
        "right_hip":              0.0,
        "left_knee":              0.0,
        "right_knee":             0.0,
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
    # Body joints are irrelevant — set to 0.0 so they are excluded from
    # both scoring and feedback entirely.
    "finger_exercise": {
        "left_elbow":             0.03,
        "right_elbow":            0.03,
        "left_shoulder":          0.02,
        "right_shoulder":         0.02,
        "torso_lean":             0.0,
        "left_hip":               0.0,
        "right_hip":              0.0,
        "left_knee":              0.0,
        "right_knee":             0.0,
        # fingers are primary
        "left_hand_thumb_curl":   0.08,
        "left_hand_index_curl":   0.09,
        "left_hand_middle_curl":  0.09,
        "left_hand_ring_curl":    0.09,
        "left_hand_pinky_curl":   0.08,
        "right_hand_thumb_curl":  0.08,
        "right_hand_index_curl":  0.09,
        "right_hand_middle_curl": 0.09,
        "right_hand_ring_curl":   0.09,
        "right_hand_pinky_curl":  0.08,
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


# ---------------------------------------------------------------------------
# Fuzzy exercise-name resolution
# ---------------------------------------------------------------------------
# User-facing / DB exercise names rarely match the curated keys exactly.
# This alias table normalises common variants so the pipeline picks up the
# right curated weights, primary angles, and phase machines.

_EXERCISE_ALIASES: dict[str, str] = {
    # Hand / finger
    "hand":            "finger_exercise",
    "hands":           "finger_exercise",
    "hand_exercise":   "finger_exercise",
    "hand_exercises":  "finger_exercise",
    "fist":            "finger_exercise",
    "fists":           "finger_exercise",
    "fist_exercise":   "finger_exercise",
    "finger":          "finger_exercise",
    "fingers":         "finger_exercise",
    "finger_curl":     "finger_exercise",
    "finger_curls":    "finger_exercise",
    "finger_exercises":"finger_exercise",
    # Wrist
    "wrist":           "wrist_curl",
    "wrists":          "wrist_curl",
    "wrist_curls":     "wrist_curl",
    # Bicep
    "bicep":           "bicep_curl",
    "biceps":          "bicep_curl",
    "bicep_curls":     "bicep_curl",
    "curl":            "bicep_curl",
    "curls":           "bicep_curl",
    "dumbbell_curl":   "bicep_curl",
    # Shoulder press
    "shoulder_presses":"shoulder_press",
    "overhead_press":  "shoulder_press",
    # Pushup
    "pushups":         "pushup",
    "push_up":         "pushup",
    "push_ups":        "pushup",
    # Squat
    "squats":          "squat",
    # Lunge
    "lunges":          "lunge",
    # Deadlift
    "deadlifts":       "deadlift",
    # Shoulder rotation
    "shoulder_rotations": "shoulder_rotation",
}


def resolve_exercise_name(exercise: str) -> str:
    """Normalise a user/DB exercise string to a curated key if possible.

    Lowercases, collapses whitespace/hyphens to underscores, strips, and
    consults ``_EXERCISE_ALIASES``. If no alias matches, returns the
    normalised string as-is so the caller can still do membership checks
    against ``EXERCISE_WEIGHTS``.
    """
    if not exercise:
        return "default"
    norm = exercise.strip().lower().replace("-", "_").replace(" ", "_")
    # Collapse double underscores.
    while "__" in norm:
        norm = norm.replace("__", "_")
    if norm in EXERCISE_WEIGHTS:
        return norm
    return _EXERCISE_ALIASES.get(norm, norm)


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
    return EXERCISE_WEIGHTS.get(resolve_exercise_name(exercise), EXERCISE_WEIGHTS["default"])


def get_relevant_joints(
    weights: dict[str, float],
    threshold: float = FEEDBACK_WEIGHT_THRESHOLD,
) -> set[str]:
    """Return joints whose weight meets the feedback significance threshold.

    Only joints in this set should appear in per-frame feedback and session
    reports. Joints below the threshold are irrelevant to the exercise and
    should never generate corrections.
    """
    return {j for j, w in weights.items() if w >= threshold}


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

    # ------------------------------------------------------------------
    # Body-region awareness: the reference video may contain incidental
    # sway in body joints that is not part of the exercise. We detect the
    # dominant region and zero out joints that are merely noise so that
    # downstream feedback/reporting stays focused on what the exercise
    # actually trains.
    # ------------------------------------------------------------------
    hand_rom_total = sum(ranges.get(j, 0.0) for j in _HAND_JOINT_NAMES)
    pose_rom_total = sum(ranges.get(j, 0.0) for j in _JOINT_NAMES)

    if hand_rom_total >= 20.0 and hand_rom_total >= 0.4 * pose_rom_total:
        # Hand-dominant exercise — body joints are mere stabilisers.
        # Keep a small anchor weight on elbow/shoulder so gross body-position
        # errors (e.g. leaning into the camera) still reduce the score.
        _BODY_ANCHOR_JOINTS = {
            "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        }
        _BODY_ANCHOR_SCALE = 0.15  # body joints collectively get at most 15% of total ROM
        max_hand_rom = max((ranges.get(j, 0.0) for j in _HAND_JOINT_NAMES), default=1.0) or 1.0
        for j in _JOINT_NAMES:
            if j in ranges:
                if j in _BODY_ANCHOR_JOINTS:
                    # Clamp to a small fraction of the dominant hand ROM so
                    # they don't overwhelm the hand score but still penalise
                    # gross posture errors.
                    ranges[j] = min(ranges[j], max_hand_rom * _BODY_ANCHOR_SCALE)
                else:
                    ranges[j] = 0.0
    else:
        # Body-dominant exercise — drop any joint whose ROM is tiny
        # compared to the most-moving joint. Incidental 5-10° drift in
        # unrelated joints would otherwise cross the 0.05 feedback
        # weight threshold after normalisation.
        max_rom = max(ranges.values()) if ranges else 0.0
        if max_rom > 0:
            cutoff = max(15.0, 0.20 * max_rom)
            for j in list(ranges.keys()):
                if ranges[j] < cutoff:
                    ranges[j] = 0.0

    total_range = sum(ranges.values())

    if total_range < 1e-6:
        weights = {j: 1.0 / len(candidate_joints) for j in candidate_joints}
    else:
        raw = {
            j: (ranges[j] / total_range) if ranges[j] > 0 else 0.0
            for j in candidate_joints
        }
        # Apply _MIN_WEIGHT floor only to surviving (non-zero) joints so
        # we don't re-introduce the joints we just zeroed out.
        weights = {
            j: (max(w, _MIN_WEIGHT) if w > 0 else 0.0)
            for j, w in raw.items()
        }
        wsum = sum(weights.values())
        if wsum > 0:
            weights = {j: w / wsum for j, w in weights.items()}

    # Re-pick primary joint from filtered ranges so it can never be a
    # joint we just suppressed.
    non_zero = {j: r for j, r in ranges.items() if r > 0}
    if non_zero:
        primary_joint = max(non_zero, key=lambda k: non_zero[k])
    else:
        primary_joint = max(ranges, key=lambda k: ranges[k]) if ranges else ""  # type: ignore[arg-type]

    return {
        "weights": weights,
        "primary_joint": primary_joint,
        "primary_min": rom_min[primary_joint],
        "primary_max": rom_max[primary_joint],
    }
