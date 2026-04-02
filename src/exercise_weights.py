"""
exercise_weights.py — Per-exercise joint importance weights.

Different exercises rely on different joints. For example, squats depend
heavily on knee and hip angles while elbows are nearly irrelevant.
This module provides weight vectors that the matcher uses to scale each
joint's contribution to the overall score.

Provides:
  - EXERCISE_WEIGHTS        — registry of weight dicts
  - get_weights()           — look up weights by exercise name
  - compute_auto_weights()  — derive weights from reference video ROM
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Weight registry
# ---------------------------------------------------------------------------
# Each dict maps joint_name → weight (must sum to 1.0).

EXERCISE_WEIGHTS: dict[str, dict[str, float]] = {
    "default": {
        "left_elbow":     1 / 9,
        "right_elbow":    1 / 9,
        "left_knee":      1 / 9,
        "right_knee":     1 / 9,
        "left_hip":       1 / 9,
        "right_hip":      1 / 9,
        "left_shoulder":  1 / 9,
        "right_shoulder": 1 / 9,
        "torso_lean":     1 / 9,
    },
    "squat": {
        "left_knee":      0.20,
        "right_knee":     0.20,
        "left_hip":       0.15,
        "right_hip":      0.15,
        "torso_lean":     0.20,
        "left_shoulder":  0.025,
        "right_shoulder": 0.025,
        "left_elbow":     0.025,
        "right_elbow":    0.025,
    },
    "pushup": {
        "left_elbow":     0.25,
        "right_elbow":    0.25,
        "left_shoulder":  0.12,
        "right_shoulder": 0.12,
        "torso_lean":     0.15,
        "left_hip":       0.03,
        "right_hip":      0.03,
        "left_knee":      0.025,
        "right_knee":     0.025,
    },
    "deadlift": {
        "left_hip":       0.20,
        "right_hip":      0.20,
        "left_knee":      0.15,
        "right_knee":     0.15,
        "torso_lean":     0.15,
        "left_shoulder":  0.05,
        "right_shoulder": 0.05,
        "left_elbow":     0.025,
        "right_elbow":    0.025,
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
}


_JOINT_NAMES = [
    "left_elbow", "right_elbow", "left_knee", "right_knee",
    "left_hip", "right_hip", "left_shoulder", "right_shoulder",
    "torso_lean",
]

_MIN_WEIGHT = 0.02


def get_weights(exercise: str = "default") -> dict[str, float]:
    """Return the joint weight vector for the given exercise.

    Parameters
    ----------
    exercise : str
        Exercise name (case-insensitive). Falls back to ``"default"``
        (equal weights) if the name is not recognised.

    Returns
    -------
    dict[str, float]
        ``{joint_name: weight}`` — values sum to ~1.0.
    """
    return EXERCISE_WEIGHTS.get(exercise.lower(), EXERCISE_WEIGHTS["default"])


def compute_auto_weights(
    gt_norm: list[dict],
) -> dict[str, Any]:
    """Derive joint weights, primary joint, and phase-detection range
    directly from a normalised reference skeleton sequence.

    Joints that move more in the reference video receive higher weights.
    The joint with the largest range of motion is designated the
    *primary joint* for rep/phase detection.

    Parameters
    ----------
    gt_norm : list[dict]
        Output of ``normalizer.normalize_skeleton`` — each frame must
        contain an ``"angles"`` dict.

    Returns
    -------
    dict
        ``{"weights": {joint: float}, "primary_joint": str,
        "primary_min": float, "primary_max": float}``
    """
    rom_min: dict[str, float] = {j: 360.0 for j in _JOINT_NAMES}
    rom_max: dict[str, float] = {j: 0.0 for j in _JOINT_NAMES}

    for frame in gt_norm:
        angles = frame.get("angles", {})
        for j in _JOINT_NAMES:
            val = angles.get(j)
            if val is None:
                continue
            rom_min[j] = min(rom_min[j], val)
            rom_max[j] = max(rom_max[j], val)

    ranges: dict[str, float] = {}
    for j in _JOINT_NAMES:
        r = max(0.0, rom_max[j] - rom_min[j])
        ranges[j] = r

    total_range = sum(ranges.values())

    if total_range < 1e-6:
        weights = {j: 1.0 / len(_JOINT_NAMES) for j in _JOINT_NAMES}
    else:
        raw = {j: ranges[j] / total_range for j in _JOINT_NAMES}
        weights = {j: max(w, _MIN_WEIGHT) for j, w in raw.items()}
        wsum = sum(weights.values())
        weights = {j: w / wsum for j, w in weights.items()}

    primary_joint = max(ranges, key=ranges.get)  # type: ignore[arg-type]

    return {
        "weights": weights,
        "primary_joint": primary_joint,
        "primary_min": rom_min[primary_joint],
        "primary_max": rom_max[primary_joint],
    }
