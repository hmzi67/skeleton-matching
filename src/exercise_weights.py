"""
exercise_weights.py — Per-exercise joint importance weights.

Different exercises rely on different joints. For example, squats depend
heavily on knee and hip angles while elbows are nearly irrelevant.
This module provides weight vectors that the matcher uses to scale each
joint's contribution to the overall score.

Provides:
  - EXERCISE_WEIGHTS — registry of weight dicts
  - get_weights()    — look up weights by exercise name
"""

from __future__ import annotations


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
        "torso_lean":     0.15,
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
