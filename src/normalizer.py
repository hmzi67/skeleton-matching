"""
normalizer.py — Normalize raw MediaPipe skeleton data for body-size,
position, and scale independence.

Steps applied per frame:
1. TRANSLATE       — centre on hip midpoint
2. SCALE           — divide by torso length
3. ROTATE          — align shoulder vector to X-axis (view-invariant)
4. JOINT ANGLES    — compute 9 anatomical joint angles in degrees
5. SYMMETRY RATIOS — min/max ratio for paired left/right joints

After all frames are processed:
6. ANGULAR VELOCITY — rate-of-change of each joint angle across frames
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _vec(a: dict, b: dict) -> np.ndarray:
    """Vector from landmark *a* to landmark *b* (3-D)."""
    return np.array([b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"]])


def compute_angle(a: dict, b: dict, c: dict) -> float:
    """Compute the angle at vertex *b* formed by segments a→b and c→b.

    Parameters
    ----------
    a, b, c : dict
        Landmark dicts with ``x``, ``y``, ``z`` keys.
        *b* is the vertex joint; *a* and *c* are the adjacent joints.

    Returns
    -------
    float
        Angle in **degrees** in the range [0, 180].
    """
    v1 = _vec(b, a)
    v2 = _vec(b, c)
    dot = float(np.dot(v1, v2))
    mag = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if mag < 1e-9:
        return 0.0
    cos_angle = np.clip(dot / mag, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _angle_from_vertical(v: np.ndarray) -> float:
    """Angle (degrees) between vector *v* and the vertical axis (0, -1, 0)."""
    vertical = np.array([0.0, -1.0, 0.0])
    dot = float(np.dot(v, vertical))
    mag = float(np.linalg.norm(v) * np.linalg.norm(vertical))
    if mag < 1e-9:
        return 0.0
    cos_angle = np.clip(dot / mag, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Rotation alignment helper
# ---------------------------------------------------------------------------


def _rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix around the Y-axis (vertical)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])


def _align_to_x_axis(landmarks: list[dict]) -> list[dict]:
    """Rotate landmarks around the Y-axis so the shoulder vector
    (left_shoulder → right_shoulder) is aligned with the X-axis.

    This removes global facing-direction variance (camera angle).
    """
    left_sh = landmarks[11]
    right_sh = landmarks[12]

    # Shoulder vector projected onto the XZ plane.
    dx = right_sh["x"] - left_sh["x"]
    dz = right_sh["z"] - left_sh["z"]

    # Angle from the X-axis in the XZ plane.
    angle = math.atan2(dz, dx)

    if abs(angle) < 1e-6:
        return landmarks  # already aligned

    rot = _rotation_matrix_y(-angle)

    rotated: list[dict] = []
    for lm in landmarks:
        pt = np.array([lm["x"], lm["y"], lm["z"]])
        rpt = rot @ pt
        rotated.append({
            "x": float(rpt[0]),
            "y": float(rpt[1]),
            "z": float(rpt[2]),
            "visibility": lm["visibility"],
        })
    return rotated


# ---------------------------------------------------------------------------
# Joint-angle definitions
# ---------------------------------------------------------------------------

# (joint_name, vertex_index, adjacent_index_a, adjacent_index_b)
_ANGLE_DEFS: list[tuple[str, int, int, int]] = [
    ("left_elbow",     13, 11, 15),
    ("right_elbow",    14, 12, 16),
    ("left_knee",      25, 23, 27),
    ("right_knee",     26, 24, 28),
    ("left_hip",       23, 11, 25),
    ("right_hip",      24, 12, 26),
    ("left_shoulder",  11, 13, 23),
    ("right_shoulder", 12, 14, 24),
]

# Paired joints for symmetry ratios
_SYMMETRY_PAIRS = [
    ("left_elbow",    "right_elbow"),
    ("left_knee",     "right_knee"),
    ("left_hip",      "right_hip"),
    ("left_shoulder", "right_shoulder"),
]


def _compute_frame_angles(lms: list[dict]) -> dict[str, float]:
    """Compute all joint angles and symmetry ratios for a single frame.

    Parameters
    ----------
    lms : list[dict]
        The 33-landmark list for the frame.

    Returns
    -------
    dict[str, float]
        Mapping of joint name → angle in degrees, plus
        ``symmetry_<pair>`` ratios in [0, 1].
    """
    angles: dict[str, float] = {}

    for name, vertex, adj_a, adj_b in _ANGLE_DEFS:
        angles[name] = compute_angle(lms[adj_a], lms[vertex], lms[adj_b])

    # Torso lean: angle of spine vector (hip_center → shoulder_center) from
    # the vertical axis.
    hip_center = {
        k: (lms[23][k] + lms[24][k]) / 2.0 for k in ("x", "y", "z")
    }
    shoulder_center = {
        k: (lms[11][k] + lms[12][k]) / 2.0 for k in ("x", "y", "z")
    }
    spine_vec = _vec(hip_center, shoulder_center)
    angles["torso_lean"] = _angle_from_vertical(spine_vec)

    # --- Symmetry ratios ---
    for left, right in _SYMMETRY_PAIRS:
        l_val = abs(angles.get(left, 0.0))
        r_val = abs(angles.get(right, 0.0))
        max_val = max(l_val, r_val)
        min_val = min(l_val, r_val)
        ratio = (min_val / max_val) if max_val > 1e-3 else 1.0
        pair_name = left.replace("left_", "")  # e.g. "elbow"
        angles[f"symmetry_{pair_name}"] = round(ratio, 4)

    return angles


# ---------------------------------------------------------------------------
# Main normalisation function
# ---------------------------------------------------------------------------


def normalize_skeleton(skeleton_data: list[dict]) -> list[dict]:
    """Normalize a skeleton sequence for body-size / position independence.

    The function applies these steps per frame:

    1. **Translate** — shift all landmarks so the hip midpoint is at the
       origin.
    2. **Scale** — divide all coordinates by the torso length (hip-centre →
       shoulder-centre distance).
    3. **Rotate** — align the shoulder vector to the X-axis to remove
       global facing-direction variance.
    4. **Joint angles** — compute 9 anatomical joint angles in degrees.
    5. **Symmetry ratios** — min/max ratio for each left/right joint pair.

    After all frames are processed:

    6. **Angular velocity** — rate-of-change of each joint angle using
       timestamp deltas.

    Frames where the computed torso length is < 0.01 are dropped (unreliable
    pose).

    Parameters
    ----------
    skeleton_data : list[dict]
        Output of ``extract_skeleton_from_video`` — each dict has
        ``frame_index``, ``timestamp_ms``, and ``landmarks``.

    Returns
    -------
    list[dict]
        Same structure, but each frame now also contains:

        * ``normalized_landmarks`` — translated + scaled + rotated coordinates
        * ``angles`` — ``dict[str, float]`` of joint angles in degrees
        * ``velocities`` — ``dict[str, float]`` of angular velocity (°/s)
    """
    normalised: list[dict] = []

    for frame in skeleton_data:
        lms = frame["landmarks"]

        # --- 1. TRANSLATE --------------------------------------------------
        hip_center_x = (lms[23]["x"] + lms[24]["x"]) / 2.0
        hip_center_y = (lms[23]["y"] + lms[24]["y"]) / 2.0
        hip_center_z = (lms[23]["z"] + lms[24]["z"]) / 2.0

        translated: list[dict] = []
        for lm in lms:
            translated.append(
                {
                    "x": lm["x"] - hip_center_x,
                    "y": lm["y"] - hip_center_y,
                    "z": lm["z"] - hip_center_z,
                    "visibility": lm["visibility"],
                }
            )

        # --- 2. SCALE ------------------------------------------------------
        shoulder_center_x = (translated[11]["x"] + translated[12]["x"]) / 2.0
        shoulder_center_y = (translated[11]["y"] + translated[12]["y"]) / 2.0
        shoulder_center_z = (translated[11]["z"] + translated[12]["z"]) / 2.0

        torso_length = math.sqrt(
            shoulder_center_x ** 2
            + shoulder_center_y ** 2
            + shoulder_center_z ** 2
        )

        if torso_length < 0.01:
            continue  # unreliable frame

        scaled: list[dict] = []
        for lm in translated:
            scaled.append(
                {
                    "x": lm["x"] / torso_length,
                    "y": lm["y"] / torso_length,
                    "z": lm["z"] / torso_length,
                    "visibility": lm["visibility"],
                }
            )

        # --- 3. ROTATE (align shoulder to X-axis) --------------------------
        rotated = _align_to_x_axis(scaled)

        # --- 4. JOINT ANGLES + 5. SYMMETRY RATIOS -------------------------
        angles = _compute_frame_angles(rotated)

        normalised.append(
            {
                "frame_index": frame["frame_index"],
                "timestamp_ms": frame["timestamp_ms"],
                "landmarks": frame["landmarks"],          # keep originals
                "normalized_landmarks": rotated,
                "angles": angles,
            }
        )

    # --- 6. ANGULAR VELOCITY (post-pass) -----------------------------------
    _joint_names_for_velocity = [
        "left_elbow", "right_elbow", "left_knee", "right_knee",
        "left_hip", "right_hip", "left_shoulder", "right_shoulder",
        "torso_lean",
    ]

    for i, frame in enumerate(normalised):
        velocities: dict[str, float] = {}
        if i == 0:
            # First frame: no velocity information.
            for j in _joint_names_for_velocity:
                velocities[j] = 0.0
        else:
            prev = normalised[i - 1]
            dt_ms = frame["timestamp_ms"] - prev["timestamp_ms"]
            dt_s = dt_ms / 1000.0 if dt_ms > 0 else 1.0 / 30.0  # fallback 30fps

            for j in _joint_names_for_velocity:
                curr_angle = frame["angles"].get(j, 0.0)
                prev_angle = prev["angles"].get(j, 0.0)
                velocities[j] = round((curr_angle - prev_angle) / dt_s, 2)

        frame["velocities"] = velocities

    return normalised
