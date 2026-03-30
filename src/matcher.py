"""
matcher.py — Compare normalised skeleton sequences using DTW alignment.

Supports:
  - Adaptive ROM-based thresholds (per-user scoring)
  - Exercise-specific joint weighting
  - Multi-dimensional DTW for better alignment
  - Symmetry penalty integration
  - Fallback to fixed thresholds when no ROM profile is provided

Provides:
  - match_single_frame()      — per-joint angle comparison for one frame pair
  - match_video_sequence()    — DTW-aligned comparison of two full sequences
  - compute_summary()         — aggregate statistics across all matched frames
"""

from __future__ import annotations

import numpy as np
from dtaidistance import dtw, dtw_ndim


# ---------------------------------------------------------------------------
# Default (fixed) thresholds — used when no ROM profile is supplied
# ---------------------------------------------------------------------------

_GOOD_THRESHOLD = 10     # degrees
_WARNING_THRESHOLD = 25  # degrees

# Maximum penalty per joint (caps contribution so score stays in [0, 100]).
_MAX_PENALTY_PER_JOINT = 45.0

# ---------------------------------------------------------------------------
# Velocity thresholds
# ---------------------------------------------------------------------------

# How much faster/slower than the reference is acceptable (as a ratio).
_VELOCITY_FAST_RATIO = 1.8   # user > 1.8× ref speed → too fast
_VELOCITY_SLOW_RATIO = 0.3   # user < 0.3× ref speed → too slow

# Absolute velocity floor — ignore velocity comparison when both are slow
# (avoids noisy penalties during near-static holds).
_VELOCITY_MIN_DEG_PER_SEC = 15.0

# Maximum velocity penalty per joint (degrees/sec excess, capped).
_MAX_VELOCITY_PENALTY = 40.0

# Blend ratio: overall = (1 - α) × angle_score + α × velocity_score
_VELOCITY_BLEND_ALPHA = 0.20

# Joint names expected in the "angles" dict produced by normalizer.
_JOINT_NAMES = [
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

# Paired joints for symmetry checking
_SYMMETRY_PAIRS = [
    ("left_elbow", "right_elbow"),
    ("left_knee", "right_knee"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "right_shoulder"),
]

_SYMMETRY_THRESHOLD = 0.85  # ratio below this flags asymmetry


# ---------------------------------------------------------------------------
# Single-frame comparison
# ---------------------------------------------------------------------------


def match_single_frame(
    gt_frame: dict,
    user_frame: dict,
    *,
    adaptive_thresholds: dict[str, dict[str, float]] | None = None,
    exercise_weights: dict[str, float] | None = None,
) -> dict:
    """Compare the joint angles of a ground-truth frame and a user frame.

    Parameters
    ----------
    gt_frame : dict
        A normalised frame dict containing an ``"angles"`` key and,
        optionally, a ``"velocities"`` key.
    user_frame : dict
        Same structure as *gt_frame*.
    adaptive_thresholds : dict, optional
        Per-joint ``{"good": float, "warning": float}`` from
        ``calibration.compute_adaptive_thresholds``. When provided, scoring
        is relative to the user's ROM rather than fixed degree thresholds.
    exercise_weights : dict, optional
        Per-joint weight ``{joint_name: float}`` from
        ``exercise_weights.get_weights``. When provided, each joint's
        contribution to the overall score is weighted accordingly.

    Returns
    -------
    dict
        ``{ joint_name: {"gt": float, "user": float, "diff": float,
        "abs_diff": float, "status": str, "velocity_status": str}, ...,
        "overall_score": float, "angle_score": float,
        "velocity_score": float, "symmetry": dict }``
    """
    gt_angles = gt_frame["angles"]
    user_angles = user_frame["angles"]
    gt_velocities = gt_frame.get("velocities", {})
    user_velocities = user_frame.get("velocities", {})
    has_velocity = bool(gt_velocities and user_velocities)

    result: dict = {}
    total_weighted_penalty = 0.0
    total_velocity_penalty = 0.0
    total_weight = 0.0

    for joint in _JOINT_NAMES:
        gt_val = gt_angles.get(joint, 0.0)
        user_val = user_angles.get(joint, 0.0)
        diff = user_val - gt_val
        abs_diff = abs(diff)

        # --- Determine thresholds (adaptive or fixed) ---
        if adaptive_thresholds and joint in adaptive_thresholds:
            good_th = adaptive_thresholds[joint]["good"]
            warn_th = adaptive_thresholds[joint]["warning"]
        else:
            good_th = _GOOD_THRESHOLD
            warn_th = _WARNING_THRESHOLD

        if abs_diff < good_th:
            status = "good"
        elif abs_diff < warn_th:
            status = "warning"
        else:
            status = "bad"

        # --- Velocity comparison ---
        velocity_status = "ok"
        velocity_excess = 0.0
        if has_velocity:
            gt_vel = abs(gt_velocities.get(joint, 0.0))
            user_vel = abs(user_velocities.get(joint, 0.0))

            # Only compare when reference motion is meaningful
            if gt_vel >= _VELOCITY_MIN_DEG_PER_SEC:
                ratio = user_vel / gt_vel if gt_vel > 1e-6 else 1.0
                if ratio > _VELOCITY_FAST_RATIO:
                    velocity_status = "too_fast"
                    velocity_excess = user_vel - gt_vel * _VELOCITY_FAST_RATIO
                elif ratio < _VELOCITY_SLOW_RATIO:
                    velocity_status = "too_slow"
                    velocity_excess = gt_vel * _VELOCITY_SLOW_RATIO - user_vel
            elif user_vel > _VELOCITY_MIN_DEG_PER_SEC * _VELOCITY_FAST_RATIO:
                # Reference is near-static but user is moving fast
                velocity_status = "too_fast"
                velocity_excess = user_vel

        result[joint] = {
            "gt": round(gt_val, 2),
            "user": round(user_val, 2),
            "diff": round(diff, 2),
            "abs_diff": round(abs_diff, 2),
            "status": status,
            "velocity_status": velocity_status,
        }

        # --- Weighted penalties ---
        weight = (exercise_weights or {}).get(joint, 1.0 / len(_JOINT_NAMES))
        penalty = min(abs_diff, _MAX_PENALTY_PER_JOINT) * weight
        vel_penalty = min(velocity_excess, _MAX_VELOCITY_PENALTY) * weight

        total_weighted_penalty += penalty
        total_velocity_penalty += vel_penalty
        total_weight += weight

    # --- Component scores (0–100) ---
    max_angle_penalty = total_weight * _MAX_PENALTY_PER_JOINT
    angle_score = (
        max(0.0, 100.0 * (1.0 - total_weighted_penalty / max_angle_penalty))
        if max_angle_penalty
        else 100.0
    )

    max_vel_penalty = total_weight * _MAX_VELOCITY_PENALTY
    velocity_score = (
        max(0.0, 100.0 * (1.0 - total_velocity_penalty / max_vel_penalty))
        if (max_vel_penalty and has_velocity)
        else 100.0
    )

    # --- Blended overall score ---
    alpha = _VELOCITY_BLEND_ALPHA if has_velocity else 0.0
    overall = (1.0 - alpha) * angle_score + alpha * velocity_score

    result["angle_score"] = round(angle_score, 1)
    result["velocity_score"] = round(velocity_score, 1)
    result["overall_score"] = round(overall, 1)

    # --- Symmetry analysis ---
    symmetry: dict[str, dict] = {}
    for left, right in _SYMMETRY_PAIRS:
        l_val = user_angles.get(left, 0.0)
        r_val = user_angles.get(right, 0.0)

        max_val = max(abs(l_val), abs(r_val))
        min_val = min(abs(l_val), abs(r_val))
        ratio = (min_val / max_val) if max_val > 1e-3 else 1.0

        pair_name = left.replace("left_", "")  # e.g. "elbow"
        symmetry[pair_name] = {
            "left": round(l_val, 2),
            "right": round(r_val, 2),
            "ratio": round(ratio, 3),
            "balanced": ratio >= _SYMMETRY_THRESHOLD,
        }

    result["symmetry"] = symmetry

    return result


# ---------------------------------------------------------------------------
# DTW-aligned sequence comparison  (multi-dimensional)
# ---------------------------------------------------------------------------


def _angle_vector(frame: dict) -> np.ndarray:
    """Extract the angle vector (one float per joint) from a normalised frame."""
    angles = frame["angles"]
    return np.array([angles.get(j, 0.0) for j in _JOINT_NAMES], dtype=np.float64)


def match_video_sequence(
    gt_skeleton: list[dict],
    user_skeleton: list[dict],
    *,
    adaptive_thresholds: dict[str, dict[str, float]] | None = None,
    exercise_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Align two normalised skeleton sequences with multi-dimensional DTW,
    then compare them frame by frame.

    Parameters
    ----------
    gt_skeleton : list[dict]
        Ground-truth normalised skeleton frames.
    user_skeleton : list[dict]
        User normalised skeleton frames.
    adaptive_thresholds : dict, optional
        Per-joint thresholds from ROM calibration.
    exercise_weights : dict, optional
        Per-joint weights for the exercise type.

    Returns
    -------
    list[dict]
        One ``match_single_frame`` result per aligned frame pair, with
        additional keys ``gt_frame_index`` and ``user_frame_index``.
    """
    gt_vecs = np.array([_angle_vector(f) for f in gt_skeleton])
    user_vecs = np.array([_angle_vector(f) for f in user_skeleton])

    # Multi-dimensional DTW on the full angle vectors (9-D).
    # dtw_ndim expects shape (n_timesteps, n_dims) with dtype double.
    gt_vecs_d = np.ascontiguousarray(gt_vecs, dtype=np.double)
    user_vecs_d = np.ascontiguousarray(user_vecs, dtype=np.double)

    path = dtw_ndim.warping_path(gt_vecs_d, user_vecs_d)

    results: list[dict] = []
    for gt_idx, user_idx in path:
        frame_result = match_single_frame(
            gt_skeleton[gt_idx],
            user_skeleton[user_idx],
            adaptive_thresholds=adaptive_thresholds,
            exercise_weights=exercise_weights,
        )
        frame_result["gt_frame_index"] = gt_skeleton[gt_idx]["frame_index"]
        frame_result["user_frame_index"] = user_skeleton[user_idx]["frame_index"]
        results.append(frame_result)

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def compute_summary(match_results: list[dict]) -> dict:
    """Aggregate match results across all aligned frames.

    Parameters
    ----------
    match_results : list[dict]
        Output of ``match_video_sequence``.

    Returns
    -------
    dict
        Keys: ``overall_score``, ``per_joint_avg_error``, ``worst_joint``,
        ``best_joint``, ``frame_scores``, ``symmetry_summary``.
    """
    if not match_results:
        return {
            "overall_score": 0.0,
            "angle_score": 0.0,
            "velocity_score": 0.0,
            "per_joint_avg_error": {},
            "velocity_issues": {},
            "worst_joint": "",
            "best_joint": "",
            "frame_scores": [],
            "symmetry_summary": {},
        }

    frame_scores = [r["overall_score"] for r in match_results]
    angle_scores = [r.get("angle_score", r["overall_score"]) for r in match_results]
    velocity_scores = [r.get("velocity_score", 100.0) for r in match_results]

    # Per-joint average absolute error.
    joint_errors: dict[str, list[float]] = {j: [] for j in _JOINT_NAMES}
    for r in match_results:
        for j in _JOINT_NAMES:
            if j in r:
                joint_errors[j].append(r[j]["abs_diff"])

    per_joint_avg = {
        j: round(float(np.mean(errs)), 2) if errs else 0.0
        for j, errs in joint_errors.items()
    }

    worst_joint = max(per_joint_avg, key=per_joint_avg.get)  # type: ignore[arg-type]
    best_joint = min(per_joint_avg, key=per_joint_avg.get)   # type: ignore[arg-type]

    # Per-joint velocity issue rate (% of frames flagged too_fast / too_slow).
    velocity_issues: dict[str, dict[str, float]] = {}
    for j in _JOINT_NAMES:
        fast_count = 0
        slow_count = 0
        total = 0
        for r in match_results:
            if j in r and isinstance(r[j], dict):
                vs = r[j].get("velocity_status", "ok")
                if vs == "too_fast":
                    fast_count += 1
                elif vs == "too_slow":
                    slow_count += 1
                total += 1
        if total > 0 and (fast_count + slow_count) > 0:
            velocity_issues[j] = {
                "too_fast_pct": round(100.0 * fast_count / total, 1),
                "too_slow_pct": round(100.0 * slow_count / total, 1),
            }

    # Symmetry summary: average ratio per pair across all frames.
    symmetry_summary: dict[str, dict] = {}
    pair_ratios: dict[str, list[float]] = {}
    for r in match_results:
        sym = r.get("symmetry", {})
        for pair_name, data in sym.items():
            pair_ratios.setdefault(pair_name, []).append(data["ratio"])

    for pair_name, ratios in pair_ratios.items():
        avg_ratio = float(np.mean(ratios))
        symmetry_summary[pair_name] = {
            "avg_ratio": round(avg_ratio, 3),
            "balanced": avg_ratio >= _SYMMETRY_THRESHOLD,
        }

    return {
        "overall_score": round(float(np.mean(frame_scores)), 1),
        "angle_score": round(float(np.mean(angle_scores)), 1),
        "velocity_score": round(float(np.mean(velocity_scores)), 1),
        "per_joint_avg_error": per_joint_avg,
        "velocity_issues": velocity_issues,
        "worst_joint": worst_joint,
        "best_joint": best_joint,
        "frame_scores": frame_scores,
        "symmetry_summary": symmetry_summary,
    }
