"""
matcher.py — Compare normalised skeleton sequences using DTW alignment.

Design goals:
  - CORRECTNESS-FOCUSED: scores are based on joint angle accuracy only.
    Speed differences between user and reference are handled by DTW alignment
    (offline) and by the TemporalAligner (live).  Velocity is computed and
    surfaced in feedback but does NOT contribute to the score.
  - GENERIC: works for any exercise.  Pose joints (9) are always scored.
    Hand joints (10) are scored when both GT and user have hand data.
  - ROBUST: joints absent in either GT or user are silently skipped so that
    old cached skeleton data (pre-hand-landmark) keeps working.

Provides:
  - match_single_frame()      — per-joint comparison for one frame pair
  - match_video_sequence()    — DTW-aligned comparison of two full sequences
  - compute_summary()         — aggregate statistics across all matched frames
"""

from __future__ import annotations

import numpy as np
from dtaidistance import dtw_ndim

from src.exercise_weights import ALL_JOINT_NAMES, _JOINT_NAMES, _HAND_JOINT_NAMES


# ---------------------------------------------------------------------------
# Angle-accuracy thresholds — used when no adaptive thresholds are supplied
# ---------------------------------------------------------------------------

_GOOD_THRESHOLD = 10     # degrees — within this → "good"
_WARNING_THRESHOLD = 25  # degrees — within this → "warning"; beyond → "bad"

# Maximum angle penalty per joint (caps contribution so score ∈ [0, 100]).
_MAX_PENALTY_PER_JOINT = 45.0

# Small natural style differences should not tank score.
_POSE_ANGLE_TOLERANCE_DEG = 5.0
_HAND_ANGLE_TOLERANCE_DEG = 8.0

# Missing-joint penalty should be gentle unless coverage is extremely low.
_COVERAGE_NO_PENALTY = 0.70
_COVERAGE_SOFT_FLOOR = 0.50

# ---------------------------------------------------------------------------
# Velocity thresholds (informational only — NOT included in score)
# ---------------------------------------------------------------------------

# Ratio of user speed to reference speed that triggers a flag.
_VELOCITY_FAST_RATIO = 1.8   # user > 1.8× ref speed → "too_fast" warning
_VELOCITY_SLOW_RATIO = 0.3   # user < 0.3× ref speed → "too_slow" warning

# Ignore velocity comparison when reference motion is near-static (noisy).
_VELOCITY_MIN_DEG_PER_SEC = 15.0

# Paired joints for symmetry checking.
_SYMMETRY_PAIRS = [
    ("left_elbow",    "right_elbow"),
    ("left_knee",     "right_knee"),
    ("left_hip",      "right_hip"),
    ("left_shoulder", "right_shoulder"),
]

_SYMMETRY_THRESHOLD = 0.85  # ratio below this flags asymmetry


def _angle_tolerance_for_joint(joint: str) -> float:
    """Return scoring dead-zone (degrees) for a joint."""
    if joint in _HAND_JOINT_NAMES:
        return _HAND_ANGLE_TOLERANCE_DEG
    return _POSE_ANGLE_TOLERANCE_DEG


def _soft_coverage_factor(coverage: float) -> float:
    """Map coverage [0,1] to a soft score multiplier.

    Coverage >= _COVERAGE_NO_PENALTY receives no penalty.
    Below that, scale down gradually with a floor to avoid collapsing scores
    for otherwise-correct form when some joints are intermittently unavailable.
    """
    c = max(0.0, min(1.0, coverage))
    if c >= _COVERAGE_NO_PENALTY:
        return 1.0
    ratio = c / _COVERAGE_NO_PENALTY if _COVERAGE_NO_PENALTY > 1e-9 else 1.0
    return _COVERAGE_SOFT_FLOOR + (1.0 - _COVERAGE_SOFT_FLOOR) * ratio


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
    """Compare joint angles of a ground-truth frame and a user frame.

    Parameters
    ----------
    gt_frame : dict
        A normalised frame dict with an ``"angles"`` key and optionally
        a ``"velocities"`` key.
    user_frame : dict
        Same structure as *gt_frame*.
    adaptive_thresholds : dict, optional
        Per-joint ``{"good": float, "warning": float}`` from
        ``calibration.compute_adaptive_thresholds``.
    exercise_weights : dict, optional
        Per-joint weight ``{joint_name: float}`` from
        ``exercise_weights.get_weights`` or ``compute_auto_weights``.

        Joints **not present** in this dict get weight **0.0** — they are
        compared and reported but excluded from the score.  This is
        intentional: unknown/irrelevant joints should never silently inflate
        or deflate the score.

        When ``None``, pose joints receive equal weight (1/9 each); hand
        joints receive weight 0.0 unless hand data is present in both frames
        (in which case they also receive equal weight among all active joints).

    Returns
    -------
    dict
        ``{ joint_name: {"gt", "user", "diff", "abs_diff", "status",
        "velocity_status"}, ..., "overall_score", "symmetry" }``
    """
    gt_angles = gt_frame.get("angles", {})
    user_angles = user_frame.get("angles", {})
    gt_velocities = gt_frame.get("velocities", {})
    user_velocities = user_frame.get("velocities", {})
    has_velocity = bool(gt_velocities and user_velocities)

    # ------------------------------------------------------------------
    # Determine which joints will be compared and scored.
    # A joint is "active" when it has a non-None value in BOTH frames.
    # ------------------------------------------------------------------
    active_joints: list[str] = []
    for j in ALL_JOINT_NAMES:
        if gt_angles.get(j) is not None and user_angles.get(j) is not None:
            active_joints.append(j)

    # ------------------------------------------------------------------
    # Resolve weights for active joints.
    # ------------------------------------------------------------------
    if exercise_weights is not None:
        # Use provided weights; joints absent from the dict → weight 0.
        joint_weights = {j: exercise_weights.get(j, 0.0) for j in active_joints}
    else:
        # Equal weight for pose joints; equal weight for hand joints only
        # when they are actually detected in this frame pair.
        pose_active = [j for j in active_joints if j in _JOINT_NAMES]
        hand_active = [j for j in active_joints if j in _HAND_JOINT_NAMES]

        n_pose = len(pose_active) or 1
        n_hand = len(hand_active) or 1

        joint_weights = {}
        for j in pose_active:
            joint_weights[j] = 1.0 / n_pose
        for j in hand_active:
            joint_weights[j] = 1.0 / n_hand

        # Blend: if hand joints are present, give them combined 30% weight.
        if hand_active and pose_active:
            for j in pose_active:
                joint_weights[j] *= 0.70
            for j in hand_active:
                joint_weights[j] *= 0.30

    # Normalise weights so they sum to 1.0 over scored joints.
    scored_joints = [j for j in active_joints if joint_weights.get(j, 0.0) > 0]
    weight_sum = sum(joint_weights.get(j, 0.0) for j in scored_joints)
    if weight_sum > 1e-9:
        for j in scored_joints:
            joint_weights[j] /= weight_sum

    # ------------------------------------------------------------------
    # Per-joint comparison
    # ------------------------------------------------------------------
    result: dict = {}
    joint_scores: dict[str, float] = {}
    joint_errors: dict[str, float] = {}
    total_weighted_penalty = 0.0
    total_weight = 0.0

    for joint in active_joints:
        gt_val = float(gt_angles[joint])
        user_val = float(user_angles[joint])
        diff = user_val - gt_val
        abs_diff = abs(diff)

        # Determine accuracy thresholds (adaptive or fixed).
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

        # Velocity comparison — informational only, not scored.
        velocity_status = "ok"
        if has_velocity:
            gt_vel = abs(gt_velocities.get(joint, 0.0))
            user_vel = abs(user_velocities.get(joint, 0.0))

            if gt_vel >= _VELOCITY_MIN_DEG_PER_SEC:
                ratio = user_vel / gt_vel if gt_vel > 1e-6 else 1.0
                if ratio > _VELOCITY_FAST_RATIO:
                    velocity_status = "too_fast"
                elif ratio < _VELOCITY_SLOW_RATIO:
                    velocity_status = "too_slow"
            elif user_vel > _VELOCITY_MIN_DEG_PER_SEC * _VELOCITY_FAST_RATIO:
                velocity_status = "too_fast"

        result[joint] = {
            "gt":             round(gt_val,   2),
            "user":           round(user_val, 2),
            "diff":           round(diff,     2),
            "abs_diff":       round(abs_diff, 2),
            "status":         status,
            "velocity_status": velocity_status,
        }

        # Per-joint normalised score (0.0–1.0) and raw error for reporting.
        # A small dead-zone keeps natural movement style differences from
        # being over-penalized.
        effective_abs_diff = max(0.0, abs_diff - _angle_tolerance_for_joint(joint))
        capped = min(effective_abs_diff, _MAX_PENALTY_PER_JOINT)
        joint_scores[joint] = round(1.0 - (capped / _MAX_PENALTY_PER_JOINT), 4)
        joint_errors[joint] = round(abs_diff, 2)

        # Accumulate score penalty only for scored (weight > 0) joints.
        weight = joint_weights.get(joint, 0.0)
        if weight > 0:
            penalty = capped * weight
            total_weighted_penalty += penalty
            total_weight += weight

    # ------------------------------------------------------------------
    # Overall score — angle accuracy only (velocity does NOT affect score).
    # ------------------------------------------------------------------
    max_angle_penalty = total_weight * _MAX_PENALTY_PER_JOINT
    if max_angle_penalty > 0:
        raw_score = max(0.0, 100.0 * (1.0 - total_weighted_penalty / max_angle_penalty))

        # Coverage penalty: compute weighted coverage and apply a soft factor.
        # This keeps scores stable when a few joints flicker out of view,
        # while still penalizing sustained low-observability sessions.
        if exercise_weights is not None:
            expected_weight_sum = sum(
                max(0.0, exercise_weights.get(j, 0.0)) for j in ALL_JOINT_NAMES
            )
            observed_weight_sum = sum(
                max(0.0, exercise_weights.get(j, 0.0)) for j in active_joints
            )
            coverage = (
                observed_weight_sum / expected_weight_sum
                if expected_weight_sum > 1e-9 else 1.0
            )
        else:
            pose_observed = len([j for j in active_joints if j in _JOINT_NAMES])
            coverage = pose_observed / len(_JOINT_NAMES) if _JOINT_NAMES else 1.0

        raw_score *= _soft_coverage_factor(coverage)

        overall_score = raw_score
    else:
        # No joints were scored (e.g. hand landmarks not detected for a hand
        # exercise). Return 0 rather than a perfect score — the user has not
        # demonstrated any correct form yet.
        overall_score = 0.0

    result["overall_score"] = round(overall_score, 1)
    result["joint_scores"] = joint_scores
    result["joint_errors"] = joint_errors

    # ------------------------------------------------------------------
    # Symmetry analysis (pose joints only)
    # ------------------------------------------------------------------
    symmetry: dict[str, dict] = {}
    for left, right in _SYMMETRY_PAIRS:
        l_val = user_angles.get(left)
        r_val = user_angles.get(right)
        if l_val is None or r_val is None:
            continue

        max_val = max(abs(l_val), abs(r_val))
        min_val = min(abs(l_val), abs(r_val))
        ratio = (min_val / max_val) if max_val > 1e-3 else 1.0

        pair_name = left.replace("left_", "")
        symmetry[pair_name] = {
            "left":     round(float(l_val), 2),
            "right":    round(float(r_val), 2),
            "ratio":    round(ratio, 3),
            "balanced": ratio >= _SYMMETRY_THRESHOLD,
        }

    result["symmetry"] = symmetry
    return result


# ---------------------------------------------------------------------------
# DTW-aligned sequence comparison  (multi-dimensional)
# ---------------------------------------------------------------------------


def _angle_vector(
    frame: dict,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Extract the (optionally weighted) angle feature vector from a frame.

    When *weights* is supplied each dimension is scaled by ``sqrt(weight)``
    so that DTW path optimisation is driven by the joints that matter most
    for the current exercise. Irrelevant joints (weight ≈ 0) contribute
    almost nothing to the distance, keeping alignment exercise-aware.

    Missing joints default to 0.0 so vectors are always the same length.
    """
    angles = frame.get("angles", {})
    n = len(ALL_JOINT_NAMES)
    result = np.zeros(n, dtype=np.float64)
    for i, j in enumerate(ALL_JOINT_NAMES):
        val = angles.get(j, 0.0)
        if weights is not None:
            # Scale by sqrt(w * n) so total energy is preserved on average.
            val *= (weights.get(j, 0.0) * n) ** 0.5
        result[i] = val
    return result


def match_video_sequence(
    gt_skeleton: list[dict],
    user_skeleton: list[dict],
    *,
    adaptive_thresholds: dict[str, dict[str, float]] | None = None,
    exercise_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Align two normalised skeleton sequences with multi-dimensional DTW,
    then compare them frame by frame.

    DTW handles speed differences between the user and reference: a user who
    performs the movement slower or faster will still be scored on correctness
    of the angles, not on timing.

    Parameters
    ----------
    gt_skeleton : list[dict]
        Ground-truth normalised skeleton frames.
    user_skeleton : list[dict]
        User normalised skeleton frames.
    adaptive_thresholds : dict, optional
        Per-joint thresholds from ROM calibration.
    exercise_weights : dict, optional
        Per-joint weights for this exercise type.

    Returns
    -------
    list[dict]
        One ``match_single_frame`` result per aligned frame pair, with
        additional keys ``gt_frame_index`` and ``user_frame_index``.
    """
    gt_vecs  = np.ascontiguousarray(
        [_angle_vector(f, exercise_weights) for f in gt_skeleton],  dtype=np.double
    )
    user_vecs = np.ascontiguousarray(
        [_angle_vector(f, exercise_weights) for f in user_skeleton], dtype=np.double
    )

    path = dtw_ndim.warping_path(gt_vecs, user_vecs)

    results: list[dict] = []
    for gt_idx, user_idx in path:
        frame_result = match_single_frame(
            gt_skeleton[gt_idx],
            user_skeleton[user_idx],
            adaptive_thresholds=adaptive_thresholds,
            exercise_weights=exercise_weights,
        )
        frame_result["gt_frame_index"]   = gt_skeleton[gt_idx]["frame_index"]
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
        ``best_joint``, ``frame_scores``, ``velocity_issues``,
        ``symmetry_summary``.
    """
    if not match_results:
        return {
            "overall_score":       0.0,
            "per_joint_avg_error": {},
            "velocity_issues":     {},
            "worst_joint":         "",
            "best_joint":          "",
            "frame_scores":        [],
            "symmetry_summary":    {},
        }

    frame_scores = [r["overall_score"] for r in match_results]

    # Per-joint average absolute error (for joints that appear in results).
    joint_errors: dict[str, list[float]] = {}
    for r in match_results:
        for j in ALL_JOINT_NAMES:
            if j in r and isinstance(r[j], dict) and "abs_diff" in r[j]:
                joint_errors.setdefault(j, []).append(r[j]["abs_diff"])

    per_joint_avg = {
        j: round(float(np.mean(errs)), 2)
        for j, errs in joint_errors.items()
        if errs
    }

    if per_joint_avg:
        worst_joint = max(per_joint_avg, key=per_joint_avg.get)  # type: ignore[arg-type]
        best_joint  = min(per_joint_avg, key=per_joint_avg.get)  # type: ignore[arg-type]
    else:
        worst_joint = best_joint = ""

    # Velocity issue rate per joint.
    velocity_issues: dict[str, dict[str, float]] = {}
    for j in ALL_JOINT_NAMES:
        fast_count = slow_count = total = 0
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

    # Symmetry summary.
    pair_ratios: dict[str, list[float]] = {}
    for r in match_results:
        for pair_name, data in r.get("symmetry", {}).items():
            pair_ratios.setdefault(pair_name, []).append(data["ratio"])

    symmetry_summary = {
        pair_name: {
            "avg_ratio": round(float(np.mean(ratios)), 3),
            "balanced":  float(np.mean(ratios)) >= _SYMMETRY_THRESHOLD,
        }
        for pair_name, ratios in pair_ratios.items()
    }

    return {
        "overall_score":       round(float(np.mean(frame_scores)), 1),
        "per_joint_avg_error": per_joint_avg,
        "velocity_issues":     velocity_issues,
        "worst_joint":         worst_joint,
        "best_joint":          best_joint,
        "frame_scores":        frame_scores,
        "symmetry_summary":    symmetry_summary,
    }
