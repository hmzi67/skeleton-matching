"""
feedback.py — Convert raw match results into actionable coaching feedback.

Supports:
  - Per-frame coaching cues with adaptive thresholds
  - Symmetry imbalance warnings
  - Velocity-based form warnings
  - Session reports with symmetry summary

Provides:
  - generate_feedback()        — per-frame coaching cues
  - generate_session_report()  — plain-text summary saved after each run
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    """Map a 0-100 score to a human-readable grade."""
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Needs Work"
    return "Poor"


# ---------------------------------------------------------------------------
# Coaching instruction generators (based on joint name + diff direction)
# ---------------------------------------------------------------------------

def _coaching_instruction(joint: str, diff: float) -> str:
    """Return a specific, actionable coaching cue for the given joint.

    Parameters
    ----------
    joint : str
        Joint name (e.g. ``"left_knee"``).
    diff : float
        ``user_angle - gt_angle`` in degrees (positive → joint angle is
        larger than the target).
    """
    side = joint.replace("_", " ").split()[0]  # "left" or "right"
    abs_d = abs(diff)

    if joint in ("left_knee", "right_knee"):
        if diff > 0:
            return f"Bend your {side} knee more - about {abs_d:.0f} deg further"
        return f"Straighten your {side} knee - reduce bend by {abs_d:.0f} deg"

    if joint in ("left_hip", "right_hip"):
        if diff > 0:
            return f"Push your {side} hip back more"
        return f"Drive your {side} hip forward"

    if joint in ("left_elbow", "right_elbow"):
        if diff > 0:
            return f"Bend your {side} arm more"
        return f"Extend your {side} arm more"

    if joint == "torso_lean":
        if diff > 0:
            return "Lean your torso forward slightly"
        return "Stand more upright - you're leaning too far forward"

    if joint in ("left_shoulder", "right_shoulder"):
        if diff > 0:
            return f"Raise your {side} arm higher - about {abs_d:.0f} deg more"
        return f"Lower your {side} arm - about {abs_d:.0f} deg less"

    return f"Adjust your {joint.replace('_', ' ')} by {abs_d:.0f} deg"


def _velocity_instruction(joint: str, velocity_status: str) -> str:
    """Return a coaching cue for velocity issues."""
    name = joint.replace("_", " ")
    if velocity_status == "too_fast":
        return f"Slow down your {name} — you're moving too fast"
    if velocity_status == "too_slow":
        return f"Speed up your {name} — try to match the reference tempo"
    return ""


def _symmetry_warning(pair_name: str, left: float, right: float) -> str:
    """Generate a warning for asymmetric joint usage."""
    diff = abs(left - right)
    if left > right:
        return (
            f"Your left {pair_name} is {diff:.0f}° more bent than your "
            f"right — try to keep both sides even"
        )
    return (
        f"Your right {pair_name} is {diff:.0f}° more bent than your "
        f"left — try to keep both sides even"
    )


# ---------------------------------------------------------------------------
# Per-frame feedback
# ---------------------------------------------------------------------------


def generate_feedback(summary: dict, current_frame_result: dict) -> dict:
    """Generate human-readable coaching feedback for the current frame.

    Parameters
    ----------
    summary : dict
        Output of ``matcher.compute_summary`` (used for overall context).
    current_frame_result : dict
        Output of ``matcher.match_single_frame`` for the frame to
        annotate.

    Returns
    -------
    dict
        Keys: ``overall_score``, ``grade``, ``headline``,
        ``joint_feedback`` (list), ``positive_feedback`` (list),
        ``priority_fix`` (str), ``symmetry_warnings`` (list).
    """
    score = current_frame_result.get("overall_score", 0)
    grade = _grade(score)

    joint_feedback: list[dict] = []
    positive_feedback: list[str] = []
    velocity_feedback: list[dict] = []

    for joint, data in current_frame_result.items():
        if joint in ("overall_score", "symmetry",
                      "gt_frame_index", "user_frame_index"):
            continue
        if not isinstance(data, dict):
            continue
        if "status" not in data:
            continue

        status = data.get("status", "good")
        if status == "good":
            positive_feedback.append(
                f"Your {joint.replace('_', ' ')} looks great!"
            )
        else:
            joint_feedback.append(
                {
                    "joint": joint,
                    "instruction": _coaching_instruction(joint, data["diff"]),
                    "severity": status,
                    "diff_degrees": data["abs_diff"],
                }
            )

        # --- Velocity feedback ---
        vel_status = data.get("velocity_status", "ok")
        if vel_status in ("too_fast", "too_slow"):
            velocity_feedback.append(
                {
                    "joint": joint,
                    "instruction": _velocity_instruction(joint, vel_status),
                    "issue": vel_status,
                }
            )

    # Sort worst-first so the priority fix is the biggest error.
    joint_feedback.sort(key=lambda x: x["diff_degrees"], reverse=True)

    priority_fix = (
        joint_feedback[0]["instruction"] if joint_feedback else "Keep it up!"
    )

    # --- Symmetry warnings ---
    symmetry_warnings: list[str] = []
    sym = current_frame_result.get("symmetry", {})
    for pair_name, data in sym.items():
        if not data.get("balanced", True):
            symmetry_warnings.append(
                _symmetry_warning(pair_name, data["left"], data["right"])
            )

    # Build a headline.
    if grade == "Excellent":
        headline = "Outstanding form! Keep it up."
    elif grade == "Good":
        if joint_feedback:
            worst = joint_feedback[0]["joint"].replace("_", " ")
            headline = f"Great form! Just work on your {worst}."
        else:
            headline = "Looking good overall!"
    elif grade == "Needs Work":
        headline = "Getting there — focus on the tips below."
    else:
        headline = "Let's work on the basics — check the corrections below."

    return {
        "overall_score": round(score),
        "grade": grade,
        "headline": headline,
        "joint_feedback": joint_feedback,
        "positive_feedback": positive_feedback,
        "velocity_feedback": velocity_feedback,
        "priority_fix": priority_fix,
        "symmetry_warnings": symmetry_warnings,
    }


# ---------------------------------------------------------------------------
# Session report
# ---------------------------------------------------------------------------


def generate_session_report(
    summary: dict,
    *,
    exercise: str = "default",
) -> str:
    """Produce a plain-text session report and save to ``output/``.

    Parameters
    ----------
    summary : dict
        Output of ``matcher.compute_summary``.
    exercise : str
        Exercise name used for this session.

    Returns
    -------
    str
        The report text (also written to ``output/report_<timestamp>.txt``).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score = summary.get("overall_score", 0)
    grade = _grade(score)
    best = summary.get("best_joint", "—")
    worst = summary.get("worst_joint", "—")
    per_joint = summary.get("per_joint_avg_error", {})
    best_err = per_joint.get(best, 0)
    worst_err = per_joint.get(worst, 0)

    lines = [
        f"Session Report — {now}",
        f"{'=' * 50}",
        f"Exercise      : {exercise}",
        f"Overall Score : {score:.0f}/100 ({grade})",
        "",
        f"Strongest     : {best.replace('_', ' ')} (avg error {best_err:.0f}°)",
        f"Needs Work    : {worst.replace('_', ' ')} (avg error {worst_err:.0f}°)",
        "",
        "Per-Joint Average Errors:",
    ]

    for joint, err in sorted(per_joint.items(), key=lambda x: x[1]):
        bar = "█" * int(err) + "░" * max(0, 45 - int(err))
        lines.append(f"  {joint:<20s}  {err:5.1f}°  {bar}")

    # Velocity issues
    vel_issues = summary.get("velocity_issues", {})
    if vel_issues:
        lines.append("")
        lines.append("Velocity Issues:")
        for joint, data in vel_issues.items():
            parts = []
            if data.get("too_fast_pct", 0) > 0:
                parts.append(f"too fast {data['too_fast_pct']:.0f}%")
            if data.get("too_slow_pct", 0) > 0:
                parts.append(f"too slow {data['too_slow_pct']:.0f}%")
            lines.append(f"  {joint:<20s}  {', '.join(parts)} of frames")

    # Symmetry summary
    sym_summary = summary.get("symmetry_summary", {})
    if sym_summary:
        lines.append("")
        lines.append("Symmetry Analysis:")
        for pair, data in sym_summary.items():
            ratio = data.get("avg_ratio", 1.0)
            status = "✓ Balanced" if data.get("balanced", True) else "⚠ Imbalanced"
            lines.append(f"  {pair:<20s}  ratio {ratio:.2f}  {status}")

    # Coaching advice for worst joint
    if worst and worst_err > 10:
        lines.append("")
        lines.append(f"💡 Focus tip: {_coaching_instruction(worst, worst_err)}")

    report = "\n".join(lines)

    # Save to output/
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"report_{ts}.txt"
    out_path.write_text(report, encoding="utf-8")

    return report
