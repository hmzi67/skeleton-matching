"""
feedback.py — High-quality, actionable coaching feedback engine.

Design principles:
  1. SEVERITY-TIERED — mild / moderate / critical language matches the error magnitude.
  2. EXERCISE-AWARE  — instructions reference the exercise context when provided.
  3. PHASE-AWARE     — coaching adapts to the current movement phase
                       (DOWN, HOLD, UP, READY).
  4. PRIORITISED     — at most 2 corrections shown per frame to avoid overloading
                       the user; the worst joint is always shown first.
  5. BILATERAL MERGE — if both left and right versions of the same joint are
                       equally wrong, a single unified cue is given instead of two.
  6. HAND-SPECIFIC   — finger curl errors produce finger-level instructions.
  7. VELOCITY AS     — speed issues are reported as coaching hints but never
     COACHING HINT     affect the score.

Provides:
  - generate_feedback()        — per-frame coaching dict
  - generate_session_report()  — plain-text summary saved after each run
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Joints with weight below this threshold are never shown in feedback/reports.
_FEEDBACK_WEIGHT_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Needs Work"
    return "Poor"


# ---------------------------------------------------------------------------
# Severity levels based on angle error magnitude
# ---------------------------------------------------------------------------

def _severity(abs_diff: float) -> str:
    """Return 'mild', 'moderate', or 'critical' based on angular error."""
    if abs_diff < 15:
        return "mild"
    if abs_diff < 30:
        return "moderate"
    return "critical"


# ---------------------------------------------------------------------------
# Coaching instruction library
#
# Convention for `diff`:  diff = user_angle − gt_angle
#   diff > 0 → user joint angle is LARGER than reference
#              (less bent for knees/elbows/hips, arm too high for shoulder)
#   diff < 0 → user joint angle is SMALLER than reference
#              (more bent than needed, arm too low for shoulder)
# ---------------------------------------------------------------------------

def _coaching_instruction(
    joint: str,
    diff: float,
    *,
    exercise: str = "",
    phase: str = "",
) -> str:
    """Return a specific, actionable coaching cue.

    Parameters
    ----------
    joint : str
        Joint name (e.g. ``"left_knee"``).
    diff : float
        ``user_angle − gt_angle`` in degrees.
    exercise : str, optional
        Exercise context (e.g. ``"squat"``).
    phase : str, optional
        Movement phase: ``"DOWN"``, ``"HOLD"``, ``"UP"``, ``"READY"``, or ``""``.
    """
    severity = _severity(abs(diff))
    ex = exercise.lower()
    ph = phase.upper()

    # Resolve side label ("left", "right", or "both" for bilateral joints).
    if joint.startswith("left_"):
        side = "left"
        bare = joint[5:]   # strip "left_"
    elif joint.startswith("right_"):
        side = "right"
        bare = joint[6:]   # strip "right_"
    else:
        side = ""
        bare = joint

    side_label = f"your {side} " if side else "your "

    # ------------------------------------------------------------------ KNEE
    if bare == "knee":
        if diff > 0:  # user's knee is more straight → needs to bend more
            msgs = {
                "mild":     f"Bend {side_label}knee a little more",
                "moderate": f"Go deeper — bend {side_label}knee further",
                "critical": f"Much deeper needed — drop {side_label}knee significantly",
            }
            if ph == "DOWN":
                msgs["moderate"] = f"Keep sinking — drive {side_label}knee down"
            if ph == "UP":
                msgs = {
                    "mild":     f"Finish the rep — don't straighten {side_label}knee yet",
                    "moderate": f"Hold the descent — {side_label}knee is not low enough",
                    "critical": f"You're coming up too early — drop {side_label}knee more",
                }
        else:         # user's knee is more bent → needs to come up
            msgs = {
                "mild":     f"Slightly straighten {side_label}knee",
                "moderate": f"Rise up more — {side_label}knee is over-bent",
                "critical": f"Stand up fully — {side_label}knee is too deep",
            }
            if ph == "UP":
                msgs["moderate"] = f"Push up harder — {side_label}knee should be straighter now"
        return msgs[severity]

    # ------------------------------------------------------------------- HIP
    if bare == "hip":
        if diff > 0:  # hip angle too large → needs more hip flexion
            msgs = {
                "mild":     f"Push {side_label}hip back slightly",
                "moderate": f"Hinge from {side_label}hip — send it further back",
                "critical": f"Drive {side_label}hip back aggressively; you're standing too upright",
            }
            if ex in ("deadlift",):
                msgs["moderate"] = f"Hinge deeper at {side_label}hip — chest forward, hips back"
        else:         # hip angle too small → needs hip extension
            msgs = {
                "mild":     f"Drive {side_label}hip forward slightly",
                "moderate": f"Extend {side_label}hip — squeeze your glute at the top",
                "critical": f"Lock out {side_label}hip fully — you're not completing the rep",
            }
        return msgs[severity]

    # ----------------------------------------------------------------- ELBOW
    if bare == "elbow":
        if diff > 0:  # elbow too straight → needs more flexion
            msgs = {
                "mild":     f"Bend {side_label}arm a little more",
                "moderate": f"Curl {side_label}arm further — elbow should be more bent",
                "critical": f"Much more elbow bend required — {side_label}arm is too straight",
            }
            if ex == "pushup":
                msgs = {
                    "mild":     "Lower your chest a bit more",
                    "moderate": "Go lower — your elbows should bend further",
                    "critical": "You need a much deeper push-up — bend both elbows fully",
                }
        else:         # elbow too bent → needs extension
            msgs = {
                "mild":     f"Extend {side_label}arm slightly",
                "moderate": f"Straighten {side_label}arm more — elbow is over-bent",
                "critical": f"Fully extend {side_label}arm — don't lock it, but open it much more",
            }
            if ex == "pushup":
                msgs = {
                    "mild":     "Push up a bit more",
                    "moderate": "Lock out your arms at the top",
                    "critical": "Fully extend your arms — you're not completing the push-up",
                }
        return msgs[severity]

    # -------------------------------------------------------------- SHOULDER
    if bare == "shoulder":
        if diff > 0:  # arm too high
            msgs = {
                "mild":     f"Lower {side_label}arm slightly",
                "moderate": f"Bring {side_label}arm down — it's raised too high",
                "critical": f"{side_label.capitalize()}arm is far too high — lower it significantly",
            }
        else:         # arm too low
            msgs = {
                "mild":     f"Raise {side_label}arm a little higher",
                "moderate": f"Lift {side_label}arm more — it's not reaching the target height",
                "critical": f"Raise {side_label}arm much higher — it's far below where it should be",
            }
            if ex == "shoulder_press":
                msgs["moderate"] = f"Press {side_label}arm fully overhead — it needs to go higher"
        return msgs[severity]

    # -------------------------------------------------------------- TORSO LEAN
    if joint == "torso_lean":
        if diff > 0:  # user leaning more than reference
            msgs = {
                "mild":     "Stand slightly more upright",
                "moderate": "Straighten your torso — you're leaning too far forward",
                "critical": "You're hunching over — lift your chest and stand tall",
            }
            if ex in ("squat", "lunge"):
                msgs["moderate"] = "Chest up — your torso is leaning too far forward"
        else:         # user not leaning enough
            msgs = {
                "mild":     "Lean your torso forward slightly",
                "moderate": "Hinge forward more with your upper body",
                "critical": "Much more forward lean needed — follow the reference posture",
            }
            if ex == "deadlift":
                msgs["moderate"] = "Lean into it — keep your chest above the bar angle"
        return msgs[severity]

    # --------------------------------------------------------- HAND / FINGER CURL
    # joint pattern: "left_hand_index_curl", "right_hand_thumb_curl", etc.
    if "hand" in joint and "curl" in joint:
        parts = joint.split("_")  # ["left", "hand", "index", "curl"]
        hand_side = parts[0] if parts[0] in ("left", "right") else ""
        finger = parts[2] if len(parts) >= 3 else "finger"

        hand_label = f"your {hand_side} " if hand_side else "your "

        if diff > 0:  # more extended than GT → needs to curl more
            msgs = {
                "mild":     f"Curl {hand_label}{finger} slightly more",
                "moderate": f"Bend {hand_label}{finger} further — curl it more",
                "critical": f"Much more curl needed — {hand_label}{finger} is far too straight",
            }
        else:         # more curled than GT → needs to extend
            msgs = {
                "mild":     f"Open {hand_label}{finger} a little",
                "moderate": f"Extend {hand_label}{finger} more — it's too curled",
                "critical": f"Straighten {hand_label}{finger} significantly — it's over-curled",
            }
        return msgs[severity]

    # ------------------------------------------------- GENERIC FALLBACK
    abs_d = abs(diff)
    return f"Adjust {joint.replace('_', ' ')} by {abs_d:.0f}°"


def _velocity_coaching(joint: str, velocity_status: str) -> str:
    """Return a speed coaching hint (not an error — purely informational)."""
    name = joint.replace("_", " ")
    if velocity_status == "too_fast":
        return f"Slow down — you're moving your {name} faster than the reference"
    if velocity_status == "too_slow":
        return f"Pick up the pace on your {name} to match the reference tempo"
    return ""


def _symmetry_cue(pair_name: str, left: float, right: float) -> str:
    """Generate a symmetry correction cue."""
    diff = abs(left - right)
    dominant = "left" if left > right else "right"
    weaker   = "right" if left > right else "left"
    return (
        f"Your {dominant} {pair_name} is {diff:.0f}° more bent than your {weaker} — "
        f"keep both sides even"
    )


# ---------------------------------------------------------------------------
# Bilateral merge: consolidate left+right cues for the same joint
# ---------------------------------------------------------------------------

def _try_bilateral_merge(
    joint_feedback: list[dict],
    exercise: str = "",
    phase: str = "",
) -> list[dict]:
    """Merge left/right cues for the same bare joint when both are flagged
    with the same direction, producing a single bilateral instruction.

    Returns a new list; the original is not modified.
    """
    from collections import defaultdict

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in joint_feedback:
        joint = item["joint"]
        if joint.startswith("left_"):
            bare = joint[5:]
        elif joint.startswith("right_"):
            bare = joint[6:]
        else:
            bare = joint
        grouped[bare].append(item)

    merged: list[dict] = []
    processed: set[str] = set()

    for item in joint_feedback:
        joint = item["joint"]
        if joint in processed:
            continue

        if joint.startswith("left_"):
            bare = joint[5:]
        elif joint.startswith("right_"):
            bare = joint[6:]
        else:
            bare = joint

        # Only merge simple pose joints (knee, hip, elbow, shoulder).
        # Hand joints carry too much side-specificity to merge cleanly.
        is_pose_joint = bare in ("knee", "hip", "elbow", "shoulder")

        pair = grouped[bare]
        if len(pair) == 2 and is_pose_joint:
            left_item  = next((x for x in pair if x["joint"].startswith("left_")),  None)
            right_item = next((x for x in pair if x["joint"].startswith("right_")), None)

            if left_item and right_item:
                # Merge if both joints have the same correction direction.
                l_diff = left_item["diff"]
                r_diff = right_item["diff"]
                same_direction = (l_diff > 0 and r_diff > 0) or (l_diff < 0 and r_diff < 0)

                if same_direction:
                    avg_diff  = (l_diff + r_diff) / 2.0
                    avg_abs   = (left_item["diff_degrees"] + right_item["diff_degrees"]) / 2.0
                    sev       = _severity(avg_abs)

                    bilateral_joint = f"both_{bare}s"

                    # Build instruction then make it bilateral.
                    instruction = _coaching_instruction(
                        f"left_{bare}", avg_diff, exercise=exercise, phase=phase
                    )
                    # "your left knee" → "both knees", "your left hip" → "both hips"
                    instruction = instruction.replace(f"your left {bare}", f"both {bare}s")
                    instruction = instruction.replace(f"Your left {bare}", f"Both {bare}s")

                    merged.append({
                        "joint":        bilateral_joint,
                        "instruction":  instruction,
                        "severity":     sev,
                        "diff_degrees": avg_abs,
                        "diff":         avg_diff,
                    })
                    processed.add(left_item["joint"])
                    processed.add(right_item["joint"])
                    continue

        # Not merged — add as-is.
        processed.add(joint)
        merged.append(item)

    return merged


# ---------------------------------------------------------------------------
# Per-frame feedback
# ---------------------------------------------------------------------------


def generate_feedback(
    summary: dict,
    current_frame_result: dict,
    *,
    exercise: str = "",
    phase: str = "",
    exercise_weights: dict[str, float] | None = None,
) -> dict:
    """Generate coaching feedback for the current frame.

    Parameters
    ----------
    summary : dict
        Output of ``matcher.compute_summary`` (used for overall context).
    current_frame_result : dict
        Output of ``matcher.match_single_frame``.
    exercise : str, optional
        Exercise name for context-aware instructions (e.g. ``"squat"``).
    phase : str, optional
        Current movement phase: ``"DOWN"``, ``"HOLD"``, ``"UP"``, or ``"READY"``.

    Returns
    -------
    dict
        Keys: ``overall_score``, ``grade``, ``headline``,
        ``joint_feedback`` (list, max 2 items), ``velocity_feedback`` (list),
        ``priority_fix`` (str), ``symmetry_warnings`` (list).
    """
    score = current_frame_result.get("overall_score", 0)
    grade = _grade(score)

    raw_joint_feedback: list[dict] = []
    velocity_feedback: list[dict] = []

    for joint, data in current_frame_result.items():
        if not isinstance(data, dict) or "status" not in data:
            continue
        if joint in ("symmetry",):
            continue
        # Skip joints that are not meaningful for this exercise.
        if exercise_weights is not None and exercise_weights.get(joint, 0.0) < _FEEDBACK_WEIGHT_THRESHOLD:
            continue

        status   = data.get("status", "good")
        diff     = data.get("diff", 0.0)
        abs_diff = data.get("abs_diff", 0.0)

        if status != "good" and abs_diff > 5.0:
            raw_joint_feedback.append(
                {
                    "joint":       joint,
                    "instruction": _coaching_instruction(
                        joint, diff, exercise=exercise, phase=phase
                    ),
                    "severity":    _severity(abs_diff),
                    "diff_degrees": abs_diff,
                    "diff":        diff,
                }
            )

        # Velocity hints (informational).
        vel_status = data.get("velocity_status", "ok")
        if vel_status in ("too_fast", "too_slow"):
            hint = _velocity_coaching(joint, vel_status)
            if hint:
                velocity_feedback.append(
                    {"joint": joint, "instruction": hint, "issue": vel_status}
                )

    # Sort worst-first.
    raw_joint_feedback.sort(key=lambda x: x["diff_degrees"], reverse=True)

    # Try bilateral merge (e.g. left+right knee → "both knees").
    merged_feedback = _try_bilateral_merge(raw_joint_feedback, exercise=exercise, phase=phase)

    # Cap at 2 corrections to avoid cognitive overload.
    joint_feedback = merged_feedback[:2]

    # Priority fix = most important single correction.
    if phase == "HOLD":
        priority_fix = "Hold steady — maintain your position"
    elif joint_feedback:
        priority_fix = joint_feedback[0]["instruction"]
    elif score >= 90:
        priority_fix = "Excellent — keep it up!"
    elif score >= 75:
        priority_fix = "Looking good — stay focused"
    else:
        priority_fix = "Follow the reference movement closely"

    # Symmetry warnings.
    symmetry_warnings: list[str] = []
    sym = current_frame_result.get("symmetry", {})
    for pair_name, sym_data in sym.items():
        if not sym_data.get("balanced", True):
            cue = _symmetry_cue(
                pair_name, sym_data["left"], sym_data["right"]
            )
            symmetry_warnings.append(cue)

    # Headline generation.
    ph_label = {"DOWN": "going down", "HOLD": "holding", "UP": "coming up"}.get(phase, "")

    if grade == "Excellent":
        headline = f"Perfect form{' — ' + ph_label if ph_label else ''}! Keep it up."
    elif grade == "Good":
        if joint_feedback:
            worst = joint_feedback[0]["joint"].replace("_", " ")
            headline = f"Great form — just work on your {worst}."
        else:
            headline = "Looking good overall!"
    elif grade == "Needs Work":
        if joint_feedback:
            worst = joint_feedback[0]["joint"].replace("_", " ")
            headline = f"Focus on your {worst} — check the tip below."
        else:
            headline = "Getting there — keep following the reference."
    else:
        if joint_feedback:
            worst = joint_feedback[0]["joint"].replace("_", " ")
            headline = f"Work on your {worst} first."
        else:
            headline = "Match the reference posture closely."

    return {
        "overall_score":     round(score),
        "grade":             grade,
        "headline":          headline,
        "joint_feedback":    joint_feedback,
        "velocity_feedback": velocity_feedback[:2],   # limit to top 2 speed hints
        "priority_fix":      priority_fix,
        "symmetry_warnings": symmetry_warnings,
    }


# ---------------------------------------------------------------------------
# Session report
# ---------------------------------------------------------------------------


def generate_session_report(
    summary: dict,
    *,
    exercise: str = "default",
    exercise_weights: dict[str, float] | None = None,
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
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score = summary.get("overall_score", 0)
    grade = _grade(score)
    per_joint  = summary.get("per_joint_avg_error", {})

    # Filter to only joints relevant to this exercise.
    if exercise_weights is not None:
        per_joint = {
            j: e for j, e in per_joint.items()
            if exercise_weights.get(j, 0.0) >= _FEEDBACK_WEIGHT_THRESHOLD
        }

    best  = min(per_joint, key=per_joint.get) if per_joint else "—"  # type: ignore[arg-type]
    worst = max(per_joint, key=per_joint.get) if per_joint else "—"  # type: ignore[arg-type]
    best_err  = per_joint.get(best,  0) if isinstance(best, str) and best != "—" else 0
    worst_err = per_joint.get(worst, 0) if isinstance(worst, str) and worst != "—" else 0

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
        lines.append(f"  {joint:<28s}  {err:5.1f}°  {bar}")

    # Velocity issues summary.
    vel_issues = summary.get("velocity_issues", {})
    if vel_issues:
        lines += ["", "Speed Issues:"]
        for joint, data in vel_issues.items():
            parts = []
            if data.get("too_fast_pct", 0) > 0:
                parts.append(f"too fast {data['too_fast_pct']:.0f}%")
            if data.get("too_slow_pct", 0) > 0:
                parts.append(f"too slow {data['too_slow_pct']:.0f}%")
            if parts:
                lines.append(f"  {joint:<28s}  {', '.join(parts)} of frames")

    # Symmetry summary.
    sym_summary = summary.get("symmetry_summary", {})
    if sym_summary:
        lines += ["", "Symmetry Analysis:"]
        for pair, data in sym_summary.items():
            ratio  = data.get("avg_ratio", 1.0)
            status = "✓ Balanced" if data.get("balanced", True) else "⚠ Imbalanced"
            lines.append(f"  {pair:<28s}  ratio {ratio:.2f}  {status}")

    # Coaching focus tip for the worst joint.
    if worst and worst_err > 10:
        tip = _coaching_instruction(worst, worst_err, exercise=exercise)
        lines += ["", f"💡 Focus tip: {tip}"]

    report = "\n".join(lines)

    out_dir  = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"report_{ts}.txt"
    out_path.write_text(report, encoding="utf-8")

    return report
