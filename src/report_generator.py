"""
report_generator.py — Pure data-transformation layer that converts
RepTracker outputs into report dicts ready for the API and frontend.

NO database imports. NO Flask imports.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.exercise_weights import (
    EXERCISE_WEIGHTS,
    compute_auto_weights,
    resolve_exercise_name,
)
from src.rep_tracker import SessionResult, SetResult


# Mirror feedback.py severity tiers.
_SEVERITY_MILD = 10.0      # < 10° → mild
_SEVERITY_MODERATE = 20.0  # < 20° → moderate; >= 20° → critical

_MAX_PROBLEM_JOINTS = 5


class ReportGenerator:
    """Build set/session report payloads from RepTracker results."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_set_report(self, set_result: SetResult, exercise_name: str) -> dict:
        weights = self._weights_for(exercise_name)
        problem_joints = self._build_problem_joints(
            self._mean_joint_errors_from_reps(set_result.reps),
            weights,
        )

        return {
            "report_type":      "set",
            "exercise":         resolve_exercise_name(exercise_name),
            "set_number":       set_result.set_number,
            "grade":            set_result.grade,
            "overall_accuracy": round(set_result.set_accuracy, 4),
            "reps": [
                {"rep": r.rep_number, "accuracy": round(r.accuracy, 4)}
                for r in set_result.reps
            ],
            "sets":             None,
            "problem_joints":   problem_joints,
            "total_reps":       len(set_result.reps),
            "completed":        len(set_result.reps) > 0,
            "generated_at":     self._now_iso(),
        }

    def build_session_report(self, session_result: SessionResult) -> dict:
        weights = self._weights_for(session_result.exercise_name)

        # Aggregate joint errors across every rep in every set.
        all_reps = [r for s in session_result.sets for r in s.reps]
        problem_joints = self._build_problem_joints(
            self._mean_joint_errors_from_reps(all_reps),
            weights,
        )

        # For the session report's "reps" array we flatten all reps with
        # globally-incrementing rep numbers so the frontend can render a
        # single timeline if it wants to.
        flat_reps: list[dict] = []
        global_rep = 0
        for s in session_result.sets:
            for r in s.reps:
                global_rep += 1
                flat_reps.append({"rep": global_rep, "accuracy": round(r.accuracy, 4)})

        return {
            "report_type":      "session",
            "exercise":         resolve_exercise_name(session_result.exercise_name),
            "set_number":       None,
            "grade":            self._grade(session_result.overall_accuracy),
            "overall_accuracy": round(session_result.overall_accuracy, 4),
            "reps":             flat_reps,
            "sets": [
                {
                    "set":      s.set_number,
                    "accuracy": round(s.set_accuracy, 4),
                    "grade":    s.grade,
                }
                for s in session_result.sets
            ],
            "problem_joints":   problem_joints,
            "total_reps":       session_result.total_reps,
            "completed":        session_result.completed,
            "generated_at":     self._now_iso(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grade(self, accuracy: float) -> str:
        if accuracy >= 0.90:
            return "Excellent"
        if accuracy >= 0.75:
            return "Good"
        if accuracy >= 0.60:
            return "Needs Work"
        return "Poor"

    def _severity(self, error_deg: float) -> str:
        if error_deg < _SEVERITY_MILD:
            return "mild"
        if error_deg < _SEVERITY_MODERATE:
            return "moderate"
        return "critical"

    def _label_joint(self, joint_name: str) -> str:
        # "left_wrist_curl" → "Left wrist curl"
        cleaned = joint_name.replace("_", " ").strip()
        if not cleaned:
            return joint_name
        return cleaned[0].upper() + cleaned[1:]

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _weights_for(self, exercise_name: str) -> dict[str, float]:
        """Look up curated weights, fall back to a permissive equal-weight
        dict that keeps every joint visible in problem_joints if the
        exercise is unknown.

        compute_auto_weights() requires a normalised skeleton, which the
        report layer does not have access to. The fallback is therefore an
        equal-weight dict over the joints we have errors for, computed
        lazily by the caller filtering on `> 0`.
        """
        resolved = resolve_exercise_name(exercise_name)
        if resolved in EXERCISE_WEIGHTS:
            return EXERCISE_WEIGHTS[resolved]
        # Sentinel: empty dict means "all joints allowed" — handled below.
        return {}

    def _mean_joint_errors_from_reps(self, reps: list) -> dict[str, float]:
        accum: dict[str, list[float]] = {}
        for r in reps:
            for j, e in r.joint_error_summary.items():
                accum.setdefault(j, []).append(float(e))
        return {j: sum(vs) / len(vs) for j, vs in accum.items() if vs}

    def _build_problem_joints(
        self,
        joint_errors: dict[str, float],
        weights: dict[str, float],
    ) -> list[dict]:
        # Filter joints by weight > 0. Empty weights dict = allow all.
        if weights:
            filtered = {
                j: e for j, e in joint_errors.items()
                if weights.get(j, 0.0) > 0.0
            }
        else:
            filtered = dict(joint_errors)

        ranked = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)
        out: list[dict] = []
        for joint, mean_err in ranked[:_MAX_PROBLEM_JOINTS]:
            error_pct = min((mean_err / 90.0) * 100.0, 100.0)
            out.append({
                "joint":           joint,
                "label":           self._label_joint(joint),
                "mean_error_deg":  round(mean_err, 2),
                "error_pct":       round(error_pct, 1),
                "severity":        self._severity(mean_err),
            })
        return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from src.rep_tracker import RepResult, SetResult, SessionResult

    fake_reps = [
        RepResult(
            rep_number=i,
            accuracy=0.70 + 0.03 * i,
            joint_error_summary={"left_knee": 8.0 + i, "right_knee": 12.0, "torso_lean": 4.0},
            joint_accuracy={"left_knee": 0.85, "right_knee": 0.78, "torso_lean": 0.92},
            frame_count=20,
            duration_ms=1500.0,
        )
        for i in range(1, 11)
    ]
    fake_set = SetResult(
        set_number=1, reps=fake_reps, set_accuracy=0.82,
        joint_problem_ranking=[], grade="Good",
    )
    fake_session = SessionResult(
        exercise_name="squat", sets=[fake_set], overall_accuracy=0.82,
        joint_problem_ranking=[], total_reps=10, completed=False,
    )

    rg = ReportGenerator()
    import json
    print("--- Set report ---")
    print(json.dumps(rg.build_set_report(fake_set, "squat"), indent=2))
    print("\n--- Session report ---")
    print(json.dumps(rg.build_session_report(fake_session), indent=2))
