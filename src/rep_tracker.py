"""
rep_tracker.py — Pure-Python aggregation of per-frame scores into reps,
sets, and sessions.

NO database imports. NO Flask imports. Safe to use from CLI (`main.py`)
and from the Flask backend (`app.py`) alike.

Domain model:
    1 ExerciseSession
    └── N Sets        (default 3)
        └── N Reps    (default 10 per set)
            └── N Frames (every processed webcam frame)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.exercise_weights import (
    ALL_JOINT_NAMES,
    EXERCISE_WEIGHTS,
    compute_auto_weights,
    resolve_exercise_name,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FrameScore:
    frame_index: int
    joint_errors: dict[str, float]   # joint_name -> degrees
    joint_scores: dict[str, float]   # joint_name -> 0.0–1.0
    overall_score: float             # 0.0–1.0 (or 0–100 — RepTracker normalises)
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000.0)


@dataclass
class RepResult:
    rep_number: int
    accuracy: float                          # mean overall_score across frames (0.0–1.0)
    joint_error_summary: dict[str, float]    # mean abs error per joint (degrees)
    joint_accuracy: dict[str, float]         # mean per-joint score (0.0–1.0)
    frame_count: int
    duration_ms: float


@dataclass
class SetResult:
    set_number: int
    reps: list[RepResult]
    set_accuracy: float                      # mean of rep accuracies (0.0–1.0)
    joint_problem_ranking: list[dict]        # sorted worst-first
    grade: str


@dataclass
class SessionResult:
    exercise_name: str
    sets: list[SetResult]
    overall_accuracy: float                  # mean of set accuracies (0.0–1.0)
    joint_problem_ranking: list[dict]
    total_reps: int
    completed: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grade(accuracy: float) -> str:
    """Mirror the thresholds in report_generator._grade."""
    if accuracy >= 0.90:
        return "Excellent"
    if accuracy >= 0.75:
        return "Good"
    if accuracy >= 0.60:
        return "Needs Work"
    return "Poor"


def _normalise_overall(score: float) -> float:
    """Accept either 0.0–1.0 or 0–100 inputs and return 0.0–1.0."""
    return score / 100.0 if score > 1.0 else score


def _resolve_weights(exercise_name: str, observed_joints: set[str] | None = None) -> dict[str, float]:
    """Resolve weights via the curated registry, falling back to auto-weights.

    `observed_joints` is unused at this layer — auto fallback requires a
    normalised skeleton which the tracker does not have. We return an
    EQUAL-weight dict over ALL_JOINT_NAMES as a final fallback so that
    `joint > 0.0` filtering still keeps every joint visible.
    """
    resolved = resolve_exercise_name(exercise_name)
    if resolved in EXERCISE_WEIGHTS:
        return EXERCISE_WEIGHTS[resolved]
    return {j: 1.0 / len(ALL_JOINT_NAMES) for j in ALL_JOINT_NAMES}


def _build_joint_problem_ranking(
    joint_error_means: dict[str, float],
    weights: dict[str, float],
    top_n: int = 5,
) -> list[dict]:
    """Sort joints by mean error desc, filter to weighted joints only, top N."""
    items = []
    for joint, mean_err in joint_error_means.items():
        if weights.get(joint, 0.0) <= 0.0:
            continue
        items.append({"joint": joint, "mean_error_deg": round(mean_err, 2)})
    items.sort(key=lambda x: x["mean_error_deg"], reverse=True)
    return items[:top_n]


# ---------------------------------------------------------------------------
# RepTracker
# ---------------------------------------------------------------------------


class RepTracker:
    """In-memory aggregator of frame -> rep -> set -> session results."""

    def __init__(
        self,
        exercise_name: str,
        reps_per_set: int = 10,
        total_sets: int = 3,
    ) -> None:
        self.exercise_name = exercise_name
        self.reps_per_set = reps_per_set
        self.total_sets = total_sets

        self._weights = _resolve_weights(exercise_name)

        self.current_set_number: int = 1
        self.current_rep_number: int = 1

        self._current_rep_frames: list[FrameScore] = []
        self._rep_start_ms: float | None = None

        self._completed_reps: list[RepResult] = []  # reset per set
        self._completed_sets: list[SetResult] = []

    # ------------------------------------------------------------------
    # Frame ingest
    # ------------------------------------------------------------------

    def on_frame(self, frame_score: FrameScore) -> None:
        if self._rep_start_ms is None:
            self._rep_start_ms = frame_score.timestamp_ms
        self._current_rep_frames.append(frame_score)

    def _build_set_result(self, reps: list[RepResult], set_number: int) -> SetResult:
        if reps:
            set_accuracy = sum(r.accuracy for r in reps) / len(reps)
        else:
            set_accuracy = 0.0

        # Aggregate joint errors across reps (mean of rep means).
        joint_err_accum: dict[str, list[float]] = {}
        for r in reps:
            for j, e in r.joint_error_summary.items():
                joint_err_accum.setdefault(j, []).append(e)
        joint_error_means = {
            j: sum(vs) / len(vs) for j, vs in joint_err_accum.items() if vs
        }
        ranking = _build_joint_problem_ranking(joint_error_means, self._weights)

        return SetResult(
            set_number=set_number,
            reps=reps,
            set_accuracy=round(set_accuracy, 4),
            joint_problem_ranking=ranking,
            grade=_grade(set_accuracy),
        )

    def current_partial_set_result(self) -> SetResult | None:
        """Return the in-progress set using only completed reps, if any."""
        if not self._completed_reps:
            return None
        return self._build_set_result(list(self._completed_reps), self.current_set_number)

    # ------------------------------------------------------------------
    # Rep aggregation
    # ------------------------------------------------------------------

    def on_rep_complete(self) -> RepResult:
        frames = self._current_rep_frames
        n = len(frames)

        if n == 0:
            rep = RepResult(
                rep_number=self.current_rep_number,
                accuracy=0.0,
                joint_error_summary={},
                joint_accuracy={},
                frame_count=0,
                duration_ms=0.0,
            )
        else:
            mean_overall = sum(_normalise_overall(f.overall_score) for f in frames) / n

            joint_err_sum: dict[str, float] = {}
            joint_err_count: dict[str, int] = {}
            joint_score_sum: dict[str, float] = {}
            joint_score_count: dict[str, int] = {}

            for f in frames:
                for j, e in f.joint_errors.items():
                    joint_err_sum[j] = joint_err_sum.get(j, 0.0) + float(e)
                    joint_err_count[j] = joint_err_count.get(j, 0) + 1
                for j, s in f.joint_scores.items():
                    joint_score_sum[j] = joint_score_sum.get(j, 0.0) + float(s)
                    joint_score_count[j] = joint_score_count.get(j, 0) + 1

            joint_error_summary = {
                j: round(joint_err_sum[j] / joint_err_count[j], 2)
                for j in joint_err_sum
            }
            joint_accuracy = {
                j: round(joint_score_sum[j] / joint_score_count[j], 4)
                for j in joint_score_sum
            }

            duration_ms = max(0.0, frames[-1].timestamp_ms - (self._rep_start_ms or frames[0].timestamp_ms))

            rep = RepResult(
                rep_number=self.current_rep_number,
                accuracy=round(mean_overall, 4),
                joint_error_summary=joint_error_summary,
                joint_accuracy=joint_accuracy,
                frame_count=n,
                duration_ms=round(duration_ms, 1),
            )

        self._completed_reps.append(rep)
        self._current_rep_frames = []
        self._rep_start_ms = None
        self.current_rep_number += 1
        return rep

    # ------------------------------------------------------------------
    # Set aggregation
    # ------------------------------------------------------------------

    def on_set_complete(self) -> SetResult:
        result = self._build_set_result(list(self._completed_reps), self.current_set_number)
        self._completed_sets.append(result)

        # Reset for next set.
        self._completed_reps = []
        self.current_set_number += 1
        self.current_rep_number = 1
        return result

    # ------------------------------------------------------------------
    # Session finalisation
    # ------------------------------------------------------------------

    def finalize_session(self) -> SessionResult:
        sets = list(self._completed_sets)

        partial_set = self.current_partial_set_result()
        if partial_set is not None:
            sets.append(partial_set)

        if sets:
            overall_accuracy = sum(s.set_accuracy for s in sets) / len(sets)
        else:
            overall_accuracy = 0.0

        # Aggregate joint errors across sets (mean of set means).
        joint_err_accum: dict[str, list[float]] = {}
        for s in sets:
            # Recompute set's joint means from its reps to avoid losing fidelity.
            joint_in_set: dict[str, list[float]] = {}
            for r in s.reps:
                for j, e in r.joint_error_summary.items():
                    joint_in_set.setdefault(j, []).append(e)
            for j, vs in joint_in_set.items():
                if vs:
                    joint_err_accum.setdefault(j, []).append(sum(vs) / len(vs))

        joint_error_means = {
            j: sum(vs) / len(vs) for j, vs in joint_err_accum.items() if vs
        }
        ranking = _build_joint_problem_ranking(joint_error_means, self._weights)

        total_reps = sum(len(s.reps) for s in sets)
        completed = len(sets) == self.total_sets

        return SessionResult(
            exercise_name=self.exercise_name,
            sets=sets,
            overall_accuracy=round(overall_accuracy, 4),
            joint_problem_ranking=ranking,
            total_reps=total_reps,
            completed=completed,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import random

    random.seed(42)
    tracker = RepTracker("squat", reps_per_set=3, total_sets=2)

    for set_idx in range(2):
        for rep_idx in range(3):
            for fi in range(5):
                tracker.on_frame(FrameScore(
                    frame_index=fi,
                    joint_errors={"left_knee": random.uniform(2, 18),
                                  "right_knee": random.uniform(2, 18),
                                  "torso_lean": random.uniform(1, 8)},
                    joint_scores={"left_knee": random.uniform(0.6, 1.0),
                                  "right_knee": random.uniform(0.6, 1.0),
                                  "torso_lean": random.uniform(0.7, 1.0)},
                    overall_score=random.uniform(0.65, 0.95),
                    timestamp_ms=fi * 50.0,
                ))
            rep = tracker.on_rep_complete()
            print(f"Set {tracker.current_set_number} Rep {rep.rep_number}: acc={rep.accuracy}")
        sr = tracker.on_set_complete()
        print(f"  Set {sr.set_number} accuracy={sr.set_accuracy} grade={sr.grade}")
        print(f"  Top joint problems: {sr.joint_problem_ranking}")

    session = tracker.finalize_session()
    print(f"\nSession: acc={session.overall_accuracy} reps={session.total_reps} completed={session.completed}")
    print(f"Joint problems: {session.joint_problem_ranking}")
