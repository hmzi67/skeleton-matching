"""
state_machine.py — Exercise phase detection and rep counting.

Tracks the exercise through phases based on a primary joint angle
(e.g. knee angle for squats):

    READY  →  DOWN  →  HOLD  →  UP  →  READY  (1 rep completed)

Provides:
  - ExerciseStateMachine — main class with rep counting
"""

from __future__ import annotations

from collections import deque


class ExerciseStateMachine:
    """Detect exercise phases and count reps from joint angle data.

    The state machine uses a primary angle (e.g. average knee angle for
    squats) and transitions through phases when the angle crosses
    configurable thresholds.

    Parameters
    ----------
    down_threshold : float
        Angle below which the user is considered to be in the DOWN phase.
        For squats, knee angle < 120° = squatting.
    up_threshold : float
        Angle above which the user is considered standing (READY/UP).
        For squats, knee angle > 155° = standing.
    hold_frames : int
        Number of consecutive frames the user must stay in DOWN before
        transitioning to HOLD.
    """

    PHASES = ("READY", "DOWN", "HOLD", "UP")

    _HYSTERESIS = 5.0

    def __init__(
        self,
        down_threshold: float = 120.0,
        up_threshold: float = 155.0,
        hold_frames: int = 2,
    ) -> None:
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.hold_frames = hold_frames

        self.phase: str = "READY"
        self.rep_count: int = 0
        self.rep_scores: list[float] = []
        self.rep_joint_issues: list[dict[str, int]] = []  # per-rep {joint: bad_frame_count}

        self._down_frame_count: int = 0
        self._current_rep_scores: list[float] = []
        self._current_rep_joint_issues: dict[str, int] = {}
        self._angle_history: deque[float] = deque(maxlen=60)
        self._last_completed_rep: int | None = None  # rep number just completed

    def update(self, primary_angle: float, frame_score: float = 0.0,
               joint_statuses: dict[str, str] | None = None) -> str:
        """Update the state machine with a new angle reading.

        Parameters
        ----------
        primary_angle : float
            The primary joint angle for this exercise (e.g. avg knee angle).
        frame_score : float
            The matching score for this frame (used to compute per-rep score).
        joint_statuses : dict, optional
            {joint_name: "good"|"warning"|"bad"} for the current frame.

        Returns
        -------
        str
            Current phase after the update.
        """
        self._angle_history.append(primary_angle)
        self._current_rep_scores.append(frame_score)
        self._last_completed_rep = None

        # Accumulate joint issues for the current rep.
        if joint_statuses:
            for joint, status in joint_statuses.items():
                if status in ("warning", "bad"):
                    self._current_rep_joint_issues[joint] = (
                        self._current_rep_joint_issues.get(joint, 0) + 1
                    )

        h = self._HYSTERESIS

        if self.phase == "READY":
            if primary_angle < self.down_threshold + h:
                self.phase = "DOWN"
                self._down_frame_count = 1
                self._current_rep_scores = [frame_score]
                self._current_rep_joint_issues = {}

        elif self.phase == "DOWN":
            if primary_angle < self.down_threshold + h:
                self._down_frame_count += 1
                if self._down_frame_count >= self.hold_frames:
                    self.phase = "HOLD"
            elif primary_angle > self.up_threshold:
                self.phase = "READY"
                self._down_frame_count = 0

        elif self.phase == "HOLD":
            if primary_angle > self.up_threshold - h:
                self.phase = "UP"

        elif self.phase == "UP":
            if primary_angle > self.up_threshold - h:
                self.rep_count += 1
                self._last_completed_rep = self.rep_count
                if self._current_rep_scores:
                    avg = sum(self._current_rep_scores) / len(self._current_rep_scores)
                    self.rep_scores.append(round(avg, 1))
                self.rep_joint_issues.append(dict(self._current_rep_joint_issues))
                self._current_rep_scores = []
                self._current_rep_joint_issues = {}
                self.phase = "READY"

        return self.phase

    @property
    def latest_rep_score(self) -> float:
        """Score of the most recently completed rep."""
        return self.rep_scores[-1] if self.rep_scores else 0.0

    @property
    def phase_instruction(self) -> str:
        """A user-friendly instruction for the current phase."""
        return {
            "READY": "Get ready...",
            "DOWN": "Go down!",
            "HOLD": "Hold it!",
            "UP": "Stand up!",
        }.get(self.phase, "")

    def reset(self) -> None:
        """Reset all state."""
        self.phase = "READY"
        self.rep_count = 0
        self.rep_scores = []
        self.rep_joint_issues = []
        self._down_frame_count = 0
        self._current_rep_scores = []
        self._current_rep_joint_issues = {}
        self._angle_history.clear()
        self._last_completed_rep = None
