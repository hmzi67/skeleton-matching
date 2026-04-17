"""
stability.py — Real-time feedback stabilizers for pose coaching.

Includes:
  - Temporal smoothing (sliding window)
  - Exponential moving average (EMA)
  - Threshold buffering (hysteresis)
  - Majority voting
  - Debounced, event-gated feedback output
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import logging

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Angle smoothing (sliding window + EMA)
# ---------------------------------------------------------------------------


@dataclass
class AngleSmoother:
    """Smooth joint angles with a sliding window mean + EMA.

    This reduces per-frame jitter while keeping latency low (small window).
    """

    window_size: int = 7
    ema_alpha: float = 0.3
    debug: bool = False
    log_every: int = 30

    _history: dict[str, deque[float]] = field(default_factory=dict)
    _ema: dict[str, float] = field(default_factory=dict)
    _frame_count: int = 0

    def update(self, angles: dict[str, float]) -> dict[str, float]:
        """Return a smoothed copy of the input angle dict."""
        smoothed: dict[str, float] = {}
        for joint, val in angles.items():
            hist = self._history.setdefault(joint, deque(maxlen=self.window_size))
            hist.append(float(val))
            window_avg = sum(hist) / len(hist)

            prev = self._ema.get(joint, window_avg)
            ema_val = (self.ema_alpha * window_avg) + ((1.0 - self.ema_alpha) * prev)
            self._ema[joint] = ema_val
            smoothed[joint] = ema_val

        if self.debug:
            self._frame_count += 1
            if self._frame_count % max(1, self.log_every) == 0 and smoothed:
                sample_joint = next(iter(smoothed.keys()))
                _LOGGER.debug(
                    "Angle smoothing sample %s: raw=%.2f smoothed=%.2f",
                    sample_joint,
                    angles.get(sample_joint, 0.0),
                    smoothed.get(sample_joint, 0.0),
                )

        return smoothed

    def reset(self) -> None:
        self._history.clear()
        self._ema.clear()
        self._frame_count = 0


# ---------------------------------------------------------------------------
# Threshold buffering (hysteresis) + majority voting for joint statuses
# ---------------------------------------------------------------------------


@dataclass
class JointStatusStabilizer:
    """Stabilize per-joint good/warning/bad status with hysteresis + voting."""

    hysteresis_buffer: float = 5.0
    majority_window: int = 7
    default_good_threshold: float = 10.0
    default_warning_threshold: float = 25.0

    _prev_status: dict[str, str] = field(default_factory=dict)
    _history: dict[str, deque[str]] = field(default_factory=dict)

    def reset(self) -> None:
        self._prev_status.clear()
        self._history.clear()

    def _apply_hysteresis(
        self,
        joint: str,
        abs_diff: float,
        good_th: float,
        warn_th: float,
        raw_status: str,
    ) -> str:
        """Keep prior state unless the value crosses buffered thresholds."""
        prev = self._prev_status.get(joint)
        buf = self.hysteresis_buffer

        if prev is None:
            return raw_status

        if prev == "good":
            if abs_diff > good_th + buf:
                return "warning" if abs_diff <= warn_th else "bad"
            return "good"

        if prev == "bad":
            if abs_diff < warn_th - buf:
                return "warning" if abs_diff >= good_th else "good"
            return "bad"

        # prev == "warning"
        if abs_diff < good_th - buf:
            return "good"
        if abs_diff > warn_th + buf:
            return "bad"
        return "warning"

    def stabilize_match(
        self,
        match_result: dict,
        *,
        adaptive_thresholds: dict[str, dict[str, float]] | None = None,
    ) -> dict:
        """Apply hysteresis + majority vote to each joint status in-place."""
        for joint, data in match_result.items():
            if not isinstance(data, dict) or "status" not in data:
                continue

            abs_diff = float(data.get("abs_diff", 0.0))
            if adaptive_thresholds and joint in adaptive_thresholds:
                good_th = float(adaptive_thresholds[joint]["good"])
                warn_th = float(adaptive_thresholds[joint]["warning"])
            else:
                good_th = self.default_good_threshold
                warn_th = self.default_warning_threshold

            raw_status = data.get("status", "good")
            hyst_status = self._apply_hysteresis(joint, abs_diff, good_th, warn_th, raw_status)

            # Majority voting across recent frames for stability.
            history = self._history.setdefault(joint, deque(maxlen=self.majority_window))
            history.append(hyst_status)
            majority = Counter(history).most_common(1)[0][0]

            data["status"] = majority
            self._prev_status[joint] = majority

        return match_result


# ---------------------------------------------------------------------------
# Debounced feedback + event gating for speech
# ---------------------------------------------------------------------------


@dataclass
class FeedbackDebouncer:
    """Debounce feedback text updates across consecutive frames."""

    debounce_frames: int = 7

    _stable_text: str = ""
    _pending_text: str = ""
    _pending_count: int = 0

    @property
    def stable_text(self) -> str:
        return self._stable_text

    def reset(self) -> None:
        self._stable_text = ""
        self._pending_text = ""
        self._pending_count = 0

    def update(self, new_text: str, *, force: bool = False) -> tuple[str, bool]:
        if force:
            changed = new_text != self._stable_text
            self._stable_text = new_text
            self._pending_text = new_text
            self._pending_count = 0
            return self._stable_text, changed

        if new_text == self._stable_text:
            self._pending_text = ""
            self._pending_count = 0
            return self._stable_text, False

        if new_text == self._pending_text:
            self._pending_count += 1
        else:
            self._pending_text = new_text
            self._pending_count = 1

        if self._pending_count >= self.debounce_frames:
            self._stable_text = new_text
            self._pending_text = ""
            self._pending_count = 0
            return self._stable_text, True

        return self._stable_text, False


@dataclass
class FeedbackStabilizer:
    """Gate feedback to stable phases and emit speech only on stable changes."""

    debounce_frames: int = 7
    phase_stable_frames: int = 5
    speech_cooldown_ms: int = 2500

    _debouncer: FeedbackDebouncer = field(init=False)
    _last_phase: str = ""
    _phase_streak: int = 0
    _last_spoken_text: str = ""
    _last_spoken_ms: float = 0.0

    def __post_init__(self) -> None:
        self._debouncer = FeedbackDebouncer(debounce_frames=self.debounce_frames)

    def reset(self) -> None:
        self._debouncer.reset()
        self._last_phase = ""
        self._phase_streak = 0
        self._last_spoken_text = ""
        self._last_spoken_ms = 0.0

    def update(
        self,
        candidate_text: str,
        phase: str,
        now_ms: float,
        *,
        rep_summary_text: str | None = None,
        force: bool = False,
    ) -> tuple[str, str | None]:
        """Return (stable_text, speech_text or None)."""
        if phase == self._last_phase:
            self._phase_streak += 1
        else:
            self._last_phase = phase
            self._phase_streak = 1

        phase_stable = self._phase_streak >= self.phase_stable_frames

        if rep_summary_text:
            stable_text, changed = self._debouncer.update(rep_summary_text, force=True)
        elif force:
            stable_text, changed = self._debouncer.update(candidate_text, force=True)
        elif phase_stable:
            stable_text, changed = self._debouncer.update(candidate_text)
        else:
            stable_text = self._debouncer.stable_text
            changed = False

        speech_text: str | None = None
        if (
            changed
            and stable_text
            and stable_text != self._last_spoken_text
            and (now_ms - self._last_spoken_ms) >= self.speech_cooldown_ms
        ):
            speech_text = stable_text
            self._last_spoken_text = stable_text
            self._last_spoken_ms = now_ms

        return stable_text, speech_text


# ---------------------------------------------------------------------------
# Rep-level feedback helper (event-based)
# ---------------------------------------------------------------------------


def build_rep_feedback(
    rep_number: int,
    rep_score: float,
    joint_issues: dict[str, int] | None,
) -> str:
    """Create a short, rep-level coaching summary."""
    issues = joint_issues or {}
    if not issues and rep_score >= 85:
        return f"Rep {rep_number}: strong form."

    if issues:
        worst_joint = max(issues, key=issues.get)
        joint_label = worst_joint.replace("_", " ")
        return f"Rep {rep_number}: focus on your {joint_label}."

    return f"Rep {rep_number} completed."
