"""
filters.py — One Euro Filter for smooth landmark tracking.

The One Euro Filter removes jitter from noisy signals (like pose landmarks)
while preserving sharp, intentional movements. It adapts its cutoff frequency
based on the signal's speed: slow movements are smoothed heavily, fast
movements pass through with minimal lag.

Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems" (CHI 2012).

Provides:
  - OneEuroFilter           — single-value filter
  - LandmarkSmoother        — applies One Euro filtering to all 33 landmarks
  - HandLandmarkSmoother    — same, for left/right hand landmarks (21 each)
  - OpticalFlowValidator    — rejects implausible landmark jumps via Lucas-Kanade
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.perf_monitor import timed


class OneEuroFilter:
    """Adaptive low-pass filter for a single scalar signal.

    Parameters
    ----------
    min_cutoff : float
        Minimum cutoff frequency in Hz. Lower = smoother but more lag.
    beta : float
        Speed coefficient. Higher = less lag for fast movements.
    d_cutoff : float
        Cutoff for the derivative filter (usually 1.0).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, t: float, x: float) -> float:
        """Filter a new sample.

        Parameters
        ----------
        t : float
            Timestamp in seconds (must be monotonically increasing).
        x : float
            Raw signal value.

        Returns
        -------
        float
            Filtered value.
        """
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        t_e = t - self._t_prev
        if t_e <= 0:
            return self._x_prev  # type: ignore[return-value]

        # Derivative (speed) estimate.
        dx = (x - self._x_prev) / t_e  # type: ignore[operator]
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff based on speed.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filtered value.
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev  # type: ignore[operator]

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat


class LandmarkSmoother:
    """Apply One Euro filtering to all 33 MediaPipe landmarks (x, y, z).

    Usage::

        smoother = LandmarkSmoother()
        for each frame:
            smoothed = smoother.smooth(timestamp_sec, raw_landmarks)
    """

    def __init__(
        self,
        min_cutoff: float = 1.5,
        beta: float = 0.05,  # Higher beta = less lag for fast movements
        n_landmarks: int = 33,
    ) -> None:
        self.n_landmarks = n_landmarks
        # 3 filters per landmark (x, y, z).
        self._filters: list[list[OneEuroFilter]] = [
            [
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta),
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta),
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta),
            ]
            for _ in range(n_landmarks)
        ]

    def smooth(
        self, t: float, landmarks: list[dict],
    ) -> list[dict]:
        """Filter a frame of landmarks.

        Parameters
        ----------
        t : float
            Timestamp in seconds.
        landmarks : list[dict]
            33 dicts with ``x``, ``y``, ``z``, ``visibility``.

        Returns
        -------
        list[dict]
            Smoothed landmarks (same structure).
        """
        result: list[dict] = []
        for i, lm in enumerate(landmarks):
            if i >= self.n_landmarks:
                result.append(lm)
                continue

            fx, fy, fz = self._filters[i]
            result.append({
                "x": fx(t, lm["x"]),
                "y": fy(t, lm["y"]),
                "z": fz(t, lm["z"]),
                "visibility": lm["visibility"],
            })

        return result

    def reset(self) -> None:
        """Reset all filters (e.g. when pose is lost)."""
        for group in self._filters:
            for f in group:
                f.reset()


class HandLandmarkSmoother:
    """Apply One Euro filtering to MediaPipe hand landmarks (21 per hand).

    Maintains independent smoothers for left and right hands.  When a hand
    disappears from the frame its smoother is reset so stale state does not
    contaminate the next detection.

    Usage::

        smoother = HandLandmarkSmoother()
        for each frame:
            raw = {"left": [...21 dicts...] | None, "right": [...21 dicts...] | None}
            smoothed = smoother.smooth(timestamp_sec, raw)
    """

    def __init__(
        self,
        min_cutoff: float = 1.5,
        beta: float = 0.07,
    ) -> None:
        self._left = LandmarkSmoother(min_cutoff=min_cutoff, beta=beta, n_landmarks=21)
        self._right = LandmarkSmoother(min_cutoff=min_cutoff, beta=beta, n_landmarks=21)

    def smooth(
        self,
        t: float,
        hand_landmarks: "dict[str, list[dict] | None]",
    ) -> "dict[str, list[dict] | None]":
        """Filter hand landmarks for both hands.

        Parameters
        ----------
        t : float
            Timestamp in seconds (monotonically increasing).
        hand_landmarks : dict
            ``{"left": [...21 dicts...] | None, "right": [...21 dicts...] | None}``

        Returns
        -------
        dict
            Smoothed landmarks in the same structure.
        """
        result: dict = {"left": None, "right": None}

        left = hand_landmarks.get("left")
        if left:
            result["left"] = self._left.smooth(t, left)
        else:
            self._left.reset()

        right = hand_landmarks.get("right")
        if right:
            result["right"] = self._right.smooth(t, right)
        else:
            self._right.reset()

        return result

    def reset(self) -> None:
        """Reset all filters (e.g. when hands are lost)."""
        self._left.reset()
        self._right.reset()


class OpticalFlowValidator:
    """Reject landmark jumps that exceed optical-flow-predicted motion.

    Uses sparse Lucas-Kanade optical flow on the previous frame's landmark
    positions to predict where each landmark *should* be in the current frame.
    If the detector's reported position differs from the flow prediction by
    more than ``max_residual_pixels``, we treat that landmark as a detection
    snap and replace it with the flow prediction (with reduced confidence).

    Accepts landmarks in either tuple form ``(x_norm, y_norm, vis)`` or
    MediaPipe dict form ``{"x", "y", "visibility", ...}``; returns the same
    shape it received, with ``None`` entries passed through unchanged.
    """

    def __init__(
        self,
        max_residual_pixels: float = 25.0,
        num_landmarks: int = 33,
        win_size: tuple[int, int] = (15, 15),
        max_level: int = 2,
    ) -> None:
        self.max_residual_pixels = float(max_residual_pixels)
        self.num_landmarks = num_landmarks
        self.win_size = win_size
        self.max_level = max_level
        self._prev_gray: np.ndarray | None = None
        self._prev_landmarks: list | None = None

    def reset(self) -> None:
        """Clear the previous-frame state (e.g. when tracking restarts)."""
        self._prev_gray = None
        self._prev_landmarks = None

    @staticmethod
    def _to_xy_vis(lm) -> tuple[float, float, float] | None:
        """Normalize an entry into (x_norm, y_norm, vis) or None."""
        if lm is None:
            return None
        if isinstance(lm, dict):
            if "x" not in lm or "y" not in lm:
                return None
            return float(lm["x"]), float(lm["y"]), float(lm.get("visibility", 1.0))
        if isinstance(lm, (tuple, list)) and len(lm) >= 2:
            x = float(lm[0])
            y = float(lm[1])
            vis = float(lm[2]) if len(lm) >= 3 else 1.0
            return x, y, vis
        return None

    @staticmethod
    def _replace_xy(original, x_norm: float, y_norm: float, vis: float):
        """Return a copy of ``original`` with its xy position overwritten."""
        if isinstance(original, dict):
            out = dict(original)
            out["x"] = x_norm
            out["y"] = y_norm
            out["visibility"] = vis
            return out
        return (x_norm, y_norm, vis)

    @timed("flow_validate")
    def validate(
        self,
        frame_bgr: np.ndarray,
        landmarks: list,
        frame_shape: tuple[int, int] | None = None,
    ) -> list:
        """Validate the current-frame landmarks against optical flow.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Current frame in BGR (H, W, 3).
        landmarks : list
            Per-landmark entries (dict, tuple, or None).
        frame_shape : tuple[int, int], optional
            (H, W). Defaults to ``frame_bgr.shape[:2]`` if omitted.

        Returns
        -------
        list
            Same shape as ``landmarks`` with snaps replaced by flow-predicted
            positions at reduced confidence.
        """
        if frame_shape is None:
            h, w = frame_bgr.shape[:2]
        else:
            h, w = frame_shape

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # First frame or fresh reset → record state, pass through unchanged.
        if self._prev_gray is None or self._prev_landmarks is None:
            self._prev_gray = gray
            self._prev_landmarks = list(landmarks)
            return landmarks

        # Collect previous-frame pixel positions for landmarks present then.
        prev_pts_list: list[list[float]] = []
        idx_map: list[int] = []
        for i, prev_lm in enumerate(self._prev_landmarks):
            prev_norm = self._to_xy_vis(prev_lm)
            if prev_norm is None:
                continue
            px, py, _ = prev_norm
            prev_pts_list.append([px * w, py * h])
            idx_map.append(i)

        if not prev_pts_list:
            self._prev_gray = gray
            self._prev_landmarks = list(landmarks)
            return landmarks

        prev_pts = np.array(prev_pts_list, dtype=np.float32).reshape(-1, 1, 2)

        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            prev_pts,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        validated = list(landmarks)
        for j, i in enumerate(idx_map):
            if i >= len(validated):
                continue
            if status[j][0] == 0:
                continue  # flow tracking failed — trust detector
            current = self._to_xy_vis(validated[i])
            if current is None:
                continue

            predicted_x, predicted_y = float(next_pts[j][0][0]), float(next_pts[j][0][1])
            actual_x = current[0] * w
            actual_y = current[1] * h

            residual = math.sqrt(
                (predicted_x - actual_x) ** 2 + (predicted_y - actual_y) ** 2
            )

            if residual > self.max_residual_pixels:
                new_vis = current[2] * 0.5
                validated[i] = self._replace_xy(
                    validated[i],
                    predicted_x / w if w > 0 else current[0],
                    predicted_y / h if h > 0 else current[1],
                    new_vis,
                )

        self._prev_gray = gray
        self._prev_landmarks = list(validated)
        return validated
