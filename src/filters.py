"""
filters.py — One Euro Filter for smooth landmark tracking.

The One Euro Filter removes jitter from noisy signals (like pose landmarks)
while preserving sharp, intentional movements. It adapts its cutoff frequency
based on the signal's speed: slow movements are smoothed heavily, fast
movements pass through with minimal lag.

Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems" (CHI 2012).

Provides:
  - OneEuroFilter       — single-value filter
  - LandmarkSmoother    — applies One Euro filtering to all 33 landmarks
"""

from __future__ import annotations

import math


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
        beta: float = 0.01,
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
