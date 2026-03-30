"""
calibration.py — Range of Motion (ROM) profiling for adaptive scoring.

A ROM profile records each user's personal min/max angle per joint,
enabling the matcher to score relative to the user's achievable range
rather than fixed degree thresholds.

Provides:
  - ROMProfile           — dataclass holding per-joint min/max/range
  - calibrate_from_skeleton() — build a ROM profile from a normalised skeleton
  - save_rom_profile() / load_rom_profile() — JSON persistence
  - compute_adaptive_thresholds()  — convert ROM ranges into scoring thresholds
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Joint names (must match normalizer / matcher)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ROM Profile
# ---------------------------------------------------------------------------


@dataclass
class JointROM:
    """Min/max angle observed for a single joint, in degrees."""

    min_angle: float = 180.0
    max_angle: float = 0.0

    @property
    def range(self) -> float:
        """Achievable range of motion in degrees."""
        return max(0.0, self.max_angle - self.min_angle)


@dataclass
class ROMProfile:
    """Per-joint ROM profile for one user."""

    joints: dict[str, JointROM] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure all expected joints exist.
        for j in _JOINT_NAMES:
            if j not in self.joints:
                self.joints[j] = JointROM()


# ---------------------------------------------------------------------------
# Build a ROM profile from a normalised skeleton sequence
# ---------------------------------------------------------------------------


def calibrate_from_skeleton(
    normalised_skeleton: list[dict],
) -> ROMProfile:
    """Scan a normalised skeleton sequence to find min/max angle per joint.

    This can be run on:
    - A dedicated calibration video (user slowly moves each joint to extremes)
    - Any exercise video (uses observed range as an approximation)

    Parameters
    ----------
    normalised_skeleton : list[dict]
        Output of ``normalizer.normalize_skeleton`` — each frame must contain
        an ``"angles"`` dict.

    Returns
    -------
    ROMProfile
        The observed ROM for each joint.
    """
    profile = ROMProfile()

    for frame in normalised_skeleton:
        angles = frame.get("angles", {})
        for joint in _JOINT_NAMES:
            val = angles.get(joint)
            if val is None:
                continue
            jr = profile.joints[joint]
            jr.min_angle = min(jr.min_angle, val)
            jr.max_angle = max(jr.max_angle, val)

    return profile


# ---------------------------------------------------------------------------
# Adaptive thresholds
# ---------------------------------------------------------------------------

# Fraction of ROM range that counts as "good" / "warning".
_GOOD_FRAC = 0.15   # within 15% of ROM range → good
_WARN_FRAC = 0.35   # within 35% → warning; beyond → bad

# Absolute floor so that joints with very small ROM still have sane thresholds.
_MIN_GOOD = 5.0     # degrees
_MIN_WARN = 12.0    # degrees


def compute_adaptive_thresholds(
    rom_profile: ROMProfile,
) -> dict[str, dict[str, float]]:
    """Convert a ROM profile into per-joint good/warning thresholds.

    Parameters
    ----------
    rom_profile : ROMProfile

    Returns
    -------
    dict[str, dict[str, float]]
        ``{joint_name: {"good": float, "warning": float}}``
    """
    thresholds: dict[str, dict[str, float]] = {}

    for joint in _JOINT_NAMES:
        jr = rom_profile.joints.get(joint, JointROM())
        rom_range = jr.range

        good = max(rom_range * _GOOD_FRAC, _MIN_GOOD)
        warn = max(rom_range * _WARN_FRAC, _MIN_WARN)

        thresholds[joint] = {
            "good": round(good, 2),
            "warning": round(warn, 2),
        }

    return thresholds


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def save_rom_profile(profile: ROMProfile, path: str) -> None:
    """Save a ROM profile to JSON."""
    data: dict[str, Any] = {}
    for joint, jr in profile.joints.items():
        data[joint] = {
            "min_angle": round(jr.min_angle, 2),
            "max_angle": round(jr.max_angle, 2),
            "range": round(jr.range, 2),
        }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"ROM profile saved → {p}")


def load_rom_profile(path: str) -> ROMProfile:
    """Load a ROM profile from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data: dict = json.load(f)

    profile = ROMProfile()
    for joint, vals in data.items():
        if joint in profile.joints:
            profile.joints[joint] = JointROM(
                min_angle=vals.get("min_angle", 180.0),
                max_angle=vals.get("max_angle", 0.0),
            )
    print(f"Loaded ROM profile from {path}")
    return profile
