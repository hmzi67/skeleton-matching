"""
visualizer.py — Kemtai-style visual feedback overlay.

Provides:
  - draw_glowing_skeleton()    — neon skeleton with color-coded segments
  - draw_hand_skeleton()       — neon hand overlay (21 landmarks per hand)
  - draw_score_bar()           — vertical green→red score bar with badge
  - draw_coaching_bubble()     — floating instruction bubble
  - draw_rep_bars()            — coloured rep history columns
  - draw_directional_arrow()   — arrow at joint showing correction direction
  - draw_phase_indicator()     — exercise phase text + rep counter
  - create_side_by_side()      — full Kemtai-style composite frame
  - save_output_video()        — write frames to MP4
  - score_over_time_chart()    — matplotlib line chart
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe pose connections
# ---------------------------------------------------------------------------
POSE_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31),
    (28, 30), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (17, 19), (18, 20),
]

# Body-segment connections with assigned colour group
_SEGMENT_CONNECTIONS: list[tuple[tuple[int, int], str]] = [
    # Torso
    ((11, 12), "torso"), ((11, 23), "torso"), ((12, 24), "torso"),
    ((23, 24), "torso"),
    # Left arm
    ((11, 13), "left_arm"), ((13, 15), "left_arm"),
    ((15, 17), "left_arm"), ((15, 19), "left_arm"), ((15, 21), "left_arm"),
    ((17, 19), "left_arm"),
    # Right arm
    ((12, 14), "right_arm"), ((14, 16), "right_arm"),
    ((16, 18), "right_arm"), ((16, 20), "right_arm"), ((16, 22), "right_arm"),
    ((18, 20), "right_arm"),
    # Left leg
    ((23, 25), "left_leg"), ((25, 27), "left_leg"),
    ((27, 29), "left_leg"), ((29, 31), "left_leg"),
    # Right leg
    ((24, 26), "right_leg"), ((26, 28), "right_leg"),
    ((28, 30), "right_leg"), ((30, 32), "right_leg"),
]

_BODY_LANDMARKS = set(range(11, 33))

# Per-segment colours (BGR)
_SEGMENT_COLORS: dict[str, tuple[int, int, int]] = {
    "torso":     (255, 255, 255),   # white
    "left_arm":  (255, 255, 0),     # cyan
    "right_arm": (255, 0, 255),     # magenta
    "left_leg":  (0, 255, 0),       # green
    "right_leg": (0, 255, 255),     # yellow
}

# Map landmark idx → segment for joint colour
_IDX_TO_SEGMENT: dict[int, str] = {}
for (i, j), seg in _SEGMENT_CONNECTIONS:
    _IDX_TO_SEGMENT[i] = seg
    _IDX_TO_SEGMENT[j] = seg

_JOINT_NAME_TO_IDX: dict[str, int] = {
    "left_elbow": 13, "right_elbow": 14,
    "left_knee": 25, "right_knee": 26,
    "left_hip": 23, "right_hip": 24,
    "left_shoulder": 11, "right_shoulder": 12,
}

# ---------------------------------------------------------------------------
# MediaPipe Hands connections (21 landmarks per hand)
# ---------------------------------------------------------------------------

# Each tuple is (start_idx, end_idx) within a 21-landmark hand list.
HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17),
]

# Finger group → colour (BGR).
_FINGER_COLORS: dict[int, tuple[int, int, int]] = {
    # thumb: orange
    1: (0, 165, 255),  2: (0, 165, 255),  3: (0, 165, 255),  4: (0, 165, 255),
    # index: cyan
    5: (255, 255, 0),  6: (255, 255, 0),  7: (255, 255, 0),  8: (255, 255, 0),
    # middle: green
    9: (0, 255, 128),  10: (0, 255, 128), 11: (0, 255, 128), 12: (0, 255, 128),
    # ring: magenta
    13: (255, 0, 255), 14: (255, 0, 255), 15: (255, 0, 255), 16: (255, 0, 255),
    # pinky: yellow
    17: (0, 255, 255), 18: (0, 255, 255), 19: (0, 255, 255), 20: (0, 255, 255),
}
_WRIST_COLOR: tuple[int, int, int] = (200, 200, 200)  # wrist landmark (idx=0)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _score_color_bgr(score: float) -> tuple[int, int, int]:
    """Map 0-100 score to BGR: red → yellow → green."""
    t = max(0.0, min(score, 100.0)) / 100.0
    if t < 0.5:
        r, g = 255, int(255 * (t / 0.5))
    else:
        r, g = int(255 * ((1.0 - t) / 0.5)), 255
    return (0, g, r)


def _rounded_rect(
    frame: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int],
    color: tuple[int, int, int], radius: int = 20,
    thickness: int = -1, alpha: float = 1.0,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if alpha < 1.0:
        ov = frame.copy()
        _draw_rounded(ov, x1, y1, x2, y2, r, color, thickness)
        cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)
    else:
        _draw_rounded(frame, x1, y1, x2, y2, r, color, thickness)


def _draw_rounded(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
    r: int, color: tuple[int, int, int], thickness: int,
) -> None:
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    if thickness == -1:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x1+r, y2-r), color, -1)
        cv2.rectangle(img, (x2-r, y1+r), (x2, y2-r), color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Glowing skeleton
# ---------------------------------------------------------------------------


def _draw_glowing_line(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    glow_passes: int = 3,
) -> None:
    """Draw a line with a neon glow effect (multi-pass)."""
    for i in range(glow_passes, 0, -1):
        alpha = 0.15 * (1.0 - i / (glow_passes + 1))
        glow_t = thickness + i * 4
        glow_col = tuple(max(0, min(255, int(c * 0.5))) for c in color)
        cv2.line(frame, pt1, pt2, glow_col, glow_t, cv2.LINE_AA)  # type: ignore[arg-type]
    # Core bright line
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def _draw_glowing_joint(
    frame: np.ndarray,
    pt: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 6,
) -> None:
    """Draw a joint dot with glow halo."""
    # Outer glow
    cv2.circle(frame, pt, radius + 4, tuple(int(c * 0.3) for c in color), -1, cv2.LINE_AA)  # type: ignore[arg-type]
    cv2.circle(frame, pt, radius + 2, tuple(int(c * 0.5) for c in color), -1, cv2.LINE_AA)  # type: ignore[arg-type]
    # Core dot
    cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
    # Bright center
    cv2.circle(frame, pt, max(2, radius // 2), (255, 255, 255), -1, cv2.LINE_AA)


def draw_glowing_skeleton(
    frame: np.ndarray,
    landmarks: list[dict],
) -> np.ndarray:
    """Draw a glowing, color-coded skeleton (Kemtai-style).

    Body segments:
    - Torso: white
    - Left arm: cyan | Right arm: magenta
    - Left leg: green | Right leg: yellow

    Returns the annotated frame.
    """
    h, w = frame.shape[:2]

    def _px(lm: dict) -> tuple[int, int]:
        return int(lm["x"] * w), int(lm["y"] * h)

    # Draw segment connections with glow.
    for (i, j), seg in _SEGMENT_CONNECTIONS:
        if i < len(landmarks) and j < len(landmarks):
            color = _SEGMENT_COLORS.get(seg, (255, 255, 255))
            _draw_glowing_line(frame, _px(landmarks[i]), _px(landmarks[j]), color,
                               thickness=3, glow_passes=3)

    # Draw joints with glow.
    for idx in _BODY_LANDMARKS:
        if idx < len(landmarks):
            seg = _IDX_TO_SEGMENT.get(idx, "torso")
            color = _SEGMENT_COLORS.get(seg, (255, 255, 255))
            _draw_glowing_joint(frame, _px(landmarks[idx]), color, radius=5)

    return frame


# Legacy API compat
def draw_skeleton_on_frame(
    frame: np.ndarray,
    landmarks: list[dict],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 3,
    joint_radius: int = 6,
    body_only: bool = True,
    joint_status: dict[str, str] | None = None,
) -> np.ndarray:
    """Draw skeleton — delegates to glowing version."""
    return draw_glowing_skeleton(frame, landmarks)


# ---------------------------------------------------------------------------
# Hand skeleton overlay
# ---------------------------------------------------------------------------


def draw_hand_skeleton(
    frame: np.ndarray,
    hand_landmarks: "dict[str, list[dict] | None]",
) -> np.ndarray:
    """Draw neon hand skeletons for both hands (when detected).

    Parameters
    ----------
    frame : np.ndarray
        BGR frame to annotate (modified in-place).
    hand_landmarks : dict
        ``{"left": [...21 dicts...] | None, "right": [...21 dicts...] | None}``
        Each dict has ``x``, ``y`` in normalised [0, 1] coordinates.

    Returns
    -------
    np.ndarray
        Annotated frame.
    """
    h, w = frame.shape[:2]

    def _px(lm: dict) -> tuple[int, int]:
        return int(lm["x"] * w), int(lm["y"] * h)

    for hand_lms in (
        hand_landmarks.get("left"),
        hand_landmarks.get("right"),
    ):
        if not hand_lms or len(hand_lms) < 21:
            continue

        # Draw connections with glow.
        for i, j in HAND_CONNECTIONS:
            color = _FINGER_COLORS.get(max(i, j), _WRIST_COLOR)
            _draw_glowing_line(frame, _px(hand_lms[i]), _px(hand_lms[j]),
                               color, thickness=2, glow_passes=2)

        # Draw joints.
        for idx, lm in enumerate(hand_lms):
            color = _FINGER_COLORS.get(idx, _WRIST_COLOR)
            _draw_glowing_joint(frame, _px(lm), color, radius=4)

    return frame


# ---------------------------------------------------------------------------
# Directional arrows for correction
# ---------------------------------------------------------------------------


def draw_directional_arrow(
    frame: np.ndarray,
    landmarks: list[dict],
    joint_name: str,
    diff: float,
) -> None:
    """Draw an arrow at a joint showing the correction direction.

    The arrow points in the direction the user should move.
    """
    idx = _JOINT_NAME_TO_IDX.get(joint_name)
    if idx is None or idx >= len(landmarks):
        return

    h, w = frame.shape[:2]
    lm = landmarks[idx]
    cx, cy = int(lm["x"] * w), int(lm["y"] * h)

    # Arrow direction based on joint type and diff sign.
    arrow_len = min(40, max(15, int(abs(diff) * 0.5)))

    if "knee" in joint_name or "hip" in joint_name:
        # Vertical correction: diff>0 = bend more (arrow down), diff<0 = straighten (arrow up)
        dy = arrow_len if diff > 0 else -arrow_len
        dx = 0
    elif "elbow" in joint_name or "shoulder" in joint_name:
        # Horizontal-ish: diff>0 = bend more (arrow inward), diff<0 = extend (arrow outward)
        side = -1 if "left" in joint_name else 1
        dx = side * arrow_len if diff < 0 else -side * arrow_len
        dy = 0
    else:
        return

    end_x, end_y = cx + dx, cy + dy
    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y),
                    (0, 180, 255), 3, cv2.LINE_AA, tipLength=0.35)


# ---------------------------------------------------------------------------
# Vertical score bar
# ---------------------------------------------------------------------------


def draw_score_bar(
    frame: np.ndarray, score: float, x_center: int,
    y_top: int, y_bottom: int, bar_width: int = 12,
) -> None:
    """Vertical score bar with gradient and score badge."""
    h_bar = y_bottom - y_top
    x1 = x_center - bar_width // 2
    x2 = x_center + bar_width // 2

    for y in range(y_top, y_bottom):
        t = (y - y_top) / max(h_bar - 1, 1)
        g = int(255 * (1.0 - t))
        r = int(255 * t)
        cv2.line(frame, (x1, y), (x2, y), (0, g, r), 1)

    cv2.rectangle(frame, (x1-1, y_top-1), (x2+1, y_bottom+1), (80, 80, 80), 1)

    score_t = 1.0 - max(0.0, min(score, 100.0)) / 100.0
    dot_y = y_top + int(score_t * h_bar)
    dot_color = _score_color_bgr(score)
    cv2.circle(frame, (x_center, dot_y), 10, dot_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x_center, dot_y), 10, (255, 255, 255), 2, cv2.LINE_AA)

    badge_w, badge_h = 60, 36
    badge_x1 = x_center - badge_w // 2
    badge_y1 = max(5, dot_y - badge_h - 16)
    badge_x2 = badge_x1 + badge_w
    badge_y2 = badge_y1 + badge_h

    _rounded_rect(frame, (badge_x1, badge_y1), (badge_x2, badge_y2),
                  (50, 50, 50), radius=10, thickness=-1, alpha=0.85)

    text = str(int(score))
    fs = 0.8
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    tx = badge_x1 + (badge_w - tw) // 2
    ty = badge_y1 + (badge_h + th) // 2
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Coaching bubble
# ---------------------------------------------------------------------------


def draw_coaching_bubble(
    frame: np.ndarray, text: str, score: float,
    x_center: int, y_center: int,
) -> None:
    """Floating coaching bubble with single instruction."""
    if not text:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2

    # Wrap text.
    max_chars = 28
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if current and len(current) + len(w) + 1 > max_chars:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip() if current else w
    if current:
        lines.append(current)
    if not lines:
        return

    line_height = 34
    max_tw = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_tw = max(max_tw, tw)

    pad_x, pad_y = 30, 18
    bubble_w = max_tw + pad_x * 2
    bubble_h = len(lines) * line_height + pad_y * 2

    bx1 = x_center - bubble_w // 2
    by1 = y_center - bubble_h // 2
    bx2 = bx1 + bubble_w
    by2 = by1 + bubble_h

    fh, fw = frame.shape[:2]
    if bx1 < 5:
        bx1, bx2 = 5, 5 + bubble_w
    if bx2 > fw - 5:
        bx2 = fw - 5; bx1 = bx2 - bubble_w
    if by2 > fh - 5:
        by2 = fh - 5; by1 = by2 - bubble_h

    if score >= 90:
        border_color = (0, 215, 255)   # gold
    elif score >= 75:
        border_color = (200, 200, 200)
    else:
        border_color = (120, 80, 200)  # pink

    _rounded_rect(frame, (bx1, by1), (bx2, by2),
                  (30, 30, 30), radius=16, thickness=-1, alpha=0.85)
    _rounded_rect(frame, (bx1, by1), (bx2, by2),
                  border_color, radius=16, thickness=3)

    text_color = (0, 235, 255) if score >= 90 else (255, 255, 255)
    y_text = by1 + pad_y + 22
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        tx = bx1 + (bubble_w - tw) // 2
        cv2.putText(frame, line, (tx, y_text), font, font_scale,
                    text_color, thickness, cv2.LINE_AA)
        y_text += line_height


# ---------------------------------------------------------------------------
# Rep bars
# ---------------------------------------------------------------------------


def draw_rep_bars(
    frame: np.ndarray, score_history: list[float],
    y_top: int, x_start: int, x_end: int,
    bar_height: int = 50, max_bars: int = 20,
) -> None:
    """Coloured rep/score history columns."""
    if not score_history:
        return
    recent = score_history[-max_bars:]
    n = len(recent)
    total_w = x_end - x_start
    gap = 3
    bar_w = max(4, (total_w - gap * (n + 1)) // max(n, 1))
    x = x_start + gap
    for s in recent:
        color = _score_color_bgr(s)
        h = max(4, int((s / 100.0) * bar_height))
        cv2.rectangle(frame, (x, y_top + bar_height - h), (x + bar_w, y_top + bar_height),
                      color, -1)
        x += bar_w + gap


# ---------------------------------------------------------------------------
# Phase indicator + rep counter
# ---------------------------------------------------------------------------


def draw_phase_indicator(
    frame: np.ndarray,
    phase: str,
    rep_count: int,
    x: int,
    y: int,
) -> None:
    """Draw exercise phase text and rep counter on the frame."""
    # Phase badge
    phase_colors = {
        "READY": (200, 200, 200),
        "DOWN":  (0, 200, 255),
        "HOLD":  (0, 255, 0),
        "UP":    (255, 200, 0),
    }
    pc = phase_colors.get(phase, (200, 200, 200))

    # Phase text
    cv2.putText(frame, phase, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, pc, 2, cv2.LINE_AA)

    # Rep counter below
    rep_text = f"Reps: {rep_count}"
    cv2.putText(frame, rep_text, (x, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main side-by-side (Kemtai-style)
# ---------------------------------------------------------------------------


def create_side_by_side(
    gt_frame: np.ndarray,
    user_frame: np.ndarray,
    gt_landmarks: list[dict],
    user_landmarks: list[dict],
    feedback: dict,
    *,
    gt_label: str = "Ground Truth",
    user_label: str = "You",
    joint_status: dict[str, str] | None = None,
    score_history: list[float] | None = None,
    phase: str = "",
    rep_count: int = 0,
    joint_errors: dict[str, float] | None = None,
    user_hand_landmarks: "dict[str, list[dict] | None] | None" = None,
    gt_hand_landmarks: "dict[str, list[dict] | None] | None" = None,
) -> np.ndarray:
    """Create a Kemtai-style side-by-side composite frame.

    User on LEFT (60%), score bar CENTER, GT on RIGHT (40%).
    Includes: glowing skeleton, hand overlay (when available),
    coaching bubble, rep bars, phase indicator, and correction arrows.

    Parameters
    ----------
    user_hand_landmarks : dict, optional
        ``{"left": [...21 dicts...] | None, "right": [...21 dicts...] | None}``
        When provided, the hand skeleton is drawn over the user panel.
    gt_hand_landmarks : dict, optional
        Same structure for the reference panel (drawn in muted tones).
    """
    target_h = 540
    user_w = int(target_h * 4 / 3)
    gt_w = int(target_h * 3 / 4)
    bar_gap = 30

    user_resized = cv2.resize(user_frame, (user_w, target_h))
    gt_resized   = cv2.resize(gt_frame,   (gt_w,   target_h))

    # Draw glowing body skeleton on USER panel.
    draw_glowing_skeleton(user_resized, user_landmarks)

    # Draw hand skeleton overlay on USER panel (when available).
    if user_hand_landmarks:
        draw_hand_skeleton(user_resized, user_hand_landmarks)

    # Draw directional correction arrows on user skeleton.
    if joint_errors:
        worst = sorted(joint_errors.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
        for jname, diff in worst:
            if abs(diff) > 15:
                draw_directional_arrow(user_resized, user_landmarks, jname, diff)

    # Phase indicator on user panel.
    if phase:
        draw_phase_indicator(user_resized, phase, rep_count, x=15, y=35)

    # Draw hand skeleton on GT panel when available (thinner, muted).
    if gt_hand_landmarks:
        draw_hand_skeleton(gt_resized, gt_hand_landmarks)

    # Assemble canvas.
    total_w = user_w + bar_gap + gt_w
    canvas = np.zeros((target_h, total_w, 3), dtype=np.uint8)
    canvas[:, :user_w]          = user_resized
    canvas[:, user_w + bar_gap:] = gt_resized
    canvas[:, user_w:user_w + bar_gap] = (20, 20, 20)

    # Score bar in the gap strip.
    score = feedback.get("overall_score", 0)
    bar_x = user_w + bar_gap // 2
    draw_score_bar(canvas, score, bar_x, y_top=40, y_bottom=target_h - 100)

    # Coaching bubble — top priority_fix or headline.
    joint_fb = feedback.get("joint_feedback", [])
    if joint_fb:
        instruction = joint_fb[0].get("instruction", "")
    elif score >= 90:
        instruction = "Awesome Job!"
    else:
        instruction = feedback.get("priority_fix", feedback.get("headline", ""))

    bubble_x = total_w // 2
    bubble_y = target_h - 80
    draw_coaching_bubble(canvas, instruction, score, bubble_x, bubble_y)

    # Bottom strip with frame-score history bars.
    strip_h = 65
    bottom = np.zeros((strip_h, total_w, 3), dtype=np.uint8)
    bottom[:] = (20, 20, 20)

    if score_history:
        draw_rep_bars(bottom, score_history, y_top=8, x_start=10,
                      x_end=total_w - 10, bar_height=strip_h - 16)

    return np.vstack([canvas, bottom])


# Legacy no-op
def draw_feedback_overlay(frame: np.ndarray, feedback: dict) -> np.ndarray:
    return frame


# ---------------------------------------------------------------------------
# Save video
# ---------------------------------------------------------------------------


def save_output_video(
    frames: list[np.ndarray], output_path: str, fps: float = 30.0,
) -> None:
    if not frames:
        print("No frames to write.")
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"Output video saved -> {path}")


# ---------------------------------------------------------------------------
# Score-over-time chart
# ---------------------------------------------------------------------------


def score_over_time_chart(
    frame_scores: list[float], output_path: str = "output/score_chart.png",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_scores, color="#4fc3f7", linewidth=1.5)
    ax.fill_between(range(len(frame_scores)), frame_scores, alpha=0.15, color="#4fc3f7")
    ax.axhline(90, color="#66bb6a", linestyle="--", linewidth=0.8, label="Excellent (90)")
    ax.axhline(75, color="#ffa726", linestyle="--", linewidth=0.8, label="Good (75)")
    ax.axhline(50, color="#ef5350", linestyle="--", linewidth=0.8, label="Needs Work (50)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_title("Pose Match Score Over Time")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"Score chart saved -> {path}")
