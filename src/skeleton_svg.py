"""
SVG skeleton diagram generator for visual accuracy reports.

This module renders a canonical MediaPipe-like skeleton (pose + simplified
hands) and overlays defective keypoints derived from report problem_joints.
"""

from __future__ import annotations

from typing import Optional


# Pose joints that can receive accuracy values.
POSE_ACCURACY_JOINTS: list[str] = [
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

# Finger-curl joints used by hand exercises.
HAND_ACCURACY_JOINTS: list[str] = [
    "left_hand_thumb_curl",
    "left_hand_index_curl",
    "left_hand_middle_curl",
    "left_hand_ring_curl",
    "left_hand_pinky_curl",
    "right_hand_thumb_curl",
    "right_hand_index_curl",
    "right_hand_middle_curl",
    "right_hand_ring_curl",
    "right_hand_pinky_curl",
]

TRACKED_JOINTS: list[str] = POSE_ACCURACY_JOINTS + HAND_ACCURACY_JOINTS

# Human-readable joint names for display
JOINT_DISPLAY_NAMES: dict[str, str] = {
    "left_elbow": "L. Elbow",
    "right_elbow": "R. Elbow",
    "left_knee": "L. Knee",
    "right_knee": "R. Knee",
    "left_hip": "L. Hip",
    "right_hip": "R. Hip",
    "left_shoulder": "L. Shoulder",
    "right_shoulder": "R. Shoulder",
    "torso_lean": "Torso",
    "left_hand_thumb_curl": "L. Thumb",
    "left_hand_index_curl": "L. Index",
    "left_hand_middle_curl": "L. Middle",
    "left_hand_ring_curl": "L. Ring",
    "left_hand_pinky_curl": "L. Pinky",
    "right_hand_thumb_curl": "R. Thumb",
    "right_hand_index_curl": "R. Index",
    "right_hand_middle_curl": "R. Middle",
    "right_hand_ring_curl": "R. Ring",
    "right_hand_pinky_curl": "R. Pinky",
}


# Canonical MediaPipe pose landmarks (front-facing, normalized to 0..100).
# Landmark indices follow MediaPipe Pose 33-landmark order.
_POSE_POINTS: dict[int, tuple[float, float]] = {
    0: (50.0, 9.6),
    1: (49.0, 9.1),
    2: (48.1, 9.0),
    3: (46.9, 9.4),
    4: (51.0, 9.1),
    5: (51.9, 9.0),
    6: (53.1, 9.4),
    7: (45.8, 10.9),
    8: (54.2, 10.9),
    9: (48.7, 12.1),
    10: (51.3, 12.1),
    11: (40.0, 22.0),
    12: (60.0, 22.0),
    13: (31.0, 37.0),
    14: (69.0, 37.0),
    15: (26.0, 52.0),
    16: (74.0, 52.0),
    17: (24.0, 54.0),
    18: (76.0, 54.0),
    19: (25.0, 53.0),
    20: (75.0, 53.0),
    21: (27.0, 54.0),
    22: (73.0, 54.0),
    23: (45.0, 48.0),
    24: (55.0, 48.0),
    25: (45.0, 68.0),
    26: (55.0, 68.0),
    27: (44.0, 89.0),
    28: (56.0, 89.0),
    29: (42.0, 92.0),
    30: (58.0, 92.0),
    31: (46.0, 93.0),
    32: (54.0, 93.0),
}


_POSE_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


# Canonical right-hand landmark offsets from wrist (landmark 0).
# Left hand is mirrored over wrist.
_HAND_RIGHT_OFFSETS: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (1.6, -0.9),
    2: (2.9, -1.9),
    3: (4.0, -3.0),
    4: (5.2, -4.3),
    5: (1.0, -1.9),
    6: (1.7, -3.8),
    7: (2.2, -5.4),
    8: (2.6, -6.9),
    9: (0.1, -2.1),
    10: (0.2, -4.2),
    11: (0.2, -6.1),
    12: (0.2, -7.8),
    13: (-0.9, -1.8),
    14: (-1.5, -3.6),
    15: (-2.0, -5.2),
    16: (-2.3, -6.7),
    17: (-1.8, -1.1),
    18: (-2.7, -2.5),
    19: (-3.4, -3.9),
    20: (-3.9, -5.2),
}


_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


_HAND_BASE_LANDMARKS: set[int] = {
    0,
    3,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
}


_POSE_JOINT_TO_INDEX: dict[str, int] = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
}


_HAND_JOINT_TO_INDEX: dict[str, tuple[str, int]] = {
    "left_hand_thumb_curl": ("left", 3),
    "left_hand_index_curl": ("left", 6),
    "left_hand_middle_curl": ("left", 10),
    "left_hand_ring_curl": ("left", 14),
    "left_hand_pinky_curl": ("left", 18),
    "right_hand_thumb_curl": ("right", 3),
    "right_hand_index_curl": ("right", 6),
    "right_hand_middle_curl": ("right", 10),
    "right_hand_ring_curl": ("right", 14),
    "right_hand_pinky_curl": ("right", 18),
}


_JOINT_ALIASES: dict[str, str] = {
    "torso_center": "torso_lean",
    "left_thumb_curl": "left_hand_thumb_curl",
    "left_index_curl": "left_hand_index_curl",
    "left_middle_curl": "left_hand_middle_curl",
    "left_ring_curl": "left_hand_ring_curl",
    "left_pinky_curl": "left_hand_pinky_curl",
    "right_thumb_curl": "right_hand_thumb_curl",
    "right_index_curl": "right_hand_index_curl",
    "right_middle_curl": "right_hand_middle_curl",
    "right_ring_curl": "right_hand_ring_curl",
    "right_pinky_curl": "right_hand_pinky_curl",
}


_SEVERITY_COLORS: dict[str, str] = {
    "mild": "#facc15",
    "moderate": "#f97316",
    "critical": "#ef4444",
}

# Glow colors with higher saturation for better visibility
_SEVERITY_GLOW_COLORS: dict[str, str] = {
    "mild": "#fef08a",
    "moderate": "#fed7aa",
    "critical": "#fecaca",
}


def _normalized_joint_name(joint_name: str) -> str:
    normalized = str(joint_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    if normalized in _JOINT_ALIASES:
        return _JOINT_ALIASES[normalized]

    if normalized.endswith("_curl"):
        if normalized.startswith("left_") and not normalized.startswith("left_hand_"):
            return "left_hand_" + normalized[len("left_"):]
        if normalized.startswith("right_") and not normalized.startswith("right_hand_"):
            return "right_hand_" + normalized[len("right_"):]

    return normalized


def _clamp_accuracy(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def accuracy_to_color(accuracy: float) -> str:
    accuracy = _clamp_accuracy(accuracy)

    if accuracy >= 0.85:
        return "#22c55e"
    if accuracy >= 0.70:
        t = (accuracy - 0.70) / 0.15
        r = int(34 + (250 - 34) * (1 - t))
        g = int(197 + (204 - 197) * (1 - t))
        b = int(94 + (21 - 94) * (1 - t))
        return f"#{r:02x}{g:02x}{b:02x}"
    if accuracy >= 0.55:
        t = (accuracy - 0.55) / 0.15
        r = int(249 + (250 - 249) * t)
        g = int(115 + (204 - 115) * t)
        b = int(22 + (21 - 22) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    t = accuracy / 0.55
    r = int(239 + (249 - 239) * t)
    g = int(68 + (115 - 68) * t)
    b = int(68 + (22 - 68) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_hand_points(wrist_x: float, wrist_y: float, *, side: str) -> dict[int, tuple[float, float]]:
    x_sign = -1.0 if side == "left" else 1.0
    scale = 1.35

    points: dict[int, tuple[float, float]] = {}
    for index, (dx, dy) in _HAND_RIGHT_OFFSETS.items():
        points[index] = (wrist_x + (x_sign * dx * scale), wrist_y + (dy * scale))
    return points


def _build_canvas_geometry(y_offset: float) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]], dict[int, tuple[float, float]], dict[str, tuple[float, float]]]:
    pose_points = {idx: (x, y + y_offset) for idx, (x, y) in _POSE_POINTS.items()}

    left_wrist_x, left_wrist_y = pose_points[15]
    right_wrist_x, right_wrist_y = pose_points[16]
    left_hand_points = _build_hand_points(left_wrist_x, left_wrist_y, side="left")
    right_hand_points = _build_hand_points(right_wrist_x, right_wrist_y, side="right")

    shoulder_mid = (
        (pose_points[11][0] + pose_points[12][0]) / 2.0,
        (pose_points[11][1] + pose_points[12][1]) / 2.0,
    )
    hip_mid = (
        (pose_points[23][0] + pose_points[24][0]) / 2.0,
        (pose_points[23][1] + pose_points[24][1]) / 2.0,
    )
    torso_point = (
        (shoulder_mid[0] + hip_mid[0]) / 2.0,
        (shoulder_mid[1] + hip_mid[1]) / 2.0,
    )

    joint_points: dict[str, tuple[float, float]] = {"torso_lean": torso_point}
    for joint_name, index in _POSE_JOINT_TO_INDEX.items():
        joint_points[joint_name] = pose_points[index]

    for joint_name, (hand_side, hand_index) in _HAND_JOINT_TO_INDEX.items():
        if hand_side == "left":
            joint_points[joint_name] = left_hand_points[hand_index]
        else:
            joint_points[joint_name] = right_hand_points[hand_index]

    return pose_points, left_hand_points, right_hand_points, joint_points


def generate_skeleton_svg(
    joint_accuracies: dict[str, float],
    width: int = 300,
    height: int = 420,
    show_labels: bool = True,
    show_legend: bool = True,
    title: Optional[str] = None,
    dark_mode: bool = True,
    defect_meta: Optional[dict[str, dict]] = None,
) -> str:
    """Generate a MediaPipe-style SVG skeleton with highlighted defects."""
    bg_color = "#0a1628" if dark_mode else "#f8fafc"
    text_color = "#e2e8f0" if dark_mode else "#1e293b"
    line_color = "#4a6fa5" if dark_mode else "#6c88ab"
    base_point_color = "#64748b" if dark_mode else "#94a3b8"
    good_joint_color = "#22c55e"

    view_height = 120 if show_legend else 108
    y_offset = 8 if title else 2

    pose_points, left_hand_points, right_hand_points, joint_points = _build_canvas_geometry(y_offset)
    defect_meta = defect_meta or {}

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 {view_height}" '
        f'width="{width}" height="{height}" style="background:{bg_color};border-radius:16px;">',
        "<defs>",
        # Glow filter for defective joints
        '<filter id="defect-glow" x="-100%" y="-100%" width="300%" height="300%">',
        '<feGaussianBlur stdDeviation="2.0" result="blur"/>',
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>',
        '</filter>',
        # Pulsing animation for critical joints
        '<style>',
        '@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }',
        '.pulse-ring { animation: pulse 1.5s ease-in-out infinite; }',
        '</style>',
        # Gradient for body lines
        '<linearGradient id="bodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        f'<stop offset="0%" style="stop-color:{line_color};stop-opacity:0.9"/>',
        f'<stop offset="100%" style="stop-color:{line_color};stop-opacity:0.7"/>',
        '</linearGradient>',
        "</defs>",
    ]

    if title:
        svg_parts.append(
            f'<text x="50" y="6" text-anchor="middle" '
            f'font-family="system-ui, -apple-system, sans-serif" font-size="3.8" font-weight="700" '
            f'fill="{text_color}">{title}</text>'
        )

    # Draw pose skeleton edges with thicker, more visible lines
    for start_idx, end_idx in _POSE_CONNECTIONS:
        x1, y1 = pose_points[start_idx]
        x2, y2 = pose_points[end_idx]
        svg_parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="url(#bodyGrad)" stroke-width="2.4" stroke-linecap="round"/>'
        )

    # Draw hand edges
    for start_idx, end_idx in _HAND_CONNECTIONS:
        lx1, ly1 = left_hand_points[start_idx]
        lx2, ly2 = left_hand_points[end_idx]
        rx1, ry1 = right_hand_points[start_idx]
        rx2, ry2 = right_hand_points[end_idx]

        svg_parts.append(
            f'<line x1="{lx1}" y1="{ly1}" x2="{lx2}" y2="{ly2}" '
            f'stroke="{line_color}" stroke-width="1.0" stroke-linecap="round" opacity="0.55"/>'
        )
        svg_parts.append(
            f'<line x1="{rx1}" y1="{ry1}" x2="{rx2}" y2="{ry2}" '
            f'stroke="{line_color}" stroke-width="1.0" stroke-linecap="round" opacity="0.55"/>'
        )

    # Draw base pose landmarks (non-tracked joints)
    for idx, (x, y) in pose_points.items():
        radius = 1.4
        if idx in (0, 11, 12, 23, 24):
            radius = 1.7
        svg_parts.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{base_point_color}" opacity="0.85"/>'
        )

    # Draw base hand landmarks
    for hand_points in (left_hand_points, right_hand_points):
        for idx, (x, y) in hand_points.items():
            if idx not in _HAND_BASE_LANDMARKS:
                continue
            radius = 0.9 if idx != 0 else 1.1
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{base_point_color}" opacity="0.65"/>'
            )

    # Overlay tracked joints with accuracy visualization
    for joint_name in TRACKED_JOINTS:
        if joint_name not in joint_points:
            continue

        x, y = joint_points[joint_name]
        is_hand_joint = joint_name in HAND_ACCURACY_JOINTS

        accuracy = joint_accuracies.get(joint_name)
        if accuracy is None:
            accuracy = 0.95
        accuracy = _clamp_accuracy(accuracy)

        color = accuracy_to_color(accuracy)
        base_radius = 1.8 if is_hand_joint else 2.8

        defect_info = defect_meta.get(joint_name)
        
        if defect_info:
            # This joint has a defect - make it highly visible
            severity = str(defect_info.get("severity") or "moderate")
            color = _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS["moderate"])
            glow_color = _SEVERITY_GLOW_COLORS.get(severity, "#fef08a")
            
            # Enlarged radius for defect joints
            radius = base_radius + (1.2 if is_hand_joint else 1.8)
            
            # Outer glow ring (pulsing for critical)
            outer_ring_radius = radius + (3.0 if is_hand_joint else 4.5)
            pulse_class = ' class="pulse-ring"' if severity == "critical" else ''
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{outer_ring_radius}" fill="none" '
                f'stroke="{glow_color}" stroke-width="0.8" opacity="0.4"{pulse_class}/>'
            )
            
            # Inner glow ring
            inner_ring_radius = radius + (1.8 if is_hand_joint else 2.8)
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{inner_ring_radius}" fill="none" '
                f'stroke="{color}" stroke-width="1.4" filter="url(#defect-glow)"/>'
            )
            
            # Main joint circle
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" '
                f'stroke="#ffffff" stroke-width="0.6"/>'
            )
            
            # Rank badge - positioned clearly beside the joint
            rank = defect_info.get("rank")
            if isinstance(rank, int) and rank > 0:
                # Position badge to the side of the joint
                badge_offset_x = 5.5 if x < 50 else -5.5
                badge_offset_y = -1.0
                badge_x = x + badge_offset_x
                badge_y = y + badge_offset_y
                
                # Badge background
                svg_parts.append(
                    f'<circle cx="{badge_x}" cy="{badge_y}" r="2.2" '
                    f'fill="#0f172a" stroke="{color}" stroke-width="0.7"/>'
                )
                # Badge number
                svg_parts.append(
                    f'<text x="{badge_x}" y="{badge_y + 0.7}" text-anchor="middle" '
                    f'font-family="system-ui, -apple-system, sans-serif" font-size="2.2" font-weight="700" '
                    f'fill="#ffffff">{rank}</text>'
                )
            
            # Show joint name label for defect joints (more prominently)
            if show_labels and not is_hand_joint:
                display_name = JOINT_DISPLAY_NAMES.get(joint_name, joint_name.replace("_", " ").title())
                pct = int(round(accuracy * 100.0))
                label_text = f"{display_name}"
                
                # Position label based on joint location
                if y > (55 + y_offset):
                    label_y_pos = y - 6.5
                else:
                    label_y_pos = y + 7.5
                    
                # Background rect for better readability
                svg_parts.append(
                    f'<rect x="{x - 8}" y="{label_y_pos - 2.2}" width="16" height="3.2" '
                    f'rx="1" fill="{bg_color}" opacity="0.85"/>'
                )
                svg_parts.append(
                    f'<text x="{x}" y="{label_y_pos}" text-anchor="middle" '
                    f'font-family="system-ui, -apple-system, sans-serif" font-size="2.0" font-weight="600" '
                    f'fill="{color}">{label_text}</text>'
                )
        else:
            # Normal joint (no defect) - show in green
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{base_radius}" fill="{good_joint_color}" '
                f'stroke="#ffffff" stroke-width="0.5" opacity="0.9"/>'
            )

    # Draw legend at the bottom
    if show_legend:
        legend_y = 112 + y_offset - 6
        
        # Legend background
        svg_parts.append(
            f'<rect x="5" y="{legend_y - 5}" width="90" height="9" rx="2" '
            f'fill="{bg_color}" stroke="{line_color}" stroke-width="0.3" opacity="0.9"/>'
        )
        
        legend_items = [
            ("#22c55e", "Good (85%+)"),
            ("#facc15", "Mild (70-84%)"),
            ("#f97316", "Moderate (55-69%)"),
            ("#ef4444", "Critical (<55%)"),
        ]

        start_x = 10.0
        spacing = 21.5
        for idx, (color, label) in enumerate(legend_items):
            x = start_x + (idx * spacing)
            svg_parts.append(f'<circle cx="{x}" cy="{legend_y}" r="1.6" fill="{color}"/>')
            svg_parts.append(
                f'<text x="{x + 2.5}" y="{legend_y + 0.7}" '
                f'font-family="system-ui, -apple-system, sans-serif" font-size="2.0" fill="{text_color}">{label}</text>'
            )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _extract_problem_joint_maps(problem_joints: list[dict]) -> tuple[dict[str, float], dict[str, dict]]:
    """Normalize report problem_joints into accuracy and defect metadata maps."""
    joint_accuracies: dict[str, float] = {}
    defect_meta: dict[str, dict] = {}

    rank_counter = 0
    for item in (problem_joints or []):
        if not isinstance(item, dict):
            continue

        joint = _normalized_joint_name(str(item.get("joint") or ""))
        if joint not in TRACKED_JOINTS:
            continue

        try:
            error_pct = float(item.get("error_pct", 0.0))
        except (TypeError, ValueError):
            error_pct = 0.0
        accuracy = _clamp_accuracy(1.0 - (error_pct / 100.0))

        if joint in joint_accuracies:
            joint_accuracies[joint] = min(joint_accuracies[joint], accuracy)
        else:
            joint_accuracies[joint] = accuracy

        if joint not in defect_meta:
            rank_counter += 1
            defect_meta[joint] = {
                "rank": rank_counter,
                "severity": str(item.get("severity") or "moderate"),
                "error_pct": error_pct,
            }

    for joint in TRACKED_JOINTS:
        if joint not in joint_accuracies:
            joint_accuracies[joint] = 0.95

    return joint_accuracies, defect_meta


def generate_problem_joints_svg(
    problem_joints: list[dict],
    width: int = 300,
    height: int = 420,
    title: Optional[str] = None,
    dark_mode: bool = True,
) -> str:
    """Generate skeleton SVG from report problem_joints list."""
    joint_accuracies, defect_meta = _extract_problem_joint_maps(problem_joints)

    return generate_skeleton_svg(
        joint_accuracies,
        width=width,
        height=height,
        title=title,
        dark_mode=dark_mode,
        defect_meta=defect_meta,
    )


def generate_problem_joints_png(
    problem_joints: list[dict],
    width: int = 760,
    height: int = 980,
    title: Optional[str] = None,
    dark_mode: bool = True,
) -> bytes:
    """Generate a PNG skeleton image from report problem_joints data."""
    import cv2
    import numpy as np

    width = max(320, int(width))
    height = max(420, int(height))
    show_legend = True
    view_height = 120 if show_legend else 108
    y_offset = 8 if title else 2

    joint_accuracies, defect_meta = _extract_problem_joint_maps(problem_joints)
    pose_points, left_hand_points, right_hand_points, joint_points = _build_canvas_geometry(y_offset)

    def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
        value = str(hex_color or "#000000").strip().lstrip("#")
        if len(value) != 6:
            return (0, 0, 0)
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return (b, g, r)

    def _to_px(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        px = int(round((x / 100.0) * (width - 1)))
        py = int(round((y / float(view_height)) * (height - 1)))
        px = min(max(px, 0), width - 1)
        py = min(max(py, 0), height - 1)
        return px, py

    bg_color = _hex_to_bgr("#0a1628" if dark_mode else "#f8fafc")
    line_color = _hex_to_bgr("#4a6fa5" if dark_mode else "#6c88ab")
    base_point_color = _hex_to_bgr("#64748b" if dark_mode else "#94a3b8")
    white = _hex_to_bgr("#ffffff")
    text_color = _hex_to_bgr("#e2e8f0" if dark_mode else "#1e293b")
    badge_bg = _hex_to_bgr("#0f172a")
    good_joint_color = _hex_to_bgr("#22c55e")

    image = np.full((height, width, 3), bg_color, dtype=np.uint8)

    body_thickness = max(3, int(round(width / 150.0)))
    hand_thickness = max(1, int(round(width / 280.0)))
    pose_point_radius = max(3, int(round(width / 200.0)))
    hand_point_radius = max(2, int(round(width / 320.0)))
    overlay_pose_radius = max(7, int(round(width / 100.0)))
    overlay_hand_radius = max(5, int(round(width / 150.0)))
    ring_thickness = max(3, int(round(width / 260.0)))

    for start_idx, end_idx in _POSE_CONNECTIONS:
        cv2.line(
            image,
            _to_px(pose_points[start_idx]),
            _to_px(pose_points[end_idx]),
            line_color,
            body_thickness,
            lineType=cv2.LINE_AA,
        )

    for start_idx, end_idx in _HAND_CONNECTIONS:
        cv2.line(
            image,
            _to_px(left_hand_points[start_idx]),
            _to_px(left_hand_points[end_idx]),
            line_color,
            hand_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            image,
            _to_px(right_hand_points[start_idx]),
            _to_px(right_hand_points[end_idx]),
            line_color,
            hand_thickness,
            lineType=cv2.LINE_AA,
        )

    for idx, point in pose_points.items():
        radius = pose_point_radius + 1 if idx in (0, 11, 12, 23, 24) else pose_point_radius
        cv2.circle(image, _to_px(point), radius, base_point_color, -1, lineType=cv2.LINE_AA)

    for hand_points in (left_hand_points, right_hand_points):
        for idx, point in hand_points.items():
            if idx not in _HAND_BASE_LANDMARKS:
                continue
            radius = hand_point_radius + 1 if idx == 0 else hand_point_radius
            cv2.circle(image, _to_px(point), radius, base_point_color, -1, lineType=cv2.LINE_AA)

    for joint_name in TRACKED_JOINTS:
        if joint_name not in joint_points:
            continue

        is_hand_joint = joint_name in HAND_ACCURACY_JOINTS
        center = _to_px(joint_points[joint_name])

        accuracy = _clamp_accuracy(joint_accuracies.get(joint_name, 0.95))
        base_radius = overlay_hand_radius if is_hand_joint else overlay_pose_radius

        defect_info = defect_meta.get(joint_name)
        if defect_info:
            # Defect joint - draw prominently with rings
            severity = str(defect_info.get("severity") or "moderate")
            color_hex = _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS["moderate"])
            glow_hex = _SEVERITY_GLOW_COLORS.get(severity, "#fef08a")
            radius = base_radius + (3 if is_hand_joint else 5)

            point_color = _hex_to_bgr(color_hex)
            glow_color = _hex_to_bgr(glow_hex)
            
            # Outer glow ring
            outer_ring_radius = radius + (max(12, int(round(width / 65.0))) if is_hand_joint else max(16, int(round(width / 50.0))))
            cv2.circle(image, center, outer_ring_radius, glow_color, ring_thickness - 1, lineType=cv2.LINE_AA)
            
            # Inner ring
            inner_ring_radius = radius + (max(7, int(round(width / 100.0))) if is_hand_joint else max(10, int(round(width / 80.0))))
            cv2.circle(image, center, inner_ring_radius, point_color, ring_thickness, lineType=cv2.LINE_AA)
            
            # Main joint circle
            cv2.circle(image, center, radius, point_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(image, center, radius, white, 2, lineType=cv2.LINE_AA)

            # Rank badge
            rank = defect_info.get("rank")
            if isinstance(rank, int) and rank > 0:
                badge_radius = max(12, int(round(width / 55.0)))
                offset = max(28, int(round(width / 26.0)))
                label_x = center[0] + offset if center[0] < (width // 2) else center[0] - offset
                label_y = center[1] - max(8, int(round(height / 100.0)))
                label_x = min(max(label_x, badge_radius + 2), width - badge_radius - 2)
                label_y = min(max(label_y, badge_radius + 2), height - badge_radius - 2)

                cv2.circle(image, (label_x, label_y), badge_radius, badge_bg, -1, lineType=cv2.LINE_AA)
                cv2.circle(image, (label_x, label_y), badge_radius, point_color, 2, lineType=cv2.LINE_AA)

                rank_text = str(rank)
                font_scale = max(0.5, width / 1600.0)
                text_thickness = max(1, int(round(width / 450.0)))
                (tw, th), baseline = cv2.getTextSize(rank_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                text_org = (label_x - (tw // 2), label_y + (th // 2) - baseline)
                cv2.putText(
                    image,
                    rank_text,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    white,
                    text_thickness,
                    lineType=cv2.LINE_AA,
                )

            # Draw joint name label for pose joints
            if not is_hand_joint:
                display_name = JOINT_DISPLAY_NAMES.get(joint_name, joint_name.replace("_", " ").title())
                font_scale = max(0.38, width / 2000.0)
                text_thickness = max(1, int(round(width / 550.0)))
                (tw, th), _ = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                
                text_y = center[1] - max(22, int(round(height / 42.0))) if center[1] > (height * 0.5) else center[1] + max(26, int(round(height / 38.0)))
                text_x = min(max(center[0] - (tw // 2), 3), width - tw - 3)
                text_y = min(max(text_y, th + 3), height - 4)
                
                # Draw background for label
                padding = 4
                cv2.rectangle(
                    image, 
                    (text_x - padding, text_y - th - padding), 
                    (text_x + tw + padding, text_y + padding),
                    bg_color, -1
                )
                cv2.putText(
                    image,
                    display_name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    point_color,
                    text_thickness,
                    lineType=cv2.LINE_AA,
                )
        else:
            # Good joint - draw in green
            cv2.circle(image, center, base_radius, good_joint_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(image, center, base_radius, white, 1, lineType=cv2.LINE_AA)

    if title:
        font_scale = max(0.55, width / 1550.0)
        text_thickness = max(1, int(round(width / 520.0)))
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        title_x = max(4, (width - tw) // 2)
        title_y = max(th + 4, int(round((6.0 / view_height) * height)))
        cv2.putText(
            image,
            title,
            (title_x, title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    if show_legend:
        legend_items = [
            ("#22c55e", "Good (85%+)"),
            ("#facc15", "Mild (70-84%)"),
            ("#f97316", "Moderate (55-69%)"),
            ("#ef4444", "Critical (<55%)"),
        ]
        legend_y_px = _to_px((50.0, 112.0 + y_offset - 6))[1]
        font_scale = max(0.42, width / 1900.0)
        text_thickness = max(1, int(round(width / 580.0)))

        # Draw legend background
        legend_bg_h = max(40, int(round(height / 24.0)))
        cv2.rectangle(
            image, 
            (int(width * 0.04), legend_y_px - int(legend_bg_h * 0.5)), 
            (int(width * 0.96), legend_y_px + int(legend_bg_h * 0.5)),
            bg_color, -1
        )
        cv2.rectangle(
            image, 
            (int(width * 0.04), legend_y_px - int(legend_bg_h * 0.5)), 
            (int(width * 0.96), legend_y_px + int(legend_bg_h * 0.5)),
            line_color, 1
        )

        for idx, (color_hex, label) in enumerate(legend_items):
            x = 8.0 + (21.5 * idx)
            cx, cy = _to_px((x, 112.0 + y_offset - 6))
            cv2.circle(image, (cx, cy), max(5, int(round(width / 150.0))), _hex_to_bgr(color_hex), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                image,
                label,
                (cx + max(12, int(round(width / 64.0))), cy + max(5, int(round(height / 220.0)))),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                text_thickness,
                lineType=cv2.LINE_AA,
            )

    encoded_ok, encoded_png = cv2.imencode(".png", image)
    if not encoded_ok:
        raise RuntimeError("Failed to encode skeleton visualization")

    return encoded_png.tobytes()


def generate_skeleton_svg_from_joint_accuracy(
    joint_accuracy: dict[str, float],
    width: int = 300,
    height: int = 420,
    title: Optional[str] = None,
    dark_mode: bool = True,
) -> str:
    """Generate skeleton SVG from a joint-accuracy dict."""
    normalized: dict[str, float] = {}

    for raw_joint_name, raw_accuracy in (joint_accuracy or {}).items():
        joint_name = _normalized_joint_name(str(raw_joint_name))
        if joint_name not in TRACKED_JOINTS:
            continue
        try:
            normalized[joint_name] = _clamp_accuracy(float(raw_accuracy))
        except (TypeError, ValueError):
            continue

    for joint in TRACKED_JOINTS:
        if joint not in normalized:
            normalized[joint] = 0.95

    return generate_skeleton_svg(
        normalized,
        width=width,
        height=height,
        title=title,
        dark_mode=dark_mode,
    )


if __name__ == "__main__":
    sample_problem_joints = [
        {"joint": "right_hand_pinky_curl", "error_pct": 21.4, "severity": "moderate"},
        {"joint": "torso_lean", "error_pct": 21.0, "severity": "moderate"},
        {"joint": "right_hand_ring_curl", "error_pct": 19.7, "severity": "moderate"},
        {"joint": "left_hip", "error_pct": 17.3, "severity": "moderate"},
        {"joint": "right_hip", "error_pct": 16.3, "severity": "moderate"},
    ]

    print(generate_problem_joints_svg(sample_problem_joints, width=380, height=520, title="MediaPipe-style Joint Accuracy"))