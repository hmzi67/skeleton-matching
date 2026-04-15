"""
app.py — Pose Matcher backend.
Auth:  JWT Bearer tokens + Session table in PostgreSQL via Prisma.
Roles: user | admin
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
import uuid
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import bcrypt
import cv2
import jwt
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_file
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from flask_cors import CORS
from prisma.errors import DataError
from prisma.errors import TableNotFoundError
from src.calibration import calibrate_from_skeleton, compute_adaptive_thresholds
from src.exercise_weights import (
    get_weights,
    compute_auto_weights,
    resolve_exercise_name,
    EXERCISE_WEIGHTS,
    FEEDBACK_WEIGHT_THRESHOLD,
)
from src.extractor import extract_skeleton_from_video, load_skeleton
from src.feedback import generate_feedback
from src.filters import LandmarkSmoother, HandLandmarkSmoother
from src.matcher import match_single_frame
from src.normalizer import normalize_skeleton, compute_hand_angles, reconcile_hand_sides
from src.stability import (
    AngleSmoother,
    FeedbackStabilizer,
    JointStatusStabilizer,
    build_rep_feedback,
)
from src.state_machine import ExerciseStateMachine
from src.rep_tracker import FrameScore, RepTracker
from src.report_generator import ReportGenerator
from src.skeleton_svg import (
    generate_skeleton_svg,
    generate_problem_joints_svg,
    generate_problem_joints_png,
    generate_skeleton_svg_from_joint_accuracy,
)
import threading
from werkzeug.utils import secure_filename

try:
    from flask_sock import Sock
except ModuleNotFoundError:
    Sock = None

load_dotenv()
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/posematcher")

from prisma import Prisma

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SECRET_KEY   = os.environ.get("SECRET_KEY", "dev-secret-change-in-production-32")
TOKEN_EXPIRY = timedelta(hours=8)

PENDING_DIR  = Path("data/pending")
APPROVED_DIR = Path("data/ground_truth")
PENDING_DIR.mkdir(parents=True, exist_ok=True)
APPROVED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB limit
SKELETON_CACHE_DIR = Path("data/cache/reference_skeletons")
SKELETON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_REFERENCE_NORM_CACHE_VERSION = 3
REMINDER_FALLBACK_FILE = Path("data/reminders.json")
if not REMINDER_FALLBACK_FILE.exists():
    REMINDER_FALLBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    REMINDER_FALLBACK_FILE.write_text("[]", encoding="utf-8")

POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [24, 26], [26, 28],
    [27, 31], [28, 32], [15, 17], [16, 18], [15, 19], [16, 20],
]

# Minimum visibility confidence for a landmark to be used in angle computation.
# Landmarks below this threshold produce unreliable angles that tank the score.
_VISIBILITY_THRESHOLD = 0.35

_ANGLE_JOINTS = [
    ("left_elbow", 13, 11, 15),
    ("right_elbow", 14, 12, 16),
    ("left_knee", 25, 23, 27),
    ("right_knee", 26, 24, 28),
    ("left_hip", 23, 11, 25),
    ("right_hip", 24, 12, 26),
    ("left_shoulder", 11, 13, 23),
    ("right_shoulder", 12, 14, 24),
]

_BONE_JOINT_MAP: dict[tuple[int, int], str] = {
    (11, 13): "left_shoulder", (13, 15): "left_elbow",
    (12, 14): "right_shoulder", (14, 16): "right_elbow",
    (23, 25): "left_hip",      (25, 27): "left_knee",
    (24, 26): "right_hip",     (26, 28): "right_knee",
    (11, 23): "torso_lean",    (12, 24): "torso_lean",
    (11, 12): "torso_lean",    (23, 24): "torso_lean",
    (27, 31): "left_knee",     (28, 32): "right_knee",
    (15, 17): "left_hand",     (16, 18): "right_hand",
    (15, 19): "left_hand",     (16, 20): "right_hand",
    (15, 21): "left_hand",     (16, 22): "right_hand",
    (17, 19): "left_hand",     (18, 20): "right_hand",
}

_JOINT_LANDMARK_IDX: dict[str, int] = {
    "left_elbow": 13, "right_elbow": 14,
    "left_knee": 25,  "right_knee": 26,
    "left_hip": 23,   "right_hip": 24,
    "left_shoulder": 11, "right_shoulder": 12,
}

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_MODEL_PATHS = {
    "lite": _MODEL_DIR / "pose_landmarker_lite.task",
    "full": _MODEL_DIR / "pose_landmarker_full.task",
    "heavy": _MODEL_DIR / "pose_landmarker_heavy.task",
}
_DEFAULT_MODEL_TIER = os.environ.get("POSE_MODEL_TIER", "lite").strip().lower()
LIVE_MATCH_WINDOW = max(1, int(os.environ.get("LIVE_MATCH_WINDOW", "8")))
LIVE_MATCH_DEBUG_TIMINGS = os.environ.get("LIVE_MATCH_DEBUG_TIMINGS", "0").strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# Live feedback stability tuning (window sizes, EMA, hysteresis, debounce).
LIVE_ANGLE_WINDOW = max(3, int(os.environ.get("LIVE_ANGLE_WINDOW", "7")))
LIVE_ANGLE_EMA_ALPHA = _clamp(float(os.environ.get("LIVE_ANGLE_EMA_ALPHA", "0.3")), 0.2, 0.4)
LIVE_STATUS_HYSTERESIS = float(os.environ.get("LIVE_STATUS_HYSTERESIS", "5.0"))
LIVE_STATUS_MAJORITY = max(3, int(os.environ.get("LIVE_STATUS_MAJORITY", "7")))
LIVE_FEEDBACK_DEBOUNCE = max(3, int(os.environ.get("LIVE_FEEDBACK_DEBOUNCE", "7")))
LIVE_PHASE_STABLE_FRAMES = max(2, int(os.environ.get("LIVE_PHASE_STABLE_FRAMES", "5")))
LIVE_SPEECH_COOLDOWN_MS = max(500, int(os.environ.get("LIVE_SPEECH_COOLDOWN_MS", "2500")))
LIVE_SMOOTH_DEBUG = os.environ.get("LIVE_SMOOTH_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model_path() -> str:
    preferred = _MODEL_PATHS.get(_DEFAULT_MODEL_TIER) or _MODEL_PATHS["full"]
    if preferred.exists():
        return str(preferred)
    for tier in ("full", "lite", "heavy"):
        candidate = _MODEL_PATHS[tier]
        if candidate.exists():
            return str(candidate)
    return str(_MODEL_PATHS["heavy"])


_MODEL_PATH = _resolve_model_path()

_pose_detector = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )
)

# MediaPipe Hands detector for fine-grained hand/finger tracking.
# Uses the Solutions API (no separate model file required).
_hands_detector = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


@dataclass
class TemporalAligner:
    """Tracks the user's progress through the reference sequence independently
    of the reference video's wall-clock position.

    Instead of anchoring to ref_time_ms (which penalises delayed users), this
    tracks the last-matched reference index and advances forward as the user
    progresses. The ref_time_ms is used only for loop detection and as a
    secondary catch-up anchor.
    """
    last_matched_idx: int = 0
    smoothed_offset_ms: float = 0.0
    _prev_ref_time_ms: float = 0.0
    _offset_ema_alpha: float = 0.15
    _forward_window: int = 12
    _backward_tolerance: int = 5

    def compute_search_range(
        self,
        ref_time_ms: float,
        ref_timestamps_ms: list[float],
        ref_len: int,
        base_window: int = 5,
    ) -> tuple[int, int]:
        """Return (start, end) search range prioritising user progress."""
        ref_idx = _closest_reference_index(ref_timestamps_ms, ref_time_ms)

        if self._detect_loop(ref_time_ms):
            self.last_matched_idx = 0
            self.smoothed_offset_ms = 0.0

        progress = self.last_matched_idx
        p_start = max(0, progress - self._backward_tolerance)
        p_end = min(ref_len - 1, progress + self._forward_window)

        r_start = max(0, ref_idx - base_window)
        r_end = min(ref_len - 1, ref_idx + base_window)

        search_start = min(p_start, r_start)
        search_end = max(p_end, r_end)

        self._prev_ref_time_ms = ref_time_ms
        return search_start, search_end

    def update_after_match(
        self,
        best_idx: int,
        ref_time_ms: float,
        ref_timestamps_ms: list[float],
    ) -> None:
        """Update progress anchor and offset estimate after a successful match."""
        if best_idx >= self.last_matched_idx - self._backward_tolerance:
            self.last_matched_idx = best_idx

        ref_idx = _closest_reference_index(ref_timestamps_ms, ref_time_ms)
        if ref_idx < len(ref_timestamps_ms) and best_idx < len(ref_timestamps_ms):
            offset = ref_timestamps_ms[ref_idx] - ref_timestamps_ms[best_idx]
            self.smoothed_offset_ms = (
                (1 - self._offset_ema_alpha) * self.smoothed_offset_ms
                + self._offset_ema_alpha * offset
            )

    def _detect_loop(self, ref_time_ms: float) -> bool:
        """Detect when the reference video loops (time jumps backward)."""
        if self._prev_ref_time_ms > 0 and ref_time_ms < self._prev_ref_time_ms - 1000:
            return True
        return False

    @property
    def offset_ms(self) -> float:
        return self.smoothed_offset_ms

    @property
    def timing_status(self) -> str:
        if abs(self.smoothed_offset_ms) < 300:
            return "synced"
        if self.smoothed_offset_ms > 0:
            return "behind" if self.smoothed_offset_ms < 2000 else "far_behind"
        return "ahead"


@dataclass
class LiveSessionState:
    user_id: str
    ref_video_id: str
    exercise: str
    gt_norm: list[dict]
    ref_timestamps_ms: list[float]
    auto_weights: dict[str, float] = field(default_factory=dict)
    adaptive_thresholds: dict[str, dict[str, float]] | None = None
    primary_joint: str = "left_knee"
    primary_min: float = 120.0
    primary_max: float = 170.0
    smoothed_score: float | None = None
    last_live_angles: dict[str, float] | None = None
    last_live_ts_ms: float | None = None
    no_pose_frame_count: int = 0
    idle_frame_count: int = 0
    target_reps: int | None = None
    temporal_aligner: TemporalAligner = field(default_factory=TemporalAligner)
    phase_machine: ExerciseStateMachine = field(default_factory=ExerciseStateMachine)
    landmark_smoother: LandmarkSmoother = field(
        default_factory=lambda: LandmarkSmoother(min_cutoff=1.5, beta=0.15),
    )
    hand_smoother: HandLandmarkSmoother = field(
        default_factory=lambda: HandLandmarkSmoother(min_cutoff=2.0, beta=0.1),
    )
    angle_smoother: AngleSmoother = field(
        default_factory=lambda: AngleSmoother(
            window_size=LIVE_ANGLE_WINDOW,
            ema_alpha=LIVE_ANGLE_EMA_ALPHA,
            debug=LIVE_SMOOTH_DEBUG,
        ),
    )
    status_stabilizer: JointStatusStabilizer = field(
        default_factory=lambda: JointStatusStabilizer(
            hysteresis_buffer=LIVE_STATUS_HYSTERESIS,
            majority_window=LIVE_STATUS_MAJORITY,
        ),
    )
    feedback_stabilizer: FeedbackStabilizer = field(
        default_factory=lambda: FeedbackStabilizer(
            debounce_frames=LIVE_FEEDBACK_DEBOUNCE,
            phase_stable_frames=LIVE_PHASE_STABLE_FRAMES,
            speech_cooldown_ms=LIVE_SPEECH_COOLDOWN_MS,
        ),
    )

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)
sock = Sock(app) if Sock is not None else None

# ---------------------------------------------------------------------------
# Prisma client (one shared instance, connected per request)
# ---------------------------------------------------------------------------

db = Prisma()
_reference_norm_cache: dict[str, list[dict]] = {}
_http_live_states: dict[str, LiveSessionState] = {}


@app.before_request
def _connect():
    if not db.is_connected():
        try:
            db.connect()
        except Exception as exc:
            abort(503, description=f"Database unavailable: {exc}")


# ---------------------------------------------------------------------------
# Seed default admin on first run
# ---------------------------------------------------------------------------

def _seed_admin() -> None:
    try:
        with Prisma() as client:
            existing = client.user.find_first(where={"role": "admin"})
            if not existing:
                client.user.create(data={
                    "id":       str(uuid.uuid4()),
                    "username": "admin",
                    "email":    "admin@posematcher.local",
                    "password": bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode(),
                    "role":     "admin",
                })
                print("[seed] Default admin created — username: admin / password: admin123")
    except Exception as exc:
        print(f"[seed] Skipping admin seed: {exc}")

_seed_admin()


# ---------------------------------------------------------------------------
# JWT + Session helpers
# ---------------------------------------------------------------------------

def _make_token(user) -> str:
    payload = {
        "sub":      user.id,
        "username": user.username,
        "role":     str(user.role),
        "exp":      datetime.now(timezone.utc) + TOKEN_EXPIRY,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def _decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def _get_bearer() -> str | None:
    auth = request.headers.get("Authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else None


def _decode_token_and_session(token: str) -> dict | None:
    payload = _decode_token(token)
    if not payload:
        return None
    session = db.session.find_unique(where={"token": token})
    if not session or session.expiresAt < datetime.now(timezone.utc):
        return None
    return payload


def _decode_data_url_to_bgr(image_data_url: str) -> np.ndarray | None:
    if not image_data_url:
        return None
    try:
        b64_part = image_data_url.split(",", 1)[1] if "," in image_data_url else image_data_url
        img_bytes = base64.b64decode(b64_part)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _extract_pose_and_hands(
    image_bgr: np.ndarray,
) -> "tuple[list[dict] | None, dict[str, list[dict] | None]]":
    """Detect pose + hand landmarks from a single BGR frame.

    Returns
    -------
    pose_landmarks : list[dict] | None
        33 pose landmark dicts, or None when no person detected.
    hand_landmarks : dict
        ``{"left": [...21 dicts...] | None, "right": [...21 dicts...] | None}``
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ---- Pose ----
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pose_results = _pose_detector.detect(mp_image)
    pose_landmarks: list[dict] | None = None
    if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
        pose_landmarks = [
            {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(lm.visibility),
            }
            for lm in pose_results.pose_landmarks[0]
        ]

    # ---- Hands ----
    rgb.flags.writeable = True
    hand_results = _hands_detector.process(rgb)
    rgb.flags.writeable = False

    hand_lms_dict: dict[str, list[dict] | None] = {"left": None, "right": None}
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for handedness, hand_lms in zip(
            hand_results.multi_handedness,
            hand_results.multi_hand_landmarks,
        ):
            label = handedness.classification[0].label.lower()
            hand_lms_dict[label] = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": 1.0}
                for lm in hand_lms.landmark
            ]

    hand_lms_dict = reconcile_hand_sides(pose_landmarks, hand_lms_dict)

    return pose_landmarks, hand_lms_dict


# Keep legacy name for any external callers.
def _extract_landmarks(image_bgr: np.ndarray) -> "list[dict] | None":
    pose_lms, _ = _extract_pose_and_hands(image_bgr)
    return pose_lms


def _vec(a: dict, b: dict) -> tuple[float, float, float]:
    return (b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"])


def _dot(v1: tuple[float, float, float], v2: tuple[float, float, float]) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _angle(a: dict, b: dict, c: dict) -> float:
    v1 = _vec(b, a)
    v2 = _vec(b, c)
    mag = _norm(v1) * _norm(v2)
    if mag < 1e-8:
        return 0.0
    cosv = max(-1.0, min(1.0, _dot(v1, v2) / mag))
    return math.degrees(math.acos(cosv))


def _landmarks_visible(landmarks: list[dict], *indices: int) -> bool:
    """Return True only if ALL listed landmark indices have visibility >= threshold."""
    for idx in indices:
        if idx >= len(landmarks):
            return False
        vis = landmarks[idx].get("visibility", 0.0)
        if vis < _VISIBILITY_THRESHOLD:
            return False
    return True


def _extract_angles(
    landmarks: list[dict],
    hand_landmarks: "dict[str, list[dict] | None] | None" = None,
) -> dict[str, float]:
    """Compute joint angles from pose (and optionally hand) landmarks.

    Joints whose constituent landmarks have visibility below
    ``_VISIBILITY_THRESHOLD`` are skipped entirely so that poorly-detected
    body parts (e.g. hips/knees when only the upper body is in frame)
    do not inject garbage angles that tank the score.
    """
    angles: dict[str, float] = {}
    for name, vertex, a, c in _ANGLE_JOINTS:
        if _landmarks_visible(landmarks, vertex, a, c):
            angles[name] = _angle(landmarks[a], landmarks[vertex], landmarks[c])

    # Torso lean — requires both hips and both shoulders to be visible.
    if _landmarks_visible(landmarks, 11, 12, 23, 24):
        hip_center = {
            "x": (landmarks[23]["x"] + landmarks[24]["x"]) / 2.0,
            "y": (landmarks[23]["y"] + landmarks[24]["y"]) / 2.0,
            "z": (landmarks[23]["z"] + landmarks[24]["z"]) / 2.0,
        }
        shoulder_center = {
            "x": (landmarks[11]["x"] + landmarks[12]["x"]) / 2.0,
            "y": (landmarks[11]["y"] + landmarks[12]["y"]) / 2.0,
            "z": (landmarks[11]["z"] + landmarks[12]["z"]) / 2.0,
        }

        spine = _vec(hip_center, shoulder_center)
        vertical = (0.0, -1.0, 0.0)
        mag = _norm(spine) * _norm(vertical)
        if mag < 1e-8:
            angles["torso_lean"] = 0.0
        else:
            cosv = max(-1.0, min(1.0, _dot(spine, vertical) / mag))
            angles["torso_lean"] = math.degrees(math.acos(cosv))

    # Hand angles — finger curl per detected hand.
    if hand_landmarks:
        hand_landmarks = reconcile_hand_sides(landmarks, hand_landmarks)
        left_hand  = hand_landmarks.get("left")
        right_hand = hand_landmarks.get("right")
        if left_hand:
            angles.update(compute_hand_angles(left_hand, "left_hand"))
        if right_hand:
            angles.update(compute_hand_angles(right_hand, "right_hand"))

    return angles


def _compute_match_score(gt_angles: dict[str, float], user_angles: dict[str, float]) -> tuple[int, dict[str, float]]:
    joint_diffs: dict[str, float] = {}
    if not gt_angles:
        return 0, joint_diffs

    total = 0.0
    for joint, gt_val in gt_angles.items():
        diff = abs(user_angles.get(joint, 0.0) - gt_val)
        joint_diffs[joint] = round(diff, 2)
        total += diff

    avg_diff = total / len(gt_angles)
    score = max(0.0, min(100.0, 100.0 - (avg_diff / 45.0) * 100.0))
    return int(round(score)), joint_diffs


def _resolve_video_path(video_id: str) -> Path | None:
    record = db.video.find_unique(where={"id": video_id})
    if not record:
        return None
    pending_path = PENDING_DIR / record.storedName
    if pending_path.exists():
        return pending_path
    approved_path = APPROVED_DIR / record.storedName
    if approved_path.exists():
        return approved_path
    return None


def _load_or_extract_reference_norm(video_id: str) -> list[dict]:
    cache_key = f"v{_REFERENCE_NORM_CACHE_VERSION}:{video_id}"
    if cache_key in _reference_norm_cache:
        return _reference_norm_cache[cache_key]

    cache_path = SKELETON_CACHE_DIR / f"{video_id}.normalized.v{_REFERENCE_NORM_CACHE_VERSION}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _reference_norm_cache[cache_key] = data
            return data

    video_path = _resolve_video_path(video_id)
    if video_path is None:
        raise FileNotFoundError("Reference video not found")

    raw_cache_path = video_path.with_suffix(".json")
    if raw_cache_path.exists():
        raw_skeleton = load_skeleton(str(raw_cache_path))
    else:
        raw_skeleton = extract_skeleton_from_video(str(video_path))

    normalized = normalize_skeleton(raw_skeleton)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f)

    _reference_norm_cache[cache_key] = normalized
    return normalized


def _closest_reference_index(ref_timestamps_ms: list[float], ref_time_ms: float) -> int:
    if not ref_timestamps_ms:
        return 0
    idx = bisect_left(ref_timestamps_ms, ref_time_ms)
    if idx <= 0:
        return 0
    if idx >= len(ref_timestamps_ms):
        return len(ref_timestamps_ms) - 1
    left = ref_timestamps_ms[idx - 1]
    right = ref_timestamps_ms[idx]
    return idx - 1 if abs(ref_time_ms - left) <= abs(right - ref_time_ms) else idx


def _load_reminder_fallback_records() -> list[dict]:
    try:
        raw = REMINDER_FALLBACK_FILE.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _save_reminder_fallback_records(records: list[dict]) -> None:
    REMINDER_FALLBACK_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


_CURATED_PRIMARY_ANGLES: dict[str, list[str]] = {
    "squat":             ["left_knee", "right_knee"],
    "lunge":             ["left_knee", "right_knee"],
    "deadlift":          ["left_hip", "right_hip"],
    "pushup":            ["left_elbow", "right_elbow"],
    "shoulder_press":    ["left_elbow", "right_elbow"],
    "bicep_curl":        ["left_elbow", "right_elbow"],
    "shoulder_rotation": ["left_shoulder", "right_shoulder"],
    "wrist_curl":        ["left_hand_middle_curl", "right_hand_middle_curl"],
    "finger_exercise":   ["left_hand_index_curl", "right_hand_index_curl"],
}

_SYMMETRY_MAP: dict[str, str] = {
    "left_knee": "right_knee", "right_knee": "left_knee",
    "left_hip": "right_hip", "right_hip": "left_hip",
    "left_elbow": "right_elbow", "right_elbow": "left_elbow",
    "left_shoulder": "right_shoulder", "right_shoulder": "left_shoulder",
}


def _primary_angle_for_exercise(
    exercise: str,
    angles: dict[str, float],
    primary_joint: str | None = None,
) -> float:
    """Return the primary angle for phase detection.

    Uses curated joint list for known exercises; for unknown exercises
    uses the ROM-derived ``primary_joint`` (averaging with its symmetry
    counterpart when available).

    Skips joints whose live value is ``None`` or ``0.0`` so that a
    temporarily-undetected hand does not freeze the state machine.
    """
    def _present(v):
        return v is not None and abs(v) > 1e-6

    curated = _CURATED_PRIMARY_ANGLES.get(exercise.lower())
    if curated:
        vals = [angles.get(j) for j in curated]
        good = [v for v in vals if _present(v)]
        if good:
            return sum(good) / len(good)
        # Nothing detected — fall through so the caller can decide what
        # to do (we return 0.0 only as an absolute last resort).

    if primary_joint:
        val = angles.get(primary_joint)
        sym = _SYMMETRY_MAP.get(primary_joint)
        sym_val = angles.get(sym) if sym else None
        if _present(val) and _present(sym_val):
            return (val + sym_val) / 2.0
        if _present(val):
            return val
        if _present(sym_val):
            return sym_val

    # Final fallback: average of knees (legacy behaviour for unknown
    # body exercises).
    lk = angles.get("left_knee")
    rk = angles.get("right_knee")
    if _present(lk) and _present(rk):
        return (lk + rk) / 2.0
    if _present(lk):
        return lk
    if _present(rk):
        return rk
    return 0.0


def _state_machine_for_exercise(exercise: str) -> ExerciseStateMachine:
    if exercise in {"squat", "lunge"}:
        return ExerciseStateMachine(down_threshold=120.0, up_threshold=155.0, hold_frames=2)
    if exercise == "deadlift":
        return ExerciseStateMachine(down_threshold=100.0, up_threshold=150.0, hold_frames=2)
    if exercise == "pushup":
        return ExerciseStateMachine(down_threshold=95.0, up_threshold=155.0, hold_frames=2)
    if exercise == "shoulder_press":
        return ExerciseStateMachine(down_threshold=90.0, up_threshold=150.0, hold_frames=2)
    if exercise == "bicep_curl":
        # Elbow flexed ~50° at top of curl, ~160° when arm extended.
        return ExerciseStateMachine(down_threshold=70.0, up_threshold=150.0, hold_frames=2)
    if exercise == "finger_exercise":
        # Index-finger curl angle: ~60° when closed (fist), ~160° extended.
        return ExerciseStateMachine(down_threshold=80.0, up_threshold=150.0, hold_frames=2)
    if exercise == "wrist_curl":
        return ExerciseStateMachine(down_threshold=80.0, up_threshold=150.0, hold_frames=2)
    return ExerciseStateMachine()


def _compute_live_velocities(
    state: LiveSessionState,
    live_angles: dict[str, float],
    now_ms: float,
) -> dict[str, float]:
    if state.last_live_angles is None or state.last_live_ts_ms is None:
        state.last_live_angles = dict(live_angles)
        state.last_live_ts_ms = now_ms
        return {joint: 0.0 for joint in live_angles}

    dt_s = max((now_ms - state.last_live_ts_ms) / 1000.0, 1e-3)
    velocities: dict[str, float] = {}
    for joint, curr in live_angles.items():
        prev = state.last_live_angles.get(joint, curr)
        velocities[joint] = round((curr - prev) / dt_s, 2)

    state.last_live_angles = dict(live_angles)
    state.last_live_ts_ms = now_ms
    return velocities


# Hand finger-curl joint names per side — used to derive an aggregate hand status.
_HAND_CURL_JOINTS: dict[str, list[str]] = {
    "left_hand":  ["left_hand_thumb_curl",  "left_hand_index_curl",  "left_hand_middle_curl",
                   "left_hand_ring_curl",   "left_hand_pinky_curl"],
    "right_hand": ["right_hand_thumb_curl", "right_hand_index_curl", "right_hand_middle_curl",
                   "right_hand_ring_curl",  "right_hand_pinky_curl"],
}
_STATUS_RANK = {"good": 0, "warning": 1, "bad": 2}


def _aggregate_hand_status(hand_key: str, best_match: dict,
                           exercise_weights: dict[str, float] | None) -> str:
    """Return the worst finger-curl status for a hand (left_hand / right_hand)."""
    curl_joints = _HAND_CURL_JOINTS.get(hand_key, [])
    relevant = [j for j in curl_joints
                if exercise_weights is None
                or exercise_weights.get(j, 0.0) >= FEEDBACK_WEIGHT_THRESHOLD]
    worst = "good"
    for j in relevant:
        if j in best_match and isinstance(best_match[j], dict):
            s = best_match[j].get("status", "good")
            if _STATUS_RANK.get(s, 0) > _STATUS_RANK.get(worst, 0):
                worst = s
    return worst


def _compute_bone_statuses(
    best_match: dict,
    exercise_weights: dict[str, float] | None = None,
) -> dict[str, str]:
    bone_statuses: dict[str, str] = {}
    for bone in POSE_CONNECTIONS:
        key = f"{bone[0]}-{bone[1]}"
        joint_name = _BONE_JOINT_MAP.get((bone[0], bone[1])) or _BONE_JOINT_MAP.get((bone[1], bone[0]))
        # Suppress bones whose underlying joint is irrelevant to the exercise
        # so the live overlay doesn't draw red outlines on the torso/legs
        # during a hand exercise.
        if (joint_name
                and exercise_weights is not None
                and exercise_weights.get(joint_name, 0.0) < FEEDBACK_WEIGHT_THRESHOLD):
            # For hand bones, check the individual curl joints instead of the
            # aggregate key (which won't be in exercise_weights directly).
            if joint_name in _HAND_CURL_JOINTS:
                status = _aggregate_hand_status(joint_name, best_match, exercise_weights)
                bone_statuses[key] = status
            else:
                bone_statuses[key] = "good"
            continue
        if joint_name in _HAND_CURL_JOINTS:
            bone_statuses[key] = _aggregate_hand_status(joint_name, best_match, exercise_weights)
        elif joint_name and joint_name in best_match and isinstance(best_match[joint_name], dict):
            bone_statuses[key] = best_match[joint_name].get("status", "good")
        else:
            bone_statuses[key] = "good"
    return bone_statuses


_JOINT_STATUS_WEIGHT_THRESHOLD = FEEDBACK_WEIGHT_THRESHOLD  # shared with feedback.py


def _compute_joint_statuses(
    best_match: dict,
    exercise_weights: dict[str, float] | None = None,
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for joint_name in _JOINT_LANDMARK_IDX:
        # Suppress irrelevant joints — mark as "good" so the frontend never
        # adds them to jointIssueTracker or lights them up as errors.
        if (exercise_weights is not None
                and exercise_weights.get(joint_name, 0.0) < _JOINT_STATUS_WEIGHT_THRESHOLD):
            statuses[joint_name] = "good"
            continue
        if joint_name in best_match and isinstance(best_match[joint_name], dict):
            statuses[joint_name] = best_match[joint_name].get("status", "good")
    if "torso_lean" in best_match and isinstance(best_match["torso_lean"], dict):
        if (exercise_weights is None
                or exercise_weights.get("torso_lean", 0.0) >= _JOINT_STATUS_WEIGHT_THRESHOLD):
            statuses["torso_lean"] = best_match["torso_lean"].get("status", "good")

    # Aggregate hand finger-curl statuses into a single per-hand status so the
    # frontend can colour the wrist/hand landmark red when the hand is wrong.
    for hand_key, curl_joints in _HAND_CURL_JOINTS.items():
        relevant = [j for j in curl_joints
                    if exercise_weights is None
                    or exercise_weights.get(j, 0.0) >= _JOINT_STATUS_WEIGHT_THRESHOLD]
        if not relevant:
            continue
        worst = "good"
        for j in relevant:
            if j in best_match and isinstance(best_match[j], dict):
                s = best_match[j].get("status", "good")
                if _STATUS_RANK.get(s, 0) > _STATUS_RANK.get(worst, 0):
                    worst = s
        statuses[hand_key] = worst

    return statuses


def _compute_correction_arrows(
    best_match: dict,
    live_landmarks: list[dict],
    exercise_weights: dict[str, float] | None = None,
) -> list[dict]:
    joint_errors: list[tuple[str, float]] = []
    for joint_name, idx in _JOINT_LANDMARK_IDX.items():
        if (exercise_weights is not None
                and exercise_weights.get(joint_name, 0.0) < _JOINT_STATUS_WEIGHT_THRESHOLD):
            continue  # skip irrelevant joints
        if joint_name in best_match and isinstance(best_match[joint_name], dict):
            data = best_match[joint_name]
            if data.get("status") != "good":
                joint_errors.append((joint_name, data.get("diff", 0.0)))

    joint_errors.sort(key=lambda x: abs(x[1]), reverse=True)
    arrows: list[dict] = []
    for joint_name, diff in joint_errors[:3]:
        idx = _JOINT_LANDMARK_IDX.get(joint_name)
        if idx is None or idx >= len(live_landmarks):
            continue
        lm = live_landmarks[idx]
        arrow_mag = min(0.08, max(0.03, abs(diff) * 0.001))
        dx, dy = 0.0, 0.0
        if "knee" in joint_name or "hip" in joint_name:
            dy = arrow_mag if diff > 0 else -arrow_mag
        elif "elbow" in joint_name or "shoulder" in joint_name:
            side = -1 if "left" in joint_name else 1
            dx = side * arrow_mag if diff < 0 else -side * arrow_mag
        arrows.append({
            "landmark_idx": idx,
            "x": lm["x"],
            "y": lm["y"],
            "dx": round(dx, 4),
            "dy": round(dy, 4),
            "joint": joint_name,
            "status": best_match[joint_name].get("status", "warning"),
        })
    return arrows


def _live_session_key(user_id: str, ref_id: str, exercise: str) -> str:
    return f"{user_id}:{ref_id}:{exercise}"


_CURATED_STATE_MACHINE_EXERCISES: frozenset[str] = frozenset({
    "squat", "lunge", "deadlift", "pushup", "shoulder_press",
    "bicep_curl", "finger_exercise", "wrist_curl",
})


def _build_session_config(
    gt_norm: list[dict], exercise: str,
) -> dict:
    """Derive auto-weights, adaptive thresholds, primary joint, and
    phase-machine parameters from the reference video ROM.

    Curated presets override auto-weights for known exercises. The
    exercise name is passed through :func:`resolve_exercise_name` so
    user/DB variants like ``"hand exercise"`` still hit the curated
    ``finger_exercise`` preset.
    """
    auto = compute_auto_weights(gt_norm)
    resolved = resolve_exercise_name(exercise)

    if resolved in EXERCISE_WEIGHTS and resolved != "default":
        weights = get_weights(resolved)
    else:
        weights = auto["weights"]

    rom_profile = calibrate_from_skeleton(gt_norm)
    adaptive_thresholds = compute_adaptive_thresholds(rom_profile)

    primary_joint = auto["primary_joint"]
    primary_min = auto["primary_min"]
    primary_max = auto["primary_max"]

    # If the resolved exercise has a curated primary angle, prefer the
    # first entry so ROM thresholds below are computed from the joint
    # the state machine will actually read.
    curated_primary = _CURATED_PRIMARY_ANGLES.get(resolved)
    if curated_primary:
        # Try to find ROM of the curated joint in the reference.
        rom_min_map: dict[str, float] = {}
        rom_max_map: dict[str, float] = {}
        for frame in gt_norm:
            for j in curated_primary:
                val = frame.get("angles", {}).get(j)
                if val is None:
                    continue
                rom_min_map[j] = min(rom_min_map.get(j, 360.0), val)
                rom_max_map[j] = max(rom_max_map.get(j, 0.0), val)
        if rom_min_map:
            primary_joint = curated_primary[0]
            primary_min = min(rom_min_map.values())
            primary_max = max(rom_max_map.values())

    rom_range = primary_max - primary_min
    down_thresh = primary_min + rom_range * 0.20
    up_thresh = primary_max - rom_range * 0.20

    if resolved in _CURATED_STATE_MACHINE_EXERCISES:
        phase_machine = _state_machine_for_exercise(resolved)
    else:
        phase_machine = ExerciseStateMachine(
            down_threshold=down_thresh,
            up_threshold=up_thresh,
            hold_frames=2,
        )

    return {
        "auto_weights": weights,
        "adaptive_thresholds": adaptive_thresholds,
        "primary_joint": primary_joint,
        "primary_min": primary_min,
        "primary_max": primary_max,
        "phase_machine": phase_machine,
        "resolved_exercise": resolved,
    }


# Bump this constant whenever weight-selection / feedback logic changes so
# in-memory sessions from a previous app start are discarded instead of
# silently reusing stale weights.
LIVE_STATE_VERSION = "2026-04-09-feedback-stability-v1"


def _get_or_create_live_state(user_id: str, ref_id: str, exercise: str) -> LiveSessionState:
    key = _live_session_key(user_id, ref_id, exercise)
    existing = _http_live_states.get(key)
    if existing is not None and getattr(existing, "state_version", None) == LIVE_STATE_VERSION:
        return existing
    if existing is not None:
        # Stale session from an older weight-logic version — drop it so
        # the fix takes effect on the very next frame.
        _http_live_states.pop(key, None)

    gt_norm = _load_or_extract_reference_norm(ref_id)
    ref_timestamps_ms = [float(frame.get("timestamp_ms", 0.0)) for frame in gt_norm]
    cfg = _build_session_config(gt_norm, exercise)
    state = LiveSessionState(
        user_id=user_id,
        ref_video_id=ref_id,
        exercise=cfg.get("resolved_exercise", exercise),
        gt_norm=gt_norm,
        ref_timestamps_ms=ref_timestamps_ms,
        auto_weights=cfg["auto_weights"],
        adaptive_thresholds=cfg["adaptive_thresholds"],
        primary_joint=cfg["primary_joint"],
        primary_min=cfg["primary_min"],
        primary_max=cfg["primary_max"],
        phase_machine=cfg["phase_machine"],
    )
    state.state_version = LIVE_STATE_VERSION  # type: ignore[attr-defined]
    _http_live_states[key] = state
    return state


def _process_live_frame_message(
    state: LiveSessionState,
    live_frame: str,
    ref_time_ms: float,
) -> dict:
    _NO_POSE_RESET_FRAME_LIMIT = 4

    total_t0 = time.perf_counter()
    decode_t0 = time.perf_counter()
    live_img = _decode_data_url_to_bgr(live_frame)
    decode_ms = (time.perf_counter() - decode_t0) * 1000.0
    if live_img is None:
        return {"type": "error", "error": "invalid_live_frame"}

    detect_t0 = time.perf_counter()
    raw_landmarks, raw_hand_landmarks = _extract_pose_and_hands(live_img)
    detect_ms = (time.perf_counter() - detect_t0) * 1000.0
    if not raw_landmarks:
        state.no_pose_frame_count += 1
        if state.no_pose_frame_count >= _NO_POSE_RESET_FRAME_LIMIT:
            state.landmark_smoother.reset()
            state.hand_smoother.reset()
            state.angle_smoother.reset()
            state.status_stabilizer.reset()
            state.feedback_stabilizer.reset()
            state.smoothed_score = None
            state.last_live_angles = None
            state.last_live_ts_ms = None
            state.no_pose_frame_count = _NO_POSE_RESET_FRAME_LIMIT
        payload = {
            "type": "no_pose",
            "pose_detected": False,
            "match_score": None,
            "connections": POSE_CONNECTIONS,
            "live_landmarks": None,
            "ref_landmarks": None,
            "joint_statuses": {},
            "bone_statuses": {},
            "correction_arrows": [],
            "coaching_text": "Step fully into frame",
            "speech_text": None,
            "feedback": {
                "headline": "No pose detected",
                "priority_fix": "Step fully into frame",
                "top_joint_feedback": [],
                "velocity_feedback": [],
                "symmetry_warnings": [],
            },
            "phase": state.phase_machine.phase,
            "rep_count": int(state.phase_machine.rep_count),
        }
        if LIVE_MATCH_DEBUG_TIMINGS:
            payload["timings"] = {
                "decode_ms": round(decode_ms, 2),
                "detect_ms": round(detect_ms, 2),
                "match_ms": 0.0,
                "total_ms": round((time.perf_counter() - total_t0) * 1000.0, 2),
            }
        return payload

    state.no_pose_frame_count = 0

    t_now = time.time()
    live_landmarks = state.landmark_smoother.smooth(t_now, raw_landmarks)
    live_hand_landmarks = state.hand_smoother.smooth(t_now, raw_hand_landmarks)
    raw_angles = _extract_angles(live_landmarks, live_hand_landmarks)
    # Temporal smoothing: sliding window mean + EMA to reduce jitter.
    live_angles = state.angle_smoother.update(raw_angles)
    now_ms = t_now * 1000.0

    _IDLE_ANGLE_THRESHOLD = 3.0
    _IDLE_FRAME_LIMIT = 10

    if state.last_live_angles is not None:
        total_angle_change = sum(
            abs(live_angles.get(j, 0.0) - state.last_live_angles.get(j, 0.0))
            for j in live_angles
        )
        if total_angle_change < _IDLE_ANGLE_THRESHOLD:
            state.idle_frame_count += 1
        else:
            state.idle_frame_count = 0
    user_is_idle = state.idle_frame_count >= _IDLE_FRAME_LIMIT

    _compute_live_velocities(state, live_angles, now_ms)

    user_frame = {
        "angles": live_angles,
    }

    match_t0 = time.perf_counter()

    aligner = state.temporal_aligner
    search_start, search_end = aligner.compute_search_range(
        ref_time_ms, state.ref_timestamps_ms,
        len(state.gt_norm), base_window=LIVE_MATCH_WINDOW,
    )

    best_idx = aligner.last_matched_idx
    best_match = None
    best_score = -1.0

    for idx in range(search_start, search_end + 1):
        candidate = state.gt_norm[idx]
        candidate_match = match_single_frame(
            candidate,
            user_frame,
            exercise_weights=state.auto_weights,
            adaptive_thresholds=state.adaptive_thresholds,
        )
        candidate_score = float(candidate_match.get("overall_score", 0.0))
        if candidate_score > best_score:
            best_score = candidate_score
            best_idx = idx
            best_match = candidate_match

    if not best_match:
        return {"type": "error", "error": "matching_failed"}

    # Guard against inflated scores while the user stays mostly static.
    # HOLD can be legitimately stable, so allow a slightly higher cap there.
    if user_is_idle:
        _IDLE_READY_SCORE_CAP = 55.0
        _IDLE_HOLD_SCORE_CAP = 75.0
        idle_cap = (
            _IDLE_HOLD_SCORE_CAP
            if state.phase_machine.phase == "HOLD"
            else _IDLE_READY_SCORE_CAP
        )
        best_score = min(best_score, idle_cap)
        best_match["overall_score"] = best_score

    aligner.update_after_match(best_idx, ref_time_ms, state.ref_timestamps_ms)
    if state.smoothed_score is None:
        state.smoothed_score = best_score
    else:
        state.smoothed_score = 0.5 * best_score + 0.5 * state.smoothed_score
    best_match["overall_score"] = state.smoothed_score

    # Threshold buffering + majority voting for stable joint statuses.
    best_match = state.status_stabilizer.stabilize_match(
        best_match, adaptive_thresholds=state.adaptive_thresholds,
    )

    summary_stub = {
        "overall_score": state.smoothed_score,
        "per_joint_avg_error": {},
        "worst_joint": "",
        "best_joint": "",
        "frame_scores": [],
    }
    primary_angle = _primary_angle_for_exercise(
        state.exercise, live_angles, primary_joint=state.primary_joint,
    )
    joint_statuses = _compute_joint_statuses(best_match, state.auto_weights)
    phase = state.phase_machine.update(primary_angle, state.smoothed_score,
                                       joint_statuses=joint_statuses)
    rep_completed = state.phase_machine._last_completed_rep
    rep_summary_text = None
    if rep_completed is not None:
        rep_score = state.phase_machine.rep_scores[-1] if state.phase_machine.rep_scores else 0.0
        rep_issues = state.phase_machine.rep_joint_issues[-1] if state.phase_machine.rep_joint_issues else {}
        rep_summary_text = build_rep_feedback(rep_completed, rep_score, rep_issues)

    feedback = generate_feedback(
        summary_stub, best_match,
        exercise=state.exercise,
        phase=phase,
        exercise_weights=state.auto_weights,
    )

    gt_frame = state.gt_norm[best_idx]

    display_ref_idx = _closest_reference_index(state.ref_timestamps_ms, ref_time_ms)
    display_ref_frame = state.gt_norm[display_ref_idx]

    joint_statuses = _compute_joint_statuses(best_match, state.auto_weights)
    bone_statuses = _compute_bone_statuses(best_match, state.auto_weights)
    correction_arrows = _compute_correction_arrows(best_match, live_landmarks, state.auto_weights)

    joint_fb = feedback.get("joint_feedback", [])
    coaching_text = joint_fb[0]["instruction"] if joint_fb else (
        "Great form!" if state.smoothed_score >= 80 else feedback.get("priority_fix", "")
    )

    if user_is_idle:
        coaching_text = "Start following the reference exercise"

    timing_status = aligner.timing_status
    offset_ms = aligner.offset_ms

    if not user_is_idle:
        if timing_status == "far_behind" and state.smoothed_score >= 70:
            coaching_text = "Good form! Try to keep up with the model"
        elif timing_status == "behind" and state.smoothed_score >= 85:
            coaching_text = coaching_text or "Great form!"

    # Debounce + phase-gate feedback, and only speak on stable changes.
    stable_text, speech_text = state.feedback_stabilizer.update(
        coaching_text,
        phase,
        now_ms,
        rep_summary_text=rep_summary_text,
    )
    coaching_text = stable_text
    feedback["priority_fix"] = coaching_text

    payload = {
        "type": "frame_result",
        "pose_detected": True,
        "match_score": int(round(state.smoothed_score)),
        "matched_ref_frame_index": int(best_idx),
        "matched_ref_source_frame": int(gt_frame.get("frame_index", best_idx)),
        "joint_diffs": {k: v.get("abs_diff", 0.0) for k, v in best_match.items() if isinstance(v, dict) and "abs_diff" in v},
        "joint_scores": best_match.get("joint_scores", {}),
        "joint_errors": best_match.get("joint_errors", {}),
        "joint_statuses": joint_statuses,
        "bone_statuses": bone_statuses,
        "correction_arrows": correction_arrows,
        "coaching_text": coaching_text,
        "speech_text": speech_text,
        "connections": POSE_CONNECTIONS,
        "live_landmarks": live_landmarks,
        "live_hand_landmarks": live_hand_landmarks,
        "ref_landmarks": display_ref_frame.get("landmarks", []),
        "ref_hand_landmarks": display_ref_frame.get("hand_landmarks", {"left": None, "right": None}),
        "exercise_weights": state.auto_weights,
        "feedback": {
            "headline": feedback.get("headline", ""),
            "priority_fix": feedback.get("priority_fix", ""),
            "top_joint_feedback": feedback.get("joint_feedback", [])[:3],
            "velocity_feedback": feedback.get("velocity_feedback", []),
            "symmetry_warnings": feedback.get("symmetry_warnings", []),
        },
        "phase": phase,
        "rep_count": int(state.phase_machine.rep_count),
        "rep_scores": state.phase_machine.rep_scores,
        "rep_joint_issues": state.phase_machine.rep_joint_issues,
        "temporal": {
            "offset_ms": round(offset_ms),
            "status": timing_status,
        },
    }

    # Emit rep_completed when a rep just finished.
    if rep_completed is not None:
        payload["rep_completed"] = {
            "rep_number": rep_completed,
            "score": state.phase_machine.rep_scores[-1] if state.phase_machine.rep_scores else 0,
            "joint_issues": state.phase_machine.rep_joint_issues[-1] if state.phase_machine.rep_joint_issues else {},
        }

    # Emit session_complete when target reps reached.
    if state.target_reps and state.phase_machine.rep_count >= state.target_reps:
        payload["session_complete"] = True
    if LIVE_MATCH_DEBUG_TIMINGS:
        payload["timings"] = {
            "decode_ms": round(decode_ms, 2),
            "detect_ms": round(detect_ms, 2),
            "match_ms": round((time.perf_counter() - match_t0) * 1000.0, 2),
            "total_ms": round((time.perf_counter() - total_t0) * 1000.0, 2),
        }
    return payload


# ---------------------------------------------------------------------------
# Auth decorators
# ---------------------------------------------------------------------------

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = _get_bearer()
        if not token:
            return jsonify({"error": "Authentication required"}), 401
        payload = _decode_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        session = db.session.find_unique(where={"token": token})
        if not session or session.expiresAt < datetime.now(timezone.utc):
            return jsonify({"error": "Session expired, please log in again"}), 401
        request.user = payload
        return f(*args, **kwargs)
    return wrapper


def require_admin(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = _get_bearer()
        if not token:
            return jsonify({"error": "Authentication required"}), 401
        payload = _decode_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        if payload.get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        session = db.session.find_unique(where={"token": token})
        if not session or session.expiresAt < datetime.now(timezone.utc):
            return jsonify({"error": "Session expired"}), 401
        request.user = payload
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Frontend routes
# ---------------------------------------------------------------------------

@app.route("/")
def page_index():     return app.send_static_file("index.html")

@app.route("/login")
def page_login():     return app.send_static_file("login.html")

@app.route("/register")
def page_register():  return app.send_static_file("register.html")

@app.route("/upload")
def page_upload():    return app.send_static_file("upload.html")

@app.route("/dashboard")
def page_dashboard(): return app.send_static_file("app.html")

@app.route("/exercise")
def page_exercise():  return app.send_static_file("exercise.html")

@app.route("/reports")
def page_reports():   return app.send_static_file("app.html")

@app.route("/profile")
def page_profile():   return app.send_static_file("app.html")

@app.route("/admin")
def page_admin():     return app.send_static_file("app.html")

@app.route("/app")
def page_app():       return app.send_static_file("app.html")

@app.route("/report.html")
def page_report():    return app.send_static_file("report.html")


# ---------------------------------------------------------------------------
# Auth API
# ---------------------------------------------------------------------------

@app.route("/api/auth/register", methods=["POST"])
def register():
    data     = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "")

    if not username or not email or not password:
        return jsonify({"error": "username, email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    if db.user.find_unique(where={"username": username}):
        return jsonify({"error": "Username already taken"}), 409
    if db.user.find_unique(where={"email": email}):
        return jsonify({"error": "Email already registered"}), 409

    user = db.user.create(data={
        "id":       str(uuid.uuid4()),
        "username": username,
        "email":    email,
        "password": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
        "role":     "user",
    })

    token = _make_token(user)
    db.session.create(data={
        "id":        str(uuid.uuid4()),
        "userId":    user.id,
        "token":     token,
        "expiresAt": datetime.now(timezone.utc) + TOKEN_EXPIRY,
    })

    return jsonify({"message": "Account created", "token": token,
                    "username": user.username, "role": str(user.role)}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "")

    user = db.user.find_unique(where={"username": username})
    if not user or not bcrypt.checkpw(password.encode(), user.password.encode()):
        return jsonify({"error": "Invalid username or password"}), 401

    token = _make_token(user)

    db.session.create(data={
        "id":        str(uuid.uuid4()),
        "userId":    user.id,
        "token":     token,
        "expiresAt": datetime.now(timezone.utc) + TOKEN_EXPIRY,
    })

    return jsonify({"token": token, "username": user.username, "role": str(user.role)})


@app.route("/api/auth/logout", methods=["POST"])
@require_auth
def logout():
    token = _get_bearer()
    db.session.delete_many(where={"token": token})
    return jsonify({"message": "Logged out"})


@app.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    return jsonify({
        "id":       request.user["sub"],
        "username": request.user["username"],
        "role":     request.user["role"],
    })


@app.route("/api/me", methods=["GET"])
@require_auth
def me_profile():
    """Get detailed user profile info."""
    user = db.user.find_unique(where={"id": request.user["sub"]})
    if not user:
        abort(404)
    return jsonify({
        "id":         user.id,
        "username":   user.username,
        "email":      user.email,
        "role":       str(user.role),
        "created_at": user.createdAt.isoformat(),
    })


# ---------------------------------------------------------------------------
# Exercises (public)
# ---------------------------------------------------------------------------

def _approved_exercise_catalog() -> list[dict]:
    videos = db.video.find_many(
        where={"status": "approved"},
        include={"uploader": True},
        order={"uploadedAt": "desc"},
    )
    seen: dict[str, dict] = {}
    for v in videos:
        if v.exercise not in seen:
            exercise_key = resolve_exercise_name(v.exercise)
            seen[v.exercise] = {
                "exercise": v.exercise,
                "exercise_key": exercise_key,
                "video_id": v.id,
                "uploaded_at": v.uploadedAt.isoformat(),
                "uploader": {
                    "id": v.uploader.id if v.uploader else None,
                    "username": v.uploader.username if v.uploader else "unknown",
                    "created_at": v.uploader.createdAt.isoformat() if v.uploader else None,
                },
            }
    return list(seen.values())

@app.route("/api/exercises", methods=["GET"])
def list_exercises():
    return jsonify(_approved_exercise_catalog())


@app.route("/api/reminders", methods=["GET"])
@require_auth
def list_reminders():
    catalog = _approved_exercise_catalog()
    catalog_map = {
        resolve_exercise_name(item.get("exercise") or ""): item
        for item in catalog
        if resolve_exercise_name(item.get("exercise") or "")
    }
    try:
        reminders = db.reminder.find_many(
            where={"userId": request.user["sub"]},
            order={"remindAt": "asc"},
        )
    except (TableNotFoundError, DataError):
        fallback = _load_reminder_fallback_records()
        payload = [
            {
                "id": item["id"],
                "exercise": resolve_exercise_name(item.get("exercise") or ""),
                "remind_at": item["remind_at"],
                "created_at": item["created_at"],
                "video_id": catalog_map.get(
                    resolve_exercise_name(item.get("exercise") or ""),
                    {},
                ).get("video_id"),
            }
            for item in fallback
            if item.get("user_id") == request.user["sub"]
            and item.get("id")
            and item.get("remind_at")
            and resolve_exercise_name(item.get("exercise") or "") in catalog_map
        ]
        payload.sort(key=lambda item: item["remind_at"])
        return jsonify(payload)

    payload: list[dict] = []
    for reminder in reminders:
        exercise_key = resolve_exercise_name(reminder.exercise)
        catalog_item = catalog_map.get(exercise_key)
        if not catalog_item:
            continue
        payload.append({
            "id": reminder.id,
            "exercise": exercise_key,
            "remind_at": reminder.remindAt.isoformat(),
            "created_at": reminder.createdAt.isoformat(),
            "video_id": catalog_item.get("video_id"),
        })

    return jsonify(payload)


@app.route("/api/reminders", methods=["POST"])
@require_auth
def create_reminder():
    data = request.get_json(silent=True) or {}
    exercise = resolve_exercise_name(data.get("exercise") or "")
    remind_at_raw = data.get("remind_at")

    if not exercise:
        return jsonify({"error": "exercise is required"}), 400
    if not remind_at_raw:
        return jsonify({"error": "remind_at is required"}), 400

    catalog = _approved_exercise_catalog()
    catalog_map = {
        resolve_exercise_name(item.get("exercise") or ""): item
        for item in catalog
        if resolve_exercise_name(item.get("exercise") or "")
    }
    catalog_item = catalog_map.get(exercise)
    if not catalog_item:
        return jsonify({"error": "exercise reference video not found"}), 400

    try:
        remind_at = datetime.fromisoformat(str(remind_at_raw).replace("Z", "+00:00"))
    except ValueError:
        return jsonify({"error": "invalid remind_at"}), 400

    try:
        reminder = db.reminder.create(data={
            "id": str(uuid.uuid4()),
            "userId": request.user["sub"],
            "exercise": exercise,
            "remindAt": remind_at,
        })
    except (TableNotFoundError, DataError):
        reminder_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        fallback = _load_reminder_fallback_records()
        fallback.append({
            "id": reminder_id,
            "user_id": request.user["sub"],
            "exercise": exercise,
            "remind_at": remind_at.isoformat(),
            "created_at": created_at.isoformat(),
        })
        _save_reminder_fallback_records(fallback)
        return jsonify({
            "id": reminder_id,
            "exercise": exercise,
            "remind_at": remind_at.isoformat(),
            "created_at": created_at.isoformat(),
            "video_id": catalog_item["video_id"],
        }), 201

    return jsonify({
        "id": reminder.id,
        "exercise": exercise,
        "remind_at": reminder.remindAt.isoformat(),
        "created_at": reminder.createdAt.isoformat(),
        "video_id": catalog_item["video_id"],
    }), 201


@app.route("/api/reminders/<reminder_id>", methods=["DELETE"])
@require_auth
def delete_reminder(reminder_id: str):
    try:
        reminder = db.reminder.find_unique(where={"id": reminder_id})
    except (TableNotFoundError, DataError):
        fallback = _load_reminder_fallback_records()
        before = len(fallback)
        filtered = [
            item
            for item in fallback
            if not (
                item.get("id") == reminder_id
                and item.get("user_id") == request.user["sub"]
            )
        ]
        if len(filtered) == before:
            return jsonify({"error": "reminder not found"}), 404
        _save_reminder_fallback_records(filtered)
        return jsonify({"ok": True})
    if not reminder or reminder.userId != request.user["sub"]:
        return jsonify({"error": "reminder not found"}), 404

    try:
        db.reminder.delete(where={"id": reminder_id})
    except (TableNotFoundError, DataError):
        return jsonify({"error": "reminders_unavailable"}), 503
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Video streaming (auth required)
# ---------------------------------------------------------------------------

@app.route("/api/video/<video_id>", methods=["GET"])
@require_auth
def stream_video(video_id: str):
    record = db.video.find_unique(where={"id": video_id})
    if not record:
        abort(404)

    path = PENDING_DIR / record.storedName
    if not path.exists():
        path = APPROVED_DIR / record.storedName
    if not path.exists():
        abort(404)

    return send_file(path, mimetype="video/mp4")


@app.route("/api/video-meta/<video_id>", methods=["GET"])
@require_auth
def video_meta(video_id: str):
    record = db.video.find_unique(where={"id": video_id}, include={"uploader": True})
    if not record:
        abort(404)

    return jsonify({
        "id": record.id,
        "exercise": record.exercise,
        "status": str(record.status),
        "uploaded_at": record.uploadedAt.isoformat(),
        "reviewed_at": record.reviewedAt.isoformat() if record.reviewedAt else None,
        "uploader": {
            "id": record.uploader.id if record.uploader else None,
            "username": record.uploader.username if record.uploader else "unknown",
            "email": record.uploader.email if record.uploader else None,
            "created_at": record.uploader.createdAt.isoformat() if record.uploader else None,
        },
    })


@app.route("/api/match-frames", methods=["POST"])
@require_auth
def match_frames():
    data = request.get_json(silent=True) or {}
    live_frame = data.get("live_frame")
    ref_frame = data.get("ref_frame")

    if not live_frame or not ref_frame:
        return jsonify({"error": "live_frame and ref_frame are required"}), 400

    live_img = _decode_data_url_to_bgr(live_frame)
    ref_img = _decode_data_url_to_bgr(ref_frame)

    if live_img is None or ref_img is None:
        return jsonify({"error": "invalid frame payload"}), 400

    live_landmarks = _extract_landmarks(live_img)
    ref_landmarks = _extract_landmarks(ref_img)

    if not live_landmarks or not ref_landmarks:
        return jsonify({
            "live_landmarks": live_landmarks,
            "ref_landmarks": ref_landmarks,
            "pose_detected": False,
            "match_score": None,
            "joint_diffs": {},
            "connections": POSE_CONNECTIONS,
        })

    live_angles = _extract_angles(live_landmarks)
    ref_angles = _extract_angles(ref_landmarks)
    match_score, joint_diffs = _compute_match_score(ref_angles, live_angles)

    return jsonify({
        "live_landmarks": live_landmarks,
        "ref_landmarks": ref_landmarks,
        "pose_detected": True,
        "match_score": match_score,
        "joint_diffs": joint_diffs,
        "connections": POSE_CONNECTIONS,
    })


@app.route("/api/live-match/<ref_id>", methods=["POST"])
@require_auth
def live_match_http(ref_id: str):
    data = request.get_json(silent=True) or {}
    live_frame = data.get("live_frame")
    ref_time_ms = float(data.get("ref_time_ms") or 0.0)
    exercise = resolve_exercise_name(data.get("exercise") or "default")

    if not live_frame:
        return jsonify({"error": "live_frame is required"}), 400

    try:
        state = _get_or_create_live_state(str(request.user["sub"]), ref_id, exercise)
        payload = _process_live_frame_message(state, live_frame, ref_time_ms)
    except FileNotFoundError:
        return jsonify({"error": "reference not found"}), 404
    except Exception as exc:
        return jsonify({"error": f"live match failed: {exc}"}), 500

    if payload.get("type") == "error":
        return jsonify(payload), 400
    return jsonify(payload)


if sock is not None:
    @sock.route("/ws/live-match")
    def ws_live_match(ws):
        if not db.is_connected():
            db.connect()

        token = request.args.get("token", "")
        ref_id = request.args.get("ref_id", "")
        exercise = resolve_exercise_name(request.args.get("exercise") or "default")
        try:
            target_reps = int(request.args.get("target_reps") or 0) or None
        except (ValueError, TypeError):
            target_reps = None

        auth_payload = _decode_token_and_session(token)
        if not auth_payload:
            ws.send(json.dumps({"type": "error", "error": "unauthorized"}))
            return
        if not ref_id:
            ws.send(json.dumps({"type": "error", "error": "ref_id is required"}))
            return

        try:
            gt_norm = _load_or_extract_reference_norm(ref_id)
        except Exception as exc:
            ws.send(json.dumps({"type": "error", "error": f"reference_unavailable: {exc}"}))
            return

        if not gt_norm:
            ws.send(json.dumps({"type": "error", "error": "reference_skeleton_empty"}))
            return

        cfg = _build_session_config(gt_norm, exercise)
        state = LiveSessionState(
            user_id=str(auth_payload["sub"]),
            ref_video_id=ref_id,
            exercise=exercise,
            gt_norm=gt_norm,
            ref_timestamps_ms=[float(frame.get("timestamp_ms", 0.0)) for frame in gt_norm],
            auto_weights=cfg["auto_weights"],
            adaptive_thresholds=cfg["adaptive_thresholds"],
            primary_joint=cfg["primary_joint"],
            primary_min=cfg["primary_min"],
            primary_max=cfg["primary_max"],
            phase_machine=cfg["phase_machine"],
            target_reps=target_reps,
        )

        ws.send(json.dumps({
            "type": "ready",
            "exercise": exercise,
            "frame_count": len(gt_norm),
            "connections": POSE_CONNECTIONS,
            "target_reps": target_reps,
        }))

        while True:
            raw = ws.receive()
            if raw is None:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                ws.send(json.dumps({"type": "error", "error": "invalid_json"}))
                continue

            if msg.get("type") == "ping":
                ws.send(json.dumps({"type": "pong"}))
                continue

            if msg.get("type") != "frame":
                continue

            live_frame = msg.get("live_frame")
            ref_time_ms = float(msg.get("ref_time_ms") or 0.0)
            if not live_frame:
                ws.send(json.dumps({"type": "error", "error": "live_frame required"}))
                continue

            payload = _process_live_frame_message(state, live_frame, ref_time_ms)
            ws.send(json.dumps(payload))


# ---------------------------------------------------------------------------
# Upload (any authenticated user)
# ---------------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/api/upload", methods=["POST"])
@require_auth
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file     = request.files["video"]
    exercise = (request.form.get("exercise") or "unknown").strip()

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not _allowed(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset position
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({"error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB"}), 400
    
    if file_size == 0:
        return jsonify({"error": "File is empty"}), 400

    video_id  = str(uuid.uuid4())
    ext       = file.filename.rsplit(".", 1)[1].lower()
    safe_name = secure_filename(f"{video_id}.{ext}")
    file.save(PENDING_DIR / safe_name)

    db.video.create(data={
        "id":           video_id,
        "originalName": file.filename,
        "storedName":   safe_name,
        "exercise":     exercise,
        "uploaderId":   request.user["sub"],
        "status":       "pending",
    })

    return jsonify({"message": "Uploaded, awaiting admin approval", "id": video_id}), 201


# ---------------------------------------------------------------------------
# My uploads
# ---------------------------------------------------------------------------

@app.route("/api/my-uploads", methods=["GET"])
@require_auth
def my_uploads():
    videos = db.video.find_many(
        where={"uploaderId": request.user["sub"]},
        order={"uploadedAt": "desc"},
    )
    return jsonify([{
        "id":           v.id,
        "original_name": v.originalName,
        "exercise":     v.exercise,
        "status":       str(v.status),
        "admin_note":   v.adminNote,
        "uploaded_at":  v.uploadedAt.isoformat(),
        "reviewed_at":  v.reviewedAt.isoformat() if v.reviewedAt else None,
    } for v in videos])


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

@app.route("/api/categories", methods=["GET"])
def list_categories():
    """List categories for dropdowns and admin moderation views.

    Behavior:
    - Admin + ?all=true: returns every category and moderation metadata.
    - Public/authenticated users: approved categories plus the requester's
      own pending suggestions (when authenticated).
    - If no category rows exist yet, returns built-in exercise names so the
      upload dropdown does not appear empty on fresh databases.
    """
    auth_header = _get_bearer()
    auth_data = _decode_token_and_session(auth_header) if auth_header else None
    requester_user_id = auth_data.get("sub") if isinstance(auth_data, dict) else None
    is_admin = False
    if auth_data:
        if auth_data.get("role") == "admin":
            is_admin = True

    if is_admin and request.args.get("all") == "true":
        categories = db.category.find_many(
            include={"suggestedBy": True},
            order={"createdAt": "desc"},
        )
    else:
        approved_categories = db.category.find_many(
            where={"status": "approved"},
            order={"name": "asc"},
        )
        categories = list(approved_categories)

        if requester_user_id:
            pending_categories = db.category.find_many(
                where={
                    "suggestedById": requester_user_id,
                    "status": "pending",
                },
                order={"createdAt": "desc"},
            )
            seen_ids = {c.id for c in categories}
            categories.extend([c for c in pending_categories if c.id not in seen_ids])

        if not categories:
            built_in_keys = sorted(
                key for key in EXERCISE_WEIGHTS.keys()
                if key != "default"
            )
            return jsonify([{
                "id": f"builtin:{key}",
                "name": key.replace("_", " ").title(),
                "description": None,
                "status": "approved",
                "admin_note": None,
                "created_at": None,
                "suggested_by": None,
                "is_builtin": True,
            } for key in built_in_keys])

    return jsonify([{
        "id":          c.id,
        "name":        c.name,
        "description": c.description,
        "status":      str(c.status),
        "admin_note":  c.adminNote if is_admin else None,
        "created_at":  c.createdAt.isoformat(),
        "suggested_by": {
            "id": c.suggestedBy.id,
            "username": c.suggestedBy.username,
        } if is_admin and hasattr(c, "suggestedBy") and c.suggestedBy else None,
        "is_builtin": False,
    } for c in categories])


@app.route("/api/categories", methods=["POST"])
@require_auth
def create_category():
    """User suggests a new category (pending admin approval)."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()

    if not name:
        return jsonify({"error": "Category name is required"}), 400
    if len(name) > 100:
        return jsonify({"error": "Category name too long (max 100 chars)"}), 400

    existing = db.category.find_first(where={"name": name})
    if existing:
        return jsonify({"error": "Category already exists"}), 409

    category = db.category.create(data={
        "name":          name,
        "description":   description or None,
        "suggestedById": request.user["sub"],
        "status":        "pending",
    })

    return jsonify({
        "message": "Category submitted for approval",
        "id":      category.id,
        "name":    category.name,
    }), 201


@app.route("/api/categories/<category_id>/approve", methods=["POST"])
@require_admin
def approve_category(category_id: str):
    """Admin approves a category."""
    record = db.category.find_unique(where={"id": category_id})
    if not record:
        abort(404)

    note = (request.get_json(silent=True) or {}).get("note", "")

    db.category.update(
        where={"id": category_id},
        data={
            "status":     "approved",
            "adminNote":  note or None,
            "reviewedAt": datetime.now(timezone.utc),
        },
    )
    return jsonify({"message": "Category approved"})


@app.route("/api/categories/<category_id>/reject", methods=["POST"])
@require_admin
def reject_category(category_id: str):
    """Admin rejects a category."""
    record = db.category.find_unique(where={"id": category_id})
    if not record:
        abort(404)

    note = (request.get_json(silent=True) or {}).get("note", "")

    db.category.update(
        where={"id": category_id},
        data={
            "status":     "rejected",
            "adminNote":  note or None,
            "reviewedAt": datetime.now(timezone.utc),
        },
    )
    return jsonify({"message": "Category rejected"})


@app.route("/api/categories/<category_id>", methods=["DELETE"])
@require_admin
def delete_category(category_id: str):
    """Admin deletes a category."""
    record = db.category.find_unique(where={"id": category_id})
    if not record:
        abort(404)

    db.category.delete(where={"id": category_id})
    return jsonify({"message": "Category deleted"})


# ---------------------------------------------------------------------------
# Admin — list all videos
# ---------------------------------------------------------------------------

@app.route("/api/videos", methods=["GET"])
@require_admin
def list_videos():
    status_filter = request.args.get("status")
    where = {"status": status_filter} if status_filter else {}
    videos = db.video.find_many(
        where=where,
        include={"uploader": True},
        order={"uploadedAt": "desc"},
    )
    return jsonify([{
        "id":           v.id,
        "original_name": v.originalName,
        "exercise":     v.exercise,
        "status":       str(v.status),
        "uploader":     v.uploader.username if v.uploader else "unknown",
        "uploader_id":  v.uploader.id if v.uploader else None,
        "uploader_profile": {
            "id": v.uploader.id if v.uploader else None,
            "username": v.uploader.username if v.uploader else "unknown",
            "email": v.uploader.email if v.uploader else None,
            "created_at": v.uploader.createdAt.isoformat() if v.uploader else None,
        },
        "admin_note":   v.adminNote,
        "uploaded_at":  v.uploadedAt.isoformat(),
        "reviewed_at":  v.reviewedAt.isoformat() if v.reviewedAt else None,
    } for v in videos])


# ---------------------------------------------------------------------------
# Admin — approve / reject
# ---------------------------------------------------------------------------

@app.route("/api/approve/<video_id>", methods=["POST"])
@require_admin
def approve_video(video_id: str):
    record = db.video.find_unique(where={"id": video_id})
    if not record:
        abort(404)

    note = (request.get_json(silent=True) or {}).get("note", "")

    src = PENDING_DIR / record.storedName
    dst = APPROVED_DIR / record.storedName
    if src.exists():
        src.rename(dst)

    db.video.update(
        where={"id": video_id},
        data={"status": "approved", "reviewedAt": datetime.now(timezone.utc), "adminNote": note},
    )
    return jsonify({"message": "Video approved", "id": video_id})


@app.route("/api/reject/<video_id>", methods=["POST"])
@require_admin
def reject_video(video_id: str):
    record = db.video.find_unique(where={"id": video_id})
    if not record:
        abort(404)

    note = (request.get_json(silent=True) or {}).get("note", "")
    db.video.update(
        where={"id": video_id},
        data={"status": "rejected", "reviewedAt": datetime.now(timezone.utc), "adminNote": note},
    )
    return jsonify({"message": "Video rejected", "id": video_id})


# ---------------------------------------------------------------------------
# Delete video (admin or owner)
# ---------------------------------------------------------------------------

@app.route("/api/videos/<video_id>", methods=["DELETE"])
@require_auth
def delete_video(video_id: str):
    """Delete a video. Admin can delete any, users can only delete their own."""
    record = db.video.find_unique(where={"id": video_id})
    if not record:
        abort(404)

    user_id = request.user["sub"]
    is_admin = request.user.get("role") == "admin"

    # Check permission: admin can delete any, user can only delete their own
    if not is_admin and record.uploaderId != user_id:
        return jsonify({"error": "You can only delete your own videos"}), 403

    # Delete the video file
    pending_path = PENDING_DIR / record.storedName
    approved_path = APPROVED_DIR / record.storedName
    
    if pending_path.exists():
        pending_path.unlink()
    if approved_path.exists():
        approved_path.unlink()

    # Delete skeleton cache if exists
    skeleton_cache = Path("data/skeletons") / f"{video_id}.json"
    if skeleton_cache.exists():
        skeleton_cache.unlink()

    # Delete from database
    db.video.delete(where={"id": video_id})

    return jsonify({"message": "Video deleted", "id": video_id})


# ---------------------------------------------------------------------------
# Exercise Reports
# ---------------------------------------------------------------------------

def calculate_grade(avg_score: float) -> str:
    """Calculate letter grade from average score."""
    if avg_score >= 90:
        return "A"
    elif avg_score >= 80:
        return "B"
    elif avg_score >= 70:
        return "C"
    elif avg_score >= 60:
        return "D"
    else:
        return "F"


def _copy_default(default):
    if isinstance(default, list):
        return list(default)
    if isinstance(default, dict):
        return dict(default)
    return default


def _safe_json(value, default):
    """Decode JSON safely while preserving backward compatibility."""
    if value is None:
        return _copy_default(default)
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value) if value else _copy_default(default)
        except Exception:
            return _copy_default(default)
    return _copy_default(default)


def _normalize_numeric_list(raw_value) -> list[float]:
    if not isinstance(raw_value, list):
        return []
    values: list[float] = []
    for item in raw_value:
        if isinstance(item, (int, float)):
            values.append(float(item))
    return values


def _normalize_issue_map(raw_value) -> dict[str, int]:
    if not isinstance(raw_value, dict):
        return {}
    result: dict[str, int] = {}
    for joint, count in raw_value.items():
        if not isinstance(joint, str) or not isinstance(count, (int, float)):
            continue
        value = int(count)
        if value > 0:
            result[joint] = value
    return result


def _normalize_rep_joint_issues(raw_value) -> list[dict[str, int]]:
    if not isinstance(raw_value, list):
        return []
    normalized: list[dict[str, int]] = []
    for item in raw_value:
        normalized.append(_normalize_issue_map(item))
    return normalized


def _normalize_feedback_summary(raw_value) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    cleaned: list[str] = []
    for item in raw_value:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
    return cleaned


def _slug_for_filename(value: str, *, fallback: str = "report") -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or fallback


def _humanize_joint_name(name: str) -> str:
    return name.replace("_", " ").strip().title() if name else "Unknown"


def _as_percentage(value) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    numeric = float(value)
    return numeric * 100.0 if numeric <= 1.0 else numeric


def _load_reportlab_modules() -> dict:
    try:
        import importlib

        return {
            "pagesizes": importlib.import_module("reportlab.lib.pagesizes"),
            "colors": importlib.import_module("reportlab.lib.colors"),
            "styles": importlib.import_module("reportlab.lib.styles"),
            "platypus": importlib.import_module("reportlab.platypus"),
        }
    except ModuleNotFoundError as exc:
        raise RuntimeError("PDF export requires the reportlab package") from exc


def _build_pdf_styles(styles_module, colors_module):
    sample = styles_module.getSampleStyleSheet()
    ParagraphStyle = styles_module.ParagraphStyle
    return {
        "title_light": ParagraphStyle(
            "title_light",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors_module.HexColor("#f8fafc"),
            spaceAfter=4,
        ),
        "meta_light": ParagraphStyle(
            "meta_light",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors_module.HexColor("#cbd5e1"),
        ),
        "badge_light": ParagraphStyle(
            "badge_light",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            alignment=1,
            textColor=colors_module.HexColor("#e6fffa"),
        ),
        "section": ParagraphStyle(
            "section",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors_module.HexColor("#0f172a"),
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors_module.HexColor("#0f172a"),
        ),
        "muted": ParagraphStyle(
            "muted",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors_module.HexColor("#64748b"),
        ),
        "metric": ParagraphStyle(
            "metric",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=16,
            textColor=colors_module.HexColor("#0f172a"),
            alignment=1,
        ),
        "metric_label": ParagraphStyle(
            "metric_label",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors_module.HexColor("#475569"),
            alignment=1,
        ),
    }


def _build_pdf_palette(colors_module) -> dict[str, object]:
    return {
        "header_bg": colors_module.HexColor("#0f172a"),
        "header_text": colors_module.HexColor("#f8fafc"),
        "header_muted": colors_module.HexColor("#cbd5e1"),
        "panel_bg": colors_module.HexColor("#f8fafc"),
        "panel_bg_alt": colors_module.HexColor("#f1f5f9"),
        "panel_border": colors_module.HexColor("#cbd5e1"),
        "table_header_bg": colors_module.HexColor("#1e293b"),
        "table_header_text": colors_module.HexColor("#f8fafc"),
        "table_row_a": colors_module.HexColor("#ffffff"),
        "table_row_b": colors_module.HexColor("#f8fafc"),
        "accent_teal": colors_module.HexColor("#0f766e"),
        "accent_info": colors_module.HexColor("#2563eb"),
        "text_main": colors_module.HexColor("#0f172a"),
        "text_muted": colors_module.HexColor("#64748b"),
    }


def _table_style_professional(
    *,
    TableStyle,
    palette: dict[str, object],
    header_bg=None,
    header_text=None,
    right_align_cols: list[int] | None = None,
):
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg or palette["table_header_bg"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), header_text or palette["table_header_text"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [palette["table_row_a"], palette["table_row_b"]]),
        ("TEXTCOLOR", (0, 1), (-1, -1), palette["text_main"]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.7),
        ("BOX", (0, 0), (-1, -1), 0.8, palette["panel_border"]),
        ("INNERGRID", (0, 0), (-1, -1), 0.55, palette["panel_border"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])
    for col in right_align_cols or []:
        style.add("ALIGN", (col, 1), (col, -1), "RIGHT")
    return style


def _build_pdf_page_callback(colors_module, footer_title: str):
    line_color = colors_module.HexColor("#dbe3ef")
    text_color = colors_module.HexColor("#64748b")

    def _callback(canvas, doc):
        canvas.saveState()
        width, _ = doc.pagesize
        footer_y = 18
        canvas.setStrokeColor(line_color)
        canvas.setLineWidth(0.6)
        canvas.line(doc.leftMargin, footer_y + 10, width - doc.rightMargin, footer_y + 10)

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(text_color)
        canvas.drawString(doc.leftMargin, footer_y, footer_title)
        canvas.drawRightString(width - doc.rightMargin, footer_y, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    return _callback


def _format_percent_text(value) -> str:
    return f"{round(_as_percentage(value), 1)}%"


def _extract_problem_joints_for_report(report_payload: dict) -> list[dict]:
    """Prefer session-level problem joints, then fall back to latest set."""
    session_report = report_payload.get("session_report") if isinstance(report_payload.get("session_report"), dict) else {}
    if isinstance(session_report.get("problem_joints"), list) and session_report.get("problem_joints"):
        return session_report.get("problem_joints")

    set_reports = report_payload.get("set_reports") if isinstance(report_payload.get("set_reports"), list) else []
    for set_payload in reversed(set_reports):
        if not isinstance(set_payload, dict):
            continue
        candidate = set_payload.get("problem_joints")
        if isinstance(candidate, list) and candidate:
            return candidate

    return []


def _append_pdf_skeleton_section(
    story: list,
    *,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    styles: dict,
    problem_joints: list[dict],
    image_width: int = 515,
    image_height: int = 665,
) -> None:
    """Append skeleton visualization block into a ReportLab story."""
    # Reserve a clean page for the skeleton so it appears centered and prominent.
    story.append(PageBreak())
    story.append(Paragraph("Joint Skeleton Visualization", styles["section"]))
    story.append(Spacer(1, 8))

    if not problem_joints:
        story.append(Paragraph("No notable joint issues detected.", styles["muted"]))
        story.append(Spacer(1, 12))
        return

    try:
        skeleton_png = generate_problem_joints_png(
            problem_joints,
            width=760,
            height=980,
            dark_mode=True,
        )
        image_flowable = Image(io.BytesIO(skeleton_png), width=image_width, height=image_height)
        image_flowable.hAlign = "CENTER"
        image_frame = Table(
            [[image_flowable]],
            colWidths=[520],
            rowHeights=[670],
        )
        image_frame.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(image_frame)
        story.append(Spacer(1, 6))
        caption = Table(
            [[Paragraph("Defect points are circled and numbered to match the problem-joint ranking.", styles["muted"])]],
            colWidths=[520],
        )
        caption.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(caption)
    except Exception:
        story.append(Paragraph("Skeleton visualization unavailable for this report.", styles["muted"]))

    story.append(Spacer(1, 12))


def _build_exercise_report_pdf(report_data: dict) -> bytes:
    modules = _load_reportlab_modules()
    pagesizes_module = modules["pagesizes"]
    colors_module = modules["colors"]
    styles_module = modules["styles"]
    platypus = modules["platypus"]
    styles = _build_pdf_styles(styles_module, colors_module)
    palette = _build_pdf_palette(colors_module)

    score_history = _normalize_numeric_list(report_data.get("score_history", []))
    rep_scores = _normalize_numeric_list(report_data.get("rep_scores", []))
    joint_issues = _normalize_issue_map(report_data.get("joint_issues", {}))
    rep_joint_issues = _normalize_rep_joint_issues(report_data.get("rep_joint_issues", []))
    feedback_summary = _normalize_feedback_summary(report_data.get("feedback_summary", []))
    set_reports = report_data.get("set_reports") if isinstance(report_data.get("set_reports"), list) else []
    session_report = report_data.get("session_report") if isinstance(report_data.get("session_report"), dict) else None
    problem_joints = _extract_problem_joints_for_report(report_data)

    Paragraph = platypus.Paragraph
    Spacer = platypus.Spacer
    Table = platypus.Table
    TableStyle = platypus.TableStyle
    SimpleDocTemplate = platypus.SimpleDocTemplate
    Image = platypus.Image
    PageBreak = platypus.PageBreak

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=pagesizes_module.LETTER,
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=26,
        title="Pose Matcher Exercise Report",
    )

    user = report_data.get("user") or {}
    user_name = user.get("username") or "unknown"
    user_email = user.get("email") or "unknown"
    story = []

    header = Table(
        [[
            Paragraph("Pose Matcher Report", styles["title_light"]),
            Paragraph(f"{report_data.get('overall_grade', 'N/A')}<br/>{_format_percent_text(report_data.get('avg_score', 0))}", styles["badge_light"]),
        ]],
        colWidths=[390, 130],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), palette["header_bg"]),
        ("BOX", (0, 0), (-1, -1), 1, palette["header_bg"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 14),
        ("RIGHTPADDING", (1, 0), (1, 0), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(header)
    story.append(Spacer(1, 8))
    meta_table = Table(
        [[
            Paragraph(
                (
                    f"User: <b>{user_name}</b> &nbsp;&nbsp; "
                    f"Email: <b>{user_email}</b><br/>"
                    f"Exercise: <b>{report_data.get('exercise', 'unknown')}</b><br/>"
                    f"Created: {report_data.get('created_at', '')}"
                ),
                styles["body"],
            ),
            Paragraph(f"Report ID<br/><b>{report_data.get('id', '')}</b>", styles["metric_label"]),
        ]],
        colWidths=[390, 130],
    )
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), palette["panel_bg"]),
        ("BOX", (0, 0), (-1, -1), 0.8, palette["panel_border"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    stats_cells = [
        ("Avg Score", _format_percent_text(report_data.get("avg_score", 0))),
        ("Min Score", _format_percent_text(report_data.get("min_score", 0))),
        ("Max Score", _format_percent_text(report_data.get("max_score", 0))),
        ("Reps", str(int(report_data.get("rep_count") or 0))),
        ("Duration", f"{int(report_data.get('duration_seconds') or 0)} sec"),
        ("Frames", str(int(report_data.get("total_frames") or 0))),
    ]
    stats_table_data = []
    row: list = []
    for idx, (label, value) in enumerate(stats_cells):
        row.append(Paragraph(f"<b>{value}</b><br/>{label}", styles["metric_label"]))
        if (idx + 1) % 3 == 0:
            stats_table_data.append(row)
            row = []
    stats_table = Table(stats_table_data, colWidths=[170, 170, 170])
    stats_table.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [palette["panel_bg"], palette["panel_bg_alt"]]),
        ("TEXTCOLOR", (0, 0), (-1, -1), palette["text_main"]),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.8, palette["panel_border"]),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, palette["panel_border"]),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Per-Rep Accuracy", styles["section"]))
    if rep_scores:
        rep_rows = [["Rep", "Accuracy"]] + [[f"Rep {idx}", f"{round(score, 1)}%"] for idx, score in enumerate(rep_scores, start=1)]
        rep_table = Table(rep_rows, colWidths=[110, 120])
        rep_table.setStyle(_table_style_professional(
            TableStyle=TableStyle,
            palette=palette,
            header_bg=palette["accent_teal"],
            right_align_cols=[1],
        ))
        story.append(rep_table)
    else:
        story.append(Paragraph("No per-rep scores recorded.", styles["muted"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Per-Rep Joint Issues", styles["section"]))
    if rep_joint_issues:
        issue_rows = [["Rep", "Issues"]]
        for idx, issues in enumerate(rep_joint_issues, start=1):
            if not issues:
                issue_rows.append([f"Rep {idx}", "None"])
                continue
            parts = [
                f"{_humanize_joint_name(joint)} ({count})"
                for joint, count in sorted(issues.items(), key=lambda item: item[1], reverse=True)
            ]
            issue_rows.append([f"Rep {idx}", ", ".join(parts)])
        issue_table = Table(issue_rows, colWidths=[80, 450])
        issue_table.setStyle(_table_style_professional(
            TableStyle=TableStyle,
            palette=palette,
            header_bg=palette["accent_teal"],
        ))
        story.append(issue_table)
    else:
        story.append(Paragraph("No per-rep joint issues recorded.", styles["muted"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Overall Joint Issues", styles["section"]))
    if joint_issues:
        joint_rows = [["Joint", "Flags"]] + [
            [_humanize_joint_name(joint), str(count)]
            for joint, count in sorted(joint_issues.items(), key=lambda item: item[1], reverse=True)
        ]
        joint_table = Table(joint_rows, colWidths=[350, 80])
        joint_table.setStyle(_table_style_professional(
            TableStyle=TableStyle,
            palette=palette,
            right_align_cols=[1],
        ))
        story.append(joint_table)
    else:
        story.append(Paragraph("No global joint issues recorded.", styles["muted"]))

    _append_pdf_skeleton_section(
        story,
        Paragraph=Paragraph,
        Spacer=Spacer,
        Table=Table,
        TableStyle=TableStyle,
        Image=Image,
        PageBreak=PageBreak,
        styles=styles,
        problem_joints=problem_joints,
    )

    if session_report or set_reports:
        story.append(Spacer(1, 14))
        story.append(Paragraph("Linked Full Session", styles["section"]))
        if session_report:
            summary_table = Table([
                ["Metric", "Value"],
                ["Overall Accuracy", _format_percent_text(session_report.get("overall_accuracy", 0))],
                ["Total Reps", str(int(session_report.get("total_reps") or 0))],
                ["Session Grade", str(session_report.get("grade") or "N/A")],
            ], colWidths=[190, 160])
            summary_table.setStyle(_table_style_professional(
                TableStyle=TableStyle,
                palette=palette,
                header_bg=palette["accent_info"],
                right_align_cols=[1],
            ))
            story.append(summary_table)
            story.append(Spacer(1, 8))

        if set_reports:
            set_rows = [["Set", "Rep", "Accuracy", "Grade"]]
            for set_item in set_reports:
                set_no = int(set_item.get("set_number") or 0)
                set_grade = str(set_item.get("grade") or "N/A")
                reps = set_item.get("reps") if isinstance(set_item.get("reps"), list) else []
                if not reps:
                    set_rows.append([
                        f"Set {set_no}",
                        "-",
                        _format_percent_text(set_item.get("overall_accuracy", 0)),
                        set_grade,
                    ])
                    continue
                for rep in reps:
                    set_rows.append([
                        f"Set {set_no}",
                        f"Rep {int(rep.get('rep') or 0)}",
                        _format_percent_text(rep.get("accuracy", 0)),
                        set_grade,
                    ])

            set_table = Table(set_rows, colWidths=[80, 80, 110, 90])
            set_table.setStyle(_table_style_professional(
                TableStyle=TableStyle,
                palette=palette,
                header_bg=palette["accent_teal"],
                right_align_cols=[2],
            ))
            story.append(set_table)

    story.append(Spacer(1, 14))
    story.append(Paragraph("Coaching Summary", styles["section"]))
    if feedback_summary:
        for tip in feedback_summary:
            story.append(Paragraph(f"- {tip}", styles["body"]))
    else:
        story.append(Paragraph("No feedback notes recorded.", styles["muted"]))

    page_callback = _build_pdf_page_callback(colors_module, "Pose Matcher Exercise Report")
    doc.build(story, onFirstPage=page_callback, onLaterPages=page_callback)
    return buffer.getvalue()


def _build_session_report_pdf(report_data: dict) -> bytes:
    modules = _load_reportlab_modules()
    pagesizes_module = modules["pagesizes"]
    colors_module = modules["colors"]
    styles_module = modules["styles"]
    platypus = modules["platypus"]
    styles = _build_pdf_styles(styles_module, colors_module)
    palette = _build_pdf_palette(colors_module)

    Paragraph = platypus.Paragraph
    Spacer = platypus.Spacer
    Table = platypus.Table
    TableStyle = platypus.TableStyle
    SimpleDocTemplate = platypus.SimpleDocTemplate
    Image = platypus.Image
    PageBreak = platypus.PageBreak

    session = report_data.get("session") or {}
    session_username = session.get("username") or "unknown"
    session_email = session.get("email") or "unknown"
    set_reports = report_data.get("set_reports") if isinstance(report_data.get("set_reports"), list) else []
    session_report = report_data.get("session_report") if isinstance(report_data.get("session_report"), dict) else {}
    problem_joints = session_report.get("problem_joints") if isinstance(session_report.get("problem_joints"), list) else []
    if not problem_joints:
        for set_payload in reversed(set_reports):
            if not isinstance(set_payload, dict):
                continue
            candidate = set_payload.get("problem_joints")
            if isinstance(candidate, list) and candidate:
                problem_joints = candidate
                break

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=pagesizes_module.LETTER,
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=26,
        title="Pose Matcher Session Report",
    )

    story = []
    header = Table(
        [[
            Paragraph("Pose Matcher Session Report", styles["title_light"]),
            Paragraph(
                f"{session_report.get('grade', 'N/A')}<br/>{_format_percent_text(session_report.get('overall_accuracy', 0))}",
                styles["badge_light"],
            ),
        ]],
        colWidths=[390, 130],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), palette["header_bg"]),
        ("BOX", (0, 0), (-1, -1), 1, palette["header_bg"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 14),
        ("RIGHTPADDING", (1, 0), (1, 0), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(header)
    story.append(Spacer(1, 8))

    meta_table = Table(
        [[
            Paragraph(
                (
                    f"Exercise: <b>{session.get('exercise_name', 'unknown')}</b><br/>"
                    f"User: <b>{session_username}</b> &nbsp;&nbsp; "
                    f"Email: <b>{session_email}</b><br/>"
                    f"Started: {session.get('started_at', '')} &nbsp;&nbsp; "
                    f"Completed: {session.get('completed_at', '') or 'in progress'}"
                ),
                styles["body"],
            ),
            Paragraph(f"Session ID<br/><b>{session.get('id', '')}</b>", styles["metric_label"]),
        ]],
        colWidths=[390, 130],
    )
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), palette["panel_bg"]),
        ("BOX", (0, 0), (-1, -1), 0.8, palette["panel_border"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    summary_table = Table([
        ["Metric", "Value"],
        ["Overall Accuracy", _format_percent_text(session_report.get("overall_accuracy", 0))],
        ["Total Reps", str(int(session_report.get("total_reps") or 0))],
        ["Sets", str(len(set_reports))],
        ["Completed", "Yes" if bool(session_report.get("completed", False)) else "No"],
    ], colWidths=[190, 150])
    summary_table.setStyle(_table_style_professional(
        TableStyle=TableStyle,
        palette=palette,
        header_bg=palette["accent_info"],
        right_align_cols=[1],
    ))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Set and Rep Breakdown", styles["section"]))
    set_rows = [["Set", "Rep", "Accuracy", "Grade"]]
    if set_reports:
        for set_item in set_reports:
            set_no = int(set_item.get("set_number") or 0)
            set_grade = str(set_item.get("grade") or "N/A")
            reps = set_item.get("reps") if isinstance(set_item.get("reps"), list) else []
            if not reps:
                set_rows.append([
                    f"Set {set_no}",
                    "-",
                    _format_percent_text(set_item.get("overall_accuracy", 0)),
                    set_grade,
                ])
                continue
            for rep in reps:
                set_rows.append([
                    f"Set {set_no}",
                    f"Rep {int(rep.get('rep') or 0)}",
                    _format_percent_text(rep.get("accuracy", 0)),
                    set_grade,
                ])
    else:
        set_rows.append(["-", "-", "-", "No sets recorded"])

    set_table = Table(set_rows, colWidths=[90, 90, 110, 110])
    set_table.setStyle(_table_style_professional(
        TableStyle=TableStyle,
        palette=palette,
        header_bg=palette["accent_teal"],
        right_align_cols=[2],
    ))
    story.append(set_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Problem Joints", styles["section"]))
    if problem_joints:
        problem_rows = [["Joint", "Severity", "Error %"]]
        for item in problem_joints:
            if not isinstance(item, dict):
                continue
            problem_rows.append([
                str(item.get("label") or _humanize_joint_name(str(item.get("joint") or ""))),
                str(item.get("severity") or "unknown"),
                f"{round(float(item.get('error_pct', 0.0)), 1)}%" if isinstance(item.get("error_pct"), (int, float)) else "0.0%",
            ])
        problem_table = Table(problem_rows, colWidths=[230, 120, 90])
        problem_table.setStyle(_table_style_professional(
            TableStyle=TableStyle,
            palette=palette,
            right_align_cols=[2],
        ))
        story.append(problem_table)
    else:
        story.append(Paragraph("No notable joint issues detected.", styles["muted"]))

    _append_pdf_skeleton_section(
        story,
        Paragraph=Paragraph,
        Spacer=Spacer,
        Table=Table,
        TableStyle=TableStyle,
        Image=Image,
        PageBreak=PageBreak,
        styles=styles,
        problem_joints=problem_joints,
    )

    page_callback = _build_pdf_page_callback(colors_module, "Pose Matcher Session Report")
    doc.build(story, onFirstPage=page_callback, onLaterPages=page_callback)
    return buffer.getvalue()


def _serialize_exercise_report(report) -> dict:
    score_history = _normalize_numeric_list(_safe_json(report.scoreHistory, []))
    phase_history = _safe_json(report.phaseHistory, [])
    if not isinstance(phase_history, list):
        phase_history = []
    joint_issues = _normalize_issue_map(_safe_json(report.jointIssues, {}))
    bone_issues = _normalize_issue_map(_safe_json(report.boneIssues, {}))
    feedback_summary = _normalize_feedback_summary(_safe_json(report.feedbackSummary, []))
    rep_scores = _normalize_numeric_list(_safe_json(getattr(report, "repScores", None), []))
    rep_joint_issues = _normalize_rep_joint_issues(_safe_json(getattr(report, "repJointIssues", None), []))
    session_id = getattr(report, "sessionId", None)

    return {
        "id": report.id,
        "session_id": session_id,
        "has_full_session_report": bool(session_id),
        "exercise": report.exercise,
        "reference_video_id": report.referenceVideoId,
        "duration_seconds": report.durationSeconds,
        "total_frames": report.totalFrames,
        "rep_count": report.repCount,
        "avg_score": report.avgScore,
        "min_score": report.minScore,
        "max_score": report.maxScore,
        "overall_grade": report.overallGrade,
        "score_history": score_history,
        "phase_history": phase_history,
        "joint_issues": joint_issues,
        "bone_issues": bone_issues,
        "feedback_summary": feedback_summary,
        "rep_scores": rep_scores,
        "rep_joint_issues": rep_joint_issues,
        "created_at": report.createdAt.isoformat(),
        "user": {
            "id": report.user.id,
            "username": report.user.username,
            "email": report.user.email,
        } if hasattr(report, "user") and report.user else None,
    }


def _fetch_session_bundle(session_id: str, *, requester_user_id: str, is_admin: bool) -> dict | None:
    if not session_id:
        return None

    sess = db.exercisesession.find_unique(
        where={"id": session_id},
        include={"user": True},
    )
    if not sess:
        return None
    if not is_admin and sess.userId != requester_user_id:
        return None

    set_rows = db.setreport.find_many(
        where={"sessionId": session_id},
        order={"setNumber": "asc"},
    )
    sess_row = db.sessionreport.find_unique(where={"sessionId": session_id})

    return {
        "session": {
            "id": sess.id,
            "exercise_name": sess.exerciseName,
            "video_id": sess.videoId,
            "started_at": sess.startedAt.isoformat(),
            "completed_at": sess.completedAt.isoformat() if sess.completedAt else None,
            "completed": sess.completed,
            "username": sess.user.username if hasattr(sess, "user") and sess.user else None,
            "email": sess.user.email if hasattr(sess, "user") and sess.user else None,
        },
        "set_reports": [_safe_json(row.reportJson, {}) for row in set_rows],
        "session_report": _safe_json(sess_row.reportJson, {}) if sess_row else None,
    }


@app.route("/api/reports", methods=["POST"])
@require_auth
def create_report():
    """Save an exercise session report."""
    user_id = request.user["sub"]
    data = request.get_json(silent=True) or {}
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required_fields = ["exercise", "reference_video_id", "duration_seconds", "score_history"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    exercise_name = str(data.get("exercise") or "Unknown").strip() or "Unknown"
    reference_video_id = str(data.get("reference_video_id") or "")

    try:
        duration_seconds = int(data.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        duration_seconds = 0
    duration_seconds = max(duration_seconds, 0)

    score_history = _normalize_numeric_list(data.get("score_history", []))
    phase_history_raw = data.get("phase_history", [])
    phase_history = phase_history_raw if isinstance(phase_history_raw, list) else []
    joint_issues = _normalize_issue_map(data.get("joint_issues", {}))
    bone_issues = _normalize_issue_map(data.get("bone_issues", {}))

    feedback_summary_raw = data.get("feedback_summary", [])
    feedback_summary = [
        item.strip()
        for item in feedback_summary_raw
        if isinstance(item, str) and item.strip()
    ] if isinstance(feedback_summary_raw, list) else []

    session_id_raw = data.get("session_id")
    session_id = str(session_id_raw).strip() if isinstance(session_id_raw, str) else None
    if session_id:
        try:
            linked_session = db.exercisesession.find_unique(where={"id": session_id})
            if not linked_session or linked_session.userId != user_id:
                return jsonify({"error": "invalid session_id"}), 400
        except Exception:
            # Older local databases may not yet have the ExerciseSession table.
            # In that case, keep the report save working without session linkage.
            session_id = None
    else:
        session_id = None

    rep_scores = _normalize_numeric_list(data.get("rep_scores", []))
    rep_joint_issues = _normalize_rep_joint_issues(data.get("rep_joint_issues", []))

    try:
        rep_count = int(data.get("rep_count") or 0)
    except (TypeError, ValueError):
        rep_count = 0
    rep_count = max(rep_count, 0)
    if rep_count == 0 and rep_scores:
        rep_count = len(rep_scores)

    scores = score_history
    
    avg_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    overall_grade = calculate_grade(avg_score)
    
    report_data = {
        "userId": user_id,
        "exercise": exercise_name,
        "referenceVideoId": reference_video_id,
        "durationSeconds": duration_seconds,
        "totalFrames": len(scores),
        "repCount": rep_count,
        "avgScore": round(avg_score, 2),
        "minScore": round(min_score, 2),
        "maxScore": round(max_score, 2),
        "scoreHistory": json.dumps(score_history),
        "phaseHistory": json.dumps(phase_history),
        "jointIssues": json.dumps(joint_issues),
        "boneIssues": json.dumps(bone_issues),
        "feedbackSummary": json.dumps(feedback_summary),
        "repScores": json.dumps(rep_scores),
        "repJointIssues": json.dumps(rep_joint_issues),
        "overallGrade": overall_grade,
    }
    if session_id:
        report_data["sessionId"] = session_id

    optional_columns = {
        "sessionId",
        "repScores",
        "repJointIssues",
    }

    report = None
    for _ in range(4):
        try:
            report = db.exercisereport.create(data=report_data)
            break
        except DataError as exc:
            message = str(exc)
            missing_column = None
            if "column" in message and "does not exist" in message:
                parts = message.split("`")
                if len(parts) >= 2:
                    missing_column = parts[1]
            if missing_column in optional_columns and missing_column in report_data:
                report_data.pop(missing_column, None)
                continue
            raise

    if report is None:
        return jsonify({"error": "report_save_failed"}), 500
    
    return jsonify({
        "message": "Report saved",
        "report_id": report.id,
        "grade": overall_grade,
        "avg_score": round(avg_score, 2),
    }), 201


@app.route("/api/reports", methods=["GET"])
@require_auth
def get_reports():
    """Get user's exercise reports."""
    user_id = request.user["sub"]
    is_admin = request.user.get("role") == "admin"
    
    # Admin can see all reports with ?all=true
    if is_admin and request.args.get("all") == "true":
        reports = db.exercisereport.find_many(
            order={"createdAt": "desc"},
            include={"user": True}
        )
    else:
        reports = db.exercisereport.find_many(
            where={"userId": user_id},
            order={"createdAt": "desc"}
        )
    
    return jsonify([_serialize_exercise_report(r) for r in reports])


@app.route("/api/reports/<report_id>", methods=["GET"])
@require_auth
def get_report(report_id: str):
    """Get a specific report."""
    user_id = request.user["sub"]
    is_admin = request.user.get("role") == "admin"
    
    report = db.exercisereport.find_unique(
        where={"id": report_id},
        include={"user": True}
    )
    
    if not report:
        abort(404)
    
    # Users can only view their own reports, admins can view all
    if not is_admin and report.userId != user_id:
        return jsonify({"error": "Access denied"}), 403

    payload = _serialize_exercise_report(report)
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id:
        session_bundle = _fetch_session_bundle(
            session_id,
            requester_user_id=user_id,
            is_admin=is_admin,
        )
        if session_bundle:
            payload.update(session_bundle)

    return jsonify(payload)


@app.route("/api/reports/<report_id>/download", methods=["GET"])
@require_auth
def download_report_pdf(report_id: str):
    """Download an exercise report as PDF (admin or owner)."""
    user_id = request.user["sub"]
    is_admin = request.user.get("role") == "admin"

    report = db.exercisereport.find_unique(
        where={"id": report_id},
        include={"user": True},
    )
    if not report:
        abort(404)
    if not is_admin and report.userId != user_id:
        return jsonify({"error": "Access denied"}), 403

    report_payload = _serialize_exercise_report(report)
    session_id = report_payload.get("session_id")
    if isinstance(session_id, str) and session_id:
        session_bundle = _fetch_session_bundle(
            session_id,
            requester_user_id=user_id,
            is_admin=is_admin,
        )
        if session_bundle:
            report_payload.update(session_bundle)

    try:
        pdf_bytes = _build_exercise_report_pdf(report_payload)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    created_stamp = report.createdAt.strftime("%Y%m%d")
    user_part = _slug_for_filename(report_payload.get("user", {}).get("username") if report_payload.get("user") else "user", fallback="user")
    exercise_part = _slug_for_filename(report.exercise, fallback="exercise")
    filename = f"report-{user_part}-{exercise_part}-{created_stamp}-{report.id[:8]}.pdf"

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/reports/<report_id>", methods=["DELETE"])
@require_auth
def delete_report(report_id: str):
    """Delete a report. Users can delete their own, admins can delete any."""
    user_id = request.user["sub"]
    is_admin = request.user.get("role") == "admin"
    
    report = db.exercisereport.find_unique(where={"id": report_id})
    
    if not report:
        abort(404)
    
    if not is_admin and report.userId != user_id:
        return jsonify({"error": "You can only delete your own reports"}), 403
    
    db.exercisereport.delete(where={"id": report_id})
    
    return jsonify({"message": "Report deleted", "id": report_id})


# ---------------------------------------------------------------------------
# Admin: Users API
# ---------------------------------------------------------------------------

@app.route("/api/users", methods=["GET"])
@require_admin
def get_users():
    """Get all users (admin only)."""
    users = db.user.find_many(
        order={"createdAt": "desc"},
        include={
            "videos": True,
            "exerciseReports": True
        }
    )
    
    return jsonify([{
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "role": str(u.role),
        "created_at": u.createdAt.isoformat(),
        "video_count": len(u.videos) if u.videos else 0,
        "report_count": len(u.exerciseReports) if u.exerciseReports else 0,
    } for u in users])


# ---------------------------------------------------------------------------
# Exercise sessions / reports API
# ---------------------------------------------------------------------------

# Module-level RepTracker registry. Flask's dev server is multi-threaded, so
# every read/write must hold _rep_trackers_lock.
_rep_trackers: dict[str, RepTracker] = {}
_rep_trackers_lock = threading.Lock()
_report_generator = ReportGenerator()


def _get_tracker(session_id: str) -> RepTracker | None:
    with _rep_trackers_lock:
        return _rep_trackers.get(session_id)


def _set_tracker(session_id: str, tracker: RepTracker) -> None:
    with _rep_trackers_lock:
        _rep_trackers[session_id] = tracker


def _drop_tracker(session_id: str) -> None:
    with _rep_trackers_lock:
        _rep_trackers.pop(session_id, None)


def _load_session_or_404(session_id: str):
    try:
        sess = db.exercisesession.find_unique(where={"id": session_id})
    except (TableNotFoundError, DataError):
        return None
    if not sess or sess.userId != request.user["sub"]:
        return None
    return sess


@app.route("/api/session/start", methods=["POST"])
@require_auth
def session_start():
    data = request.get_json(silent=True) or {}
    exercise_name = (data.get("exercise_name") or "").strip()
    video_id = data.get("video_id")
    if not exercise_name:
        return jsonify({"error": "exercise_name is required"}), 400

    try:
        sess = db.exercisesession.create(data={
            "id":           str(uuid.uuid4()),
            "userId":       request.user["sub"],
            "exerciseName": exercise_name,
            "videoId":      video_id,
        })
    except (TableNotFoundError, DataError):
        return jsonify({"error": "session_reporting_unavailable"}), 503

    _set_tracker(sess.id, RepTracker(exercise_name=exercise_name))
    return jsonify({"session_id": sess.id}), 201


@app.route("/api/session/<session_id>/frame", methods=["POST"])
@require_auth
def session_frame(session_id: str):
    if not _load_session_or_404(session_id):
        return jsonify({"error": "session not found"}), 404
    tracker = _get_tracker(session_id)
    if tracker is None:
        return jsonify({"error": "tracker not active"}), 410

    payload = (request.get_json(silent=True) or {}).get("frame_score") or {}
    try:
        fs = FrameScore(
            frame_index=int(payload.get("frame_index", 0)),
            joint_errors=dict(payload.get("joint_errors") or {}),
            joint_scores=dict(payload.get("joint_scores") or {}),
            overall_score=float(payload.get("overall_score", 0.0)),
        )
    except (TypeError, ValueError):
        return jsonify({"error": "invalid frame_score"}), 400

    tracker.on_frame(fs)
    return jsonify({"ok": True})


@app.route("/api/session/<session_id>/rep-complete", methods=["POST"])
@require_auth
def session_rep_complete(session_id: str):
    if not _load_session_or_404(session_id):
        return jsonify({"error": "session not found"}), 404
    tracker = _get_tracker(session_id)
    if tracker is None:
        return jsonify({"error": "tracker not active"}), 410

    rep = tracker.on_rep_complete()
    return jsonify({
        "rep_number": rep.rep_number,
        "accuracy":   rep.accuracy,
        "rep_result": {
            "rep_number":          rep.rep_number,
            "accuracy":            rep.accuracy,
            "joint_error_summary": rep.joint_error_summary,
            "joint_accuracy":      rep.joint_accuracy,
            "frame_count":         rep.frame_count,
            "duration_ms":         rep.duration_ms,
        },
    })


@app.route("/api/session/<session_id>/set-complete", methods=["POST"])
@require_auth
def session_set_complete(session_id: str):
    sess = _load_session_or_404(session_id)
    if not sess:
        return jsonify({"error": "session not found"}), 404
    tracker = _get_tracker(session_id)
    if tracker is None:
        return jsonify({"error": "tracker not active"}), 410

    set_result = tracker.on_set_complete()
    report = _report_generator.build_set_report(set_result, sess.exerciseName)

    db.setreport.create(data={
        "id":              str(uuid.uuid4()),
        "sessionId":       sess.id,
        "setNumber":       set_result.set_number,
        "overallAccuracy": float(set_result.set_accuracy),
        "grade":           set_result.grade,
        "reportJson":      json.dumps(report),
    })

    return jsonify(report)


@app.route("/api/session/<session_id>/end", methods=["POST"])
@require_auth
def session_end(session_id: str):
    sess = _load_session_or_404(session_id)
    if not sess:
        return jsonify({"error": "session not found"}), 404
    tracker = _get_tracker(session_id)
    if tracker is None:
        return jsonify({"error": "tracker not active"}), 410

    body = request.get_json(silent=True) or {}
    completed_flag = bool(body.get("completed", False))

    partial_set = tracker.current_partial_set_result()
    session_result = tracker.finalize_session()
    # Honour explicit completion flag from client (e.g. early exit).
    session_result.completed = session_result.completed and completed_flag if completed_flag else session_result.completed
    if completed_flag:
        session_result.completed = True

    if partial_set is not None:
        report = _report_generator.build_set_report(partial_set, sess.exerciseName)
        db.setreport.create(data={
            "id":              str(uuid.uuid4()),
            "sessionId":       sess.id,
            "setNumber":       partial_set.set_number,
            "overallAccuracy": float(partial_set.set_accuracy),
            "grade":           partial_set.grade,
            "reportJson":      json.dumps(report),
        })

    report = _report_generator.build_session_report(session_result)

    db.sessionreport.upsert(
        where={"sessionId": sess.id},
        data={
            "create": {
                "id":              str(uuid.uuid4()),
                "sessionId":       sess.id,
                "overallAccuracy": float(session_result.overall_accuracy),
                "grade":           report["grade"],
                "completed":       bool(session_result.completed),
                "reportJson":      json.dumps(report),
            },
            "update": {
                "overallAccuracy": float(session_result.overall_accuracy),
                "grade":           report["grade"],
                "completed":       bool(session_result.completed),
                "reportJson":      json.dumps(report),
            },
        },
    )

    db.exercisesession.update(
        where={"id": sess.id},
        data={
            "completed":   bool(session_result.completed),
            "completedAt": datetime.now(timezone.utc),
        },
    )

    _drop_tracker(session_id)
    return jsonify(report)


def _parse_report_json(value) -> dict:
    """Prisma may hand back a Json column as either str or dict."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return value or {}


@app.route("/api/session/<session_id>/report", methods=["GET"])
@require_auth
def session_get_report(session_id: str):
    bundle = _fetch_session_bundle(
        session_id,
        requester_user_id=request.user["sub"],
        is_admin=request.user.get("role") == "admin",
    )
    if not bundle:
        return jsonify({"error": "session not found"}), 404
    return jsonify(bundle)


@app.route("/api/session/<session_id>/report/download", methods=["GET"])
@require_auth
def session_download_report_pdf(session_id: str):
    """Download the per-session report as PDF."""
    is_admin = request.user.get("role") == "admin"
    sess = db.exercisesession.find_unique(
        where={"id": session_id},
        include={"user": True},
    )
    if not sess:
        return jsonify({"error": "session not found"}), 404
    if not is_admin and sess.userId != request.user["sub"]:
        return jsonify({"error": "Access denied"}), 403

    payload = _fetch_session_bundle(
        session_id,
        requester_user_id=request.user["sub"],
        is_admin=is_admin,
    )
    if not payload:
        return jsonify({"error": "session not found"}), 404

    try:
        pdf_bytes = _build_session_report_pdf(payload)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    created_stamp = sess.startedAt.strftime("%Y%m%d")
    user_part = _slug_for_filename(sess.user.username if hasattr(sess, "user") and sess.user else "user", fallback="user")
    exercise_part = _slug_for_filename(sess.exerciseName, fallback="exercise")
    filename = f"session-report-{user_part}-{exercise_part}-{created_stamp}-{sess.id[:8]}.pdf"

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/session/<session_id>/skeleton-svg", methods=["GET"])
@require_auth
def session_skeleton_svg(session_id: str):
    """
    Generate an SVG skeleton diagram showing joint accuracy for a session.
    
    Query params:
        width: SVG width in pixels (default 300)
        height: SVG height in pixels (default 400)
        dark: Use dark mode (default true)
    """
    is_admin = request.user.get("role") == "admin"
    
    bundle = _fetch_session_bundle(
        session_id,
        requester_user_id=request.user["sub"],
        is_admin=is_admin,
    )
    if not bundle:
        return jsonify({"error": "session not found"}), 404
    
    # Get parameters
    try:
        width = int(request.args.get("width", 300))
        height = int(request.args.get("height", 400))
    except ValueError:
        width, height = 300, 400
    dark_mode = request.args.get("dark", "true").lower() != "false"
    
    session_payload = bundle.get("session") if isinstance(bundle.get("session"), dict) else {}
    session_report = bundle.get("session_report") if isinstance(bundle.get("session_report"), dict) else {}

    problem_joints = (
        session_report.get("problem_joints")
        if isinstance(session_report.get("problem_joints"), list)
        else []
    )

    if not problem_joints:
        set_reports = bundle.get("set_reports") if isinstance(bundle.get("set_reports"), list) else []
        for set_payload in reversed(set_reports):
            if not isinstance(set_payload, dict):
                continue
            candidate = set_payload.get("problem_joints")
            if isinstance(candidate, list) and candidate:
                problem_joints = candidate
                break

    exercise_name = session_report.get("exercise") if isinstance(session_report.get("exercise"), str) else None
    if not exercise_name:
        exercise_name = session_payload.get("exercise_name") if isinstance(session_payload.get("exercise_name"), str) else "Exercise"
    
    # Generate SVG
    svg = generate_problem_joints_svg(
        problem_joints,
        width=width,
        height=height,
        title=f"{exercise_name} - Joint Accuracy",
        dark_mode=dark_mode,
    )
    
    return svg, 200, {"Content-Type": "image/svg+xml"}


@app.route("/api/set/<set_id>/skeleton-svg", methods=["GET"])
@require_auth
def set_skeleton_svg(set_id: str):
    """
    Generate an SVG skeleton diagram showing joint accuracy for a specific set.
    
    Query params:
        width: SVG width in pixels (default 300)
        height: SVG height in pixels (default 400)
        dark: Use dark mode (default true)
    """
    set_report = db.setreport.find_unique(
        where={"id": set_id},
        include={"session": True},
    )
    if not set_report:
        return jsonify({"error": "set not found"}), 404
    
    is_admin = request.user.get("role") == "admin"
    if not is_admin and set_report.session.userId != request.user["sub"]:
        return jsonify({"error": "access denied"}), 403
    
    # Get parameters
    try:
        width = int(request.args.get("width", 300))
        height = int(request.args.get("height", 400))
    except ValueError:
        width, height = 300, 400
    dark_mode = request.args.get("dark", "true").lower() != "false"
    
    # Parse report JSON
    try:
        report_data = json.loads(set_report.reportJson) if set_report.reportJson else {}
    except json.JSONDecodeError:
        report_data = {}
    
    problem_joints = report_data.get("problem_joints", [])
    set_num = report_data.get("set_number", set_report.setNumber)
    exercise = report_data.get("exercise", set_report.session.exerciseName)
    
    svg = generate_problem_joints_svg(
        problem_joints,
        width=width,
        height=height,
        title=f"Set {set_num} - {exercise}",
        dark_mode=dark_mode,
    )
    
    return svg, 200, {"Content-Type": "image/svg+xml"}


@app.route("/api/sessions", methods=["GET"])
@require_auth
def list_sessions():
    sessions = db.exercisesession.find_many(
        where={"userId": request.user["sub"]},
        order={"startedAt": "desc"},
        include={"sessionReport": True},
    )
    return jsonify([
        {
            "id":               s.id,
            "exercise_name":    s.exerciseName,
            "started_at":       s.startedAt.isoformat(),
            "completed":        s.completed,
            "overall_accuracy": s.sessionReport.overallAccuracy if s.sessionReport else None,
            "grade":            s.sessionReport.grade if s.sessionReport else None,
        }
        for s in sessions
    ])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5001)
