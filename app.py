"""
app.py — Pose Matcher backend.
Auth:  JWT Bearer tokens + Session table in PostgreSQL via Prisma.
Roles: user | admin
"""

from __future__ import annotations

import base64
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
from prisma import Prisma
from src.calibration import calibrate_from_skeleton, compute_adaptive_thresholds
from src.exercise_weights import get_weights, compute_auto_weights, EXERCISE_WEIGHTS
from src.extractor import extract_skeleton_from_video, load_skeleton
from src.feedback import generate_feedback
from src.filters import LandmarkSmoother
from src.matcher import match_single_frame
from src.normalizer import normalize_skeleton
from src.state_machine import ExerciseStateMachine
from werkzeug.utils import secure_filename

try:
    from flask_sock import Sock
except ModuleNotFoundError:
    Sock = None

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SECRET_KEY   = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
TOKEN_EXPIRY = timedelta(hours=8)

PENDING_DIR  = Path("data/pending")
APPROVED_DIR = Path("data/ground_truth")
PENDING_DIR.mkdir(parents=True, exist_ok=True)
APPROVED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB limit
SKELETON_CACHE_DIR = Path("data/cache/reference_skeletons")
SKELETON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [24, 26], [26, 28],
    [27, 31], [28, 32], [15, 17], [16, 18], [15, 19], [16, 20],
]

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
    (15, 17): "left_elbow",    (16, 18): "right_elbow",
    (15, 19): "left_elbow",    (16, 20): "right_elbow",
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
LIVE_MATCH_WINDOW = max(1, int(os.environ.get("LIVE_MATCH_WINDOW", "5")))
LIVE_MATCH_DEBUG_TIMINGS = os.environ.get("LIVE_MATCH_DEBUG_TIMINGS", "0").strip().lower() in {"1", "true", "yes", "on"}


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
    _forward_window: int = 8
    _backward_tolerance: int = 3

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
    smoothed_score: float = 0.0
    last_live_angles: dict[str, float] | None = None
    last_live_ts_ms: float | None = None
    idle_frame_count: int = 0
    temporal_aligner: TemporalAligner = field(default_factory=TemporalAligner)
    phase_machine: ExerciseStateMachine = field(default_factory=ExerciseStateMachine)
    landmark_smoother: LandmarkSmoother = field(
        default_factory=lambda: LandmarkSmoother(min_cutoff=2.5, beta=0.4),
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
        db.connect()


# ---------------------------------------------------------------------------
# Seed default admin on first run
# ---------------------------------------------------------------------------

def _seed_admin() -> None:
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


def _extract_landmarks(image_bgr: np.ndarray) -> list[dict] | None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = _pose_detector.detect(mp_image)
    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return None

    landmarks: list[dict] = []
    for lm in results.pose_landmarks[0]:
        landmarks.append({
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        })
    return landmarks


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


def _extract_angles(landmarks: list[dict]) -> dict[str, float]:
    angles: dict[str, float] = {}
    for name, vertex, a, c in _ANGLE_JOINTS:
        angles[name] = _angle(landmarks[a], landmarks[vertex], landmarks[c])

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
    if video_id in _reference_norm_cache:
        return _reference_norm_cache[video_id]

    cache_path = SKELETON_CACHE_DIR / f"{video_id}.normalized.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _reference_norm_cache[video_id] = data
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

    _reference_norm_cache[video_id] = normalized
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


_CURATED_PRIMARY_ANGLES: dict[str, list[str]] = {
    "squat":          ["left_knee", "right_knee"],
    "lunge":          ["left_knee", "right_knee"],
    "deadlift":       ["left_hip", "right_hip"],
    "pushup":         ["left_elbow", "right_elbow"],
    "shoulder_press": ["left_elbow", "right_elbow"],
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
    """
    curated = _CURATED_PRIMARY_ANGLES.get(exercise.lower())
    if curated:
        vals = [angles.get(j, 0.0) for j in curated]
        return sum(vals) / max(len(vals), 1)

    if primary_joint:
        val = angles.get(primary_joint, 0.0)
        sym = _SYMMETRY_MAP.get(primary_joint)
        if sym and sym in angles:
            val = (val + angles[sym]) / 2.0
        return val

    return (angles.get("left_knee", 0.0) + angles.get("right_knee", 0.0)) / 2.0


def _state_machine_for_exercise(exercise: str) -> ExerciseStateMachine:
    if exercise in {"squat", "lunge"}:
        return ExerciseStateMachine(down_threshold=120.0, up_threshold=155.0, hold_frames=2)
    if exercise == "deadlift":
        return ExerciseStateMachine(down_threshold=100.0, up_threshold=150.0, hold_frames=2)
    if exercise == "pushup":
        return ExerciseStateMachine(down_threshold=95.0, up_threshold=155.0, hold_frames=2)
    if exercise == "shoulder_press":
        return ExerciseStateMachine(down_threshold=90.0, up_threshold=150.0, hold_frames=2)
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


def _compute_bone_statuses(best_match: dict) -> dict[str, str]:
    bone_statuses: dict[str, str] = {}
    for bone in POSE_CONNECTIONS:
        key = f"{bone[0]}-{bone[1]}"
        joint_name = _BONE_JOINT_MAP.get((bone[0], bone[1])) or _BONE_JOINT_MAP.get((bone[1], bone[0]))
        if joint_name and joint_name in best_match and isinstance(best_match[joint_name], dict):
            bone_statuses[key] = best_match[joint_name].get("status", "good")
        else:
            bone_statuses[key] = "good"
    return bone_statuses


def _compute_joint_statuses(best_match: dict) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for joint_name in _JOINT_LANDMARK_IDX:
        if joint_name in best_match and isinstance(best_match[joint_name], dict):
            statuses[joint_name] = best_match[joint_name].get("status", "good")
    if "torso_lean" in best_match and isinstance(best_match["torso_lean"], dict):
        statuses["torso_lean"] = best_match["torso_lean"].get("status", "good")
    return statuses


def _compute_correction_arrows(
    best_match: dict, live_landmarks: list[dict],
) -> list[dict]:
    joint_errors: list[tuple[str, float]] = []
    for joint_name, idx in _JOINT_LANDMARK_IDX.items():
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


def _build_session_config(
    gt_norm: list[dict], exercise: str,
) -> dict:
    """Derive auto-weights, adaptive thresholds, primary joint, and
    phase-machine parameters from the reference video ROM.

    Curated presets override auto-weights for known exercises.
    """
    auto = compute_auto_weights(gt_norm)

    if exercise.lower() in EXERCISE_WEIGHTS and exercise.lower() != "default":
        weights = get_weights(exercise)
    else:
        weights = auto["weights"]

    rom_profile = calibrate_from_skeleton(gt_norm)
    adaptive_thresholds = compute_adaptive_thresholds(rom_profile)

    primary_joint = auto["primary_joint"]
    primary_min = auto["primary_min"]
    primary_max = auto["primary_max"]

    rom_range = primary_max - primary_min
    down_thresh = primary_min + rom_range * 0.20
    up_thresh = primary_max - rom_range * 0.20

    if exercise.lower() in {"squat", "lunge", "deadlift", "pushup", "shoulder_press"}:
        phase_machine = _state_machine_for_exercise(exercise)
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
    }


def _get_or_create_live_state(user_id: str, ref_id: str, exercise: str) -> LiveSessionState:
    key = _live_session_key(user_id, ref_id, exercise)
    existing = _http_live_states.get(key)
    if existing is not None:
        return existing

    gt_norm = _load_or_extract_reference_norm(ref_id)
    ref_timestamps_ms = [float(frame.get("timestamp_ms", 0.0)) for frame in gt_norm]
    cfg = _build_session_config(gt_norm, exercise)
    state = LiveSessionState(
        user_id=user_id,
        ref_video_id=ref_id,
        exercise=exercise,
        gt_norm=gt_norm,
        ref_timestamps_ms=ref_timestamps_ms,
        auto_weights=cfg["auto_weights"],
        adaptive_thresholds=cfg["adaptive_thresholds"],
        primary_joint=cfg["primary_joint"],
        primary_min=cfg["primary_min"],
        primary_max=cfg["primary_max"],
        phase_machine=cfg["phase_machine"],
    )
    _http_live_states[key] = state
    return state


def _process_live_frame_message(
    state: LiveSessionState,
    live_frame: str,
    ref_time_ms: float,
) -> dict:
    total_t0 = time.perf_counter()
    decode_t0 = time.perf_counter()
    live_img = _decode_data_url_to_bgr(live_frame)
    decode_ms = (time.perf_counter() - decode_t0) * 1000.0
    if live_img is None:
        return {"type": "error", "error": "invalid_live_frame"}

    detect_t0 = time.perf_counter()
    raw_landmarks = _extract_landmarks(live_img)
    detect_ms = (time.perf_counter() - detect_t0) * 1000.0
    if not raw_landmarks:
        state.landmark_smoother.reset()
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

    live_landmarks = state.landmark_smoother.smooth(
        time.time(), raw_landmarks,
    )
    live_angles = _extract_angles(live_landmarks)
    now_ms = time.time() * 1000.0

    _IDLE_ANGLE_THRESHOLD = 3.0
    _IDLE_FRAME_LIMIT = 10
    _IDLE_SCORE_CAP = 30.0

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

    if user_is_idle:
        best_score = min(best_score, _IDLE_SCORE_CAP)

    aligner.update_after_match(best_idx, ref_time_ms, state.ref_timestamps_ms)
    state.smoothed_score = 0.5 * best_score + 0.5 * state.smoothed_score
    best_match["overall_score"] = state.smoothed_score

    summary_stub = {
        "overall_score": state.smoothed_score,
        "per_joint_avg_error": {},
        "worst_joint": "",
        "best_joint": "",
        "frame_scores": [],
    }
    feedback = generate_feedback(summary_stub, best_match)
    primary_angle = _primary_angle_for_exercise(
        state.exercise, live_angles, primary_joint=state.primary_joint,
    )
    phase = state.phase_machine.update(primary_angle, state.smoothed_score)

    gt_frame = state.gt_norm[best_idx]

    display_ref_idx = _closest_reference_index(state.ref_timestamps_ms, ref_time_ms)
    display_ref_frame = state.gt_norm[display_ref_idx]

    joint_statuses = _compute_joint_statuses(best_match)
    bone_statuses = _compute_bone_statuses(best_match)
    correction_arrows = _compute_correction_arrows(best_match, live_landmarks)

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

    payload = {
        "type": "frame_result",
        "pose_detected": True,
        "match_score": int(round(state.smoothed_score)),
        "matched_ref_frame_index": int(best_idx),
        "matched_ref_source_frame": int(gt_frame.get("frame_index", best_idx)),
        "joint_diffs": {k: v.get("abs_diff", 0.0) for k, v in best_match.items() if isinstance(v, dict) and "abs_diff" in v},
        "joint_statuses": joint_statuses,
        "bone_statuses": bone_statuses,
        "correction_arrows": correction_arrows,
        "coaching_text": coaching_text,
        "connections": POSE_CONNECTIONS,
        "live_landmarks": live_landmarks,
        "ref_landmarks": display_ref_frame.get("landmarks", []),
        "feedback": {
            "headline": feedback.get("headline", ""),
            "priority_fix": feedback.get("priority_fix", ""),
            "top_joint_feedback": feedback.get("joint_feedback", [])[:3],
            "velocity_feedback": feedback.get("velocity_feedback", []),
            "symmetry_warnings": feedback.get("symmetry_warnings", []),
        },
        "phase": phase,
        "rep_count": int(state.phase_machine.rep_count),
        "temporal": {
            "offset_ms": round(offset_ms),
            "status": timing_status,
        },
    }
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

@app.route("/api/exercises", methods=["GET"])
def list_exercises():
    videos = db.video.find_many(
        where={"status": "approved"},
        include={"uploader": True},
        order={"uploadedAt": "desc"},
    )
    seen: dict[str, dict] = {}
    for v in videos:
        if v.exercise not in seen:
            seen[v.exercise] = {
                "exercise": v.exercise,
                "video_id": v.id,
                "uploaded_at": v.uploadedAt.isoformat(),
                "uploader": {
                    "id": v.uploader.id if v.uploader else None,
                    "username": v.uploader.username if v.uploader else "unknown",
                    "email": v.uploader.email if v.uploader else None,
                    "created_at": v.uploader.createdAt.isoformat() if v.uploader else None,
                },
            }
    return jsonify(list(seen.values()))


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
    exercise = (data.get("exercise") or "default").strip().lower()

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
        exercise = (request.args.get("exercise") or "default").strip().lower()

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
        )

        ws.send(json.dumps({
            "type": "ready",
            "exercise": exercise,
            "frame_count": len(gt_norm),
            "connections": POSE_CONNECTIONS,
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
    """List approved categories (public) or all categories (admin)."""
    auth_header = _get_bearer()
    is_admin = False
    if auth_header:
        data = _decode_token_and_session(auth_header)
        if data and data.get("role") == "admin":
            is_admin = True

    if is_admin and request.args.get("all") == "true":
        categories = db.category.find_many(
            include={"suggestedBy": True},
            order={"createdAt": "desc"},
        )
    else:
        categories = db.category.find_many(
            where={"status": "approved"},
            order={"name": "asc"},
        )

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


@app.route("/api/reports", methods=["POST"])
@require_auth
def create_report():
    """Save an exercise session report."""
    user_id = request.user["sub"]
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required_fields = ["exercise", "reference_video_id", "duration_seconds", "score_history"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    score_history = data.get("score_history", [])
    scores = [s for s in score_history if isinstance(s, (int, float))]
    
    avg_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    overall_grade = calculate_grade(avg_score)
    
    import json
    
    report = db.exercisereport.create(
        data={
            "userId": user_id,
            "exercise": data.get("exercise", "Unknown"),
            "referenceVideoId": data.get("reference_video_id", ""),
            "durationSeconds": data.get("duration_seconds", 0),
            "totalFrames": len(scores),
            "repCount": data.get("rep_count", 0),
            "avgScore": round(avg_score, 2),
            "minScore": round(min_score, 2),
            "maxScore": round(max_score, 2),
            "scoreHistory": json.dumps(score_history),
            "phaseHistory": json.dumps(data.get("phase_history", [])),
            "jointIssues": json.dumps(data.get("joint_issues", {})),
            "boneIssues": json.dumps(data.get("bone_issues", {})),
            "feedbackSummary": json.dumps(data.get("feedback_summary", [])),
            "overallGrade": overall_grade,
        }
    )
    
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
    
    import json
    
    return jsonify([{
        "id": r.id,
        "exercise": r.exercise,
        "reference_video_id": r.referenceVideoId,
        "duration_seconds": r.durationSeconds,
        "total_frames": r.totalFrames,
        "rep_count": r.repCount,
        "avg_score": r.avgScore,
        "min_score": r.minScore,
        "max_score": r.maxScore,
        "overall_grade": r.overallGrade,
        "score_history": json.loads(r.scoreHistory),
        "phase_history": json.loads(r.phaseHistory),
        "joint_issues": json.loads(r.jointIssues),
        "bone_issues": json.loads(r.boneIssues),
        "feedback_summary": json.loads(r.feedbackSummary),
        "created_at": r.createdAt.isoformat(),
        "user": {
            "id": r.user.id,
            "username": r.user.username
        } if hasattr(r, 'user') and r.user else None
    } for r in reports])


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
    
    import json
    
    return jsonify({
        "id": report.id,
        "exercise": report.exercise,
        "reference_video_id": report.referenceVideoId,
        "duration_seconds": report.durationSeconds,
        "total_frames": report.totalFrames,
        "rep_count": report.repCount,
        "avg_score": report.avgScore,
        "min_score": report.minScore,
        "max_score": report.maxScore,
        "overall_grade": report.overallGrade,
        "score_history": json.loads(report.scoreHistory),
        "phase_history": json.loads(report.phaseHistory),
        "joint_issues": json.loads(report.jointIssues),
        "bone_issues": json.loads(report.boneIssues),
        "feedback_summary": json.loads(report.feedbackSummary),
        "created_at": report.createdAt.isoformat(),
        "user": {
            "id": report.user.id,
            "username": report.user.username
        } if report.user else None
    })


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
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5001)
