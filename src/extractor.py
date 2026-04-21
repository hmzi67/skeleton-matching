"""
extractor.py — Extract skeleton landmarks from video using MediaPipe Pose + Hands.

Uses the MediaPipe Tasks API (v0.10+) for pose detection and the MediaPipe
Solutions API for hand landmark detection.  Both are run on every frame so
the system captures both full-body form and fine-grained hand/finger movement.

Model file: models/pose_landmarker_heavy.task  (or lite / full)
Hand model: bundled in the mediapipe package (no separate download required)

Provides:
  - extract_skeleton_from_video()  → per-frame dicts with pose + hand landmarks
  - save_skeleton() / load_skeleton()  → JSON serialisation
  - LANDMARK_NAMES     → MediaPipe Pose landmark index → name  (33 landmarks)
  - HAND_LANDMARK_NAMES → MediaPipe Hand landmark index → name (21 landmarks)
  - MODEL_PATH         → default path to the pose landmarker model file
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from src.filters import OpticalFlowValidator
from src.normalizer import reconcile_hand_sides
from src.perf_monitor import timed

# ---------------------------------------------------------------------------
# Model path (heavy = model_complexity ≈ 2)
# ---------------------------------------------------------------------------
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_heavy.task")

# ---------------------------------------------------------------------------
# MediaPipe Pose landmark index → name mapping (all 33 landmarks)
# ---------------------------------------------------------------------------
LANDMARK_NAMES: dict[int, str] = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# ---------------------------------------------------------------------------
# MediaPipe Hand landmark index → name mapping (21 landmarks per hand)
# ---------------------------------------------------------------------------
HAND_LANDMARK_NAMES: dict[int, str] = {
    0: "wrist",
    1: "thumb_cmc",  2: "thumb_mcp",  3: "thumb_ip",   4: "thumb_tip",
    5: "index_mcp",  6: "index_pip",  7: "index_dip",  8: "index_tip",
    9: "middle_mcp", 10: "middle_pip", 11: "middle_dip", 12: "middle_tip",
    13: "ring_mcp",  14: "ring_pip",  15: "ring_dip",  16: "ring_tip",
    17: "pinky_mcp", 18: "pinky_pip", 19: "pinky_dip", 20: "pinky_tip",
}

# Minimum mean visibility across all pose landmarks to keep a frame.
_MIN_MEAN_VISIBILITY = 0.5

# Minimum per-landmark visibility to include in output (0.0–1.0).
MIN_LANDMARK_VISIBILITY = 0.5

# Minimum hand detection confidence (score) to include hand results.
MIN_HAND_DETECTION_CONFIDENCE = 0.6

# Ensemble (MediaPipe + MoveNet) placeholder. Blocked on tflite-runtime
# compatibility with Python 3.12 — revisit when runtime story is clearer.
# Keep False; there is no ensemble code path today.
USE_ENSEMBLE: bool = False

# MediaPipe Solutions hands module (no separate model file required).
_mp_hands = mp.solutions.hands


# ---------------------------------------------------------------------------
# Wrist reconciliation
# ---------------------------------------------------------------------------


def _reconcile_wrist(
    pose_wrist: dict | None,
    hand_wrist_lm0: dict | None,
    hand_detection_score: float,
    min_hand_confidence: float = MIN_HAND_DETECTION_CONFIDENCE,
) -> tuple[dict | None, str]:
    """Reconcile pose and hand model wrist landmarks, preferring hand when confident.

    Parameters
    ----------
    pose_wrist : dict | None
        Wrist landmark from PoseLandmarker (if available and not filtered).
    hand_wrist_lm0 : dict | None
        Wrist landmark from hand model (lm0 of the hand landmark list).
    hand_detection_score : float
        Detection confidence score for the hand.
    min_hand_confidence : float
        Threshold above which to prefer hand wrist.

    Returns
    -------
    tuple[dict | None, str]
        (best_wrist_landmark, source_label) where source_label is
        "hand", "pose", or None if both are unavailable.
    """
    if hand_wrist_lm0 is not None and hand_detection_score >= min_hand_confidence:
        return hand_wrist_lm0, "hand"
    return pose_wrist, "pose" if pose_wrist is not None else None


# ---------------------------------------------------------------------------
# Timed wrappers
# ---------------------------------------------------------------------------


@timed("pose_detect")
def _detect_pose_timed(pose_landmarker, mp_image, timestamp_ms: int):
    """Timed wrapper around pose_landmarker.detect_for_video."""
    return pose_landmarker.detect_for_video(mp_image, timestamp_ms)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_skeleton_from_video(
    video_path: str,
    *,
    model_path: str = MODEL_PATH,
    min_mean_visibility: float = _MIN_MEAN_VISIBILITY,
) -> list[dict]:
    """Extract MediaPipe Pose + Hand landmarks from every frame of a video.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    model_path:
        Path to the ``pose_landmarker_*.task`` model file.
    min_mean_visibility:
        Frames whose mean pose landmark visibility is below this threshold
        are skipped (unreliable pose detection).

    Returns
    -------
    list[dict]
        Each dict has keys:
        - ``frame_index``   — int
        - ``timestamp_ms``  — float
        - ``landmarks``     — list of 33 pose landmark dicts  {x, y, z, visibility}
        - ``hand_landmarks`` — dict with ``"left"`` and ``"right"`` keys, each
                               either a list of 21 hand landmark dicts or ``None``
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create PoseLandmarker with VIDEO running mode.
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    # Optical flow validator rejects implausible landmark jumps.
    flow_validator = OpticalFlowValidator()

    skeleton_data: list[dict] = []
    frame_index = 0
    skipped = 0

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    # Run hand detector as a context manager alongside the pose landmarker.
    with _mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands_detector, progress:

        task = progress.add_task("Extracting poses + hands", total=total_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Convert BGR → RGB once; reused for both detectors.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # ---- Pose detection ----
            pose_results = _detect_pose_timed(pose_landmarker, mp_image, int(timestamp_ms))

            if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
                pose = pose_results.pose_landmarks[0]
                landmarks_list = []
                for lm in pose:
                    # Filter out low-visibility landmarks by setting to None.
                    if lm.visibility < MIN_LANDMARK_VISIBILITY:
                        landmarks_list.append(None)
                    else:
                        landmarks_list.append({
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility,
                        })

                # Compute mean visibility from non-None landmarks.
                valid_visibilities = [l["visibility"] for l in landmarks_list if l is not None]
                mean_vis = sum(valid_visibilities) / len(valid_visibilities) if valid_visibilities else 0.0

                if mean_vis >= min_mean_visibility:
                    # Optical flow validation: reject landmarks that jumped
                    # further than flow predicts (catches detector snaps).
                    landmarks_list = flow_validator.validate(
                        frame, landmarks_list, frame_shape=frame.shape[:2]
                    )
                    # ---- Hand detection ----
                    # The solutions API needs a writable array.
                    frame_rgb.flags.writeable = True
                    hand_results = hands_detector.process(frame_rgb)
                    frame_rgb.flags.writeable = False

                    hand_lms_dict: dict[str, list[dict] | None] = {
                        "left": None,
                        "right": None,
                    }
                    if (
                        hand_results.multi_hand_landmarks
                        and hand_results.multi_handedness
                    ):
                        for handedness, hand_lms in zip(
                            hand_results.multi_handedness,
                            hand_results.multi_hand_landmarks,
                        ):
                            # Filter hand results by detection confidence.
                            score = handedness.classification[0].score
                            if score < MIN_HAND_DETECTION_CONFIDENCE:
                                continue  # skip low-confidence detections

                            # MediaPipe reports handedness from the camera's POV.
                            label = handedness.classification[0].label.lower()
                            hand_lms_dict[label] = [
                                {
                                    "x": lm.x,
                                    "y": lm.y,
                                    "z": lm.z,
                                    "visibility": 1.0,  # hands API has no visibility
                                }
                                for lm in hand_lms.landmark
                            ]

                    hand_lms_dict = reconcile_hand_sides(landmarks_list, hand_lms_dict)

                    # Wrist reconciliation: prefer hand wrist when confidence is high.
                    wrist_source = {"left": None, "right": None}
                    hand_confidence_scores = {}

                    # Re-scan hand results to capture scores for reconciliation.
                    if hand_results.multi_handedness and hand_results.multi_hand_landmarks:
                        for handedness, hand_lms in zip(
                            hand_results.multi_handedness,
                            hand_results.multi_hand_landmarks,
                        ):
                            label = handedness.classification[0].label.lower()
                            score = handedness.classification[0].score
                            hand_confidence_scores[label] = score

                    for side in ("left", "right"):
                        pose_wrist_idx = 15 if side == "left" else 16
                        pose_wrist = landmarks_list[pose_wrist_idx] if pose_wrist_idx < len(landmarks_list) else None

                        hand_wrist = None
                        hand_score = 0.0
                        if hand_lms_dict.get(side):
                            hand_wrist = hand_lms_dict[side][0]  # first landmark is wrist
                            hand_score = hand_confidence_scores.get(side, 0.0)

                        _, source = _reconcile_wrist(pose_wrist, hand_wrist, hand_score)
                        wrist_source[side] = source

                    skeleton_data.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_ms": round(timestamp_ms, 3),
                            "landmarks": landmarks_list,
                            "hand_landmarks": hand_lms_dict,
                            "wrist_source": wrist_source,
                        }
                    )
                else:
                    flow_validator.reset()
                    skipped += 1
            else:
                flow_validator.reset()
                skipped += 1

            frame_index += 1
            progress.update(task, advance=1)

    cap.release()
    pose_landmarker.close()

    progress.console.print(
        f"[green]✓[/green] Done — "
        f"[bold]{len(skeleton_data)}[/bold] frames extracted, "
        f"[dim]{skipped} skipped (low visibility)[/dim]"
    )

    return skeleton_data


# ---------------------------------------------------------------------------
# JSON persistence helpers
# ---------------------------------------------------------------------------


def save_skeleton(skeleton_data: list[dict], output_path: str) -> None:
    """Save skeleton data (pose + hand landmarks) to a JSON file.

    Parameters
    ----------
    skeleton_data:
        The list of frame dicts returned by ``extract_skeleton_from_video``.
    output_path:
        Destination file path (will be created / overwritten).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(skeleton_data, f, indent=2)
    print(f"Skeleton data saved → {path}")


def load_skeleton(path: str) -> list[dict]:
    """Load skeleton data from a previously saved JSON file.

    Backward-compatible: files saved before hand-landmark support was added
    will have no ``hand_landmarks`` key; the rest of the pipeline handles
    this gracefully via ``.get("hand_landmarks", {})``.

    Parameters
    ----------
    path:
        Path to the JSON file written by ``save_skeleton``.

    Returns
    -------
    list[dict]
        Same structure as ``extract_skeleton_from_video`` output.
    """
    with open(path, "r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)
    print(f"Loaded {len(data)} frames from {path}")
    return data
