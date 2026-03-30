"""
extractor.py — Extract skeleton landmarks from video using MediaPipe Pose.

Uses the new mediapipe.tasks API (v0.10+) with the PoseLandmarker task.
Model file: models/pose_landmarker_heavy.task

Provides:
  - extract_skeleton_from_video()  → process a video and return per-frame dicts
  - save_skeleton() / load_skeleton()  → JSON serialisation to avoid re-processing
  - LANDMARK_NAMES  → mapping of MediaPipe landmark indices to human-readable names
  - MODEL_PATH  → default path to the pose landmarker model file
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

# Minimum mean visibility across all landmarks to keep a frame.
_MIN_MEAN_VISIBILITY = 0.5


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_skeleton_from_video(
    video_path: str,
    *,
    model_path: str = MODEL_PATH,
    min_mean_visibility: float = _MIN_MEAN_VISIBILITY,
) -> list[dict]:
    """Extract MediaPipe Pose landmarks from every frame of a video.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    model_path:
        Path to the ``pose_landmarker_heavy.task`` model file.
    min_mean_visibility:
        Frames whose mean landmark visibility is below this threshold are
        skipped (i.e. pose was not reliably detected).

    Returns
    -------
    list[dict]
        Each dict has keys ``frame_index``, ``timestamp_ms``, and
        ``landmarks`` (a list of 33 dicts with x/y/z/visibility).
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create PoseLandmarker with VIDEO running mode.
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = PoseLandmarker.create_from_options(options)

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

    with progress:
        task = progress.add_task("Extracting poses", total=total_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Convert BGR → RGB and wrap in mediapipe Image.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detect pose.  timestamp must be monotonically increasing (ms).
            results = landmarker.detect_for_video(mp_image, int(timestamp_ms))

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                pose = results.pose_landmarks[0]  # first (only) person
                landmarks_list = [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    }
                    for lm in pose
                ]

                mean_vis = sum(l["visibility"] for l in landmarks_list) / len(
                    landmarks_list
                )

                if mean_vis >= min_mean_visibility:
                    skeleton_data.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_ms": round(timestamp_ms, 3),
                            "landmarks": landmarks_list,
                        }
                    )
                else:
                    skipped += 1
            else:
                skipped += 1

            frame_index += 1
            progress.update(task, advance=1)

    cap.release()
    landmarker.close()

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
    """Save skeleton data to a JSON file.

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
