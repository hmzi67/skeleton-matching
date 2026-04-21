"""
main.py — CLI entry point for the Pose Matcher system.

Usage
-----
  # Compare two videos
  python main.py --gt data/ground_truth/squat.mp4 --user data/user_input/my_squat.mp4

  # Live webcam comparison against ground truth
  python main.py --gt data/ground_truth/squat.mp4 --live

  # Extract skeleton only
  python main.py --extract-only data/ground_truth/squat.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.extractor import extract_skeleton_from_video, load_skeleton, save_skeleton
from src.feedback import generate_feedback, generate_session_report
from src.matcher import compute_summary, match_single_frame, match_video_sequence
from src.normalizer import normalize_skeleton
from src.stability import AngleSmoother, FeedbackStabilizer, JointStatusStabilizer
from src.visualizer import (
    create_side_by_side,
    draw_feedback_overlay,
    draw_skeleton_on_frame,
    save_output_video,
    score_over_time_chart,
)

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skeleton_cache_path(video_path: str) -> Path:
    """Return the expected JSON cache path for a given video."""
    return Path(video_path).with_suffix(".json")


def _get_skeleton(video_path: str) -> list[dict]:
    """Load cached skeleton JSON or extract from video and save it."""
    cache = _skeleton_cache_path(video_path)
    if cache.exists():
        console.print(f"[dim]Loading cached skeleton → {cache}[/dim]")
        return load_skeleton(str(cache))
    console.print(f"[bold]Extracting skeleton from [cyan]{video_path}[/cyan]…[/bold]")
    data = extract_skeleton_from_video(video_path)
    if not data:
        console.print("[red]✗ No pose detected in video.[/red]")
        sys.exit(1)
    save_skeleton(data, str(cache))
    return data


def _print_summary(summary: dict) -> None:
    """Print a rich-formatted summary table to the terminal."""
    score = summary["overall_score"]
    grade = (
        "Excellent" if score >= 90
        else "Good" if score >= 75
        else "Needs Work" if score >= 50
        else "Poor"
    )
    grade_color = (
        "green" if score >= 90
        else "yellow" if score >= 75
        else "dark_orange" if score >= 50
        else "red"
    )

    console.print()
    console.print(
        Panel(
            f"[bold {grade_color}]{score:.0f}/100  —  {grade}[/bold {grade_color}]",
            title="[bold]Overall Score[/bold]",
            border_style=grade_color,
            width=50,
        )
    )

    # Per-joint table.
    table = Table(title="Per-Joint Average Error", show_lines=True)
    table.add_column("Joint", style="cyan")
    table.add_column("Avg Error (°)", justify="right")
    table.add_column("Status")

    for joint, err in sorted(
        summary["per_joint_avg_error"].items(), key=lambda x: x[1]
    ):
        if err < 10:
            color, status = "green", "✓ Good"
        elif err < 25:
            color, status = "yellow", "⚠ Warning"
        else:
            color, status = "red", "✗ Bad"
        table.add_row(
            joint.replace("_", " ").title(),
            f"[{color}]{err:.1f}°[/{color}]",
            f"[{color}]{status}[/{color}]",
        )
    console.print(table)

    # Top tips.
    worst = summary["worst_joint"].replace("_", " ").title()
    best = summary["best_joint"].replace("_", " ").title()
    console.print(f"\n[bold green]💪 Best joint:[/bold green]  {best}")
    console.print(f"[bold red]🔧 Work on:[/bold red]    {worst}")
    console.print()


# ---------------------------------------------------------------------------
# Mode: extract-only
# ---------------------------------------------------------------------------


def _run_extract_only(video_path: str) -> None:
    """Extract skeleton from a single video and save as JSON."""
    if not Path(video_path).exists():
        console.print(f"[red]✗ File not found: {video_path}[/red]")
        sys.exit(1)
    data = extract_skeleton_from_video(video_path)
    if not data:
        console.print("[red]✗ No pose detected.[/red]")
        sys.exit(1)
    out = str(Path(video_path).with_suffix(".json"))
    save_skeleton(data, out)
    console.print(f"[green]✓ Saved → {out}[/green]")


# ---------------------------------------------------------------------------
# Mode: live webcam
# ---------------------------------------------------------------------------


def _run_live(gt_path: str) -> None:
    """Compare webcam feed against a ground-truth video in real time."""
    if not Path(gt_path).exists():
        console.print(f"[red]✗ GT file not found: {gt_path}[/red]")
        sys.exit(1)

    gt_skeleton = _get_skeleton(gt_path)
    gt_norm = normalize_skeleton(gt_skeleton)
    if not gt_norm:
        console.print("[red]✗ Could not normalise ground truth.[/red]")
        sys.exit(1)

    # Open GT video for frame playback.
    gt_cap = cv2.VideoCapture(gt_path)
    if not gt_cap.isOpened():
        console.print("[red]✗ Cannot open GT video for playback.[/red]")
        sys.exit(1)
    gt_fps = gt_cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Pre-read all GT video frames so we can loop them easily.
    console.print("[dim]Loading GT video frames for side-by-side display...[/dim]")
    gt_video_frames: list = []
    while True:
        ret, gf = gt_cap.read()
        if not ret:
            break
        gt_video_frames.append(gf)
    gt_cap.release()

    if not gt_video_frames:
        console.print("[red]✗ GT video has no readable frames.[/red]")
        sys.exit(1)
    console.print(f"[dim]Loaded {len(gt_video_frames)} GT frames[/dim]")

    # Build a lookup from frame_index → skeleton landmarks for the GT.
    gt_lm_lookup: dict[int, list[dict]] = {
        f["frame_index"]: f["landmarks"] for f in gt_skeleton
    }

    # Open webcam.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print("[red]✗ Cannot open webcam.[/red]")
        sys.exit(1)

    console.print("[bold green]Live mode - press Q to quit[/bold green]")

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions as _BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker as _PoseLandmarker,
        PoseLandmarkerOptions as _PoseLandmarkerOptions,
        PoseLandmarkerResult as _PoseLandmarkerResult,
        RunningMode as _RunningMode,
    )
    from src.extractor import MODEL_PATH

    # Shared state for the async callback.
    _latest_result: list[_PoseLandmarkerResult | None] = [None]

    def _on_result(
        result: _PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        _latest_result[0] = result

    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=_RunningMode.LIVE_STREAM,
        num_poses=1,
        result_callback=_on_result,
    )
    landmarker = _PoseLandmarker.create_from_options(options)

    gt_idx = 0
    frame_ts = 0  # monotonically increasing timestamp (ms)
    score_history: list[float] = []
    webcam_frame_count = 0

    # --- Smoothing / stability state ---

    # Exponential moving average for the displayed score.
    _SCORE_EMA_ALPHA = 0.3  # lower = smoother (0.3 gives ~3-frame lag)
    smoothed_score: float = 50.0

    # Angle smoothing + feedback stability (window + EMA + hysteresis + debounce).
    _ANGLE_WINDOW = 7
    _ANGLE_EMA_ALPHA = 0.3
    _STATUS_HYSTERESIS = 5.0
    _STATUS_MAJORITY = 7
    _FEEDBACK_DEBOUNCE = 7

    angle_smoother = AngleSmoother(window_size=_ANGLE_WINDOW, ema_alpha=_ANGLE_EMA_ALPHA)
    status_stabilizer = JointStatusStabilizer(
        hysteresis_buffer=_STATUS_HYSTERESIS,
        majority_window=_STATUS_MAJORITY,
    )
    feedback_stabilizer = FeedbackStabilizer(
        debounce_frames=_FEEDBACK_DEBOUNCE,
        phase_stable_frames=1,
        speech_cooldown_ms=0,
    )

    # GT video pacing: advance every N webcam frames.
    # With webcam ~30fps, GT_ADVANCE_EVERY=2 → GT plays at ~15fps.
    _GT_ADVANCE_EVERY = 2

    # Create a named window so we can resize it.
    cv2.namedWindow("Pose Matcher - Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Matcher - Live", 1200, 605)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        webcam_frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        frame_ts += 33  # ~30 fps
        landmarker.detect_async(mp_image, frame_ts)

        # --- GT video: sequential playback at controlled pace ---
        gt_video_idx = gt_idx % len(gt_video_frames)
        gt_frame_img = gt_video_frames[gt_video_idx].copy()

        gt_norm_frame = gt_norm[gt_idx % len(gt_norm)]
        gt_frame_index = gt_norm_frame["frame_index"]
        gt_lm = gt_lm_lookup.get(
            gt_frame_index,
            gt_skeleton[gt_idx % len(gt_skeleton)]["landmarks"],
        )

        # Advance GT at a steady pace (not every webcam frame).
        if webcam_frame_count % _GT_ADVANCE_EVERY == 0:
            gt_idx += 1

        results = _latest_result[0]
        if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
            pose = results.pose_landmarks[0]
            user_landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in pose
            ]

            user_frame_raw = {
                "frame_index": 0,
                "timestamp_ms": 0.0,
                "landmarks": user_landmarks,
            }
            user_norm = normalize_skeleton([user_frame_raw])

            if user_norm:
                user_angles = user_norm[0].get("angles", {})
                smoothed_angles = angle_smoother.update(user_angles)

                # Use smoothed angles for matching to reduce jitter.
                result = match_single_frame(gt_norm_frame, {"angles": smoothed_angles})
                status_stabilizer.stabilize_match(result)

                raw_score = result["overall_score"]

                # --- EMA score smoothing ---
                smoothed_score = _SCORE_EMA_ALPHA * raw_score + (1 - _SCORE_EMA_ALPHA) * smoothed_score

                # Use smoothed score in the feedback.
                result_smoothed = dict(result)
                result_smoothed["overall_score"] = smoothed_score

                summary_stub = {
                    "overall_score": smoothed_score,
                    "per_joint_avg_error": {},
                    "worst_joint": "",
                    "best_joint": "",
                    "frame_scores": [],
                }
                feedback = generate_feedback(summary_stub, result_smoothed)

                joint_fb = feedback.get("joint_feedback", [])
                candidate_text = joint_fb[0]["instruction"] if joint_fb else (
                    "Great form!" if smoothed_score >= 90 else feedback.get("priority_fix", "")
                )
                stable_text, _ = feedback_stabilizer.update(
                    candidate_text,
                    "READY",
                    time.time() * 1000.0,
                )

                if stable_text:
                    if joint_fb:
                        joint_fb[0]["instruction"] = stable_text
                    feedback["priority_fix"] = stable_text
                else:
                    feedback["joint_feedback"] = []
                    feedback["priority_fix"] = ""

                feedback["overall_score"] = int(smoothed_score)
                score_history.append(smoothed_score)

                combined = create_side_by_side(
                    gt_frame_img, frame, gt_lm, user_landmarks,
                    feedback,
                    score_history=score_history,
                )
                cv2.imshow("Pose Matcher - Live", combined)
            else:
                angle_smoother.reset()
                status_stabilizer.reset()
                feedback_stabilizer.reset()
                _show_raw_side_by_side(frame, gt_frame_img)
        else:
            angle_smoother.reset()
            status_stabilizer.reset()
            feedback_stabilizer.reset()
            _show_raw_side_by_side(frame, gt_frame_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


def _show_raw_side_by_side(
    user_frame: "np.ndarray",
    gt_frame: "np.ndarray",
) -> None:
    """Show a basic side-by-side when no user pose is detected (Kemtai layout)."""
    import numpy as np

    target_h = 540
    user_w = int(target_h * 4 / 3)
    gt_w = int(target_h * 3 / 4)
    bar_gap = 30

    user_resized = cv2.resize(user_frame, (user_w, target_h))
    gt_resized = cv2.resize(gt_frame, (gt_w, target_h))

    total_w = user_w + bar_gap + gt_w
    canvas = np.zeros((target_h, total_w, 3), dtype=np.uint8)
    canvas[:, :user_w] = user_resized
    canvas[:, user_w + bar_gap:] = gt_resized
    canvas[:, user_w:user_w + bar_gap] = (20, 20, 20)

    # "Waiting" text overlay on user panel.
    cv2.putText(user_resized, "Step into frame...", (20, target_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
    canvas[:, :user_w] = user_resized

    # Bottom strip.
    strip_h = 65
    strip = np.zeros((strip_h, total_w, 3), dtype=np.uint8)
    strip[:] = (20, 20, 20)

    cv2.imshow("Pose Matcher - Live", np.vstack([canvas, strip]))


# ---------------------------------------------------------------------------
# Mode: video comparison
# ---------------------------------------------------------------------------


def _run_comparison(gt_path: str, user_path: str) -> None:
    """Compare a user video against a ground-truth video."""
    for p, label in [(gt_path, "Ground truth"), (user_path, "User")]:
        if not Path(p).exists():
            console.print(f"[red]✗ {label} file not found: {p}[/red]")
            sys.exit(1)

    # 1. Extract / load skeletons.
    gt_skeleton = _get_skeleton(gt_path)
    user_skeleton = _get_skeleton(user_path)

    # 2. Normalise.
    gt_norm = normalize_skeleton(gt_skeleton)
    user_norm = normalize_skeleton(user_skeleton)

    if not gt_norm or not user_norm:
        console.print("[red]✗ Could not normalise one or both skeletons.[/red]")
        sys.exit(1)

    # 3. Match.
    console.print("[bold]Running DTW alignment…[/bold]")
    match_results = match_video_sequence(gt_norm, user_norm)

    # 4. Summary.
    summary = compute_summary(match_results)
    _print_summary(summary)

    # 5. Session report.
    report = generate_session_report(summary)
    console.print(Panel(report, title="Session Report", border_style="cyan"))

    # 6. Score chart.
    score_over_time_chart(summary["frame_scores"])

    # 7. Generate output video.
    console.print("[bold]Generating output video…[/bold]")
    gt_cap = cv2.VideoCapture(gt_path)
    user_cap = cv2.VideoCapture(user_path)
    fps = gt_cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_frames: list = []
    for mr in match_results:
        gt_fidx = mr["gt_frame_index"]
        user_fidx = mr["user_frame_index"]

        gt_cap.set(cv2.CAP_PROP_POS_FRAMES, gt_fidx)
        user_cap.set(cv2.CAP_PROP_POS_FRAMES, user_fidx)

        ret_g, gt_frame = gt_cap.read()
        ret_u, user_frame = user_cap.read()
        if not ret_g or not ret_u:
            continue

        # Find landmarks for these indices.
        gt_lm = next(
            (f["landmarks"] for f in gt_skeleton if f["frame_index"] == gt_fidx),
            None,
        )
        user_lm = next(
            (f["landmarks"] for f in user_skeleton if f["frame_index"] == user_fidx),
            None,
        )
        if gt_lm is None or user_lm is None:
            continue

        joint_status = {
            j: mr[j]["status"]
            for j in mr
            if isinstance(mr[j], dict) and "status" in mr[j]
        }

        feedback = generate_feedback(summary, mr)
        combined = create_side_by_side(
            gt_frame, user_frame, gt_lm, user_lm,
            feedback, joint_status=joint_status,
        )
        output_frames.append(combined)

    gt_cap.release()
    user_cap.release()

    if output_frames:
        save_output_video(output_frames, "output/result.mp4", fps)

        # Playback.
        console.print("[bold]Playing result — press Q to quit[/bold]")
        for f in output_frames:
            cv2.imshow("Pose Matcher — Result", f)
            delay = max(1, int(1000 / fps))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    else:
        console.print("[yellow]No aligned frames to render.[/yellow]")


# ---------------------------------------------------------------------------
# README generator
# ---------------------------------------------------------------------------


def write_readme() -> None:
    """Generate a README.md with setup and usage instructions."""
    readme = """\
# 🏋️ Pose Matcher

Compare your exercise form against a ground-truth video using MediaPipe
skeleton analysis and Dynamic Time Warping.

## Installation

```bash
uv sync
```

## Quick Start

### 1. Add a ground-truth video

Place a reference video (e.g. from Fit3D) into:

```
data/ground_truth/squat.mp4
```

### 2. Run comparison (recorded video)

```bash
uv run python main.py --gt data/ground_truth/squat.mp4 \\
                       --user data/user_input/my_squat.mp4
```

### 3. Run live webcam mode

```bash
uv run python main.py --gt data/ground_truth/squat.mp4 --live
```

### 4. Extract skeleton only

```bash
uv run python main.py --extract-only data/ground_truth/squat.mp4
```

## Output

| File | Description |
|---|---|
| `output/result.mp4` | Side-by-side comparison video |
| `output/score_chart.png` | Score consistency over time |
| `output/report_*.txt` | Session report with coaching tips |

## Score & Grades

| Score | Grade |
|---|---|
| 90-100 | Excellent |
| 75-89 | Good |
| 50-74 | Needs Work |
| 0-49 | Poor |

## Joint Names

| Joint | Landmarks Used |
|---|---|
| left / right knee | hip → knee → ankle |
| left / right hip | shoulder → hip → knee |
| left / right elbow | shoulder → elbow → wrist |
| left / right shoulder | elbow → shoulder → hip |
| torso lean | spine angle from vertical |
"""
    Path("README.md").write_text(readme, encoding="utf-8")
    console.print("[green]✓ README.md generated[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="🏋️  Pose Matcher — Compare exercise form against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gt", type=str, help="Path to ground-truth video",
    )
    parser.add_argument(
        "--user", type=str, help="Path to user video",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live webcam feed instead of --user video",
    )
    parser.add_argument(
        "--extract-only", type=str, metavar="VIDEO",
        help="Extract skeleton from a video and save as JSON, then exit",
    )
    parser.add_argument(
        "--generate-readme", action="store_true",
        help="Generate README.md and exit",
    )
    parser.add_argument(
        "--perf", action="store_true",
        help="Print per-stage latency report at end of run",
    )

    args = parser.parse_args()

    # Welcome banner.
    console.print(
        Panel(
            "[bold cyan]🏋️  Pose Matcher[/bold cyan]\n"
            "[dim]Skeleton-based exercise form comparison[/dim]",
            border_style="cyan",
            width=50,
        )
    )

    if args.generate_readme:
        write_readme()
        return

    if args.extract_only:
        _run_extract_only(args.extract_only)
        return

    if args.live:
        if not args.gt:
            console.print("[red]✗ --gt is required for live mode.[/red]")
            sys.exit(1)
        _run_live(args.gt)
        return

    if args.gt and args.user:
        _run_comparison(args.gt, args.user)
        if args.perf:
            _print_perf_report()
        return

    parser.print_help()


def _print_perf_report() -> None:
    """Print the latency monitor's rolling-window report."""
    from src.perf_monitor import monitor

    report = monitor.report()
    stages = report["stages"]
    total = report["total_ms"]
    fps = report["fps_estimate"]
    budget = report["budget_ms"]
    warning = report["warning_ms"]

    status_color = (
        "red" if report["over_budget"]
        else "yellow" if report["over_warning"]
        else "green"
    )

    table = Table(title="Per-Stage Latency (ms, rolling average)", show_lines=True)
    table.add_column("Stage", style="cyan")
    table.add_column("Avg ms", justify="right")
    for stage, avg_ms in sorted(stages.items(), key=lambda x: -x[1]):
        table.add_row(stage, f"{avg_ms:.2f}")
    console.print()
    console.print(table)
    console.print(
        f"[bold]Total:[/bold] [{status_color}]{total:.2f} ms/frame[/{status_color}]  "
        f"(budget {budget:.0f} ms, warn {warning:.0f} ms)  "
        f"→ ~{fps:.1f} FPS"
    )


if __name__ == "__main__":
    main()
