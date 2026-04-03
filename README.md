crea# 🏋️ Pose Matcher

Compare your exercise form against a ground-truth video using MediaPipe
skeleton analysis and Dynamic Time Warping.

## Installation

Python 3.12 is required for local development and `uv sync` on macOS Intel or Apple Silicon.

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
uv run python main.py --gt data/ground_truth/squat.mp4 \
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
