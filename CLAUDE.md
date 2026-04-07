# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Python 3.12 required; pyproject.toml targets macOS only)
uv sync

# Run Flask backend
uv run python app.py

# CLI: compare two videos
uv run python main.py --gt data/ground_truth/squat.mp4 --user data/user_input/my_squat.mp4

# CLI: live webcam mode
uv run python main.py --gt data/ground_truth/squat.mp4 --live

# CLI: extract skeleton only (saves JSON alongside video)
uv run python main.py --extract-only data/ground_truth/squat.mp4

# Database migrations (Prisma) — requires Node.js (package.json has the prisma npm package)
npm install          # once, to get the prisma CLI
uv run prisma migrate dev
uv run prisma generate
```

## Architecture

The project has two entry points that share the `src/` library:

- **`main.py`** — CLI tool for offline video-vs-video comparison or live webcam comparison. No database required.
- **`app.py`** — Flask REST API with JWT auth, PostgreSQL via Prisma, and optional WebSocket support (`flask-sock`). Serves the `frontend/` HTML pages.

### Core pipeline (`src/`)

Data flows in this order:

1. **`extractor.py`** — Runs MediaPipe `PoseLandmarker` (Tasks API v0.10+) **and** `mp.solutions.hands.Hands` on each frame. Outputs `{frame_index, timestamp_ms, landmarks[33], hand_landmarks:{left,right}}`. Old cached files without `hand_landmarks` are handled gracefully.
2. **`normalizer.py`** — Translates/scales/rotates pose landmarks; computes 9 pose angles + up to 10 hand finger-curl angles (5 per hand). Exports `compute_hand_angles()` for `app.py`.
3. **`matcher.py`** — DTW alignment (`dtaidistance`) + per-joint angle scoring. **Velocity is NOT part of the score** — only angle accuracy is scored. Joints absent from either frame are skipped. Exercise weights default unlisted joints to weight 0.
4. **`calibration.py`** — Builds a ROM profile from the skeleton and derives per-joint adaptive thresholds.
5. **`exercise_weights.py`** — Per-exercise joint weights. Exports `ALL_JOINT_NAMES` (9 pose + 10 hand). Exercises include squat, deadlift, pushup, shoulder_press, bicep_curl, wrist_curl, finger_exercise. `compute_auto_weights()` auto-detects dominant joints.
6. **`feedback.py`** — Severity-tiered (mild/moderate/critical), exercise-aware, phase-aware coaching. Bilateral merge for symmetric corrections (left+right knee → "both knees"). Max 2 corrections per frame. Velocity shown as hints only.
7. **`filters.py`** — `LandmarkSmoother` (33 pose landmarks) and `HandLandmarkSmoother` (21 landmarks × 2 hands) both use One Euro Filter.
8. **`state_machine.py`** — `ExerciseStateMachine` detects READY→DOWN→HOLD→UP phases and counts reps.
9. **`visualizer.py`** — Side-by-side OpenCV frames with glowing body skeleton, `draw_hand_skeleton()` overlay, score bar, coaching bubble, and correction arrows.

### `app.py` internals

- JWT tokens (8-hour expiry) stored in a `Session` table; `require_auth` / `require_admin` decorators wrap protected routes.
- `TemporalAligner` dataclass tracks user's progress through the reference sequence independently of wall-clock time to avoid penalising delayed users.
- Three model tiers: `lite` (default via `POSE_MODEL_TIER` env var), `full`, `heavy`. Falls back gracefully if the preferred model file is missing.
- Reference skeleton JSONs are cached in `data/cache/reference_skeletons/`.
- Uploaded videos go to `data/pending/` (status `pending`) and move to `data/ground_truth/` on admin approval.

### Database (Prisma + PostgreSQL)

Schema: `User`, `Session`, `Video`, `Category`, `ExerciseReport`. All IDs are UUIDs. Python client uses sync interface (`interface = "sync"`).

Required env vars (`.env`):
```
DATABASE_URL=postgresql://user:pass@localhost:5432/posematcher
SECRET_KEY=<random 32+ char string>
POSE_MODEL_TIER=lite   # optional: lite | full | heavy
```

### Models

MediaPipe `.task` files must exist in `models/`:
- `models/pose_landmarker_lite.task`
- `models/pose_landmarker_full.task`
- `models/pose_landmarker_heavy.task`

Currently only `lite` and `heavy` are present in the repo. All three are available from the MediaPipe model card. `full` falls back gracefully if missing.

### Frontend

Plain HTML/CSS/JS in `frontend/`. `auth.js` handles JWT storage and redirect logic shared across pages. No build step.

### Dataset

`dataset/ground_truth/fit3d_test/` contains FIT3D benchmark data (camera parameters + keypoints) used for evaluation/research. Not required for normal operation.
