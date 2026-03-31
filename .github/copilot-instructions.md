# Project Guidelines

## Code Style
- Prefer Python type hints and small focused helper functions, matching existing patterns in app.py and src/*.py.
- Keep module-level docstrings for purpose and exported behavior, as used across src modules.
- Preserve current API response shape conventions (snake_case JSON keys for frontend-consumed payloads).
- For frontend pages in frontend/*.html, keep logic lightweight and framework-free (vanilla JS + shared Auth helper in frontend/auth.js).

## Architecture
- Two entry points:
  - CLI workflow in main.py for skeleton extraction and comparison.
  - Web app in app.py (Flask + Prisma + JWT sessions).
- Core pose pipeline is in src/:
  - extractor.py: MediaPipe landmark extraction and skeleton JSON caching.
  - normalizer.py, matcher.py, feedback.py: normalization, DTW-based scoring, and feedback.
  - visualizer.py: output overlays/charts/video rendering.
- Web app boundaries:
  - frontend/ contains static HTML/CSS/JS pages.
  - app.py exposes page routes and /api/* endpoints.
  - prisma/schema.prisma defines User, Session, Video, and role/status enums.
- Data layout:
  - data/pending for uploaded videos awaiting review.
  - data/ground_truth for approved reference videos.

## Build and Test
- Install dependencies: uv sync
- Run CLI comparison: uv run python main.py --gt <ground_truth_video> --user <user_video>
- Run live mode: uv run python main.py --gt <ground_truth_video> --live
- Run extract-only mode: uv run python main.py --extract-only <video>
- Run web server: uv run python app.py

## Conventions
- Use frontend/auth.js Auth.guard() and Auth.apiFetch() for authenticated page/API interactions.
- Keep auth and role enforcement in backend decorators (require_auth, require_admin), not in duplicated route logic.
- When changing video/user APIs, update both backend payload keys and frontend field usage together.
- Preserve current storage flow for uploads:
  - Upload -> data/pending with status=pending
  - Admin approve -> move file to data/ground_truth and mark approved
- Keep Prisma model naming and relationships unchanged unless migration updates are included.

## Pitfalls
- The app depends on a working PostgreSQL DATABASE_URL for Prisma.
- SECRET_KEY defaults to a dev value; avoid relying on this for production-like environments.
- MediaPipe model file is required at models/pose_landmarker_heavy.task.
- Avoid mixing camelCase and snake_case in API responses consumed by frontend pages.

## References
- See README.md for usage examples and output expectations.
- See prisma/schema.prisma for database contracts and relations.
