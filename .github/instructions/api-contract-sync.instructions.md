---
description: "Use when changing Flask API endpoints in app.py or frontend pages that consume /api responses. Enforces response contract consistency, snake_case payload keys, and synchronized backend/frontend updates."
name: "API Contract Sync"
applyTo:
  - app.py
  - frontend/*.html
  - frontend/auth.js
---
# API Contract Sync

- Keep JSON payload keys in snake_case for frontend-consumed API responses.
- Hard rule: when changing response fields in app.py, update all affected frontend consumers in the same task.
- Prefer additive changes over breaking renames. If a rename is required, migrate all consumers and remove old keys only after all pages are updated.
- Include uploader context for video-centric UI where relevant:
  - Admin views should receive uploader identity/profile data required for moderation.
  - User views showing reference videos should expose uploader attribution (username only).
  - Do not expose uploader email in user-facing pages.
- Keep auth and role checks in backend decorators (`require_auth`, `require_admin`) instead of duplicating access logic in each route.
- Use `Auth.apiFetch()` for authenticated frontend API calls so Bearer token behavior stays consistent.

## Quick Validation Checklist

- Did changed endpoints keep snake_case keys?
- Did all frontend pages using the endpoint update field names?
- Did admin and user video views still render uploader details correctly?
- Did user-facing views avoid displaying uploader email?
- Did protected endpoints keep `require_auth` or `require_admin`?
