"""
app.py — Pose Matcher backend.
Auth:  JWT Bearer tokens + Session table in PostgreSQL via Prisma.
Roles: user | admin
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import bcrypt
import jwt
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_file
from flask_cors import CORS
from prisma import Prisma
from werkzeug.utils import secure_filename

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

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# ---------------------------------------------------------------------------
# Prisma client (one shared instance, connected per request)
# ---------------------------------------------------------------------------

db = Prisma()


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
        # Verify session still exists in DB
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
def page_dashboard(): return app.send_static_file("dashboard.html")

@app.route("/exercise")
def page_exercise():  return app.send_static_file("exercise.html")

@app.route("/admin")
def page_admin():     return app.send_static_file("admin.html")


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

    # Upsert session (one active session per login; old tokens still valid until expiry)
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


# ---------------------------------------------------------------------------
# Exercises (public)
# ---------------------------------------------------------------------------

@app.route("/api/exercises", methods=["GET"])
def list_exercises():
    videos = db.video.find_many(where={"status": "approved"})
    seen: dict[str, str] = {}
    for v in videos:
        if v.exercise not in seen:
            seen[v.exercise] = v.id
    return jsonify([{"exercise": ex, "video_id": vid} for ex, vid in seen.items()])


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
        "originalName": v.originalName,
        "exercise":     v.exercise,
        "status":       str(v.status),
        "adminNote":    v.adminNote,
        "uploadedAt":   v.uploadedAt.isoformat(),
        "reviewedAt":   v.reviewedAt.isoformat() if v.reviewedAt else None,
    } for v in videos])


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
        "originalName": v.originalName,
        "exercise":     v.exercise,
        "status":       str(v.status),
        "uploader":     v.uploader.username if v.uploader else "unknown",
        "adminNote":    v.adminNote,
        "uploadedAt":   v.uploadedAt.isoformat(),
        "reviewedAt":   v.reviewedAt.isoformat() if v.reviewedAt else None,
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
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
