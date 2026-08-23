import os
import json
import logging
import tempfile
import time
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from dotenv import set_key


from app.services.face_recognition.face_service import (
    detect_person,
    detect_face,
    crop_face,
    generate_embedding,
    compare_embedding,
    fetch_details,
)
from app.database.db import get_db_connection, save_conversation
from app.services.reminder_app.calendar_service import create_reminder, get_upcoming_reminders
from app.services.reminder_app.google_auth import get_auth_url, exchange_code_for_token
from app.services.redis_client import get_redis
from app.config import get_settings
from pydantic import BaseModel

# ── Merged-in REST/CRUD API (formerly app/app.py, port 8000 — merged here so
# there's a single backend; see TECH_DEBT.md) ───────────────────────────────
from app.routes.main_routes import main_router
from app.routes.audio_routes import audio_router
from app.routes.interaction_routes import interaction_router
from app.ai_models.reminders.reminder_routes import reminder_router
from app.api.routes import (
    users,
    caregivers,
    interactions,
    sessions,
    memory,
    notes,
    calendar_events,
    audio,
    emotions,
)
from app.services.session_service import SessionManager

load_dotenv()
logger = logging.getLogger(__name__)

# 🤫 SILENCE SPAM
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

USER_ID = int(os.getenv("USER_ID", "1"))

# ---------------------------------------------------------------------------
# Session State Machine
# ---------------------------------------------------------------------------
# States: "idle" | "session_active" | "grace_period" | "processing"

_session = {
    "state": "idle",
    "person_id": None,
    "person_name": None,
    "interaction_id": None,
    "session_started_at": 0,
    "last_presence_check": 0,   # timestamp of last confirmed face presence
    "grace_started_at": 0,       # when grace period countdown started
    # For deferred unknown registration
    "pending_unknown_embedding": None,  # stored embedding for end-of-session registration
    # Unknown face retry counter (prevents transient false-positive unknown detections)
    "unknown_streak": 0,         # consecutive unknown detections in idle state
    "pending_audio_path": None,
    "interaction_save_failed": False,  # True if _save_initial_interaction_with_retry gave up
}
UNKNOWN_STREAK_THRESHOLD = 3   # need this many consecutive unknowns to start unknown session
# RLock (not Lock) — _start_session/_start_unknown_session/_end_session already take
# this lock internally, and /identify's own critical sections call into them while
# still holding it. A plain Lock would deadlock on that same-thread re-acquisition.
_session_lock = threading.RLock()

# Persisted across _end_session so /register-new can use the original embedding
_saved_unknown_embedding = None

PRESENCE_CHECK_INTERVAL = 10   # seconds between presence checks during session
GRACE_PERIOD_SECONDS = 5       # seconds to wait before ending session when face is gone

# ── Redis write-through mirror ───────────────────────────────────────────────
# _session above stays the authoritative, fast, in-process store for every
# read (single-process deployment — see the migration plan's locking-model
# rationale). Every mutation also best-effort mirrors the whole blob to
# Redis (log-and-continue on failure, never raise) so a crash/restart can at
# least be logged with what was in flight — see lifespan's startup read.
# A torn session can't actually be *resumed* (the audio-recording thread
# itself doesn't survive a restart), only reported.
SESSION_KEY = "agos:session:current"

# Short, bounded retry — same reasoning and same numbers as
# SessionManager._mirror_session in session_service.py (structure B):
# absorbs a brief Redis blip without leaving the mirror stale, but stays
# short since this runs inline with request handling — a sustained outage
# should fail fast, not block /identify's hot polling loop.
MUTATE_RETRY_ATTEMPTS = 2
MUTATE_RETRY_DELAY_SECONDS = 0.3

# Counts how many times _mutate_session's retry budget was fully exhausted.
# Pure instrumentation, mirrors SessionManager._mirror_exhausted_count in
# session_service.py — same motivation: real numbers on how often this
# actually happens, before deciding whether more durability work is
# warranted here too.
_mutate_exhausted_count = 0
_mutate_exhausted_lock = threading.Lock()


def _get_mutate_exhausted_count() -> int:
    with _mutate_exhausted_lock:
        return _mutate_exhausted_count


def _mutate_session(updates: dict) -> None:
    """Caller must hold _session_lock. Mutates the in-process _session dict
    (always succeeds) and best-effort mirrors the whole blob to Redis,
    retrying a bounded number of times before giving up."""
    global _mutate_exhausted_count
    _session.update(updates)
    payload = json.dumps(_session)
    last_err = None
    for attempt in range(MUTATE_RETRY_ATTEMPTS):
        try:
            get_redis().set(SESSION_KEY, payload)
            return
        except redis.exceptions.RedisError as e:
            last_err = e
            if attempt < MUTATE_RETRY_ATTEMPTS - 1:
                time.sleep(MUTATE_RETRY_DELAY_SECONDS)

    with _mutate_exhausted_lock:
        _mutate_exhausted_count += 1
        count = _mutate_exhausted_count
    logger.error(
        f"[MUTATE_WRITE_EXHAUSTED] session mirror: all {MUTATE_RETRY_ATTEMPTS} "
        f"write attempts failed (total exhaustion count this process: {count}): {last_err}"
    )


# ── DB-save retry + last-error surface (bug fix: previously, if the DB write
# that creates a conversation row failed for a KNOWN person, the session
# proceeded anyway, recorded audio, and that audio was silently orphaned at
# session end — no registration modal fires for a known person, so nothing
# ever read the leftover pending_audio_path, and the temp .wav leaked on
# disk forever). Now: retry with backoff, and if still failing, mark the
# session so _end_session discards (rather than orphans) the audio, and
# surface the failure to the frontend via both _push_log (already polled at
# /live-log) and this small error hint (polled at /session-status).
DB_SAVE_MAX_ATTEMPTS = 3
DB_SAVE_RETRY_DELAYS = (1, 2)   # seconds between attempts 1->2 and 2->3

_last_session_error = None   # {"type": ..., "person_name": ..., "ts": ...} | None
_last_session_error_lock = threading.Lock()


def _set_last_session_error(error_type: str, person_name: str) -> None:
    global _last_session_error
    with _last_session_error_lock:
        _last_session_error = {
            "type": error_type,
            "person_name": person_name,
            "ts": datetime.now().strftime("%H:%M:%S"),
        }


def _clear_last_session_error() -> None:
    global _last_session_error
    with _last_session_error_lock:
        _last_session_error = None


def _get_last_session_error():
    with _last_session_error_lock:
        return dict(_last_session_error) if _last_session_error else None


def _save_initial_interaction_with_retry(person_id: int) -> tuple[int | None, bool]:
    """Retry _save_initial_interaction with backoff. Returns
    (interaction_id, save_failed) — save_failed=True means every attempt
    failed and the caller must not treat interaction_id as usable."""
    for attempt in range(DB_SAVE_MAX_ATTEMPTS):
        interaction_id = _save_initial_interaction(person_id)
        if interaction_id is not None:
            return interaction_id, False
        if attempt < DB_SAVE_MAX_ATTEMPTS - 1:
            delay = DB_SAVE_RETRY_DELAYS[attempt]
            _push_log(f"[DB ERROR] Retry {attempt + 1}/{DB_SAVE_MAX_ATTEMPTS - 1} in {delay}s...")
            time.sleep(delay)
    return None, True

# Live event log for frontend polling
_live_log = []   # list of {ts, message} dicts
_live_log_lock = threading.Lock()

def _push_log(message: str):
    """Push an event to the live log ring buffer (last 50 entries)."""
    with _live_log_lock:
        _live_log.append({"ts": datetime.now().strftime("%H:%M:%S"), "message": message})
        if len(_live_log) > 50:
            _live_log.pop(0)
    print(message)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Starting AG-OS API...")

    # This module's own live-camera `_session` (structure A): read the Redis
    # mirror purely to LOG what was in flight when the process last stopped —
    # a torn visit-session can't actually be resumed (the audio-recording
    # thread itself is gone), only reported. Always reset to idle either way.
    try:
        raw = get_redis().get(SESSION_KEY)
        if raw:
            prior = json.loads(raw)
            if prior.get("state") not in ("idle", None):
                logger.warning(
                    f"Found an in-flight session from before restart: "
                    f"state={prior.get('state')} person={prior.get('person_name')} "
                    f"interaction_id={prior.get('interaction_id')} — resetting to idle "
                    "(the audio-recording thread that session depended on did not survive the restart)."
                )
    except redis.exceptions.RedisError as e:
        logger.warning(f"Could not read prior session state from Redis at startup: {e}")
    with _session_lock:
        _mutate_session({
            "state": "idle", "person_id": None, "person_name": None,
            "interaction_id": None, "session_started_at": 0, "last_presence_check": 0,
            "grace_started_at": 0, "pending_unknown_embedding": None,
            "unknown_streak": 0, "pending_audio_path": None, "interaction_save_failed": False,
        })

    # The merged-in REST API's 30-min session chunking (SessionManager,
    # structure B) — genuinely restores (not just logs) since nothing there
    # depends on an unrecoverable physical resource, unlike A above. This is
    # what lets a still-pending Celery timer (already Redis-durable) find its
    # session's accumulated summaries again after a restart, instead of the
    # previous unconditional wipe that guaranteed it wouldn't.
    settings = get_settings()
    SessionManager.restore_and_sweep(max_age_minutes=2 * settings.SESSION_DURATION_MINUTES)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down AG-OS API...")


app = FastAPI(
    title="AG-OS API",
    description="AI-powered memory assistant — face recognition, session engine, audio transcription, reminders, and CRUD API.",
    version="2.0.0",
    lifespan=lifespan,
)

os.makedirs("app/static", exist_ok=True)
app.mount("/dashboard", StaticFiles(directory="app/static", html=True), name="static")

print("\n" + "="*50)
print("🚀 AG-OS SESSION ENGINE IS ONLINE")
print("📡 Port: 8004 | Mode: Stateful Session Management")
print("="*50 + "\n")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=["*"] + allow_credentials=True is an invalid combination
    # Starlette resolves by echoing back the caller's Origin, enabling
    # credentialed cross-origin requests. Explicit False — nothing here uses
    # cookie/session auth.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    from app.db.base import get_engine
    from sqlalchemy import text

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

# ---------------------------------------------------------------------------
# Reminder Schemas & Templates
# ---------------------------------------------------------------------------
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

class ReminderRequest(BaseModel):
    title: str
    date: str
    time: str

REMINDER_BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("face_recognition", "reminder_app")
templates = Jinja2Templates(directory=os.path.join(REMINDER_BASE_DIR, "templates"))

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_frame(file: UploadFile) -> np.ndarray:
    raw = file.file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image encoding")
    return frame


def _run_pipeline(frame: np.ndarray):
    """YOLO / Haar detect → crop → DeepFace embed. Returns (embedding, used_fallback)."""
    detected, bbox = detect_face(frame)
    if detected:
        roi = crop_face(frame, bbox)
        if roi is not None:
            embedding = generate_embedding(roi)
            if embedding:
                return embedding, False
    return None, False


def _has_presence(frame: np.ndarray) -> bool:
    """Lightweight check — YOLO/Haar only, NO DeepFace. ~30ms on CPU."""
    detected, _ = detect_person(frame)
    return detected


# ── Threadpool-offload helpers ───────────────────────────────────────────────
# cv2/YOLO/DeepFace/psycopg2 calls below are all blocking, CPU- or I/O-bound
# work. Calling them directly inside an `async def` route stalls uvicorn's
# single-threaded event loop for every other in-flight request (including
# /health) until they finish — confirmed in practice, where a burst of
# /idle-audio calls froze the whole server for minutes. Route handlers hand
# this work to `run_in_threadpool` instead of calling these directly; that's
# more important, not less, on a CPU as weak as a Pi 3.

def _sync_decode_and_check_presence(file: UploadFile) -> bool:
    """Sync decode+detect for /identify's session_active/grace_period presence
    checks and for /check-presence."""
    frame = _decode_frame(file)
    return _has_presence(frame)


def _sync_identify_pipeline(file: UploadFile):
    """Sync decode+embed+match for /identify's idle-state full pipeline."""
    frame = _decode_frame(file)
    embedding, _ = _run_pipeline(frame)
    if embedding is None:
        return None, None, None, None
    pid, score, status = compare_embedding(embedding)
    return embedding, pid, score, status


def _sync_decode_and_embed(file: UploadFile):
    """Sync decode+embed for /register and /register-new's fallback path."""
    frame = _decode_frame(file)
    return _run_pipeline(frame)


def _sync_process_idle_audio(temp_path: str) -> dict:
    """VAD-gate + transcribe + save for /idle-audio. Whisper inference is
    blocking CPU work; save_conversation() blocks on network I/O to Postgres —
    both need to run off the event loop, not on it."""
    import app.services.voice_app.transcription_service as ts

    if not ts.has_speech(temp_path):
        return {"status": "empty", "reason": "no_speech_detected"}

    text = ts.transcribe_audio(temp_path)
    if not text or not text.strip() or len(text.strip()) < 2:
        return {"status": "empty"}

    interaction_id = save_conversation(
        userid=USER_ID,
        personid=None,
        transcribed_text=text,
        summarized_text=None,
        detected_emotion=None,
        location="Living Room",
    )
    _push_log(f"[IDLE AUDIO] Transcribed: {text[:50]}...")
    return {"status": "saved", "transcription": text, "interactionid": interaction_id}


def _sync_register_new_person(
    name: str, relationship: str, priority: int, embedding: list[float]
) -> tuple[int, int]:
    """Insert knownperson + faceencoding for a brand-new registration."""
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO public.knownperson (name, relationshiptype, prioritylevel, notes)
            VALUES (%s, %s, %s, %s)
            RETURNING personid;
            """,
            (name, relationship, priority, "Registered via live camera"),
        )
        person_id = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO public.faceencoding (personid, encodingdata, confidencescore)
            VALUES (%s, %s, %s)
            RETURNING faceencodingid;
            """,
            (person_id, json.dumps(embedding), 1.0),
        )
        encoding_id = cur.fetchone()[0]
        conn.commit()
        _push_log(f"[REGISTER] ✅ Registered: {name} ({relationship}) | personid={person_id}")
        return person_id, encoding_id
    except Exception as exc:
        if conn is not None:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def _sync_register_existing_person(personid: int, embedding: list[float]) -> int | None:
    """Insert a faceencoding row for an existing knownperson."""
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO public.faceencoding (personid, encodingdata, confidencescore)
            VALUES (%s, %s, %s)
            RETURNING faceencodingid;
            """,
            (personid, json.dumps(embedding), 1.0),
        )
        row = cur.fetchone()
        conn.commit()

        from app.services.face_recognition.face_service import clear_face_cache
        clear_face_cache()

        return row[0] if row else None
    except Exception as exc:
        if conn is not None:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def _save_initial_interaction(person_id: int) -> int | None:
    """Create a placeholder conversation row for the session."""
    try:
        interaction_id = save_conversation(
            userid=USER_ID,
            personid=person_id,
            transcribed_text="[Session started via face recognition]",
            summarized_text="Session in progress.",
            detected_emotion="Neutral",
            location="Living Room",
        )
        return interaction_id
    except Exception as e:
        print(f"[DB ERROR] _save_initial_interaction failed: {e}", flush=True)
        _push_log(f"[DB ERROR] Could not save conversation row: {e}")
        return None


def _start_session(person_id: int, person_name: str):
    """Begin a new session for a known person.

    The DB save is retried with backoff (_save_initial_interaction_with_retry)
    rather than a single attempt — previously a single transient DB failure
    here silently orphaned the whole recorded conversation (see _end_session).
    The session state briefly goes through "processing" (a state /identify
    already knew how to handle defensively but nothing ever set) so this
    retry — up to ~3s worst case — doesn't hold _session_lock the whole time.
    """
    with _session_lock:
        if _session["state"] != "idle":
            return  # Already in a session
        _mutate_session({"state": "processing"})

    interaction_id, save_failed = _save_initial_interaction_with_retry(person_id)

    with _session_lock:
        _mutate_session({
            "state": "session_active",
            "person_id": person_id,
            "person_name": person_name,
            "interaction_id": interaction_id,
            "interaction_save_failed": save_failed,
            "session_started_at": time.time(),
            "last_presence_check": time.time(),
            "grace_started_at": 0,
            "pending_unknown_embedding": None,
            "pending_audio_path": None,
        })

    if save_failed:
        _set_last_session_error("db_save_failed", person_name)
        _push_log(
            f"[SESSION] ⚠️ Started for {person_name} WITHOUT a saved interaction "
            f"row — database unreachable after {DB_SAVE_MAX_ATTEMPTS} attempts. "
            "Audio will be discarded (not saved) when this session ends."
        )
    else:
        _clear_last_session_error()
        _push_log(f"[SESSION] ▶ Started for {person_name} (interaction_id={interaction_id})")

    # Start continuous background recording — audio recorded until session ends
    import app.services.voice_app.recorder_util as ru
    filename = f"session_{interaction_id}.wav" if interaction_id else f"session_known_{int(time.time())}.wav"
    ru.start_session_recording(filename=filename)
    _push_log(f"[MIC] 🔴 Continuous recording started (file: {filename})")


def _start_unknown_session():
    """Begin a session for an unidentified person. Registration deferred to end."""
    with _session_lock:
        if _session["state"] != "idle":
            return
        _mutate_session({
            "state": "session_active",
            "person_id": None,
            "person_name": "Unknown",
            "interaction_id": None,
            "interaction_save_failed": False,
            "session_started_at": time.time(),
            "last_presence_check": time.time(),
            "grace_started_at": 0,
            "pending_unknown_embedding": None,
            "pending_audio_path": None,
        })
    filename = f"session_unknown_{int(time.time())}.wav"
    _push_log(f"[SESSION] ▶ Started for UNKNOWN PERSON — recording audio, registration deferred (file: {filename})")
    # Start continuous background recording
    import app.services.voice_app.recorder_util as ru
    ru.start_session_recording(filename=filename)
    _push_log("[MIC] 🔴 Continuous recording started")


def _end_session(reason: str = "face_lost"):
    """
    End the current session:
    1. Stop continuous recording.
    2. Kick off transcription + summarization in background (only NOW, at session end).
    3. Reset all session state.
    Returns pending_unknown_embedding if registration is needed.
    """
    global _saved_unknown_embedding
    with _session_lock:
        if _session["state"] == "idle":
            return None
        name          = _session["person_name"]
        interaction_id = _session["interaction_id"]
        pending_emb   = _session["pending_unknown_embedding"]
        interaction_save_failed = _session["interaction_save_failed"]
        _mutate_session({
            "state": "idle",
            "person_id": None,
            "person_name": None,
            "interaction_id": None,
            "grace_started_at": 0,
            "pending_unknown_embedding": None,
            "unknown_streak": 0,
            "interaction_save_failed": False,
        })

    # Persist the embedding OUTSIDE the session dict so /register-new can use it later
    if pending_emb is not None:
        _saved_unknown_embedding = pending_emb

    _push_log(f"[SESSION] ■ Ended ({reason}) for {name}")

    # Stop recording and get the saved audio file path
    import app.services.voice_app.recorder_util as ru
    filepath = ru.stop_session_recording()
    _push_log("[MIC] ⏹ Recording stopped")

    # Only process audio if we have an interaction to update and a file to process
    if filepath and interaction_id:
        _push_log("[MIC] ⚙️ Starting transcription & summarization...")
        ru.process_recording_in_background(interaction_id, filepath)
    elif filepath and not interaction_id and interaction_save_failed:
        # Known person, but the DB save that would have given us an
        # interaction_id failed even after retries (see _start_session).
        # Previously this fell into the "unknown person" branch below and
        # got silently orphaned in pending_audio_path forever, since nothing
        # for a KNOWN person ever triggers the registration flow that reads
        # it back out. Discard explicitly instead.
        try:
            os.remove(filepath)
            _push_log(f"[SESSION] 🗑️ Discarded unsaved audio for {name} (database unreachable) — {filepath}")
        except OSError as e:
            _push_log(f"[SESSION] ⚠️ Could not delete orphaned audio file {filepath}: {e}")
    elif filepath and not interaction_id:
        # Unknown person — audio file will be processed after they register
        _push_log("[MIC] 🔖 Audio saved — will process after registration")
        with _session_lock:
            _session["pending_audio_path"] = filepath

    return pending_emb


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """
    Called by the frontend every ~1.5s.
    - If idle: run full identify pipeline to detect + match face.
    - If session_active: do lightweight presence check every PRESENCE_CHECK_INTERVAL seconds.
    - If grace_period: do lightweight presence check; reset timer if face found.
    """
    import app.services.voice_app.recorder_util as ru
    now = time.time()
    with _session_lock:
        state = _session["state"]

    # ── During active session or grace: lightweight presence only ───────────
    if state in ("session_active", "grace_period", "processing"):

        if state == "session_active":
            # Only decode frame + run YOLO when the 10s interval has elapsed
            # Between checks just return session info (zero extra CPU work)
            with _session_lock:
                time_since_check = now - _session["last_presence_check"]
            if time_since_check >= PRESENCE_CHECK_INTERVAL:
                present = await run_in_threadpool(_sync_decode_and_check_presence, file)
                with _session_lock:
                    _session["last_presence_check"] = now
                    session_started_at = _session["session_started_at"]
                    if not present:
                        _session["state"] = "grace_period"
                        _session["grace_started_at"] = now
                if present:
                    _push_log(f"[PRESENCE] ✅ Face confirmed at {int(now - session_started_at)}s")
                else:
                    _push_log(f"[GRACE] ⚠️ Face lost after {int(now - session_started_at)}s — {GRACE_PERIOD_SECONDS}s to return")

        elif state == "grace_period":
            # During grace we check every call (fast 500ms loop) so countdown is accurate
            present = await run_in_threadpool(_sync_decode_and_check_presence, file)
            with _session_lock:
                grace_elapsed = now - _session["grace_started_at"]
                if present:
                    # Face returned — reset presence timer and go back to session
                    _session["state"] = "session_active"
                    _session["last_presence_check"] = now

            if present:
                _push_log("[GRACE] ✅ Face returned — session continues")
            elif grace_elapsed >= GRACE_PERIOD_SECONDS:
                pending_emb = await run_in_threadpool(_end_session, "face_lost_grace_expired")
                return JSONResponse({
                    "session_state": "ended",
                    "reason": "grace_expired",
                    "needs_registration": pending_emb is not None,
                    "message": "Session ended — face not found in time."
                })
            else:
                countdown = int(GRACE_PERIOD_SECONDS - grace_elapsed)
                with _session_lock:
                    person_name = _session["person_name"]
                return JSONResponse({
                    "session_state": "grace_period",
                    "grace_countdown": countdown,
                    "person_name": person_name,
                    "is_recording": ru.IS_RECORDING,
                    "is_summarizing": ru.IS_SUMMARIZING,
                })

        with _session_lock:
            person_name = _session["person_name"]
        return JSONResponse({
            "session_state": state,
            "person_name": person_name,
            "is_recording": ru.IS_RECORDING,
            "is_summarizing": ru.IS_SUMMARIZING,
        })

    # ── Idle: run full identify pipeline ────────────────────────────────────
    embedding, pid, score, status = await run_in_threadpool(_sync_identify_pipeline, file)

    if embedding is None:
        return JSONResponse({
            "session_state": "idle",
            "person_detected": False,
            "match_status": "no_face",
        })

    if status in ("confirmed", "uncertain"):
        # Reset unknown streak on a successful match
        with _session_lock:
            _session["unknown_streak"] = 0
        details = fetch_details(pid) or {}
        name = details.get("name", f"Person {pid}")
        _push_log(f"[FACE] {status.upper()} — {name} (confidence={score:.2f})")
        await run_in_threadpool(_start_session, pid, name)
        return JSONResponse({
            "session_state": "session_started",
            "person_detected": True,
            "match_status": status,
            "confidence": score,
            "person_name": name,
            "relationship": details.get("relationship"),
            "last_visit": details.get("last_date"),
            "last_summary": details.get("last_summary"),
            "last_emotion": details.get("last_emotion"),
            "last_conversation": details.get("last_conversation"),
        })
    else:
        # Unknown person — require UNKNOWN_STREAK_THRESHOLD consecutive unknowns
        # before starting a session. Prevents transient misdetections.
        with _session_lock:
            _session["unknown_streak"] = _session.get("unknown_streak", 0) + 1
            streak = _session["unknown_streak"]
        _push_log(f"[FACE] UNKNOWN (score={score:.2f}) — streak {streak}/{UNKNOWN_STREAK_THRESHOLD}")

        if streak < UNKNOWN_STREAK_THRESHOLD:
            return JSONResponse({
                "session_state": "idle",
                "person_detected": True,
                "match_status": "unknown",
                "confidence": score,
                "unknown_streak": streak,
                "message": f"Unknown face ({streak}/{UNKNOWN_STREAK_THRESHOLD}) — waiting for confirmation.",
            })

        # Confirmed unknown after N consecutive detections — start session
        with _session_lock:
            _session["unknown_streak"] = 0
        _push_log(f"[FACE] UNKNOWN confirmed — starting session, registration deferred to end")
        await run_in_threadpool(_start_unknown_session)
        with _session_lock:
            _session["pending_unknown_embedding"] = embedding
        return JSONResponse({
            "session_state": "session_started",
            "person_detected": True,
            "match_status": "unknown",
            "confidence": score,
            "message": "Unknown person confirmed — recording started, will ask name after session ends.",
        })


@app.get("/session-status")
def session_status():
    """Frontend polls this to get full current state."""
    import app.services.voice_app.recorder_util as ru
    now = time.time()
    with _session_lock:
        session_snapshot = dict(_session)

    grace_countdown = 0
    if session_snapshot["state"] == "grace_period":
        grace_countdown = max(0, int(GRACE_PERIOD_SECONDS - (now - session_snapshot["grace_started_at"])))
    return {
        "state": session_snapshot["state"],
        "person_name": session_snapshot["person_name"],
        "session_duration": int(now - session_snapshot["session_started_at"]) if session_snapshot["session_started_at"] else 0,
        "grace_countdown": grace_countdown,
        "is_recording": ru.IS_RECORDING,
        "is_summarizing": ru.IS_SUMMARIZING,
        "needs_registration": session_snapshot.get("pending_unknown_embedding") is not None,
        "last_error": _get_last_session_error(),
    }


@app.get("/live-log")
def live_log():
    """Returns last 50 internal system events for the HUD debug panel."""
    with _live_log_lock:
        return {"logs": list(_live_log)}


@app.post("/check-presence")
async def check_presence(file: UploadFile = File(...)):
    """Lightweight endpoint — YOLO/Haar only, no DeepFace. Used during sessions."""
    present = await run_in_threadpool(_sync_decode_and_check_presence, file)
    return {"person_present": present}


MAX_IDLE_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB — one ~10s chunk is a few hundred KB


@app.post("/idle-audio")
async def idle_audio(file: UploadFile = File(...)):
    """
    Client-side idle-state audio capture (replaces the old server-side mic
    recording — record_audio_with_vad/record_audio_from_mic captured the HOST
    machine's own mic, not the caller's, which doesn't work for anything beyond
    a single local desktop). The frontend records fixed-interval chunks from the
    CLIENT's microphone while sessionState is 'idle' and posts each one here.

    Only processes while the session state machine is actually idle — if a
    session started between the frontend deciding to send and this request
    arriving, the chunk is dropped rather than mixed into an active session's
    transcript.
    """
    with _session_lock:
        state = _session["state"]
    if state != "idle":
        return JSONResponse({"status": "skipped", "reason": f"session state is '{state}', not idle"})

    content = await file.read(MAX_IDLE_AUDIO_BYTES + 1)
    if len(content) > MAX_IDLE_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Idle audio chunk too large.")
    if not content:
        return JSONResponse({"status": "empty"})

    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".webm")
        os.close(temp_fd)
        with open(temp_path, "wb") as f:
            f.write(content)

        result = await run_in_threadpool(_sync_process_idle_audio, temp_path)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Idle audio processing failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/register-new")
async def register_new(
    name: str = Form(...),
    relationship: str = Form(...),
    priority: int = Form(3),
    file: UploadFile = File(None),  # frame is optional — we prefer the stored embedding
):
    """
    Register a brand-new person AFTER their session has ended.
    Uses the embedding captured during the session (stored in _saved_unknown_embedding).
    Falls back to extracting from the uploaded frame if no stored embedding is available.
    """
    global _saved_unknown_embedding

    # Prefer the embedding already captured during the session
    embedding = _saved_unknown_embedding

    if embedding is None and file is not None:
        # Fallback: try to extract from the uploaded frame
        _push_log("[REGISTER] No stored embedding, extracting from uploaded frame...")
        embedding, _ = await run_in_threadpool(_sync_decode_and_embed, file)

    if embedding is None:
        raise HTTPException(
            status_code=422,
            detail="No face embedding available. Please ensure a face was detected during the session."
        )

    # ── Registration: must commit before anything below, since _save_initial_interaction
    # opens a SEPARATE connection and inserts a row with an FK to this knownperson row —
    # that insert would hang/deadlock waiting on an uncommitted referenced row. So this
    # transaction boundary is real, not just convenience: everything after the commit is
    # best-effort cleanup whose failure must NOT be reported as a failed registration,
    # since the person/encoding are already permanently saved by that point.
    person_id, encoding_id = await run_in_threadpool(
        _sync_register_new_person, name, relationship, priority, embedding
    )

    # ── Best-effort post-registration side effects — none of these can un-succeed
    # the registration above, so their failures are logged, not raised.
    _saved_unknown_embedding = None

    with _session_lock:
        pending_audio = _session.get("pending_audio_path")
        _session["pending_audio_path"] = None

    interaction_id = None
    try:
        interaction_id = await run_in_threadpool(_save_initial_interaction, person_id)
    except Exception as e:
        logger.error(f"Post-registration: failed to create interaction for person {person_id}: {e}")

    try:
        from app.services.face_recognition.face_service import clear_face_cache
        clear_face_cache()
    except Exception as e:
        logger.error(f"Post-registration: failed to clear face cache: {e}")

    try:
        await run_in_threadpool(_end_session, "registered")
    except Exception as e:
        logger.error(f"Post-registration: failed to end session cleanly: {e}")

    if pending_audio and interaction_id:
        _push_log(f"[MIC] ⚙️ Processing session audio for {name}...")
        import app.services.voice_app.recorder_util as ru
        try:
            ru.process_recording_in_background(interaction_id, pending_audio)
        except Exception as e:
            logger.error(f"Post-registration: failed to kick off audio processing: {e}")

    return JSONResponse({
        "message": f"{name} registered successfully.",
        "personid": person_id,
        "faceencodingid": encoding_id,
    })


@app.get("/config/provider")
def get_provider():
    """Current LLM provider — lets the frontend reconcile its localStorage cache
    with actual server state on load, instead of trusting a stale local value."""
    return {"provider": os.environ.get("LLM_PROVIDER", "groq")}


@app.post("/config/provider")
def set_provider(payload: dict):
    """Set LLM provider at runtime by updating .env.
    Expected payload: {"provider": "openai"|"groq"|"ollama"}
    """
    provider = payload.get("provider", "").lower()
    if provider not in {"openai", "groq", "ollama"}:
        raise HTTPException(status_code=400, detail="Invalid provider")
    # Update .env file in project root
    env_path = Path(__file__).resolve().parents[2] / ".env"
    set_key(env_path, "LLM_PROVIDER", provider)
    # Update the running process environment immediately
    os.environ["LLM_PROVIDER"] = provider
    # Reset cached LLM clients in both service modules so next call recreates them
    try:
        from app.ai_models.interaction.interaction_service import reset_llm_client as reset1
        reset1()
    except Exception:
        pass
    try:
        from app.services.conversation_summarizer import reset_llm_client as reset2
        reset2()
    except Exception:
        pass
    return {"status": "ok", "provider": provider}


@app.post("/register")
async def register(
    file: UploadFile = File(...),
    personid: int = Form(..., description="ID in public.knownperson"),
):
    """Register a face embedding for an existing person."""
    embedding, _ = await run_in_threadpool(_sync_decode_and_embed, file)
    if embedding is None:
        raise HTTPException(status_code=422, detail="Could not generate face embedding.")
    faceencoding_id = await run_in_threadpool(_sync_register_existing_person, personid, embedding)
    return JSONResponse({
        "message": "Face registered successfully",
        "personid": personid,
        "faceencodingid": faceencoding_id,
    })


@app.get("/system-status")
def system_status():
    """Legacy endpoint — kept for backward compat."""
    import app.services.voice_app.recorder_util as ru
    return {
        "is_recording": ru.IS_RECORDING,
        "is_summarizing": ru.IS_SUMMARIZING,
        "is_resting": False,
    }


# ---------------------------------------------------------------------------
# Reminder Endpoints
# ---------------------------------------------------------------------------

@app.get("/auth")
def auth():
    url = get_auth_url()
    return RedirectResponse(url)

@app.get("/oauth2callback")
def oauth_callback(code: str):
    exchange_code_for_token(code)
    return RedirectResponse("/")

@app.post("/create-reminder")
def add_reminder(data: ReminderRequest):
    try:
        event = create_reminder(data.title, data.date, data.time)
        return {"status": "created", "event_id": event.get("id")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-reminders")
def reminders():
    try:
        events = get_upcoming_reminders()
        results = [{"id": e.get("id"), "summary": e.get("summary"), "start": e.get("start", {}).get("dateTime")} for e in events]
        return JSONResponse(results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Merged-in REST/CRUD API (formerly app/app.py, port 8000)
# ---------------------------------------------------------------------------

app.include_router(main_router)
app.include_router(audio_router)
app.include_router(interaction_router)
app.include_router(reminder_router)

app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(caregivers.router, prefix="/api/caregivers", tags=["Caregivers"])
app.include_router(interactions.router, prefix="/api/interactions", tags=["Interactions"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])
app.include_router(memory.router, prefix="/api/memory", tags=["Memory"])
app.include_router(notes.router, prefix="/api/notes", tags=["Notes"])
app.include_router(calendar_events.router, prefix="/api/calendar", tags=["Calendar"])
app.include_router(audio.router, prefix="/api/sessions/audio", tags=["Session Audio"])
app.include_router(emotions.router, prefix="/api/emotions", tags=["Emotions"])

def health():
    return {"status": "healthy", "service": "face_recognition", "user_id": USER_ID}
