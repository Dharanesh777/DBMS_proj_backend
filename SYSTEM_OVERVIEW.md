# AG-OS — Current System Implementation

Snapshot of everything currently implemented in this repository, as of 2026-08-17.
AG-OS is an AI-powered cognitive assistant for people with short-term memory loss:
it recognizes visitors by face, records and transcribes conversations, summarizes
them with an LLM, detects emotional tone, and surfaces reminders/notes — backed
by a PostgreSQL database.

> **Note on architecture:** this used to be two parallel FastAPI apps (a live
> camera/session-engine app on port 8004, and a separate, unused REST/CRUD API
> on port 8000). They have been merged into **one backend, on port 8004** —
> `app/services/face_recognition/main.py` — so there is now a single entry
> point for everything described below. See `TECH_DEBT.md` for the merge note.

---

## 1. The backend (port 8004)

Entry point: `app/services/face_recognition/main.py`. Started by `start.bat`,
`docker compose up`, the `Dockerfile` `CMD`, or directly via
`uvicorn app.services.face_recognition.main:app --port 8004`.

It combines two things in one process:
- A **stateful, in-process session engine** driving live face recognition,
  continuous audio recording, and LLM summarization for the camera-driven
  frontend flow.
- A **layered REST/CRUD API** (routers → service classes → SQLAlchemy ORM →
  Postgres) for users, caregivers, interactions, sessions, memory, notes,
  calendar events, and emotion records.

Both halves read/write the same PostgreSQL schema. Two DB access layers
coexist within the merged app: the SQLAlchemy ORM (`app/db/`, used by the
`/api/*` CRUD routers) and raw `psycopg2` (`app/database/db.py`, used by the
face-recognition session engine and `interaction_service.py`'s ML pipeline).
`TECH_DEBT.md` documents that a third, now-deleted, duplicate face/person-CRUD
route set (`/api/face/*`, `/api/persons/*`) was removed earlier because it was
dead code.

### 1.1 Session state machine (live camera flow)

A single global session (`_session` dict, guarded by an `RLock`) tracks one
active visitor at a time:

- **`idle`** — every ~1.5s the frontend posts a camera frame to `/identify`.
  Runs the full pipeline: YOLOv8 person-detect → Haar-cascade face crop →
  DeepFace (`Facenet512`) embedding → cosine-similarity match against all
  stored `faceencoding` rows (in-memory cache, refreshed on registration).
  - similarity ≥ **0.70** → `confirmed` match
  - similarity ≥ **0.60** → `uncertain` match (still treated as known)
  - below that → `unknown`, but a session only starts after **3 consecutive**
    unknown detections (`UNKNOWN_STREAK_THRESHOLD`), to absorb transient
    misreads.
- **`session_active`** — a conversation row is created immediately (known
  person) or deferred (unknown person); continuous audio recording starts via
  `recorder_util.BackgroundRecorder` (uses `sounddevice`, retries mic-open on
  macOS CoreAudio teardown races). Presence is re-checked every 10s using a
  lightweight YOLO/Haar-only check (no DeepFace) to save CPU.
- **`grace_period`** — triggered when presence check fails; polls every
  request (fast ~500ms loop) for **5s** (`GRACE_PERIOD_SECONDS`); face
  returning cancels grace and resumes the session; timeout ends it.
- On session end: recording stops, and (in a background thread)
  **Whisper transcription → `conversation_summarizer.analyze_conversation`
  (LLM summary + emotion + extracted calendar events) → DB update
  (`update_conversation_results`) → auto-push any extracted events to
  `/create-reminder`** (retried up to 3x).
- **Unknown-person flow**: registration is deferred to session end. The
  frontend is told `needs_registration: true`; a modal collects name +
  relationship, posted to `/register-new`, which inserts `knownperson` +
  `faceencoding`, links `userknownperson`, clears the face-encoding cache,
  and (if audio was pending) kicks off transcription/summarization for that
  now-identified interaction.

### 1.2 Endpoints — face-recognition / session engine (prefix-less)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Welcome/health message |
| GET | `/health` | Pings the DB (`SELECT 1`) via the SQLAlchemy engine, reports `healthy`/`unhealthy` |
| POST | `/identify` | Main polling loop — face detect/match or presence check depending on state |
| GET | `/session-status` | Full current session state for the frontend HUD |
| GET | `/live-log` | Last 50 internal event-log entries (debug HUD) |
| POST | `/check-presence` | Lightweight YOLO/Haar-only presence check |
| POST | `/idle-audio` | Client-mic audio chunk capture while idle (transcribed, saved as a standalone conversation row with no person) |
| POST | `/register-new` | Register a brand-new person after their session ended |
| POST | `/register` | Add a face encoding for an *existing* `knownperson` |
| GET/POST | `/config/provider` | Get/set the active LLM provider (openai/groq/ollama) at runtime, persisted to `.env` |
| GET | `/auth`, `/oauth2callback` | Google Calendar OAuth (PKCE) flow |
| POST | `/create-reminder` | Create a Google Calendar event |
| GET | `/get-reminders` | Next 24h of Google Calendar events |
| GET | `/system-status` | Legacy recording/summarizing flags |
| GET | `/dashboard` | Static file mount (`app/static/`) |

### 1.3 Endpoints — REST/CRUD API (merged in from the former port-8000 app)

**Users — `/api/users`**: CRUD (`POST /`, `GET /{id}`, `GET /` paginated,
`PUT /{id}`, `DELETE /{id}`), plus `GET /{id}/caregivers` and
`GET /{id}/persons`. Delete is rejected with 409 if the DB doesn't yet have
cascade constraints (guards against a partially migrated schema).

**Caregivers — `/api/caregivers`**: CRUD scoped by `user_id` query param
(every get/update/delete requires the caller to pass the owning `user_id` —
closes an ID-enumeration hole noted in `CHANGELOG.txt`), plus `POST /assign`
and `POST /unassign` against the `usercaregiver` junction table.

**Interactions — `/api/interactions`**:
- `POST /start` — creates a `conversation` row and starts session 1 for a
  user+person pair. Rejects (409) if the user already has an active
  interaction (only one active interaction per user, enforced via
  `SessionManager`'s in-memory state).
- `POST /end` — ends the interaction: cancels the timer, summarizes the final
  session if there's unsaved transcript, **merges all session summaries into
  one interaction summary + overall emotion via the LLM**, writes
  `conversation.summarytext`/`emotiondetected`.

**Sessions — `/api/sessions/append` and session engine**: `SessionManager`
(`app/services/session_service.py`) implements a **30-minute rolling session
model**, independent of and unrelated to the live camera state machine in
§1.1 (different in-memory structures, different DB access path — SQLAlchemy
vs raw psycopg2 — to the same `conversation` table). Each REST-API
interaction is chopped into `SESSION_DURATION_MINUTES`-long sessions; each
session's accumulated transcript is summarized by the LLM when its timer
fires (via a Celery ETA task, migrated off the old in-process APScheduler so
timers survive a web process restart — they live in Redis), and a new
session starts automatically. `POST /api/sessions/append` appends a
transcript chunk to the active session's `conversation.conversation` text
column.

**Memory — `GET /api/memory/{person_id}`**: Fast, LLM-free retrieval of the
last `MEMORY_CONTEXT_LIMIT` (default 3) completed interaction summaries for a
person, scoped to `user_id`.

**Notes — `POST /api/notes`**: Creates a `note` row linked to an interaction
(user ownership resolved via the interaction, no separate `user_id` param
needed).

**Calendar events — `POST /api/calendar/events`**: Creates a `calendarevent`
row; validates `related_person_id` is actually linked to `user_id` via
`userknownperson` before allowing creation (a fix noted in `CHANGELOG.txt`).
This is pure DB storage — **not** synced to Google Calendar (that only
happens in the live session engine's post-session pipeline, §1.1).

**Session audio — `/api/sessions/audio`** (`app/api/routes/audio.py`):
- `POST /transcribe` — accepts an uploaded audio file (25MB cap, content-type
  allow-list), transcribes with Whisper, appends the text to the active
  session's transcript.
- `POST /record` — records from the **server's own microphone** (explicitly
  documented as dev/demo-only — not viable for a remote client).

**Emotions — `/api/emotions`**: Full CRUD for `emotionrecord`, all
read/delete endpoints scoped by `user_id` → interaction ownership (same
enumeration-hole fix pattern as caregivers). Intended to be called by an
external emotion-detection system ("Member A's" pipeline per the docstrings)
as well as the built-in LLM-based emotion tagging.

**Legacy/simple routes**:
- `app/routes/audio_routes.py` — `POST /api/audio/upload`, a simpler
  transcribe-and-save-conversation endpoint (no session concept), used by
  `app/controllers/audio_controller.py`.
- `app/routes/interaction_routes.py` (`/api/interaction/*`) — an alternate
  "Option A" single-shot workflow: `POST /detect_person` (fast presence
  check), `POST /process` (runs the full YOLO→DeepFace→Whisper→summarize
  pipeline on one frame + one audio blob, used for a non-continuous
  dashboard-driven flow), `POST /resolve_unknown` (completes registration
  for a person flagged `needs_registration` by `/process`, using an
  in-memory `temp_sessions` dict keyed by a UUID session id). Pre-existing
  duplicate of the live session engine's own pipeline — not removed by the
  backend merge, flagged as tech debt (see §7).

**Reminders (Celery + Redis)** — `app/ai_models/reminders/reminder_routes.py`
(prefix `/api`): `POST /api/schedule-reminder` schedules a Celery task with
an ETA (interprets naive datetimes as IST, converts to UTC); `GET
/api/get-notifications/{user_id}` drains a Redis list of fired reminders
(capped at 500 per request). `app/ai_models/reminders/tasks.py`'s
`remind_user` task auto-retries on failure and pushes the message onto
`notifications:{user_id}` in Redis. `celery_config.py` configures the Celery
app with a Redis broker/backend, `Asia/Kolkata` timezone, and discovers
`session_service.py`'s tasks too.

### 1.4 Startup

A `lifespan` hook runs `SessionManager.clear_all_sessions()` on startup to
clear orphaned in-memory 30-minute-session state left over from a previous
process — unrelated to and doesn't touch the live camera `_session` dict
(§1.1), which is naturally reset since it's a fresh process either way.

### 1.5 Frontend (`frontend/`, React + Vite)

- `App.jsx` — full-screen "AR HUD" style UI: live camera overlay
  (`CameraOverlay.jsx`), left panel system log, right panel session info +
  reminders, bottom LLM-provider switcher (openai/groq/ollama, calls
  `/config/provider`), registration modal for unknown visitors.
- Polls `/session-status` every 400ms, `/live-log` every 800ms, `/get-reminders`
  every 30s; each poll is abort-controlled to avoid overlap. Surfaces a "backend
  unreachable" banner if polling fails.
- `CameraOverlay.jsx` (319 lines) drives the actual `getUserMedia` camera feed
  and posts frames to `/identify`/`/idle-audio`.

---

## 2. Data model (PostgreSQL, schema `public`)

ORM models live in `app/models/`; the actual deployed schema is managed via
Alembic (`alembic/versions/74bf5b968796_...py` is the current single baseline
migration). `database/schema.sql` is a `pg_dump` snapshot that **predates**
that migration (it still shows `users.google_token_json` and lacks the
`ON DELETE CASCADE`/indexes the migration adds) — treat the Alembic migration
and ORM models as the source of truth, not the SQL dump.

| Table | Key columns | Notes |
|---|---|---|
| **users** | userid PK, name, age, medicalcondition, emergencycontact, email (unique) | `google_token_json` column dropped by migration — Google auth is now file-based only (`token.json`), not per-user in the DB |
| **knownperson** | personid PK, name, relationshiptype, prioritylevel, notes | People the user knows (family, caregivers, friends) |
| **caregiver** | caregiverid PK, name, relationshiptouser, accesslevel | |
| **conversation** | interactionid PK, userid FK→users (CASCADE), personid FK→knownperson (SET NULL), interactiondatetime, location, conversation (raw transcript text), summarytext, emotiondetected | The central "interaction" table — one row per visit/conversation |
| **emotionrecord** | emotionid PK, interactionid FK→conversation (CASCADE), emotiontype, confidencelevel | Supports multiple emotion samples per interaction |
| **calendarevent** | eventid PK, userid FK→users (CASCADE), relatedpersonid FK→knownperson (SET NULL), eventtitle, eventdatetime, remindertime | |
| **note** | noteid PK, interactionid FK→conversation (CASCADE), content, createdat | (schema.sql also shows an `importancelevel` column not present in the current ORM model/migration) |
| **faceencoding** | faceencodingid PK, personid FK→knownperson (CASCADE), encodingdata (TEXT, JSON-serialized float vector), confidencescore, createdat | Cosine similarity is computed in application code (NumPy), not in SQL |
| **usercaregiver** | userid+caregiverid composite PK, both CASCADE | M:M junction |
| **userknownperson** | userid+personid composite PK, both CASCADE | M:M junction |

---

## 3. AI/ML pipeline components

| Component | File | Detail |
|---|---|---|
| **Person/face detection** | `face_service.py` | YOLOv8n (`yolov8n.pt`) for person bounding boxes, falling back to Haar cascade (`haarcascade_frontalface_default.xml`) if YOLO finds nothing; face localized within the person box (or top-25%-of-body heuristic if the cascade fails) |
| **Face embedding** | `face_service.py` | DeepFace, `Facenet512` model, 512-d vectors, `enforce_detection=False` |
| **Face matching** | `face_service.py` | Cosine similarity against an in-memory cache of all `faceencoding` rows loaded from Postgres; thresholds 0.70 (confirmed) / 0.60 (uncertain) |
| **Fast presence check** | `interaction_service.py::check_face_fast` | Pure Haar-cascade-only check to avoid PyTorch/MPS threading deadlocks on Mac, for 1-FPS polling |
| **Speech-to-text** | `voice_app/transcription_service.py` | OpenAI Whisper (`small` model, local, auto language detection — tuned for English + Tamil) |
| **Conversation summarization (live session engine)** | `conversation_summarizer.py::analyze_conversation` | LLM call (provider-switchable) returning `{summary, emotion, events[]}` as structured JSON in one shot, including calendar-event extraction |
| **Conversation summarization (REST API sessions)** | `llm_service.py` | OpenAI-only; two-stage — per-session summary+emotion, then a merge step across all sessions in an interaction; explicit prompt-injection mitigation (transcript wrapped in `<transcript>` tags with an instruction to treat it as data) |
| **LLM provider abstraction** | `conversation_summarizer.py`, `interaction_service.py` (duplicated) | Runtime-switchable OpenAI / Groq (`llama-3.1-8b-instant` default) / Ollama (local `llama3` default) via an OpenAI-compatible client, cached per-provider, resettable after a `/config/provider` change |

---

## 4. Infrastructure & auxiliary services

- **Celery + Redis** — shared broker for (a) scheduled reminder notifications
  and (b) the REST API's 30-minute session-expiry timers. Requires a running
  `celery worker` process (not started automatically by any script observed
  here — an operational prerequisite).
- **Google Calendar OAuth** — file-based token storage (`token.json`,
  `credentials.json`) via PKCE (`google_auth.py`); a *separate*, now-removed
  DB-backed per-user OAuth path was deleted as redundant dead code (see
  `calendar_service.py`/`note_service.py` docstrings).
- **Alembic** — one baseline migration in place; adds indexes and
  `ON DELETE CASCADE`/`SET NULL` semantics across all FKs, drops
  `users.google_token_json`.
- **Docker** — `Dockerfile` builds the port-8004 app (installs ffmpeg,
  OpenCV/torch system deps, portaudio; runs as non-root `appuser`; healthcheck
  hits `/session-status`).
- **Config** — `app/config.py` centralizes all env vars via
  `pydantic-settings` (DB connection, OpenAI key/model, Google OAuth,
  session/LLM timing knobs, CORS). The live session engine's LLM provider
  selection is a separate, simpler `os.getenv("LLM_PROVIDER")` mechanism,
  runtime-mutable via `.env` rewriting (`python-dotenv`'s `set_key`).
- **Tests** — `tests/test_api.py`: pytest integration tests against a running
  `app.services.face_recognition.main:app` (port 8004), auto-skipped if no
  server is reachable at `/health`. Covers user/caregiver CRUD and assignment
  flows.

---

## 5. Known gaps / documented tech debt (`TECH_DEBT.md`)

- **No tenant/auth isolation on face matching**: `face_service.compare_embedding()`
  matches an incoming face against *every* `faceencoding` row in the database,
  with no per-user scoping anywhere in the call chain. There is currently no
  authentication layer or enforced multi-user concept at all — the app should
  be treated as single-tenant only until this is redesigned.
- A prior duplicate face/person CRUD route set (`/api/face/*`, `/api/persons/*`)
  was confirmed dead (unused by the frontend) and has been deleted.
- `database/schema.sql` is a stale dump — do not use it as the schema
  reference; the Alembic migration + ORM models are authoritative.
- Two independent copies of the LLM-provider-client logic exist
  (`conversation_summarizer.py` and `ai_models/interaction/interaction_service.py`)
  with near-identical code — not yet consolidated.
- `app/routes/interaction_routes.py`'s `/api/interaction/*` "Option A" flow
  duplicates the live session engine's own face+audio pipeline — pre-existing
  overlap, not removed by the single-backend merge.
- Session timers and reminder scheduling both depend on a continuously
  running Celery worker + Redis; if unreachable, interaction creation
  degrades gracefully (no auto-expiry timer) rather than failing.

---

## 6. Security-relevant fixes already applied (per `CHANGELOG.txt`)

- CORS: `allow_credentials` set to `False` (was combined with
  `allow_origins=["*"]`, an invalid/unsafe combination).
- Google OAuth PKCE verifier is now a random per-flow secret (was hardcoded).
- Caregiver and emotion-record endpoints now require and enforce a
  `user_id`-scoped ownership check (previously any ID was readable/writable
  by guessing).
- Calendar event creation validates the related person is actually linked to
  the requesting user.
- Audio upload `userid` is now required (was silently defaulting to `1`).
- LLM prompts wrap untrusted transcript text in explicit delimiters with
  anti-prompt-injection instructions.
- Audio uploads are capped at 25MB with a content-type allow-list; a
  magic-byte sniff check gates raw audio bytes before they reach ffmpeg/Whisper.
- `requirements.txt`/`.gitignore` reconciled; secret files (`token.json`,
  `credentials.json`) explicitly ignored rather than a blanket `*.json` rule.
