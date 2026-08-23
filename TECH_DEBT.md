# Tech Debt

## Resolved: `detect_face()`'s Haar cascade sometimes returned the WRONG PERSON'S face

Found while investigating an unrelated accuracy anomaly during face-backend
evaluation (`docs/face_recognition_backend_evaluation.md` — read that for
the full investigation, reproduction scripts, and evidence). Not a Pi 3 /
Redis / migration issue — this affects the live app today, on any hardware.

`app/services/face_recognition/face_service.py::detect_face()`'s Haar
cascade step takes the *largest* detected blob with no confidence or
plausibility check (`faces = sorted(faces, key=lambda f: f[2]*f[3],
reverse=True); return faces[0]`). Reproduced concretely: fed real photos
(LFW) through the app's actual `detect_face()`/`crop_face()` and confirmed
— by looking at the resulting crops directly, not just similarity scores —
that in a multi-face frame it sometimes crops a **different person's face**
than the one intended, or a badly-off-target non-face region. Quantified:
9 of 15 tested identities had at least one crop scoring at or below
average *impostor* similarity to their own correctly-cropped photos.

**Why this is a priority, not just an accuracy nit:** if this happens
during `/register-new` (enrolling a new known person while someone else is
also in frame, e.g. a caregiver or family member standing nearby), the
face template saved for that person could be of the *wrong person
entirely* — not a tuning problem, a potential misidentification one. If it
happens during `/identify` matching, it can cause both false rejects (own
face, but a bystander got cropped instead) and, less likely but not
impossible, false accepts (a bystander's face happens to match someone
else's registered encoding).

**Detector comparison** (`docs/face_detector_replacement_recommendation.md`,
`scripts/compare_face_detectors.py`) compared four candidates (`mtcnn`,
`yunet`, `ssd`, `fastmtcnn`). Headline finding: **the "reject low-confidence
detections" fix direction doesn't work as assumed** — 3 of 4 detectors
return confidence saturated near 1.0 regardless of correctness, unusable
for thresholding; only `yunet` shows real variance, and even its own worst
failure case in testing wasn't caught by it. `fastmtcnn` had zero
reproductions of the wrong-face bug in that test (vs. Haar's 9/15) and
tied for best FRR (44.76% → 30.00% for Facenet512).

**Fix implemented:** `face_service.py::detect_face()` now uses
`FACE_LOCALIZATION_DETECTOR = "fastmtcnn"` via `DeepFace.extract_faces()`
directly on the full frame (no more YOLO-person-box + Haar-cascade), always
selecting the *highest-confidence* detected face rather than the largest
Haar blob. `detect_person()` (used elsewhere for lightweight presence
checks, unrelated to embedding) is untouched.

**Important subtlety found during live-path re-verification, not just the
benchmark:** an initial implementation used `align=False` (since alignment
itself was already shown not to help — see
`docs/face_recognition_backend_evaluation.md`) and *regressed* on the exact
photo that originally exposed this bug. Root cause: in a multi-face frame,
fastmtcnn frequently returns **every** face at the identical confidence
(1.0000, tied) — so `max()`'s tie-breaking picks whichever face is *first*
in the returned list, and that list's order changes depending on the
unrelated `align` parameter. `align=True` happened to order the correct
face first on the photos tested; `align=False` didn't. Switched to
`align=True` and re-confirmed 0/15 bad crops on the live
`detect_face()`/`crop_face()`/`generate_embedding()` path (not just the
isolated benchmark script), full FAR/FRR on the live pipeline: **FRR
44.76% → 30.48%, FAR 0.00%** (420 genuine / 420 impostor pairs, same
identities as prior rounds). **Flagging plainly: this fix's reliability
partly depends on DeepFace's internal tie-breaking order for `align=True`,
observed to work correctly on every case tested, not something guaranteed
correct by construction** — if a future case reproduces the wrong-face bug
again, re-run `scripts/inspect_face_crop_quality.py` first, this tie-break
sensitivity is the first thing to check.

**Verified regression-safe:** an empty/no-face frame (tested with both
random noise and a blank frame) still correctly returns "no face detected"
— `extract_faces()` with `enforce_detection=False` returns a full-frame
fallback box at confidence exactly 0.0 rather than an empty list, which the
new code explicitly gates on (confirmed this matters — an earlier version
without this check would have made `detect_face()` report a face on every
single frame, breaking the idle/session state machine).

**Follow-up: Option A flow fixed too, same bug class, confirmed
independently.** `app/ai_models/interaction/interaction_service.py`'s
"Option A" flow (`process_interaction_payload`) was confirmed from code —
not assumed — to be *worse* than the pre-fix `detect_face()`: it fed
`fs.detect_person()`'s *person* box (YOLO, not face-specific at all)
directly into `crop_face()`, with no face-localization step whatsoever.
Fixed by reusing the already-fixed `fs.detect_face()` in place of
`fs.detect_person()` (one call-site swap — no new detection logic
duplicated). Re-verified two ways against the real code, not a new
isolated benchmark: (1) called the actual `process_interaction_payload()`
end-to-end (LLM call mocked to avoid a real external call, DB match mocked
to force the no-write branch) on the exact photo that originally exposed
the bug — its returned embedding is byte-identical to calling
`detect_face()`/`crop_face()`/`generate_embedding()` directly, confirming
the fix is correctly wired through; (2) re-ran the full 15-identity
worst-crop check through this flow's actual input path (JPEG-encode/decode
round trip, since that's genuinely different from the direct benchmark) —
**0/15 bad crops**, matching the primary flow's result.

**Enrolled-data exposure from this specific flow — checked, not
theoretical:** `process_interaction_payload` can return an embedding that
`/resolve_unknown` (`interaction_routes.py`) persists via
`save_person()`/`save_faceencoding()`, so this flow does share the
capability to write corrupted enrollments. But `save_person()` calls from
that path pass no `notes` argument — checked the real DB and **all 4
currently enrolled people have `notes='Registered via live camera'`**,
the string set only by the *other* (`/register-new`) flow. None of
today's 4 enrollments came through Option A's registration path — it
adds no new people to the re-enrollment list below, but was a live latent
risk for any future use of that flow until this fix.

**Blast radius on already-enrolled data — assessed, NOT fixable
retroactively, re-enrollment recommended.** Confirmed from code
(`app/models/face_encoding.py`'s schema, and grepping the whole
registration/identification call chain for any image-write) that **no raw
image or crop is ever retained anywhere** — not in the DB (`faceencoding`
stores only the derived embedding vector as text), not on disk (frames are
processed in-memory only). Checked the actual production DB (read-only):
**4 enrolled people today, every one with exactly 1 stored face
encoding — zero have more than one**, so even the partial audit of
"cross-check a person's multiple stored encodings against each other" has
no candidates. **There is no way to retroactively determine whether any of
today's 4 enrolled templates were corrupted by this bug.** Recommendation:
**re-enroll all 4 currently-known persons** (delete and re-run
`/register-new` or add a fresh `/register` encoding for each) now that the
detector fix is in place, rather than leaving this as a decision for
whoever notices a misidentification later — this is an explicit
recommendation, not something this change enforces automatically; nothing
here deletes or flags the existing 4 `faceencoding` rows.

**Follow-up done — a helper exists now, this isn't just a paragraph
anymore:** `scripts/reenroll_known_persons.py`. `--list` shows who's
enrolled (read-only); running it interactively (or with
`--personid N --photo path.jpg`) re-registers a person through the exact
same functions the real `/register` endpoint uses
(`_sync_decode_and_embed`, `_sync_register_existing_person` in
`main.py`) — not reimplemented — asserts `FACE_LOCALIZATION_DETECTOR ==
"fastmtcnn"` before writing anything (aborts loudly instead of silently
re-enrolling through a reverted detector), and reports the before/after
encoding count. `--delete-old-encodings` optionally removes the
superseded encoding(s) after the new one is confirmed written (off by
default — it's a destructive delete). Verified end-to-end against a
throwaway scratch person created and deleted for this purpose only — did
**not** run this against the real 4 people (no real photos of them
available in this environment; re-enrolling them is still an open action
item for whoever has camera access to Tejas R P, Dharanesh, Praneeth, and
Harsha, using this script).

## Resolved: Redis persistence was off by default, silently defeating the state migration

Follow-up to the Redis migration below: verified (`redis-cli config get save`
/ `appendonly` against both the running instance and Homebrew's packaged
`redis.conf`) that a default Redis install has AOF off and only loose RDB
snapshot points (as infrequent as one save/hour under light load). Confirmed
empirically that a bare `redis-server` with no persistence loses all
mirrored state across a `kill -9` — i.e. the write-through mirroring below
protected against an *app* restart but not a *Redis* restart or power loss,
which was the actual point of doing it for a Pi 3 deployment.

Fixed: `redis.conf` at the repo root sets `appendonly yes` /
`appendfsync everysec` (not `always` — real SD-card wear concern on a Pi 3
over the device's lifetime). `howtorun.txt` and
`app/ai_models/reminders/HowToUse.txt` now reference it; the latter's old
Windows-dev instructions (`redis-cli config set save ""`, which disables
persistence entirely) are now explicitly marked as local-dev-only, not for
any real deployment. Re-verified the `kill -9` scenario with this config
loaded — state now survives (once the ~1s `everysec` fsync window has
passed).

**OPEN GAP — not yet validated on target hardware:** every persistence test
so far (this entry and the `kill -9` one above) has run on the dev Mac only
(Homebrew Redis, APFS on SSD/NVMe). None of it has run on an actual
Raspberry Pi 3 or any SD-card-backed storage. The whole reason
`appendfsync everysec` vs `always` matters is fsync latency to the actual
storage medium — an SSD and a Pi 3's SD card are not comparable there, and
this hasn't been checked on the real thing. The `kill -9` reproduction now
lives as a proper, runnable script — `tests/hardware/verify_redis_persistence.py`
(see `tests/hardware/README.md`) — instead of only in chat history. Whoever
gets Pi 3 access: `python tests/hardware/verify_redis_persistence.py` and
update this entry with the result either way.

**Also added:** `SessionManager` now counts and distinctly logs
(`[MIRROR_WRITE_EXHAUSTED]`) every time `_mirror_session`'s retry budget is
fully exhausted (`SessionManager.mirror_exhausted_count()`), so it's
possible to see from real operation how often mirror writes are actually
failing — informs whether the local-disk dirty-marker mentioned in
`restore_and_sweep()`'s docstring (for the residual "restores stale-but-
under-threshold data" gap) is worth building, instead of guessing. Not
built yet — instrumentation only, per explicit request.

`main.py`'s `_mutate_session` (structure A, the live-camera `_session`
mirror) had the same gap as B before this: no retry at all, no counter, no
distinctly-tagged log — a single transient Redis blip failed the mirror
permanently with no visibility. Brought in line with B: same bounded retry
(2 attempts, ~0.3s apart), `_mutate_exhausted_count` /
`_get_mutate_exhausted_count()`, and a `[MUTATE_WRITE_EXHAUSTED]` log tag.
Both structures' retry/counter/log-tag behavior have a runnable
reproduction at `tests/hardware/verify_mirror_instrumentation.py`.

## Resolved: four process-local state structures moved to Redis (Pi 3 deployment)

Ahead of a Raspberry Pi 3 deployment, four in-process-only structures that
lost all state on a crash/restart were migrated to Redis, plus a real bug
this uncovered: `app/services/face_recognition/main.py`'s `_session` dict
(the live camera visit-session state machine), `app/services/session_service.py`'s
`SessionManager._active_sessions`/`_session_task_ids` (30-min REST
sub-session chunking — this one had a genuinely live bug: a Celery timer
survived a restart but the state it needed didn't, so a restart mid-session
silently dropped accumulated summaries), `app/services/face_recognition/face_service.py`'s
`_face_encodings_cache` (now a two-tier L1-in-process/L2-Redis cache), and
`app/routes/interaction_routes.py`'s `temp_sessions` (now Redis-backed with
a real 30-minute TTL — it had none before and leaked forever).

New shared client: `app/services/redis_client.py`. Uses `REDIS_STATE_DB`
(db=2 by default), distinct from Celery's own broker (db=0) / results
backend (db=1).

**Known inconsistency, not fixed by this change:** `app/config.py`'s
`Settings` class is this codebase's documented single source for env-var
config, but the pre-existing Celery/reminders stack
(`app/ai_models/reminders/{celery_config,tasks,reminder_routes}.py`) reads
`REDIS_HOST`/`REDIS_PORT` directly via `os.getenv()`, bypassing it — same
env vars, same defaults, just two different code paths reading them. Left
alone here to keep this change scoped to the four structures above; worth a
small dedicated cleanup pass later to route those three files through
`Settings` too.

## Resolved: merged the two backends into one

This repo used to ship two separate FastAPI apps against the same database:
`app/app.py` (`app.app:app`, run via `server.py`, port 8000 — a layered
REST/CRUD API for users/caregivers/interactions/sessions/memory/notes/
calendar/emotions) and `app/services/face_recognition/main.py` (port 8004 —
the live face-recognition/session-engine app actually started by
`start.bat`/`Dockerfile`/the frontend). The port-8000 app was never started
by anything in the deployed product.

Both apps have been merged into a single one: `app/app.py` and `server.py`
are deleted, and all of the REST/CRUD routers (plus the `/health` endpoint,
the `lifespan` startup hook, and the `/dashboard` static mount) have been
folded into `app/services/face_recognition/main.py`. There is now exactly
one backend, on port 8004. See `SYSTEM_OVERVIEW.md` for the current route
inventory.

## Resolved: duplicate face-recognition routes removed

`/api/face/*` (`app/routes/face_routes.py`, `app/controllers/face_controller.py`)
and `/api/persons/*` (`app/api/routes/persons.py`, `app/services/person_service.py`)
have been deleted. Neither was ever called by the frontend — confirmed by grepping
`frontend/src` for `api/face`/`api/persons` (zero hits) and checking `frontend/src/config.js`'s
`API_BASE` (defaults to `http://localhost:8004`, matching `app/services/face_recognition/main.py`,
the entrypoint `Dockerfile`/`howtorun.txt` actually run — not `app.app:app`, which
`server.py` runs separately on port 8000 and which hosted both dead routes).

The live face-recognition path is exclusively:
- `app/services/face_recognition/main.py` — routes `/identify`, `/register`,
  `/register-new`, `/session-status`, etc.
- `app/services/face_recognition/face_service.py` — detection/embedding/comparison
  logic, called directly in-process (no HTTP indirection)

## Still open: `face_service.compare_embedding()` has no tenant isolation

`SELECT personid, encodingdata FROM public.faceencoding` runs with **no `WHERE`
clause and no user/tenant filter anywhere in the call chain** — every `/identify`
request matches against every registered face in the system.

Removing `/api/persons/*` closed off the consolidation path described in the
previous version of this doc (that plan was to route matching through
`PersonService`'s `userknownperson`-scoped query — that service no longer exists).
This bug is now**only fixable by adding real scoping directly to
`face_service.py`**, which in turn depends on what "tenant" even means for this
app — there's currently no auth layer and no multi-user concept enforced anywhere
(explicitly deferred). Revisit this once auth is designed; until then, this app
should be treated as single-tenant-only in any deployment.
