# Tech Debt

## Duplicate face-recognition routes: `/api/face/*` vs `/api/persons/*`

Two independent implementations of "recognize/register a known person by face" exist
side by side after merging `backend/app/` into `app/`. Both were kept as-is per an
explicit decision not to consolidate them yet.

- `/api/face/identify`, `/api/face/register` — [app/routes/face_routes.py](app/routes/face_routes.py),
  [app/controllers/face_controller.py](app/controllers/face_controller.py),
  [app/services/face_recognition/face_service.py](app/services/face_recognition/face_service.py)
- `/api/persons/identify`, `/api/persons/register` — [app/api/routes/persons.py](app/api/routes/persons.py),
  [app/services/person_service.py](app/services/person_service.py)

### Storage layer: compatible

Both read/write the same physical tables — `public.knownperson` and
`public.faceencoding` (`encodingdata` as JSON-serialized TEXT) — using the same
DeepFace `Facenet512` embedding (512-d). A face registered through one path is
fully readable by the other at the storage layer.

### Behavioral differences

| | `/api/face/*` | `/api/persons/*` |
|---|---|---|
| Input | Raw image upload — runs YOLO detect → crop → DeepFace embed server-side | Pre-computed `encoding: list[float]` in the JSON body — no image-processing step exists in this path |
| Scope | Global — compares against every row in `faceencoding` | Per-user — joins through `userknownperson`, scoped to `request.user_id` |
| Threshold | 3-tier: confirmed ≥0.85, uncertain 0.70–0.85, else unknown | Single cutoff, `FACE_SIMILARITY_THRESHOLD` (default 0.60) |
| Register | Adds a photo/encoding to an *existing* `personid` | Always creates a *new* person + encoding + user link in one call |
| Response shape | Ad hoc dict (`person_detected`, `match_status`, `person_name`, ...) | Pydantic model (`person_id`, `name`, `memory_context: [...]` with last 3 interactions) |

### Bug: `/api/face/identify` has no tenant isolation

`face_service.compare_embedding()` runs
`SELECT personid, encodingdata FROM public.faceencoding` with **no `WHERE` clause
and no user/tenant filter anywhere in the call chain** (verified: zero references
to `user_id`/`userid` in `face_service.py`, `face_controller.py`, or
`face_routes.py`). Every identify request matches against every registered face
in the system. Concretely: User A's webcam frame can be matched to a person that
only User B ever registered, and User A's `/api/face/register` call adds a face
usable by anyone else's `/api/face/identify` calls.

This is a pre-existing bug independent of the `/api/persons` duplication — it
would need fixing even if `/api/persons` didn't exist.

### Proposed consolidation direction (not yet done)

Keep `/api/face`'s image-upload pipeline (YOLO detect → crop → DeepFace embed) as
the embedding-generation step, but have it call into `PersonService`
(`app/services/person_service.py`) for the actual matching/registration instead
of the global unscoped cache in `face_service.py`. This gives image-upload
convenience on the frontend while inheriting `PersonService`'s user scoping and
richer `memory_context` response — and fixes the tenant-isolation bug as a side
effect, since matching would go through the `userknownperson`-scoped query.

Until this consolidation happens, do not build new frontend features against
`/api/face/*` without being aware neither the isolation bug nor the route
duplication has been resolved.
