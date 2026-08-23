# Tech Debt

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
