import json
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import redis

from app.ai_models.interaction.interaction_service import process_interaction_payload, check_face_fast
from app.database.db import (
    save_person,
    save_faceencoding,
    save_userknownperson,
    save_conversation
)
from app.services.redis_client import get_redis

interaction_router = APIRouter(prefix="/api/interaction", tags=["Interaction"])

# Pending-registration hold for the "needs_registration" case below — was
# previously a bare process-local dict with no TTL, so an abandoned
# registration (user never calls /resolve_unknown) leaked forever until the
# next restart. Redis gives it both a real TTL and durability across a
# restart of the web process itself.
TEMP_SESSION_KEY_PREFIX = "agos:interaction:tempsession:"
TEMP_SESSION_TTL_SECONDS = 30 * 60


def _temp_session_key(session_id: str) -> str:
    return f"{TEMP_SESSION_KEY_PREFIX}{session_id}"


def _pop_temp_session(session_id: str) -> Optional[dict]:
    """Atomically fetch-and-delete via GETDEL (Redis >= 6.2). Falls back to a
    non-atomic GET+DEL on older servers — acceptable here since double-
    resolving the same UUID isn't a realistic concurrent scenario for this
    single-user registration flow."""
    r = get_redis()
    key = _temp_session_key(session_id)
    try:
        raw = r.getdel(key)
    except redis.exceptions.ResponseError:
        raw = r.get(key)
        if raw is not None:
            r.delete(key)
    return json.loads(raw) if raw is not None else None


class ResolveUnknownRequest(BaseModel):
    session_id: str
    userid: int = 1
    name: str
    relationship_type: str


@interaction_router.post("/detect_person")
async def detect_person(frame: UploadFile = File(...)):
    """
    Lightweight endpoint for frontend to ping 1-FPS to check if a person is in the frame.
    Returns: {"person_detected": true/false}
    """
    frame_bytes = await frame.read()
    detected = await run_in_threadpool(check_face_fast, frame_bytes)
    return JSONResponse({"person_detected": detected})


@interaction_router.post("/process")
async def process_interaction(
    userid: int = Form(1),
    frame: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    """
    Called by the dashboard after recording the conversation.
    Runs the full YOLO → DeepFace → Whisper → Summarization pipeline.
    """
    frame_bytes = await frame.read()
    audio_bytes = await audio.read()

    result = await run_in_threadpool(process_interaction_payload, userid, frame_bytes, audio_bytes)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    if result.get("status") == "needs_registration":
        session_id = str(uuid.uuid4())

        payload = {
            "embedding": result["embedding"],
            "transcription": result["transcription"],
            "summary": result["summary"],
            "emotion": result["emotion"],
            "confidence": result["confidence"],
        }
        try:
            get_redis().set(
                _temp_session_key(session_id),
                json.dumps(payload),
                ex=TEMP_SESSION_TTL_SECONDS,
            )
        except redis.exceptions.RedisError as e:
            # No in-process fallback for this one (unlike the session-engine
            # structures) — narrow blast radius though, just this one
            # in-flight registration attempt. Caller can retry /process.
            raise HTTPException(status_code=503, detail=f"Could not hold pending registration state: {e}")

        # Remove embedding from response to keep JSON light
        response_payload = result.copy()
        del response_payload["embedding"]
        response_payload["temp_session_id"] = session_id

        return JSONResponse(response_payload)

    # If known person, just return success
    return JSONResponse(result)


@interaction_router.post("/resolve_unknown")
def resolve_unknown(body: ResolveUnknownRequest):
    """
    Completes the registration for a previously unknown person.
    Requires the `session_id` returned from `/process`.
    """
    try:
        session_data = _pop_temp_session(body.session_id)
    except redis.exceptions.RedisError as e:
        raise HTTPException(status_code=503, detail=f"Could not retrieve pending registration state: {e}")

    if session_data is None:
        # Covers both "never existed" and "TTL expired" now — previously
        # this dict leaked forever with no expiry at all.
        raise HTTPException(status_code=404, detail="Session expired or invalid.")

    try:
        # 1. Save to knownperson
        new_person_id = save_person(
            name=body.name,
            relationship_type=body.relationship_type
        )
        if not new_person_id:
            raise Exception("Failed to insert into knownperson.")

        # 2. Save face encoding
        save_faceencoding(
            personid=new_person_id,
            embedding_vector=session_data["embedding"],
            confidencescore=session_data["confidence"]
        )

        # Invalidate the face-match cache (both the in-process L1 and the
        # Redis L2 — see face_service.py) so this person is recognized on
        # the very next /process or /identify call, not after the L2's
        # 6-hour TTL or a restart. This endpoint was the one registration
        # path that didn't do this — main.py's /register and /register-new
        # already did.
        from app.services.face_recognition.face_service import clear_face_cache
        clear_face_cache()

        # 3. Save userknownperson mapping
        save_userknownperson(userid=body.userid, personid=new_person_id)

        # 4. Save the conversation that just happened
        interaction_id = save_conversation(
            userid=body.userid,
            personid=new_person_id,
            transcribed_text=session_data["transcription"],
            summarized_text=session_data["summary"],
            detected_emotion=session_data["emotion"],
            location="Living Room via Dashboard"
        )

        return JSONResponse({
            "message": f"Successfully registered and saved interaction for {body.name}.",
            "personid": new_person_id,
            "interactionid": interaction_id
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
