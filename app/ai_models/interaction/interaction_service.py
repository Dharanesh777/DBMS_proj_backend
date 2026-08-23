import os
import cv2
import json
import logging
import tempfile
import numpy as np
from typing import Dict, Any, Tuple
from openai import OpenAI

from app.services.face_recognition import face_service as fs
from app.services.voice_app.transcription_service import transcribe_audio
from app.database.db import save_conversation

logger = logging.getLogger(__name__)

# ── LLM client cache ──────────────────────────────────────────────────────────
_openai_client = None
_cached_provider = None


def get_llm_client() -> tuple[OpenAI, str]:
    """Return a configured LLM client and provider name.
    Supported providers: 'openai', 'groq', 'ollama'.
    The client is recreated whenever LLM_PROVIDER changes at runtime.
    """
    global _openai_client, _cached_provider
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if _openai_client is None or provider != _cached_provider:
        logger.info(f"[LLM] Building new client for provider: {provider}")
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment variables.")
            _openai_client = OpenAI(api_key=api_key)
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set in environment variables.")
            _openai_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        elif provider == "ollama":
            _openai_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
        _cached_provider = provider
    return _openai_client, provider


def reset_llm_client():
    """Force the LLM client to be recreated on next call (call after provider switch)."""
    global _openai_client, _cached_provider
    _openai_client = None
    _cached_provider = None


def summarize_conversation_and_emotion(text: str) -> Tuple[str, str]:
    if not text or not text.strip() or len(text.strip()) < 2:
        return "No conversation detected.", "Neutral"

    client, provider = get_llm_client()
    if not client:
        return "Summary unavailable (missing API key).", "Neutral"

    # Choose model based on provider
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "groq":
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    elif provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3")
    else:
        model = "gpt-4o-mini"

    prompt = f"""
    Analyze the following conversation text:
    "{text}"

    Please provide a structured JSON response with exactly two keys:
    1. "summary": A very brief 1-sentence summary of what was discussed.
    2. "emotion": The single most dominant emotion detected (e.g. Happy, Concerned, Neutral, Angry, Sad, Enthusiastic).

    Respond ONLY in valid JSON.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        summary = data.get("summary", "No summary.")
        emotion = data.get("emotion", "Neutral")
        return summary, emotion
    except Exception as e:
        logger.error(f"LLM error during summarize/emotion: {e}")
        return "Summary failed.", "Neutral"


def check_face_fast(frame_bytes: bytes) -> bool:
    """
    Ultra-fast, thread-safe detector for the 1-FPS frontend polling.
    Uses pure OpenCV Haar Cascades to avoid PyTorch/MPS threading deadlocks on Mac.
    """
    from app.services.face_recognition.face_service import get_face_cascade

    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = get_face_cascade()
    if cascade is None or cascade.empty():
        logger.error("Haar cascade failed to load — cannot run fast face check.")
        return False
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    return len(faces) > 0


def _looks_like_audio(data: bytes) -> bool:
    """Cheap magic-byte check before handing untrusted client bytes to ffmpeg/Whisper."""
    if len(data) < 12:
        return False
    header = data[:12]
    if header[:4] == b"\x1a\x45\xdf\xa3":  # WebM / Matroska (EBML)
        return True
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":  # WAV
        return True
    if header[:4] == b"OggS":  # OGG
        return True
    if header[:3] == b"ID3" or header[:2] == b"\xff\xfb":  # MP3
        return True
    if header[4:8] == b"ftyp":  # MP4 / M4A
        return True
    return False


def process_interaction_payload(userid: int, frame_bytes: bytes, audio_bytes: bytes) -> Dict[str, Any]:
    """
    Option A workflow: receives the captured frame and audio from the browser,
    runs the ML pipeline, and stores interaction if known.
    """
    # 1. Image Processing
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image payload."}

    # Was fs.detect_person() (a YOLO PERSON box, not a face box) fed directly
    # into crop_face() — no face-specific localization at all, an even
    # coarser version of the wrong-face-crop bug fixed in detect_face()
    # (see TECH_DEBT.md). Reuses that same fixed function rather than
    # reimplementing the fastmtcnn/confidence-selection/align=True/
    # empty-frame-gating logic separately.
    detected, bbox = fs.detect_face(frame)
    if not detected:
        return {"error": "No face detected in the provided frame."}

    face_image = fs.crop_face(frame, bbox)
    if face_image is None:
        return {"error": "Face detected, but crop failed."}

    embedding = fs.generate_embedding(face_image)
    if embedding is None:
        return {"error": "Could not generate face embedding."}

    # 2. Audio Processing (Whisper)
    if not _looks_like_audio(audio_bytes):
        return {"error": "Audio payload does not look like a supported audio format."}

    temp_fd, temp_path = tempfile.mkstemp(suffix=".webm")
    os.close(temp_fd)

    transcribed_text = ""
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        transcribed_text = transcribe_audio(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 3. Model Logic
    best_person_id, similarity, match_status = fs.compare_embedding(embedding)
    summary, emotion = summarize_conversation_and_emotion(transcribed_text)

    # 4. Route Execution
    if match_status == "unknown" or best_person_id is None:
        return {
            "status": "needs_registration",
            "message": "Encountered an unknown person.",
            "transcription": transcribed_text,
            "summary": summary,
            "emotion": emotion,
            "embedding": embedding,
            "confidence": round(similarity, 4) if similarity else 0.0
        }

    # Known Person
    details = fs.fetch_details(best_person_id)
    person_name = details["name"] if details else "Unknown"

    interaction_id = save_conversation(
        userid=userid,
        personid=best_person_id,
        transcribed_text=transcribed_text,
        summarized_text=summary,
        detected_emotion=emotion,
        location="Living Room via Dashboard"
    )

    return {
        "status": "success",
        "message": f"Interaction recorded with known person: {person_name}.",
        "match_status": match_status,
        "person_name": person_name,
        "relationship_type": details["relationship"] if details else None,
        "confidence": similarity,
        "transcription": transcribed_text,
        "summary": summary,
        "emotion": emotion,
        "interactionid": interaction_id
    }
