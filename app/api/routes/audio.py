"""
api/routes/audio.py — Audio transcription endpoints (from legacy app)
"""
import os
import logging
import tempfile
import speech_recognition as sr
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.audio import (
    AudioTranscribeResponse,
    MicRecordRequest,
    MicRecordResponse,
)
from app.services.voice_app.transcription_service import transcribe_audio as transcribe_audio_file
from app.services.session_service import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_AUDIO_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/webm", "audio/ogg", "audio/flac",
    "audio/x-m4a", "audio/mp4",
}


async def _read_upload_with_limits(upload: UploadFile, max_bytes: int = MAX_AUDIO_BYTES) -> bytes:
    """Read an UploadFile in chunks, enforcing a content-type allow-list and a size cap."""
    content_type = (upload.content_type or "").lower()
    if content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio content type: {content_type or 'unknown'}",
        )

    chunks = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file exceeds the {max_bytes // (1024 * 1024)}MB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/transcribe", response_model=AudioTranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    interaction_id: int = Form(..., gt=0),
    db: Session = Depends(get_db),
):
    """
    Transcribe uploaded audio file and append to active session.

    This endpoint:
    1. Receives audio file upload
    2. Transcribes using Whisper
    3. Appends transcript to active session
    """
    temp_path = None
    try:
        # Save to temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)

        # Save uploaded file
        with open(temp_path, "wb") as f:
            content = await _read_upload_with_limits(audio)
            f.write(content)

        # Transcribe
        text = transcribe_audio_file(temp_path)
        if text is None:
            raise HTTPException(status_code=500, detail="Transcription failed.")
        logger.info(f"Transcribed audio for interaction {interaction_id}: {text[:50]}...")

        # Append to session
        session_manager = SessionManager(db)
        await session_manager.append_transcript(
            interaction_id=interaction_id,
            transcript_chunk=text,
        )
        
        return AudioTranscribeResponse(
            transcription=text,
            interaction_id=interaction_id,
        )
        
    except ValueError as e:
        logger.warning(f"Session not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/record", response_model=MicRecordResponse)
async def record_from_microphone(
    request: MicRecordRequest,
    db: Session = Depends(get_db),
):
    """
    Record audio from server's microphone and transcribe.
    
    NOTE: This uses the microphone of the machine hosting the backend.
    For production, use the /transcribe endpoint with client-side recording.
    """
    temp_path = None
    recognizer = sr.Recognizer()
    
    try:
        # Record from microphone
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logger.info(f"Recording for {request.duration_seconds} seconds...")
            audio_data = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=request.duration_seconds
            )
        
        # Save to temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        
        with open(temp_path, "wb") as f:
            f.write(audio_data.get_wav_data())
        
        # Transcribe
        text = transcribe_audio_file(temp_path)
        if text is None:
            raise HTTPException(status_code=500, detail="Transcription failed.")
        logger.info(f"Transcribed microphone recording: {text[:50]}...")

        return MicRecordResponse(transcription=text)
        
    except sr.WaitTimeoutError:
        logger.warning("Microphone timeout - no speech detected")
        raise HTTPException(status_code=408, detail="No speech detected within timeout period")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error recording from microphone: {e}")
        raise HTTPException(status_code=500, detail=f"Recording failed: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
