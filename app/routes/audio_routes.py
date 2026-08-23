from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from app.controllers.audio_controller import process_audio_upload

audio_router = APIRouter(prefix="/api/audio", tags=["Audio"])


@audio_router.post("/upload")
async def audio_upload(
    audio: UploadFile = File(..., description="Audio file to transcribe (WAV/MP3)"),
    userid: int = Form(..., description="ID of the user this recording belongs to"),
    personid: Optional[int] = Form(None),
):
    """
    Upload an audio file for transcription.

    - **audio**: WAV or MP3 file
    - **userid**: ID of the user (required)
    - **personid**: ID of the person being spoken to (optional)
    """
    return await process_audio_upload(audio=audio, userid=userid, personid=personid)
