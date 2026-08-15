import os
import tempfile
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from app.services.voice_app.transcription_service import transcribe_audio as transcribe_audio_file

# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

async def process_audio_upload(
    audio: UploadFile,
    userid: int,
    personid: Optional[int] = None,
):
    """
    Accept an uploaded audio file, transcribe it with Whisper and save to DB.
    """
    from app.database.db import save_conversation
    temp_path = None
    try:
        contents = await audio.read()
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)

        with open(temp_path, "wb") as f:
            f.write(contents)

        text = transcribe_audio_file(temp_path)
        if text is None:
            raise HTTPException(status_code=500, detail="Transcription failed.")
        interaction_id = save_conversation(userid, personid, text, None, None)

        return JSONResponse({
            "message": "Audio processed successfully",
            "transcription": text,
            "interactionid": interaction_id,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
