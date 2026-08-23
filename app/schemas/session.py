"""
schemas/session.py — Pydantic schemas for session transcript appending
"""
from pydantic import BaseModel, Field


class SessionAppendRequest(BaseModel):
    """POST /api/sessions/append — request payload"""
    interaction_id: int = Field(..., gt=0)
    transcript_chunk: str = Field(..., min_length=1, max_length=10000)


class SessionAppendResponse(BaseModel):
    """POST /api/sessions/append — response payload"""
    message: str = "Transcript appended successfully"
