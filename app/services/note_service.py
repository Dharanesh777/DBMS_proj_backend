"""
services/note_service.py — Note creation

Previously also synced notes to Google Tasks via a per-user DB-stored OAuth
token (users.google_token_json). That was a second, redundant Google OAuth
mechanism alongside app/services/reminder_app/google_auth.py's file-based
token.json — the only one actually exercised by the live app. The DB-token
path was never wired to anything the frontend calls, so it and its sync call
were removed; this now does pure DB storage.
"""
import logging
from sqlalchemy.orm import Session

from app.models.note import Note

logger = logging.getLogger(__name__)


class NoteService:
    """Service for creating notes"""

    def __init__(self, db: Session):
        self.db = db

    def create_note(
        self,
        interaction_id: int,
        content: str,
        user_id: int,
    ) -> int:
        """
        Create a note.

        Args:
            interaction_id: Interaction ID
            content: Note content
            user_id: User ID (kept for API-shape compatibility; no longer used
                for a Google sync lookup)

        Returns:
            note_id
        """
        note = Note(
            interactionid=interaction_id,
            content=content,
        )
        self.db.add(note)
        self.db.commit()
        self.db.refresh(note)

        note_id = note.noteid
        logger.info(f"Created note {note_id}")
        return note_id
