"""
services/calendar_service.py — Calendar event creation

Previously also synced events to Google Calendar via a per-user DB-stored OAuth
token (users.google_token_json). That was a second, redundant Google OAuth
mechanism alongside app/services/reminder_app/google_auth.py's file-based
token.json — the only one actually exercised by the live app (the frontend's
reminders panel and the post-session reminder pipeline both go through it).
The DB-token path was never wired to anything the frontend calls, so it and
its sync call were removed; this now does pure DB storage.
"""
import logging
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.calendar_event import CalendarEvent
from app.models.user import User
from app.models.junction_tables import userknownperson

logger = logging.getLogger(__name__)


class CalendarService:
    """Service for creating calendar events"""

    def __init__(self, db: Session):
        self.db = db

    def create_event(
        self,
        user_id: int,
        event_title: str,
        event_datetime: datetime,
        related_person_id: int | None = None,
        reminder_time: datetime | None = None,
    ) -> int:
        """
        Create a calendar event.

        Args:
            user_id: User ID
            event_title: Event title
            event_datetime: Event date/time
            related_person_id: Optional related person ID
            reminder_time: Optional reminder time

        Returns:
            event_id

        Raises:
            ValueError: If user_id doesn't exist, or related_person_id isn't linked to user_id
        """
        user = self.db.get(User, user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        if related_person_id is not None:
            linked = self.db.execute(
                select(userknownperson).where(
                    userknownperson.c.userid == user_id,
                    userknownperson.c.personid == related_person_id,
                )
            ).first()
            if not linked:
                raise ValueError(
                    f"Person {related_person_id} is not linked to user {user_id}"
                )

        event = CalendarEvent(
            userid=user_id,
            relatedpersonid=related_person_id,
            eventtitle=event_title,
            eventdatetime=event_datetime,
            remindertime=reminder_time,
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)

        event_id = event.eventid
        logger.info(f"Created calendar event {event_id}")
        return event_id
