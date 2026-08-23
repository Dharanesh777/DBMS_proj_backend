"""
api/routes/calendar_events.py — Calendar event creation endpoint
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.calendar_event import CalendarEventCreateRequest, CalendarEventCreateResponse
from app.services.calendar_service import CalendarService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/events", response_model=CalendarEventCreateResponse, status_code=201)
async def create_calendar_event(
    request: CalendarEventCreateRequest,
    db: Session = Depends(get_db),
):
    """
    Create a calendar event.
    """
    try:
        calendar_service = CalendarService(db)

        event_id = calendar_service.create_event(
            user_id=request.user_id,
            event_title=request.event_title,
            event_datetime=request.event_datetime,
            related_person_id=request.related_person_id,
            reminder_time=request.reminder_time,
        )

        return CalendarEventCreateResponse(event_id=event_id)

    except ValueError as e:
        logger.warning(f"Calendar event creation rejected: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error creating calendar event: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
