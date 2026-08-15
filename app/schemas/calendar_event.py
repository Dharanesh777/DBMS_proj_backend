"""
schemas/calendar_event.py — Pydantic schemas for calendar event creation
"""
from pydantic import BaseModel, Field, model_validator
from datetime import datetime


class CalendarEventCreateRequest(BaseModel):
    """POST /api/calendar/events — request payload"""
    user_id: int = Field(..., gt=0)
    related_person_id: int | None = Field(None, gt=0)
    event_title: str = Field(..., min_length=1, max_length=100)
    event_datetime: datetime
    reminder_time: datetime | None = None

    @model_validator(mode="after")
    def validate_reminder_before_event(self):
        if self.reminder_time is not None and self.reminder_time >= self.event_datetime:
            raise ValueError("reminder_time must be before event_datetime")
        return self


class CalendarEventCreateResponse(BaseModel):
    """POST /api/calendar/events — response payload"""
    event_id: int
    message: str = "Calendar event created successfully"
