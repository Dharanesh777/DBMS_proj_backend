import os
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime
from zoneinfo import ZoneInfo
from app.ai_models.reminders.tasks import remind_user
from pydantic import BaseModel
import redis

reminder_router = APIRouter(prefix="/api", tags=["Reminders"])

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=0,
)

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

# Safety cap on how many queued notifications one request will drain, so a
# pathologically large backlog can't block the request thread indefinitely.
MAX_NOTIFICATIONS_PER_REQUEST = 500


class ReminderRequest(BaseModel):
    user_id: str
    message: str
    remind_at: str   # ISO 8601 — assumed to be in IST


@reminder_router.post("/schedule-reminder")
def schedule_reminder(body: ReminderRequest):
    """
    Schedule a reminder via Celery.

    - **user_id**: target user identifier
    - **message**: reminder text
    - **remind_at**: ISO datetime string in IST (e.g. `2026-04-19T10:30:00`)
    """
    remind_at = datetime.fromisoformat(body.remind_at)

    # If the caller's string had no offset, treat it as IST per the documented
    # contract. If it DID carry an offset, respect that instead of re-interpreting
    # it as IST on top (that used to double-shift the time).
    if remind_at.tzinfo is None:
        remind_at = remind_at.replace(tzinfo=IST)
    remind_at_utc = remind_at.astimezone(UTC)

    print(f"DEBUG: Local Target: {remind_at}")
    print(f"DEBUG: Internal UTC Target: {remind_at_utc}")

    remind_user.apply_async(args=[body.user_id, body.message], eta=remind_at_utc)
    return JSONResponse({"status": "Reminder scheduled", "at": str(remind_at)})


@reminder_router.get("/get-notifications/{user_id}")
def get_notifications(user_id: str):
    """
    Retrieve and drain pending notifications for a user from Redis.
    """
    msgs = []
    for _ in range(MAX_NOTIFICATIONS_PER_REQUEST):
        item = r.rpop(f"notifications:{user_id}")
        if not item:
            break
        msgs.append(item.decode())
    return JSONResponse(msgs)