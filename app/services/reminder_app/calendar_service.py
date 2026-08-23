from googleapiclient.discovery import build
from app.services.reminder_app.google_auth import get_credentials
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
CALENDAR_ID = os.getenv("CALENDAR_ID", "primary")

def create_reminder(title: str, date: str, time: str):
    creds = get_credentials()
    service = build("calendar", "v3", credentials=creds)

    # Combine date + time into ISO format
    start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    end_dt = start_dt + timedelta(minutes=30)  # 30 min event by default

    event = {
        "summary": title,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Kolkata"},
        "end":   {"dateTime": end_dt.isoformat(),   "timeZone": "Asia/Kolkata"},
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": 10},
            ]
        }
    }

    created = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
    return created

def get_upcoming_reminders():
    creds = get_credentials()
    service = build("calendar", "v3", credentials=creds)

    now = datetime.utcnow().isoformat() + "Z"
    soon = (datetime.utcnow() + timedelta(hours=24)).isoformat() + "Z"

    result = service.events().list(
        calendarId=CALENDAR_ID,
        timeMin=now,
        timeMax=soon,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    return result.get("items", [])