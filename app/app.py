import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.routes.main_routes import main_router
from app.routes.audio_routes import audio_router
from app.routes.face_routes import face_router
from app.routes.interaction_routes import interaction_router
from app.ai_models.reminders.reminder_routes import reminder_router
from app.api.routes import (
    users,
    caregivers,
    persons,
    interactions,
    sessions,
    memory,
    notes,
    calendar_events,
    audio,
    emotions,
)
from app.core.scheduler import start_scheduler, shutdown_scheduler, get_scheduler
from app.services.session_service import SessionManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Starting DBMS Project API...")

    start_scheduler()
    logger.info("APScheduler started")

    # Startup recovery: clear orphaned session state from the memory-assistant backend
    SessionManager.clear_all_sessions()
    scheduler = get_scheduler()
    scheduler.remove_all_jobs()
    logger.info("Cleared orphaned session state and timers")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down DBMS Project API...")
    shutdown_scheduler()
    logger.info("APScheduler shut down")


app = FastAPI(
    title="DBMS Project API",
    description="AI-powered memory assistant — audio transcription & face recognition.",
    version="1.0.0",
    lifespan=lifespan,
)

os.makedirs("app/static", exist_ok=True)
app.mount("/dashboard", StaticFiles(directory="app/static", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    from app.db.base import get_engine
    from sqlalchemy import text

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}


app.include_router(main_router)
app.include_router(audio_router)
app.include_router(face_router)
app.include_router(interaction_router)
app.include_router(reminder_router)

app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(caregivers.router, prefix="/api/caregivers", tags=["Caregivers"])
app.include_router(persons.router, prefix="/api/persons", tags=["Persons"])
app.include_router(interactions.router, prefix="/api/interactions", tags=["Interactions"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])
app.include_router(memory.router, prefix="/api/memory", tags=["Memory"])
app.include_router(notes.router, prefix="/api/notes", tags=["Notes"])
app.include_router(calendar_events.router, prefix="/api/calendar", tags=["Calendar"])
app.include_router(audio.router, prefix="/api/sessions/audio", tags=["Session Audio"])
app.include_router(emotions.router, prefix="/api/emotions", tags=["Emotions"])