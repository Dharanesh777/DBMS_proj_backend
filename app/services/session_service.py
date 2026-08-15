"""
services/session_service.py — Session management with Celery-scheduled timers

Previously used APScheduler (in-process, memory-only job store). Migrated to
Celery so session timers share the same broker/worker as the reminders system
instead of running two separate schedulers — see app/ai_models/reminders/.
"""
import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List
from sqlalchemy.orm import Session

from ..models.conversation import Conversation
from ..models.user import User
from ..models.person import KnownPerson
from ..ai_models.reminders.celery_config import celery_app
from ..db.session import create_session
from .llm_service import LLMService
from ..config import get_settings

logger = logging.getLogger(__name__)


class SessionState:
    """In-memory state for an active session"""
    def __init__(self, interaction_id: int, session_number: int, user_id: int, person_id: int):
        self.interaction_id = interaction_id
        self.session_number = session_number
        self.user_id = user_id
        self.person_id = person_id
        self.session_summaries: List[str] = []  # Accumulates summaries for this interaction
        self.started_at = datetime.utcnow()


class SessionManager:
    """
    Manages 30-minute session boundaries using Celery ETA tasks.

    Architecture:
    - Active sessions tracked in-memory (lost on restart)
    - Transcripts accumulated in DB (conversation.conversation column)
    - Session summaries stored in-memory buffer
    - On interaction end, all session summaries merged into conversation.summarytext

    Note: unlike the previous in-memory APScheduler job store, Celery ETA tasks
    live in Redis and survive a web-process restart. clear_all_sessions() (called
    on startup) still clears the in-memory _active_sessions dict, so any timer
    that fires after a restart hits the "session state not found" early-return
    in _on_session_timer_expire and no-ops safely — it just isn't proactively
    cancelled the way APScheduler's memory store implicitly discarded jobs on
    restart. This requires a Celery worker to be running continuously (the same
    operational requirement the reminders system already has).
    """

    # Class-level state (shared across all instances)
    _active_sessions: Dict[int, SessionState] = {}  # interaction_id -> SessionState
    _session_task_ids: Dict[int, str] = {}  # interaction_id -> Celery task id
    # Guards all reads/writes/iterations of the dicts above — mutated from request
    # handlers, an async LLM callback, and Celery task bodies, all potentially concurrent.
    # threading.Lock (not asyncio.Lock) because this class mixes sync and async methods
    # and the dict operations it guards are quick, non-blocking, in-memory work.
    _lock = threading.Lock()

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.llm_service = LLMService()

    def start_session(
        self,
        interaction_id: int,
        user_id: int,
        person_id: int,
        session_number: int = 1,
    ) -> None:
        """
        Start a new session with a 30-minute timer.
        
        Args:
            interaction_id: DB interaction ID
            user_id: User ID
            person_id: Person ID
            session_number: Session number (1, 2, 3, ...)
        """
        # Create session state
        session_state = SessionState(
            interaction_id=interaction_id,
            session_number=session_number,
            user_id=user_id,
            person_id=person_id,
        )
        with self._lock:
            self._active_sessions[interaction_id] = session_state

        # Schedule timer to expire after SESSION_DURATION_MINUTES
        run_at = datetime.utcnow() + timedelta(minutes=self.settings.SESSION_DURATION_MINUTES)

        # Revoke any previous timer task for this interaction first (mirrors
        # APScheduler's replace_existing=True — shouldn't normally happen, but
        # start_session could in principle be called again before the last one fires).
        with self._lock:
            prev_task_id = self._session_task_ids.get(interaction_id)
        if prev_task_id:
            try:
                celery_app.control.revoke(prev_task_id)
            except Exception as e:
                logger.warning(f"Could not revoke previous timer task {prev_task_id}: {e}")

        # NOTE: the task body opens its own fresh DB session — this fires minutes
        # from now, long after the request that called start_session() has ended
        # and its DB session (self.db, from Depends(get_db)) has been closed.
        #
        # Unlike the old in-process APScheduler, scheduling this requires reaching
        # Redis. The old scheduler could never fail this way, so if the broker is
        # unreachable, degrade to "no 30-minute auto-summarization timer for this
        # session" rather than failing the whole interaction-start request — the
        # conversation can still be recorded and manually ended.
        try:
            result = expire_session_timer_task.apply_async(args=[interaction_id], eta=run_at)
            with self._lock:
                self._session_task_ids[interaction_id] = result.id
            logger.info(
                f"Started session {session_number} for interaction {interaction_id}, "
                f"timer expires at {run_at} (celery task {result.id})"
            )
        except Exception as e:
            logger.error(
                f"Could not schedule session timer for interaction {interaction_id} "
                f"(is Redis/Celery reachable?): {e}. Session started without an "
                "auto-expiry timer — it will only end when end_interaction() is called."
            )

    async def append_transcript(self, interaction_id: int, transcript_chunk: str) -> None:
        """
        Append transcript chunk to the conversation.conversation column.

        Args:
            interaction_id: DB interaction ID
            transcript_chunk: Text to append

        Raises:
            ValueError: If no active session exists for this interaction
        """
        # Check if session is active
        with self._lock:
            has_session = interaction_id in self._active_sessions
        if not has_session:
            raise ValueError(f"No active session for interaction {interaction_id}")

        # Append to DB
        conversation = self.db.get(Conversation, interaction_id)
        if not conversation:
            raise ValueError(f"Conversation {interaction_id} not found in database")
        
        if conversation.conversation is None:
            conversation.conversation = transcript_chunk
        else:
            conversation.conversation += "\n" + transcript_chunk
        
        self.db.commit()
        logger.debug(f"Appended transcript to interaction {interaction_id}")

    async def _on_session_timer_expire(self, interaction_id: int) -> None:
        """
        Called by APScheduler when a session timer expires.
        
        This method:
        1. Retrieves the accumulated transcript from DB
        2. Generates a session summary via LLM
        3. Stores summary in in-memory buffer
        4. Checks if person is still present (for now, assume they are)
        5. Starts a new session if person is still present
        """
        logger.info(f"Session timer expired for interaction {interaction_id}")

        with self._lock:
            session_state = self._active_sessions.get(interaction_id)
        if not session_state:
            logger.warning(f"Session state not found for interaction {interaction_id}")
            return
        
        # Get conversation from DB
        conversation = self.db.get(Conversation, interaction_id)
        if not conversation or not conversation.conversation:
            logger.warning(f"No transcript found for interaction {interaction_id}, skipping summarization")
            # Start next session anyway
            await self._start_next_session(interaction_id, session_state)
            return
        
        # Get user and person context for LLM
        user = self.db.get(User, session_state.user_id)
        person = self.db.get(KnownPerson, session_state.person_id)
        
        user_context = user.medicalcondition if user else None
        person_relationship = person.relationshiptype if person else None
        
        # Generate session summary
        try:
            result = await self.llm_service.summarize_session(
                transcript=conversation.conversation,
                user_context=user_context,
                person_relationship=person_relationship,
            )
            summary = result.get("summary", "[Session summary empty]")
            session_state.session_summaries.append(summary)
            logger.info(f"Generated session {session_state.session_number} summary for interaction {interaction_id}")
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")
            session_state.session_summaries.append("[Session summary generation failed]")
        
        # For V1, assume person is still present and start next session
        # In production, you'd check with Member A's detection system
        await self._start_next_session(interaction_id, session_state)

    async def _start_next_session(self, interaction_id: int, session_state: SessionState) -> None:
        """Start the next session for an ongoing interaction"""
        next_session_number = session_state.session_number + 1
        self.start_session(
            interaction_id=interaction_id,
            user_id=session_state.user_id,
            person_id=session_state.person_id,
            session_number=next_session_number,
        )

    def cancel_session_timer(self, interaction_id: int) -> None:
        """
        Cancel the active session timer for an interaction.

        Called when person leaves (interaction ends).
        """
        with self._lock:
            task_id = self._session_task_ids.pop(interaction_id, None)
        if not task_id:
            logger.warning(f"No active session to cancel for interaction {interaction_id}")
            return

        try:
            celery_app.control.revoke(task_id)
            logger.info(f"Cancelled session timer task {task_id}")
        except Exception as e:
            logger.debug(f"Could not revoke timer task {task_id} (may have already fired): {e}")

    def get_session_summaries(self, interaction_id: int) -> List[str]:
        """Get all session summaries for an interaction"""
        with self._lock:
            session_state = self._active_sessions.get(interaction_id)
        if not session_state:
            return []
        return session_state.session_summaries.copy()

    def clear_session_state(self, interaction_id: int) -> None:
        """Clear in-memory session state after interaction ends"""
        with self._lock:
            existed = self._active_sessions.pop(interaction_id, None) is not None
            self._session_task_ids.pop(interaction_id, None)
        if existed:
            logger.info(f"Cleared session state for interaction {interaction_id}")

    @classmethod
    def clear_all_sessions(cls) -> None:
        """Clear all active sessions (called on startup recovery)"""
        with cls._lock:
            cls._active_sessions.clear()
            cls._session_task_ids.clear()
        logger.info("Cleared all active session state")

    @classmethod
    def snapshot_active_sessions(cls) -> Dict[int, SessionState]:
        """A locked, shallow-copied snapshot of active sessions — safe to iterate
        without racing concurrent mutations. Callers should use this instead of
        touching _active_sessions directly."""
        with cls._lock:
            return dict(cls._active_sessions)


async def _run_session_timer_expire(interaction_id: int) -> None:
    """
    Session timer expiry logic, run in its own DB session.

    Runs minutes after the request that scheduled it — that request's DB session
    is long closed by then, so this opens and closes its own session rather than
    reusing any SessionManager instance's self.db.
    """
    db = create_session()
    try:
        manager = SessionManager(db)
        await manager._on_session_timer_expire(interaction_id)
    finally:
        db.close()


@celery_app.task(name="session_service.expire_session_timer")
def expire_session_timer_task(interaction_id: int) -> None:
    """
    Celery task entrypoint for session timer expiry — fires `SESSION_DURATION_MINUTES`
    after start_session() schedules it, via apply_async(eta=...).

    Celery tasks are sync; the actual logic (_on_session_timer_expire) is async
    because it awaits an LLM call, so it's run here via asyncio.run().
    """
    asyncio.run(_run_session_timer_expire(interaction_id))
