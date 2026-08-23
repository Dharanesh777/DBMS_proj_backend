"""
services/session_service.py — Session management with Celery-scheduled timers

Previously used APScheduler (in-process, memory-only job store). Migrated to
Celery so session timers share the same broker/worker as the reminders system
instead of running two separate schedulers — see app/ai_models/reminders/.
"""
import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import redis
from sqlalchemy.orm import Session

from ..models.conversation import Conversation
from ..models.user import User
from ..models.person import KnownPerson
from ..ai_models.reminders.celery_config import celery_app
from ..db.session import create_session
from .llm_service import LLMService
from .redis_client import get_redis
from ..config import get_settings

logger = logging.getLogger(__name__)

# ── Redis write-through mirror ───────────────────────────────────────────────
# _active_sessions/_session_task_ids below stay the authoritative, fast,
# in-process store for every read (this remains a single-process deployment —
# see the migration plan's locking-model rationale). Every mutation also
# best-effort mirrors to Redis (log-and-continue on failure, never raise) so
# a still-pending Celery timer (which already lives durably in Redis) can
# find its session's accumulated summaries again after a process restart —
# closing the exact gap the "session state not found" no-op below used to
# hit silently.
SESSION_HASH_PREFIX = "agos:sessionmgr:session:"
ACTIVE_IDS_KEY = "agos:sessionmgr:active_ids"


def _session_hash_key(interaction_id: int) -> str:
    return f"{SESSION_HASH_PREFIX}{interaction_id}"


def _restore_session_state(interaction_id: int, h: dict) -> Optional["SessionState"]:
    try:
        state = SessionState(
            interaction_id=interaction_id,
            session_number=int(h["session_number"]),
            user_id=int(h["user_id"]),
            person_id=int(h["person_id"]) if h.get("person_id") not in (None, "") else None,
        )
        state.started_at = datetime.fromisoformat(h["started_at"])
        state.session_summaries = json.loads(h.get("session_summaries", "[]"))
        return state
    except (KeyError, ValueError) as e:
        logger.warning(f"Could not deserialize mirrored session {interaction_id}: {e}")
        return None


def _mirror_reference_time(h: dict, state: "SessionState") -> datetime:
    """Staleness clock for restore_and_sweep. Prefers `mirrored_at` (the
    wall-clock time of the last SUCCESSFUL write) over `started_at` (when the
    current 30-min sub-session began) — the latter resets on every session
    transition regardless of whether mirroring is actually succeeding, so a
    session stuck failing to mirror for hours would still look "fresh" by
    started_at alone every time a new sub-session begins. mirrored_at doesn't
    advance unless a write actually landed, which is what "how stale is this"
    should mean. Falls back to started_at only for a hash written before this
    field existed."""
    raw = h.get("mirrored_at")
    if raw:
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return state.started_at


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
    live in Redis and survive a web-process restart. _active_sessions is now
    ALSO write-through mirrored to Redis (see _mirror_session/restore_and_sweep
    below) — a timer firing after a restart finds its session's accumulated
    summaries again via that mirror, instead of the "session state not found"
    no-op this used to hit unconditionally. This requires a Celery worker to be
    running continuously (the same operational requirement the reminders
    system already has).
    """

    # Class-level state (shared across all instances)
    _active_sessions: Dict[int, SessionState] = {}  # interaction_id -> SessionState
    _session_task_ids: Dict[int, str] = {}  # interaction_id -> Celery task id
    # Guards all reads/writes/iterations of the dicts above — mutated from request
    # handlers, an async LLM callback, and Celery task bodies, all potentially concurrent.
    # threading.Lock (not asyncio.Lock) because this class mixes sync and async methods
    # and the dict operations it guards are quick, non-blocking, in-memory work.
    _lock = threading.Lock()

    # Counts how many times _mirror_session's retry budget was fully
    # exhausted (i.e. the mirror write genuinely failed, not just retried
    # once and recovered). Pure instrumentation — no behavior depends on
    # this. Exists to answer "how often does this actually happen in real
    # operation" before deciding whether the local-disk dirty-marker for the
    # residual restore_and_sweep staleness gap (see its docstring) is worth
    # building, rather than guessing. Guarded by _lock like everything else
    # here; read via mirror_exhausted_count().
    _mirror_exhausted_count = 0

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.llm_service = LLMService()

    # Short, bounded retry — absorbs a brief blip (e.g. a Redis process
    # restart taking a few hundred ms to a couple seconds) so it doesn't
    # silently leave the mirror stale. Deliberately NOT a long/backoff retry:
    # this runs synchronously inline with request handling and Celery task
    # bodies, and a sustained outage should fail fast and move on rather than
    # block. See the docstring on restore_and_sweep/_mirror_reference_time
    # for why this does not, by itself, fully close the staleness gap for
    # outages that outlast this retry budget.
    MIRROR_RETRY_ATTEMPTS = 2
    MIRROR_RETRY_DELAY_SECONDS = 0.3

    @classmethod
    def _mirror_session(cls, interaction_id: int) -> None:
        """Best-effort write-through mirror of one session's current
        in-process state to Redis. Caller should hold _lock or have just
        released it after the mutation this follows. Never raises."""
        with cls._lock:
            state = cls._active_sessions.get(interaction_id)
            task_id = cls._session_task_ids.get(interaction_id)
        if state is None:
            return
        mapping = {
            "session_number": state.session_number,
            "user_id": state.user_id,
            "person_id": state.person_id if state.person_id is not None else "",
            "started_at": state.started_at.isoformat(),
            "session_summaries": json.dumps(state.session_summaries),
            "celery_task_id": task_id or "",
            "mirrored_at": datetime.utcnow().isoformat(),
        }
        last_err = None
        for attempt in range(cls.MIRROR_RETRY_ATTEMPTS):
            try:
                r = get_redis()
                r.hset(_session_hash_key(interaction_id), mapping=mapping)
                r.sadd(ACTIVE_IDS_KEY, str(interaction_id))
                return
            except redis.exceptions.RedisError as e:
                last_err = e
                if attempt < cls.MIRROR_RETRY_ATTEMPTS - 1:
                    time.sleep(cls.MIRROR_RETRY_DELAY_SECONDS)

        # Every attempt failed — this interaction's mirror is now stale
        # (restore_and_sweep will restore whatever it last successfully
        # wrote, not this current state, if the process crashes before the
        # next successful mirror). Distinct log tag + counter so this is
        # greppable/countable as its own event, not lost among routine
        # Redis warnings — see the _mirror_exhausted_count docstring.
        with cls._lock:
            cls._mirror_exhausted_count += 1
            count = cls._mirror_exhausted_count
        logger.error(
            f"[MIRROR_WRITE_EXHAUSTED] session {interaction_id}: all "
            f"{cls.MIRROR_RETRY_ATTEMPTS} mirror-write attempts failed "
            f"(total exhaustion count this process: {count}): {last_err}"
        )

    @classmethod
    def mirror_exhausted_count(cls) -> int:
        """How many times _mirror_session's retry budget has been fully
        exhausted since this process started. See _mirror_exhausted_count."""
        with cls._lock:
            return cls._mirror_exhausted_count

    @classmethod
    def _unmirror_session(cls, interaction_id: int) -> None:
        """Best-effort removal of one session's Redis mirror. Never raises."""
        try:
            r = get_redis()
            r.delete(_session_hash_key(interaction_id))
            r.srem(ACTIVE_IDS_KEY, str(interaction_id))
        except redis.exceptions.RedisError as e:
            logger.warning(f"Could not remove session {interaction_id} mirror from Redis: {e}")

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

        self._mirror_session(interaction_id)

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

        self._mirror_session(interaction_id)

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

        self._mirror_session(interaction_id)

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
        self._unmirror_session(interaction_id)

    @classmethod
    def clear_all_sessions(cls) -> None:
        """Clear all active sessions, both tiers. Kept for tests/manual ops —
        NOT called on startup anymore (see restore_and_sweep), since that
        would defeat the entire point of mirroring session state to Redis."""
        with cls._lock:
            ids = list(cls._active_sessions.keys())
            cls._active_sessions.clear()
            cls._session_task_ids.clear()
        for interaction_id in ids:
            cls._unmirror_session(interaction_id)
        logger.info("Cleared all active session state")

    @classmethod
    def restore_and_sweep(cls, max_age_minutes: int) -> None:
        """Called on startup instead of clear_all_sessions(). Rehydrates
        _active_sessions/_session_task_ids from the Redis mirror — this is
        what lets a still-pending Celery timer find its session again after
        a process restart, closing the "session state not found" gap this
        class's docstring used to document as permanent. Entries whose mirror
        hasn't been successfully written to in more than max_age_minutes are
        treated as genuinely orphaned (e.g. a crash between
        cancel_session_timer and clear_session_state, or a timer that itself
        never fired) and swept instead of restored. Soft-fails to an empty
        state if Redis is unreachable at startup, matching this method's
        previous (clear_all_sessions) behavior.

        KNOWN RESIDUAL GAP (do not remove this note without re-reading it):
        staleness here is judged by _mirror_reference_time (last successful
        write), not by whether every in-process change actually reached
        Redis. If Redis drops mid-session and the process crashes before the
        next successful mirror write, the LAST successful write — however
        far behind it's fallen — is what gets restored here, silently, as if
        it were current. This is strictly better than the pre-migration
        behavior (which lost the session entirely on every restart), but it
        is not "never restores stale data" — it's "never restores data
        older than max_age_minutes since it was last confirmed accurate."
        Closing this fully requires a durability mechanism that doesn't
        depend on Redis being reachable at the moment of failure (e.g. a
        small local-disk dirty marker per session) — not implemented here;
        see the migration follow-up notes."""
        try:
            r = get_redis()
            ids = r.smembers(ACTIVE_IDS_KEY)
        except redis.exceptions.RedisError as e:
            logger.warning(
                f"Could not read session mirror from Redis at startup — "
                f"starting with no active sessions: {e}"
            )
            return

        now = datetime.utcnow()
        restored, swept = 0, 0
        for id_str in ids:
            interaction_id = int(id_str)
            try:
                h = r.hgetall(_session_hash_key(interaction_id))
            except redis.exceptions.RedisError as e:
                logger.warning(f"Could not read mirrored session {interaction_id}: {e}")
                continue

            if not h:
                r.srem(ACTIVE_IDS_KEY, id_str)
                continue

            state = _restore_session_state(interaction_id, h)
            if state is None:
                r.delete(_session_hash_key(interaction_id))
                r.srem(ACTIVE_IDS_KEY, id_str)
                continue

            age_minutes = (now - _mirror_reference_time(h, state)).total_seconds() / 60
            if age_minutes > max_age_minutes:
                r.delete(_session_hash_key(interaction_id))
                r.srem(ACTIVE_IDS_KEY, id_str)
                swept += 1
                continue

            with cls._lock:
                cls._active_sessions[interaction_id] = state
                task_id = h.get("celery_task_id") or None
                if task_id:
                    cls._session_task_ids[interaction_id] = task_id
            restored += 1

        logger.info(
            f"Session mirror restore: {restored} session(s) restored, "
            f"{swept} stale session(s) swept."
        )

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


@celery_app.task(
    name="session_service.expire_session_timer",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
)
def expire_session_timer_task(interaction_id: int) -> None:
    """
    Celery task entrypoint for session timer expiry — fires `SESSION_DURATION_MINUTES`
    after start_session() schedules it, via apply_async(eta=...).

    Celery tasks are sync; the actual logic (_on_session_timer_expire) is async
    because it awaits an LLM call, so it's run here via asyncio.run().

    Auto-retries (matching tasks.py::remind_user's existing pattern) so a
    transient Redis blip exactly when this fires — reading the mirrored
    session state, or an LLM/DB hiccup — gets retried instead of permanently
    dropping that session's summary.
    """
    asyncio.run(_run_session_timer_expire(interaction_id))
