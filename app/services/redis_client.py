"""
services/redis_client.py — Shared pooled Redis client for application state
(session engine, sub-session chunking, face-cache, temp registration holds).

Deliberately separate from Celery's own broker (db=0) / results backend
(db=1) connections in app/ai_models/reminders/{celery_config,tasks}.py — this
uses REDIS_STATE_DB (db=2 by default) so application state never shares
keyspace with Celery internals.

decode_responses=True because every value stored under this client is JSON
text — callers never need to .decode() raw bytes. Celery's own Redis clients
stay untouched (raw bytes, as they already are).
"""
import redis

from app.config import get_settings

_pool: redis.ConnectionPool | None = None


def get_redis() -> redis.Redis:
    """Return a pooled Redis client for application state.

    Short connect/socket timeouts — a caller on this codebase's target
    hardware (a Pi 3) must not hang waiting on a dead Redis; callers are
    expected to catch redis.exceptions.RedisError themselves and decide
    per-call whether that's a hard failure or a soft, logged degradation
    (see individual call sites — most are write-through mirrors that log
    and continue, not raise).
    """
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = redis.ConnectionPool(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_STATE_DB,
            decode_responses=True,
            retry_on_timeout=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )
    return redis.Redis(connection_pool=_pool)
