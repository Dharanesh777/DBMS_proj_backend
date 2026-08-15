"""
db/session.py — Session factory and get_db() FastAPI dependency.
"""
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from app.db.base import get_engine

# One SessionLocal factory shared across the app — thread-safe
_SessionLocal: sessionmaker | None = None


def _get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields one DB session per request, always closes it."""
    factory = _get_session_factory()
    db: Session = factory()
    try:
        yield db
    finally:
        db.close()


def create_session() -> Session:
    """Create a standalone Session for code that runs outside a FastAPI request
    (e.g. Celery task bodies). Caller is responsible for closing it —
    do NOT reuse a request-scoped Session (from get_db()) after the request ends,
    it will already be closed by then."""
    factory = _get_session_factory()
    return factory()
