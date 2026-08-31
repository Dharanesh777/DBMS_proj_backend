# Re-export from app.services.voice_app.db
from app.services.voice_app.db import (
    get_db_connection,
    update_conversation_results,
)

__all__ = ["get_db_connection", "update_conversation_results"]
