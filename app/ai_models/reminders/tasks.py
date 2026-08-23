import os
from app.ai_models.reminders.celery_config import celery_app
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

_redis_pool = redis.ConnectionPool(
    host=REDIS_HOST, port=REDIS_PORT, db=0, retry_on_timeout=True,
)
r = redis.Redis(connection_pool=_redis_pool)

@celery_app.task(autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5})
def remind_user(user_id, message):
    try:
        import sys
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"DEBUG [{now}]: Firing reminder for {user_id}: {message}")
        sys.stdout.flush()
        r.lpush(f"notifications:{user_id}", message)
        print(f"DEBUG [{now}]: Done! Pushed to Redis for key notifications:{user_id}")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR in remind_user: {str(e)}")
        raise e