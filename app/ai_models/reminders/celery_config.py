import os
from celery import Celery

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

celery_app = Celery(
    'reminders',
    broker=f'redis://{REDIS_HOST}:{REDIS_PORT}/0',
    backend=f'redis://{REDIS_HOST}:{REDIS_PORT}/1',
    # Both task modules must be listed so a `celery worker` process discovers
    # tasks decorated with @celery_app.task in either — session_service.py now
    # schedules session-expiry timers here too (replaces the old APScheduler).
    include=['app.ai_models.reminders.tasks', 'app.services.session_service'],
)

celery_app.conf.update(
    timezone='Asia/Kolkata',
    enable_utc=True,
    task_track_started=True,
)