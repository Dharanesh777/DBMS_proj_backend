import requests
from datetime import datetime, timedelta
import time

URL = "http://127.0.0.1:5000/api/schedule-reminder"
GET_URL = "http://127.0.0.1:5000/api/get-notifications/test_user"

# Schedule for 5 seconds from now
remind_time = (datetime.now() + timedelta(seconds=5)).isoformat()
print(f"Scheduling reminder for {remind_time}...")

resp = requests.post(URL, json={
    "user_id": "test_user",
    "message": "Hello from test script!",
    "remind_at": remind_time
})

print("Response:", resp.json())

print("Waiting 10 seconds...")
time.sleep(60)

print("Checking notifications...")
resp = requests.get(GET_URL)
print("Notifications:", resp.json())
