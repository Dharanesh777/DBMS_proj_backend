import requests
import time

GET_URL = "http://127.0.0.1:5000/api/get-notifications/test_immediate"

# This endpoint doesn't support immediate directly, but we can try to send a time in the past
URL = "http://127.0.0.1:5000/api/schedule-reminder"
past_time = "2020-01-01T00:00:00"

print(f"Scheduling immediate reminder...")
resp = requests.post(URL, json={
    "user_id": "test_immediate",
    "message": "Immediate message!",
    "remind_at": past_time
})

print("Response:", resp.json())

print("Waiting 5 seconds...")
time.sleep(5)

print("Checking notifications...")
resp = requests.get(GET_URL)
print("Notifications:", resp.json())
