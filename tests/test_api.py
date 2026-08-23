"""
tests/test_api.py — Integration tests against a running
app.services.face_recognition.main:app server (start it with
`uvicorn app.services.face_recognition.main:app --port 8004`, `start.bat`,
or `docker compose up`).

Previously a standalone script (`python test_api.py`) with a hand-rolled
main() threading IDs through positional function args. Running it under
pytest broke, because pytest interprets a test function's parameters as
requested fixture names — `def test_caregiver_management(user_id):` looked
like a request for a fixture literally named `user_id`, which didn't exist.
Rewritten as real pytest fixtures below so `pytest tests/test_api.py` works.

All tests here are skipped automatically if no server is reachable at
BASE_URL, rather than failing with a confusing connection error.

Note: person creation has no HTTP endpoint on this server — /api/persons/*
was removed as a dead, unused-by-the-frontend duplicate of the live
face-recognition module (see TECH_DEBT.md and the git history for the
removal). The person_id fixture below inserts directly via SQLAlchemy instead.
"""
import pytest
import requests
from datetime import datetime

from app.db.session import create_session
from app.models.person import KnownPerson
from app.models.junction_tables import userknownperson

BASE_URL = "http://localhost:8004"


def _server_reachable() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=2).status_code == 200
    except requests.exceptions.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=f"No server reachable at {BASE_URL} — start it with "
           f"`uvicorn app.services.face_recognition.main:app --port 8004` first.",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def user_id():
    user_data = {
        "name": "Test User",
        "email": f"test_{datetime.now().timestamp()}@example.com",
        "age": 65,
        "medicalcondition": "Short-term memory loss",
        "emergencycontact": "+1234567890",
    }
    response = requests.post(f"{BASE_URL}/api/users/", json=user_data)
    assert response.status_code == 201, response.text
    uid = response.json()["userid"]
    yield uid
    requests.delete(f"{BASE_URL}/api/users/{uid}")


@pytest.fixture(scope="module")
def caregiver_id(user_id):
    caregiver_data = {"name": "Test Caregiver", "relationshiptouser": "daughter", "accesslevel": "admin"}
    response = requests.post(f"{BASE_URL}/api/caregivers/", json=caregiver_data)
    assert response.status_code == 201, response.text
    cid = response.json()["caregiverid"]

    assign = requests.post(
        f"{BASE_URL}/api/caregivers/assign",
        json={"user_id": user_id, "caregiver_id": cid},
    )
    assert assign.status_code == 200, assign.text

    yield cid
    requests.delete(f"{BASE_URL}/api/caregivers/{cid}?user_id={user_id}")


@pytest.fixture(scope="module")
def person_id(user_id):
    """No HTTP endpoint creates a person anymore — see module docstring."""
    db = create_session()
    try:
        person = KnownPerson(name="Test Person", relationshiptype="colleague", prioritylevel=3)
        db.add(person)
        db.flush()
        db.execute(userknownperson.insert().values(userid=user_id, personid=person.personid))
        db.commit()
        pid = person.personid
    finally:
        db.close()

    yield pid

    db = create_session()
    try:
        p = db.get(KnownPerson, pid)
        if p:
            db.delete(p)
            db.commit()
    finally:
        db.close()


@pytest.fixture(scope="module")
def interaction_id(user_id, person_id):
    response = requests.post(
        f"{BASE_URL}/api/interactions/start",
        json={"user_id": user_id, "person_id": person_id, "location": "Living Room"},
    )
    assert response.status_code == 201, response.text
    return response.json()["interaction_id"]


@pytest.fixture(scope="module")
def emotion_id(interaction_id):
    response = requests.post(
        f"{BASE_URL}/api/emotions/",
        json={"interaction_id": interaction_id, "emotiontype": "happy", "confidencelevel": 0.85},
    )
    assert response.status_code == 201, response.text
    return response.json()["emotionid"]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_check():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200


def test_read_user(user_id):
    response = requests.get(f"{BASE_URL}/api/users/{user_id}")
    assert response.status_code == 200
    assert response.json()["userid"] == user_id


def test_list_users():
    response = requests.get(f"{BASE_URL}/api/users/?skip=0&limit=10")
    assert response.status_code == 200
    assert "total" in response.json()


def test_update_user(user_id):
    response = requests.put(f"{BASE_URL}/api/users/{user_id}", json={"age": 66})
    assert response.status_code == 200
    assert response.json()["age"] == 66


def test_read_caregiver(caregiver_id, user_id):
    response = requests.get(f"{BASE_URL}/api/caregivers/{caregiver_id}?user_id={user_id}")
    assert response.status_code == 200


def test_list_caregivers(user_id):
    response = requests.get(f"{BASE_URL}/api/caregivers/?user_id={user_id}&skip=0&limit=10")
    assert response.status_code == 200
    assert "total" in response.json()


def test_user_has_caregiver(user_id, caregiver_id):
    response = requests.get(f"{BASE_URL}/api/users/{user_id}/caregivers")
    assert response.status_code == 200
    ids = [c["caregiverid"] for c in response.json()]
    assert caregiver_id in ids


def test_interaction_append_transcript(interaction_id):
    response = requests.post(
        f"{BASE_URL}/api/sessions/append",
        json={"interaction_id": interaction_id, "transcript_chunk": "Hello, how are you today?"},
    )
    assert response.status_code == 200


def test_emotion_record_read(emotion_id, user_id):
    response = requests.get(f"{BASE_URL}/api/emotions/{emotion_id}?user_id={user_id}")
    assert response.status_code == 200


def test_emotions_for_interaction(interaction_id, emotion_id, user_id):
    response = requests.get(f"{BASE_URL}/api/emotions/interaction/{interaction_id}?user_id={user_id}")
    assert response.status_code == 200
    ids = [e["emotionid"] for e in response.json()]
    assert emotion_id in ids


def test_list_emotions(user_id):
    response = requests.get(f"{BASE_URL}/api/emotions/?user_id={user_id}&skip=0&limit=10")
    assert response.status_code == 200
    assert "total" in response.json()


def test_memory_retrieval(person_id, user_id):
    response = requests.get(f"{BASE_URL}/api/memory/{person_id}?user_id={user_id}")
    assert response.status_code == 200
    assert response.json()["person_id"] == person_id
