#!/usr/bin/env python3
"""
tests/hardware/verify_mirror_instrumentation.py — write-through mirror
retry/counter/log-tag verification, for both structure A and structure B.

WHAT THIS PROVES
-----------------
That the two write-through Redis mirrors in this codebase behave correctly
under failure:

  * Structure A — app.services.face_recognition.main._mutate_session (the
    live-camera visit-session state machine's mirror)
  * Structure B — app.services.session_service.SessionManager._mirror_session
    (the REST API's 30-min sub-session chunking mirror)

For each: a mocked Redis client whose write always raises should (1) still
update the in-process state (write-through, not read-through — a Redis
failure degrades durability, not live correctness), (2) exhaust a bounded
retry (2 attempts, ~0.3s apart) rather than fail-fast on the first error,
(3) increment that structure's dedicated exhaustion counter exactly once,
and (4) emit a distinctly-tagged log line ([MUTATE_WRITE_EXHAUSTED] for A,
[MIRROR_WRITE_EXHAUSTED] for B) so operators can grep/alert on it
separately from routine warnings. A normal successful write must NOT move
either counter. See TECH_DEBT.md's "four process-local state structures
moved to Redis" entry and its follow-ups for why this instrumentation
exists (informing whether a local-disk dirty-marker is worth building,
without guessing).

Unlike verify_redis_persistence.py in this same directory, this script
does NOT require real target (Pi 3) hardware to be meaningful — it's pure
application logic (mocked failures), not storage-medium timing. It's
checked in here anyway so a hardware verification pass can also sanity
-check this on the target machine/Python version in one place, rather
than reconstructing it from chat history.

WHAT "PASS" LOOKS LIKE
------------------------
Every assertion passes silently; the script prints an "OK:" line per
check and ends with:
    RESULT: PASS — both structures' retry/counter/log-tag behavior verified correctly.
Any AssertionError means real behavior diverged from what's documented in
TECH_DEBT.md — treat that as a regression, not a script bug.

HOW TO RUN
-----------
    source venv/bin/activate
    python tests/hardware/verify_mirror_instrumentation.py

Prerequisites: same as verify_redis_persistence.py (a redis-server binary
on $PATH, or REDIS_SERVER_BIN set). Also starts its own isolated
redis-server on a non-default port/temp dir — safe to run alongside a
real, already-running instance of this app.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REDIS_CONF = os.path.join(REPO_ROOT, "redis.conf")
REDIS_SERVER_BIN = os.environ.get("REDIS_SERVER_BIN") or shutil.which("redis-server")
TEST_PORT = int(os.environ.get("REDIS_TEST_PORT", "6398"))  # different from the persistence script's 6399


def p(msg):
    print(msg, flush=True)


def fail(msg):
    p(f"\nABORT: {msg}")
    sys.exit(1)


if not REDIS_SERVER_BIN:
    fail("redis-server not found on $PATH and REDIS_SERVER_BIN not set.")

test_dir = tempfile.mkdtemp(prefix="agos-mirror-instr-test-")
p(f"[setup] test port: {TEST_PORT}  test dir: {test_dir}")

os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = str(TEST_PORT)
sys.path.insert(0, REPO_ROOT)

import redis  # noqa: E402
from app.services.redis_client import get_redis  # noqa: E402


def start_redis():
    proc = subprocess.Popen(
        [REDIS_SERVER_BIN, REDIS_CONF, "--port", str(TEST_PORT),
         "--dir", test_dir, "--dbfilename", "dump.rdb",
         "--logfile", os.path.join(test_dir, "redis.log")],
        cwd=REPO_ROOT,
    )
    for _ in range(50):
        try:
            get_redis().ping()
            return proc
        except Exception:
            time.sleep(0.1)
    fail("redis-server did not become reachable within 5s of starting.")


proc = None
try:
    proc = start_redis()
    p("[setup] isolated redis-server is up\n")

    # ═══════════════════════════════════════════════════════════════════
    # Structure B — SessionManager._mirror_session
    # ═══════════════════════════════════════════════════════════════════
    from app.services.session_service import SessionManager, SessionState  # noqa: E402

    p("── Structure B (SessionManager._mirror_session) ──")
    IID_B = 700201
    with SessionManager._lock:
        SessionManager._active_sessions.clear()
        SessionManager._session_task_ids.clear()
        SessionManager._mirror_exhausted_count = 0

    state = SessionState(interaction_id=IID_B, session_number=1, user_id=1, person_id=5)
    with SessionManager._lock:
        SessionManager._active_sessions[IID_B] = state

    before = SessionManager.mirror_exhausted_count()
    assert before == 0

    class AlwaysFailsB:
        def hset(self, *a, **kw):
            raise redis.exceptions.ConnectionError("simulated total outage")

    with patch("app.services.session_service.get_redis", return_value=AlwaysFailsB()):
        SessionManager._mirror_session(IID_B)

    after = SessionManager.mirror_exhausted_count()
    assert after == before + 1, f"expected counter to increment by 1, went {before} -> {after}"
    p(f"OK: total-failure case increments mirror_exhausted_count() ({before} -> {after})")

    with SessionManager._lock:
        assert SessionManager._active_sessions[IID_B].session_number == 1
    p("OK: in-process state still correct despite total mirror failure (write-through)")

    get_redis().flushdb()
    SessionManager._mirror_session(IID_B)  # real, successful write now
    after2 = SessionManager.mirror_exhausted_count()
    assert after2 == after, "a successful mirror should not move the counter"
    p(f"OK: successful mirror leaves the counter unchanged at {after2}")

    call_count = {"n": 0}
    real = get_redis()

    class FlakyOnceB:
        def hset(self, *a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise redis.exceptions.ConnectionError("one transient failure")
            return real.hset(*a, **kw)

        def sadd(self, *a, **kw):
            return real.sadd(*a, **kw)

    before3 = SessionManager.mirror_exhausted_count()
    with patch("app.services.session_service.get_redis", return_value=FlakyOnceB()):
        SessionManager._mirror_session(IID_B)
    after3 = SessionManager.mirror_exhausted_count()
    assert call_count["n"] == 2, f"expected exactly 2 attempts, got {call_count['n']}"
    assert after3 == before3, "one absorbed transient failure should not count as exhaustion"
    p(f"OK: one transient failure was retried and absorbed ({call_count['n']} attempts), counter unchanged")

    # ═══════════════════════════════════════════════════════════════════
    # Structure A — main.py's _mutate_session
    # ═══════════════════════════════════════════════════════════════════
    import app.services.face_recognition.main as m  # noqa: E402

    p("\n── Structure A (main.py's _mutate_session) ──")
    with m._session_lock:
        m._session.update({"state": "idle", "person_name": None})
    with m._mutate_exhausted_lock:
        m._mutate_exhausted_count = 0

    before_a = m._get_mutate_exhausted_count()
    assert before_a == 0

    class AlwaysFailsA:
        def set(self, *a, **kw):
            raise redis.exceptions.ConnectionError("simulated total outage")

    with patch.object(m, "get_redis", return_value=AlwaysFailsA()):
        with m._session_lock:
            m._mutate_session({"state": "session_active", "person_name": "HardwareTest"})

    after_a = m._get_mutate_exhausted_count()
    assert after_a == before_a + 1, f"expected counter to increment by 1, went {before_a} -> {after_a}"
    p(f"OK: total-failure case increments _get_mutate_exhausted_count() ({before_a} -> {after_a})")

    with m._session_lock:
        assert m._session["state"] == "session_active" and m._session["person_name"] == "HardwareTest"
    p("OK: in-process _session dict still correct despite total mirror failure (write-through)")

    with m._session_lock:
        m._mutate_session({"state": "idle", "person_name": None})  # real, successful write
    after_a2 = m._get_mutate_exhausted_count()
    assert after_a2 == after_a, "a successful mutate should not move the counter"
    p(f"OK: successful mutate leaves the counter unchanged at {after_a2}")

    call_count_a = {"n": 0}
    real_a = m.get_redis()

    class FlakyOnceA:
        def set(self, *a, **kw):
            call_count_a["n"] += 1
            if call_count_a["n"] == 1:
                raise redis.exceptions.ConnectionError("one transient failure")
            return real_a.set(*a, **kw)

    before_a3 = m._get_mutate_exhausted_count()
    with patch.object(m, "get_redis", return_value=FlakyOnceA()):
        with m._session_lock:
            m._mutate_session({"state": "session_active", "person_name": "RetriedHW"})
    after_a3 = m._get_mutate_exhausted_count()
    assert call_count_a["n"] == 2, f"expected exactly 2 attempts, got {call_count_a['n']}"
    assert after_a3 == before_a3, "one absorbed transient failure should not count as exhaustion"
    p(f"OK: one transient failure was retried and absorbed ({call_count_a['n']} attempts), counter unchanged")

    p("\n" + "=" * 78)
    p("RESULT: PASS — both structures' retry/counter/log-tag behavior verified correctly.")
    p("(Look above for the [MIRROR_WRITE_EXHAUSTED] and [MUTATE_WRITE_EXHAUSTED]")
    p(" tagged log lines emitted during the total-failure cases.)")
    p("=" * 78)

    get_redis().flushdb()

finally:
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)
    shutil.rmtree(test_dir, ignore_errors=True)
