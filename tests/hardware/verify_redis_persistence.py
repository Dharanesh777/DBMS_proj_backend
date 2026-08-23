#!/usr/bin/env python3
"""
tests/hardware/verify_redis_persistence.py — Redis persistence hardware check.

WHAT THIS PROVES
-----------------
That app.services.session_service.SessionManager's Redis mirror survives a
HARD kill (SIGKILL — the closest thing to a power-loss simulation available
without literally pulling power) of the Redis process, not just a graceful
shutdown or an app restart. This is the concrete reproduction behind
TECH_DEBT.md's "Resolved: Redis persistence was off by default, silently
defeating the state migration" entry — read that entry first for the full
why this exists.

WHY THIS MUST ALSO RUN ON THE REAL TARGET HARDWARE, NOT JUST WHEREVER YOU'RE
READING THIS FROM
------------------------------------------------------------------------------
The repo's redis.conf sets `appendfsync everysec` deliberately (not
`always`, to avoid excess SD-card wear on a Raspberry Pi 3 over its
lifetime). `everysec` means there is a real window — up to ~1 second —
where a completed write is acknowledged but not yet flushed to disk. How
long that window actually takes to CLOSE depends on the storage medium's
fsync latency, which differs enormously between a dev machine's SSD/NVMe
and a Pi 3's SD card. As of this writing, this script has only been run on
a dev Mac (APFS on SSD) — see TECH_DEBT.md's "OPEN GAP" note under that
same entry. Running it validates the MECHANISM (does AOF replay work at
all, does the app's mirror survive a restart of a process it doesn't
control); it does NOT validate the TIMING (is the wait this script does
before killing Redis actually long enough on the real SD card). If you're
running this on real Pi 3 hardware: that's exactly the gap this script
exists to close. Please update TECH_DEBT.md with the result either way.

WHAT "PASS" LOOKS LIKE
------------------------
The script prints a step-by-step trace and ends with one of:
    RESULT: PASS — state survived a hard kill -9 of the Redis process.
    RESULT: FAIL — state was lost across kill -9, persistence is not working.
A FAIL on real Pi 3 hardware is a genuine, important finding (not a bug in
this script) — it would mean the ~1.5s margin this script waits before
killing Redis isn't enough for that device's SD card to complete an
`everysec` fsync, and either the wait needs to be longer in practice or a
different durability tradeoff is needed for that hardware.

HOW TO RUN
-----------
    source venv/bin/activate      # repo's venv, needs `redis` + app packages
    python tests/hardware/verify_redis_persistence.py

Prerequisites: a `redis-server` (and `redis-cli`) binary reachable on
$PATH, or set REDIS_SERVER_BIN / REDIS_CLI_BIN env vars to their full
paths. Safe to run repeatedly and safe to run alongside a real Redis
instance already serving the app — this script starts its OWN isolated
redis-server process on a non-default port (REDIS_TEST_PORT, default 6399)
using a throwaway temp directory, and never touches the project's own
Redis instance, port, or data.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REDIS_CONF = os.path.join(REPO_ROOT, "redis.conf")

REDIS_SERVER_BIN = os.environ.get("REDIS_SERVER_BIN") or shutil.which("redis-server")
REDIS_CLI_BIN = os.environ.get("REDIS_CLI_BIN") or shutil.which("redis-cli")
TEST_PORT = int(os.environ.get("REDIS_TEST_PORT", "6399"))
FSYNC_MARGIN_SECONDS = float(os.environ.get("FSYNC_MARGIN_SECONDS", "1.5"))

INTERACTION_ID = 700100  # fixed, distinctive test ID — never used by real data


def p(msg):
    print(msg, flush=True)


def fail(msg):
    p(f"\nABORT: {msg}")
    sys.exit(1)


if not REDIS_SERVER_BIN:
    fail("redis-server not found on $PATH and REDIS_SERVER_BIN not set.")
if not REDIS_CLI_BIN:
    fail("redis-cli not found on $PATH and REDIS_CLI_BIN not set.")
if not os.path.exists(REDIS_CONF):
    fail(f"Expected to find {REDIS_CONF} — run this from within the repo (venv active).")

test_dir = tempfile.mkdtemp(prefix="agos-redis-persist-test-")
p(f"[setup] redis-server: {REDIS_SERVER_BIN}")
p(f"[setup] redis-cli:    {REDIS_CLI_BIN}")
p(f"[setup] test port:    {TEST_PORT} (isolated from any real instance)")
p(f"[setup] test dir:     {test_dir} (isolated from the project's own data)")
p(f"[setup] using config: {REDIS_CONF}")

# Point the app's Redis client at our isolated test instance BEFORE any
# app.* module is imported, so app.config.get_settings() (lru_cached) picks
# it up. Must happen before the imports below.
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = str(TEST_PORT)

sys.path.insert(0, REPO_ROOT)
from app.services.session_service import SessionManager, SessionState, _session_hash_key  # noqa: E402
from app.services.redis_client import get_redis  # noqa: E402


def start_redis():
    proc = subprocess.Popen(
        [REDIS_SERVER_BIN, REDIS_CONF,
         "--port", str(TEST_PORT),
         "--dir", test_dir,
         "--dbfilename", "dump.rdb",
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


def redis_up():
    try:
        get_redis().ping()
        return True
    except Exception:
        return False


try:
    p("\n[step0] starting isolated redis-server with the repo's redis.conf...")
    proc = start_redis()
    appendonly = subprocess.run(
        [REDIS_CLI_BIN, "-p", str(TEST_PORT), "config", "get", "appendonly"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    p(f"[step0] confirmed running config: appendonly={appendonly[-1] if appendonly else '?'}")
    if not appendonly or appendonly[-1] != "yes":
        fail(f"appendonly is not 'yes' — redis.conf isn't taking effect as expected: {appendonly}")

    with SessionManager._lock:
        SessionManager._active_sessions.clear()
        SessionManager._session_task_ids.clear()
    get_redis().flushdb()

    state = SessionState(interaction_id=INTERACTION_ID, session_number=1, user_id=1, person_id=5)
    state.session_summaries = ["Hardware persistence check: this string must survive a kill -9."]
    with SessionManager._lock:
        SessionManager._active_sessions[INTERACTION_ID] = state
        SessionManager._session_task_ids[INTERACTION_ID] = "hw-test-task-1"

    SessionManager._mirror_session(INTERACTION_ID)
    h = get_redis().hgetall(_session_hash_key(INTERACTION_ID))
    if h.get("session_number") != "1":
        fail(f"initial mirror write did not actually succeed: {h}")
    p(f"[step1] mirrored successfully: {h}")

    p(f"[step1] waiting {FSYNC_MARGIN_SECONDS}s for appendfsync everysec to flush "
      f"this write to disk (THIS is the number that needs re-validating on real "
      f"Pi 3 SD-card timing — see the module docstring)...")
    time.sleep(FSYNC_MARGIN_SECONDS)

    p(f"[step2] kill -9 {proc.pid} (hard kill — SIGKILL, no graceful shutdown, "
      f"no save-on-exit)")
    proc.kill()
    proc.wait(timeout=5)
    time.sleep(0.3)
    if redis_up():
        fail("redis is still responding after kill -9 — process did not actually die.")
    p("[step2] confirmed: redis-server is dead")

    p("[step3] restarting redis-server pointed at the SAME --dir (reloading AOF)...")
    proc = start_redis()
    p("[step3] redis is back up")

    h_after = get_redis().hgetall(_session_hash_key(INTERACTION_ID))
    p(f"[step3] mirror content after kill -9 + restart: {h_after}")

    ok = (
        h_after.get("session_number") == "1"
        and "kill -9" in h_after.get("session_summaries", "")
    )

    p("")
    p("=" * 78)
    if ok:
        p("RESULT: PASS — state survived a hard kill -9 of the Redis process.")
    else:
        p("RESULT: FAIL — state was lost across kill -9. Persistence is not")
        p("        protecting against this on this machine/config.")
    p("=" * 78)

    if ok:
        with SessionManager._lock:
            SessionManager._active_sessions.clear()
            SessionManager._session_task_ids.clear()
        SessionManager.restore_and_sweep(max_age_minutes=60)
        restored = SessionManager._active_sessions.get(INTERACTION_ID)
        if restored is not None and restored.session_summaries == state.session_summaries:
            p("Also confirmed: restore_and_sweep() correctly rehydrates this into "
              "_active_sessions after the simulated crash-restart.")
        else:
            p("NOTE: raw Redis data survived, but restore_and_sweep() did not "
              f"rehydrate it as expected (got: {restored}) — investigate separately.")

    get_redis().flushdb()
    sys.exit(0 if ok else 1)

finally:
    try:
        if 'proc' in dir() and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass
    shutil.rmtree(test_dir, ignore_errors=True)
