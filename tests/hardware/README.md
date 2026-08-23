# Hardware verification scripts

Standalone scripts (not pytest — they orchestrate real subprocesses and
timing, not the kind of thing you want auto-collected into a normal
`pytest` run) for verifying behavior that depends on the actual deployment
target, not just "does the code run." See `TECH_DEBT.md` at the repo root
for the full background on why these exist.

Both scripts are self-contained: they start their own isolated
`redis-server` process on a non-default port with a throwaway temp
directory, and clean up after themselves. Safe to run repeatedly, and safe
to run alongside a real, already-running instance of this app — neither
script touches the app's actual configured Redis instance, port, or data.

## `verify_redis_persistence.py`

**Needs real target (Pi 3) hardware to be conclusive — has so far only run
on a dev Mac.** Confirms the app's Redis-mirrored session state survives a
hard `kill -9` of the Redis process (a power-loss stand-in), using the
repo's `redis.conf` (`appendonly yes` / `appendfsync everysec`). The
`everysec` fsync policy has a real, storage-medium-dependent latency
window — this script validates the *mechanism* on whatever machine runs it
but the *timing margin* only means something on the actual target SD card.
See the module docstring for the full explanation, and `TECH_DEBT.md`'s
"OPEN GAP" note.

```
python tests/hardware/verify_redis_persistence.py
```

PASS/FAIL is printed explicitly at the end. A FAIL on real Pi 3 hardware is
a genuine finding — update `TECH_DEBT.md` either way.

## `verify_mirror_instrumentation.py`

Does **not** need real hardware to be meaningful (pure application logic,
not storage timing) — checked in here so a hardware-verification pass can
sanity-check it on the target machine/Python version too, without having
to reconstruct it from chat history. Confirms both write-through Redis
mirrors (`main.py`'s `_mutate_session`, `session_service.py`'s
`SessionManager._mirror_session`) retry a transient failure, count and
distinctly log (`[MUTATE_WRITE_EXHAUSTED]` / `[MIRROR_WRITE_EXHAUSTED]`)
a fully-exhausted failure, and never let a Redis failure corrupt the
in-process (write-through) state.

```
python tests/hardware/verify_mirror_instrumentation.py
```

## Prerequisites (both scripts)

- Repo venv active (`source venv/bin/activate`), so `redis` and the app's
  own packages are importable.
- A `redis-server` binary reachable on `$PATH`. If it's somewhere else, set
  `REDIS_SERVER_BIN=/path/to/redis-server` (and `REDIS_CLI_BIN` for the
  persistence script specifically).
