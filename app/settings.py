"""Runtime settings for the durable-job (Redis) path.

Everything here is read from the environment so a deployment can switch between
the in-process job store (no Redis configured -- the original behaviour, jobs
lost on restart) and the durable queue+worker split simply by setting
``KGGEN_REDIS_URL``.
"""

from __future__ import annotations

import os


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Connection string for the broker/state store. When unset, the service falls
# back to the in-memory job store and runs generation on the API event loop --
# i.e. exactly the pre-split behaviour, which keeps local dev and the test suite
# runnable without Redis.
REDIS_URL: str | None = os.environ.get("KGGEN_REDIS_URL") or None

# Key namespace, so one Redis instance can host several environments.
KEY_PREFIX: str = os.environ.get("KGGEN_KEY_PREFIX", "kggen")

# Stream + consumer group the workers read from.
STREAM_KEY: str = f"{KEY_PREFIX}:jobs"
CONSUMER_GROUP: str = os.environ.get("KGGEN_CONSUMER_GROUP", "workers")
DEAD_LETTER_KEY: str = f"{KEY_PREFIX}:jobs:dead"

# Approximate cap on stream length. Entries are XDEL'd once acked, so this is
# only a backstop against unacked growth if a worker misbehaves.
STREAM_MAXLEN: int = _int_env("KGGEN_STREAM_MAXLEN", 10_000)

# How many jobs one worker process runs concurrently. Deliberately 1 by default:
# each in-flight job holds its whole corpus and intermediate graph in RAM on top
# of the ~600 MB shared embedding/ML baseline, so scaling out by replicas is more
# predictable than raising per-process concurrency. See the RAM notes in
# docs/durable-jobs.md.
WORKER_CONCURRENCY: int = _int_env("KGGEN_WORKER_CONCURRENCY", 1)

# While a job runs, the worker refreshes its claim on the stream entry and
# flushes buffered progress this often.
HEARTBEAT_SECONDS: int = _int_env("KGGEN_HEARTBEAT_SECONDS", 30)

# How long an idle worker parks in a blocking XREADGROUP before looping (and
# re-checking for abandoned entries).
CLAIM_BLOCK_MS: int = _int_env("KGGEN_CLAIM_BLOCK_MS", 5_000)

# Socket read timeout for the Redis client. Set explicitly, and comfortably above
# CLAIM_BLOCK_MS, because the worker's claim is a *blocking* read: a socket
# timeout shorter than the block would surface as a spurious TimeoutError every
# idle poll. Left unset (None) a genuinely dead connection could hang forever
# instead, so an explicit generous value beats either default.
SOCKET_TIMEOUT_SECONDS: int = _int_env(
    "KGGEN_SOCKET_TIMEOUT_SECONDS", CLAIM_BLOCK_MS // 1000 + 10
)

# A stream entry idle for longer than this is considered abandoned (its worker
# died) and is reclaimed by another worker. Must comfortably exceed
# HEARTBEAT_SECONDS -- a live worker keeps resetting the idle timer, so this
# bound does NOT have to exceed the job's total runtime.
RECLAIM_IDLE_MS: int = _int_env("KGGEN_RECLAIM_IDLE_MS", HEARTBEAT_SECONDS * 1000 * 4)

# How many times a job may be delivered before it is treated as poison and
# dead-lettered instead of redelivered forever. Guards against a deterministic
# crash (e.g. a request whose max_tokens exceeds the model's context window)
# looping through every worker indefinitely.
MAX_ATTEMPTS: int = _int_env("KGGEN_MAX_ATTEMPTS", 3)

# Expiry for a finished job's status hash and result blob. Results are held in
# Redis memory, so this bound matters: a 15-document corpus produced a ~7 MB
# graph in testing.
RESULT_TTL_SECONDS: int = _int_env("KGGEN_RESULT_TTL_SECONDS", 24 * 3600)

# Reject a corpus larger than this outright rather than letting a single request
# consume an unbounded slice of Redis memory.
MAX_PAYLOAD_BYTES: int = _int_env("KGGEN_MAX_PAYLOAD_BYTES", 64 * 1024 * 1024)


def redis_enabled() -> bool:
    return REDIS_URL is not None
