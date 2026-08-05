"""Durable job queue + job-state store, both backed by Redis.

Why one module for both: a queue alone cannot serve ``GET /api/jobs/{id}`` --
a stream is consumed, not randomly queried -- so durable jobs need a queue *and*
a keyed store. They share a Redis client and key namespace here, so splitting
them into separate modules would only add indirection.

Layout in Redis (all keys under ``settings.KEY_PREFIX``)::

    kggen:jobs                 stream    work queue, read by a consumer group
    kggen:jobs:dead            stream    poison jobs, for inspection
    kggen:job:{id}             hash      status/progress/attempts/error
    kggen:payload:{id}         hash      raw corpus + ontology bytes + params
    kggen:result:{id}          string    gzipped KnowledgeGraph JSON
    kggen:jobs:index           zset      job ids by creation time, for listing

Crash recovery: a worker XREADGROUPs an entry and only XACKs it after the job
finishes. While working it periodically XCLAIMs its own entry, which resets the
entry's idle timer. A worker that dies stops refreshing, so the entry goes idle
and another worker reclaims it via XAUTOCLAIM -- that is the automatic restart.
Because a live worker keeps the timer fresh, ``RECLAIM_IDLE_MS`` only has to
exceed the heartbeat interval, not the job's (possibly hours-long) runtime.

Redelivery is at-least-once, so a reclaimed job restarts from scratch and
overwrites its own result key. Generation is a pure function of the payload, so
re-running is safe; there is no partial-progress resume.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import time
import uuid
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app import settings
from app.jobs import JobStatus
from app.kggen_logger import kggen_logger


def _job_key(job_id: str) -> str:
    return f"{settings.KEY_PREFIX}:job:{job_id}"


def _payload_key(job_id: str) -> str:
    return f"{settings.KEY_PREFIX}:payload:{job_id}"


def _result_key(job_id: str) -> str:
    return f"{settings.KEY_PREFIX}:result:{job_id}"


def _index_key() -> str:
    return f"{settings.KEY_PREFIX}:jobs:index"


class JobNotFound(Exception):
    """No job with that id is known (never existed, or its TTL expired)."""


class JobNotFinished(Exception):
    """The job exists but has no result yet."""

    def __init__(self, status: str):
        super().__init__(f"Job not finished (status: {status})")
        self.status = status


class JobFailed(Exception):
    """The job finished unsuccessfully."""

    def __init__(self, error: str, status_code: int):
        super().__init__(error)
        self.error = error
        self.status_code = status_code


class RedisJobQueue:
    """Durable job queue and job-state store.

    A single instance is shared by the API process (which enqueues and reads
    status) and by each worker process (which consumes and writes status).
    """

    def __init__(self, url: str):
        # decode_responses=False: result blobs are gzipped bytes. Field names and
        # small values are decoded explicitly where needed.
        self._redis = aioredis.from_url(
            url,
            decode_responses=False,
            socket_timeout=settings.SOCKET_TIMEOUT_SECONDS,
            socket_keepalive=True,
            # A worker sits idle in a blocking read for long stretches; without a
            # health check a silently dropped connection is only noticed on the
            # next command.
            health_check_interval=30,
        )

    @property
    def redis(self) -> aioredis.Redis:
        return self._redis

    async def close(self) -> None:
        await self._redis.aclose()

    async def ping(self) -> bool:
        try:
            await self._redis.ping()
            return True
        except Exception as exc:  # pragma: no cover - depends on deployment
            kggen_logger.warning(f"Redis ping failed: {exc}")
            return False

    async def ensure_group(self) -> None:
        """Create the consumer group (and the stream) if absent.

        ``id="0"`` rather than ``"$"`` so a group created after entries were
        already produced still picks those entries up instead of skipping to
        new arrivals only.
        """
        try:
            await self._redis.xgroup_create(
                name=settings.STREAM_KEY,
                groupname=settings.CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
            kggen_logger.info(
                f"Created consumer group {settings.CONSUMER_GROUP!r} on "
                f"{settings.STREAM_KEY!r}"
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # -- producer side (API process) ---------------------------------------

    async def enqueue(
        self,
        *,
        total_docs: int,
        corpus_bytes: bytes,
        ontology_bytes: Optional[bytes],
        generation_params: dict[str, Any],
        api_key: Optional[str],
    ) -> str:
        """Persist a job's payload and publish it to the work stream.

        The raw *bytes* are stored rather than the parsed objects: the API has
        already validated them (so malformed input still gets a synchronous
        4xx), and re-parsing in the worker avoids serialising rdflib graphs and
        Pydantic models across the process boundary.
        """
        job_id = uuid.uuid4().hex
        now = time.time()

        payload: dict[bytes | str, bytes | str] = {
            "corpus": corpus_bytes,
            "generation_params": json.dumps(generation_params),
        }
        if ontology_bytes:
            payload["ontology"] = ontology_bytes
        if api_key:
            # Travels with the payload because the worker needs it for every LM
            # call, including after a redelivery. Note this means client API keys
            # live in Redis for the job's lifetime -- see the security note in
            # docs/durable-jobs.md before enabling Redis persistence.
            payload["api_key"] = api_key

        pipe = self._redis.pipeline(transaction=True)
        pipe.hset(_payload_key(job_id), mapping=payload)
        pipe.hset(
            _job_key(job_id),
            mapping={
                "job_id": job_id,
                "status": JobStatus.pending.value,
                "total_docs": str(total_docs),
                "processed_docs": "0",
                "attempts": "0",
                "created_at": str(now),
            },
        )
        pipe.zadd(_index_key(), {job_id: now})
        pipe.xadd(
            settings.STREAM_KEY,
            {"job_id": job_id},
            maxlen=settings.STREAM_MAXLEN,
            approximate=True,
        )
        await pipe.execute()

        kggen_logger.info(f"[job {job_id}] enqueued ({total_docs} docs)")
        return job_id

    # -- status / result reads (API process) -------------------------------

    async def get_status(self, job_id: str) -> dict[str, Any]:
        raw = await self._redis.hgetall(_job_key(job_id))
        if not raw:
            raise JobNotFound(job_id)
        return _status_dict(_decode_hash(raw))

    async def list_statuses(self, limit: int = 128) -> list[dict[str, Any]]:
        """Job statuses, newest first.

        Ids whose hash has expired are pruned from the index as they are found,
        so the index does not grow without bound.
        """
        ids = await self._redis.zrevrange(_index_key(), 0, limit - 1)
        if not ids:
            return []

        pipe = self._redis.pipeline(transaction=False)
        for jid in ids:
            pipe.hgetall(_job_key(_as_str(jid)))
        results = await pipe.execute()

        statuses: list[dict[str, Any]] = []
        stale: list[bytes] = []
        for jid, raw in zip(ids, results):
            if raw:
                statuses.append(_status_dict(_decode_hash(raw)))
            else:
                stale.append(jid)
        if stale:
            await self._redis.zrem(_index_key(), *stale)
        return statuses

    async def get_result(self, job_id: str) -> tuple[dict[str, Any], dict[str, str]]:
        """Return ``(knowledge_graph, stat_headers)`` for a completed job."""
        raw = await self._redis.hgetall(_job_key(job_id))
        if not raw:
            raise JobNotFound(job_id)
        fields = _decode_hash(raw)

        status = fields.get("status")
        if status == JobStatus.failed.value:
            raise JobFailed(
                fields.get("error") or "unknown error",
                int(fields.get("status_code") or 500),
            )
        if status != JobStatus.completed.value:
            raise JobNotFinished(status or "unknown")

        blob = await self._redis.get(_result_key(job_id))
        if blob is None:
            # Hash outlived the result blob (or vice versa) -- treat as gone
            # rather than returning a confusingly empty graph.
            raise JobNotFound(job_id)

        graph = json.loads(gzip.decompress(blob))
        headers = json.loads(fields.get("headers") or "{}")
        return graph, headers

    # -- consumer side (worker process) ------------------------------------

    async def claim_next(
        self, consumer: str, block_ms: Optional[int] = None
    ) -> Optional[tuple[str, str]]:
        """Return ``(entry_id, job_id)`` for the next job to run, or None.

        Abandoned entries (idle beyond ``RECLAIM_IDLE_MS`` because their worker
        died) are reclaimed before new ones are read, so a crashed job restarts
        promptly instead of waiting for the queue to drain.
        """
        if block_ms is None:
            block_ms = settings.CLAIM_BLOCK_MS

        reclaimed = await self._reclaim_abandoned(consumer)
        if reclaimed is not None:
            return reclaimed

        try:
            response = await self._redis.xreadgroup(
                groupname=settings.CONSUMER_GROUP,
                consumername=consumer,
                streams={settings.STREAM_KEY: ">"},
                count=1,
                block=block_ms,
            )
        except RedisTimeoutError:
            # The blocking read simply found nothing. Not an error -- returning
            # None keeps an idle worker's logs quiet instead of reporting a
            # failure once per poll.
            return None
        if not response:
            return None
        _stream, entries = response[0]
        if not entries:
            return None
        entry_id, fields = entries[0]
        job_id = _as_str(fields.get(b"job_id"))
        return _as_str(entry_id), job_id

    async def _reclaim_abandoned(self, consumer: str) -> Optional[tuple[str, str]]:
        try:
            cursor, entries, _deleted = await self._redis.xautoclaim(
                name=settings.STREAM_KEY,
                groupname=settings.CONSUMER_GROUP,
                consumername=consumer,
                min_idle_time=settings.RECLAIM_IDLE_MS,
                start_id="0-0",
                count=1,
            )
        except ResponseError as exc:  # pragma: no cover - old Redis
            kggen_logger.warning(f"XAUTOCLAIM unavailable ({exc}); skipping reclaim")
            return None

        for entry_id, fields in entries or []:
            job_id = _as_str(fields.get(b"job_id"))
            kggen_logger.warning(
                f"[job {job_id}] reclaimed after being idle > "
                f"{settings.RECLAIM_IDLE_MS}ms (previous worker presumed dead)"
            )
            return _as_str(entry_id), job_id
        return None

    async def load_payload(self, job_id: str) -> Optional[dict[str, Any]]:
        raw = await self._redis.hgetall(_payload_key(job_id))
        if not raw:
            return None
        return {
            "corpus_bytes": raw.get(b"corpus") or b"",
            "ontology_bytes": raw.get(b"ontology"),
            "generation_params": json.loads(_as_str(raw.get(b"generation_params"))),
            "api_key": _as_str(raw[b"api_key"]) if b"api_key" in raw else None,
        }

    async def mark_running(self, job_id: str) -> int:
        """Flag the job as running and return its (1-based) attempt number."""
        pipe = self._redis.pipeline(transaction=True)
        pipe.hincrby(_job_key(job_id), "attempts", 1)
        pipe.hset(
            _job_key(job_id),
            mapping={
                "status": JobStatus.running.value,
                "started_at": str(time.time()),
                "heartbeat_at": str(time.time()),
            },
        )
        results = await pipe.execute()
        return int(results[0])

    async def heartbeat(
        self, entry_id: str, consumer: str, job_id: str, processed_docs: int
    ) -> None:
        """Refresh the claim on the stream entry and flush buffered progress.

        The XCLAIM is what keeps a long-running job from looking abandoned; the
        HSET is only progress reporting. Both are best-effort: a transient Redis
        blip should not kill an otherwise healthy job, it will just look briefly
        stale (and, if the blip outlasts RECLAIM_IDLE_MS, get duplicated by a
        reclaiming worker -- acceptable under at-least-once).
        """
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.xclaim(
                name=settings.STREAM_KEY,
                groupname=settings.CONSUMER_GROUP,
                consumername=consumer,
                min_idle_time=0,
                message_ids=[entry_id],
                justid=True,
            )
            pipe.hset(
                _job_key(job_id),
                mapping={
                    "processed_docs": str(processed_docs),
                    "heartbeat_at": str(time.time()),
                },
            )
            await pipe.execute()
        except Exception as exc:
            kggen_logger.warning(f"[job {job_id}] heartbeat failed: {exc}")

    async def complete(
        self,
        entry_id: str,
        job_id: str,
        graph_json: str,
        headers: dict[str, str],
        total_docs: int,
    ) -> None:
        """Store the result, ack the stream entry, and drop the payload.

        The result is gzipped: graphs are verbose JSON (a 15-document corpus
        produced ~7 MB in testing) and they sit in Redis memory until their TTL.
        """
        blob = gzip.compress(graph_json.encode("utf-8"), compresslevel=6)
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(_result_key(job_id), blob, ex=settings.RESULT_TTL_SECONDS)
        pipe.hset(
            _job_key(job_id),
            mapping={
                "status": JobStatus.completed.value,
                "processed_docs": str(total_docs),
                "finished_at": str(time.time()),
                "headers": json.dumps(headers),
            },
        )
        pipe.expire(_job_key(job_id), settings.RESULT_TTL_SECONDS)
        pipe.delete(_payload_key(job_id))
        pipe.xack(settings.STREAM_KEY, settings.CONSUMER_GROUP, entry_id)
        pipe.xdel(settings.STREAM_KEY, entry_id)
        await pipe.execute()
        kggen_logger.info(
            f"[job {job_id}] completed ({len(blob)} bytes gzipped result)"
        )

    async def fail(
        self,
        entry_id: str,
        job_id: str,
        error: str,
        status_code: int = 500,
        *,
        retryable: bool,
    ) -> None:
        """Record a failure.

        ``retryable=True`` leaves the entry unacked so it is redelivered once it
        goes idle. Otherwise the job is terminal: acked, payload dropped, and a
        copy pushed to the dead-letter stream for inspection.
        """
        if retryable:
            await self._redis.hset(
                _job_key(job_id),
                mapping={"error": error, "status_code": str(status_code)},
            )
            kggen_logger.warning(
                f"[job {job_id}] attempt failed, leaving for redelivery: {error}"
            )
            return

        pipe = self._redis.pipeline(transaction=True)
        pipe.hset(
            _job_key(job_id),
            mapping={
                "status": JobStatus.failed.value,
                "error": error,
                "status_code": str(status_code),
                "finished_at": str(time.time()),
            },
        )
        pipe.expire(_job_key(job_id), settings.RESULT_TTL_SECONDS)
        pipe.delete(_payload_key(job_id))
        pipe.xack(settings.STREAM_KEY, settings.CONSUMER_GROUP, entry_id)
        pipe.xdel(settings.STREAM_KEY, entry_id)
        pipe.xadd(
            settings.DEAD_LETTER_KEY,
            {"job_id": job_id, "error": error[:1000]},
            maxlen=settings.STREAM_MAXLEN,
            approximate=True,
        )
        await pipe.execute()
        kggen_logger.error(f"[job {job_id}] failed permanently: {error}")


# -- helpers ---------------------------------------------------------------


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def _decode_hash(raw: dict[bytes, bytes]) -> dict[str, str]:
    return {_as_str(k): _as_str(v) for k, v in raw.items()}


def _status_dict(fields: dict[str, str]) -> dict[str, Any]:
    """Shape a job hash like ``Job.to_status_dict`` so both stores agree.

    Keeping the two response shapes identical means clients (and
    ``fetch_*`` benchmark scripts) work against either backend unchanged.
    """

    def _f(name: str) -> Optional[float]:
        value = fields.get(name)
        return float(value) if value else None

    total = int(fields.get("total_docs") or 0)
    processed = int(fields.get("processed_docs") or 0)
    started_at = _f("started_at")
    finished_at = _f("finished_at")

    now = finished_at or time.time()
    elapsed = now - started_at if started_at is not None else None
    pct = 100.0 * processed / total if total else None

    eta = None
    if started_at is not None and processed > 0 and elapsed and elapsed > 0:
        rate = processed / elapsed
        if rate > 0:
            eta = max(total - processed, 0) / rate

    return {
        "job_id": fields.get("job_id"),
        "status": fields.get("status"),
        "processed_docs": processed,
        "total_docs": total,
        "percent_complete": round(pct, 1) if pct is not None else None,
        "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
        "eta_seconds": round(eta, 1) if eta is not None else None,
        "created_at": _f("created_at"),
        "started_at": started_at,
        "finished_at": finished_at,
        "error": fields.get("error"),
        "attempts": int(fields.get("attempts") or 0),
    }


# Keyed by event loop, not a single global: a redis.asyncio client binds its
# connection pool to the loop that first used it, and reusing it from a different
# loop raises "Event loop is closed". A server process has exactly one loop, so
# this holds one entry in production -- but harnesses that drive the ASGI app
# with a fresh loop per request (Starlette's TestClient outside a context
# manager, which is how this repo's API tests are written) would otherwise break.
_queues: dict[asyncio.AbstractEventLoop, RedisJobQueue] = {}


def get_queue() -> Optional[RedisJobQueue]:
    """Queue handle for the running event loop, or None if Redis is unconfigured."""
    if not settings.redis_enabled():
        return None

    loop = asyncio.get_event_loop()
    for stale in [lp for lp in _queues if lp.is_closed()]:
        _queues.pop(stale, None)

    queue = _queues.get(loop)
    if queue is None:
        queue = RedisJobQueue(settings.REDIS_URL)  # type: ignore[arg-type]
        _queues[loop] = queue
    return queue


async def reset_queue() -> None:
    """Close and drop the handle for the running loop (shutdown / tests)."""
    loop = asyncio.get_event_loop()
    queue = _queues.pop(loop, None)
    if queue is not None:
        await queue.close()
