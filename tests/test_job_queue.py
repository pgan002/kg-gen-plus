"""Tests for the durable (Redis-backed) job queue and the worker's recovery path.

These need a reachable Redis. Point ``KGGEN_TEST_REDIS_URL`` at one and they run;
otherwise they skip, so the default ``pytest`` run stays dependency-free::

    docker run -d --rm -p 63790:6379 redis:7-alpine \
        redis-server --save "" --appendonly no
    KGGEN_TEST_REDIS_URL=redis://localhost:63790/0 pytest tests/test_job_queue.py
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import uuid

import pytest
import pytest_asyncio

from app import settings
from app.job_queue import (
    JobFailed,
    JobNotFinished,
    JobNotFound,
    RedisJobQueue,
    _job_key,
    _payload_key,
    _result_key,
)
from app.jobs import JobStatus

REDIS_URL = os.environ.get("KGGEN_TEST_REDIS_URL")

pytestmark = [
    pytest.mark.skipif(
        not REDIS_URL, reason="set KGGEN_TEST_REDIS_URL to run durable-queue tests"
    ),
    pytest.mark.asyncio,
]


@pytest_asyncio.fixture
async def queue(monkeypatch):
    """A queue on a per-test key namespace, torn down afterwards.

    Namespacing per test keeps the stream, consumer group and job keys isolated
    so tests can run in any order against the same Redis.
    """
    prefix = f"kggen-test-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(settings, "KEY_PREFIX", prefix)
    monkeypatch.setattr(settings, "STREAM_KEY", f"{prefix}:jobs")
    monkeypatch.setattr(settings, "DEAD_LETTER_KEY", f"{prefix}:jobs:dead")

    q = RedisJobQueue(REDIS_URL)
    await q.ensure_group()
    yield q

    keys = [k async for k in q.redis.scan_iter(match=f"{prefix}:*")]
    if keys:
        await q.redis.delete(*keys)
    await q.close()


async def _enqueue(q, *, total_docs=2, api_key=None, ontology=None):
    corpus = b'{"id":"d1","text":"t1"}\n{"id":"d2","text":"t2"}\n'
    return await q.enqueue(
        total_docs=total_docs,
        corpus_bytes=corpus,
        ontology_bytes=ontology,
        generation_params={"model": "test-model"},
        api_key=api_key,
    )


async def test_enqueue_then_status_is_pending(queue):
    job_id = await _enqueue(queue)
    status = await queue.get_status(job_id)

    assert status["job_id"] == job_id
    assert status["status"] == JobStatus.pending.value
    assert status["total_docs"] == 2
    assert status["processed_docs"] == 0
    assert status["attempts"] == 0


async def test_unknown_job_raises_not_found(queue):
    with pytest.raises(JobNotFound):
        await queue.get_status("does-not-exist")
    with pytest.raises(JobNotFound):
        await queue.get_result("does-not-exist")


async def test_result_before_completion_raises_not_finished(queue):
    job_id = await _enqueue(queue)
    with pytest.raises(JobNotFinished) as exc:
        await queue.get_result(job_id)
    assert exc.value.status == JobStatus.pending.value


async def test_claim_returns_enqueued_job_and_payload_roundtrips(queue):
    job_id = await _enqueue(queue, api_key="secret-key", ontology=b"# ttl")

    claimed = await queue.claim_next("worker-1", block_ms=100)
    assert claimed is not None
    _entry_id, claimed_job_id = claimed
    assert claimed_job_id == job_id

    payload = await queue.load_payload(job_id)
    assert payload["corpus_bytes"].startswith(b'{"id":"d1"')
    assert payload["ontology_bytes"] == b"# ttl"
    assert payload["api_key"] == "secret-key"
    assert payload["generation_params"] == {"model": "test-model"}


async def test_empty_queue_returns_none(queue):
    assert await queue.claim_next("worker-1", block_ms=50) is None


async def test_complete_stores_gzipped_result_and_acks(queue):
    job_id = await _enqueue(queue)
    entry_id, _ = await queue.claim_next("worker-1", block_ms=100)
    await queue.mark_running(job_id)

    graph = {"entities": {"e0": {"surface_form": "Alice"}}, "relations": []}
    await queue.complete(
        entry_id, job_id, json.dumps(graph), {"X-KG-Gen-Time": "1.5"}, total_docs=2
    )

    status = await queue.get_status(job_id)
    assert status["status"] == JobStatus.completed.value
    assert status["processed_docs"] == 2
    assert status["percent_complete"] == 100.0

    fetched, headers = await queue.get_result(job_id)
    assert fetched == graph
    assert headers == {"X-KG-Gen-Time": "1.5"}

    # Stored compressed, payload dropped, nothing left pending on the stream.
    raw = await queue.redis.get(_result_key(job_id))
    assert json.loads(gzip.decompress(raw)) == graph
    assert await queue.redis.exists(_payload_key(job_id)) == 0
    pending = await queue.redis.xpending(settings.STREAM_KEY, settings.CONSUMER_GROUP)
    assert pending["pending"] == 0


async def test_terminal_failure_dead_letters_and_stops_redelivery(queue):
    job_id = await _enqueue(queue)
    entry_id, _ = await queue.claim_next("worker-1", block_ms=100)
    await queue.mark_running(job_id)

    await queue.fail(entry_id, job_id, "bad input", 422, retryable=False)

    with pytest.raises(JobFailed) as exc:
        await queue.get_result(job_id)
    assert exc.value.status_code == 422
    assert "bad input" in exc.value.error

    pending = await queue.redis.xpending(settings.STREAM_KEY, settings.CONSUMER_GROUP)
    assert pending["pending"] == 0
    dead = await queue.redis.xrange(settings.DEAD_LETTER_KEY)
    assert len(dead) == 1
    assert dead[0][1][b"job_id"].decode() == job_id


async def test_retryable_failure_leaves_entry_pending(queue):
    job_id = await _enqueue(queue)
    entry_id, _ = await queue.claim_next("worker-1", block_ms=100)
    await queue.mark_running(job_id)

    await queue.fail(entry_id, job_id, "transient blip", 500, retryable=True)

    # Still unacked, so it is eligible for redelivery once idle.
    pending = await queue.redis.xpending(settings.STREAM_KEY, settings.CONSUMER_GROUP)
    assert pending["pending"] == 1
    # And it must not look finished to a polling client.
    with pytest.raises(JobNotFinished):
        await queue.get_result(job_id)


async def test_dead_worker_job_is_reclaimed_by_another_worker(queue, monkeypatch):
    """The core crash-recovery guarantee: a job whose worker dies gets re-run."""
    monkeypatch.setattr(settings, "RECLAIM_IDLE_MS", 50)

    job_id = await _enqueue(queue)
    entry_id, _ = await queue.claim_next("worker-dies", block_ms=100)
    await queue.mark_running(job_id)
    # ...and now that worker "dies": no ack, no further heartbeat.

    await asyncio.sleep(0.1)

    reclaimed = await queue.claim_next("worker-survives", block_ms=100)
    assert reclaimed is not None
    reclaimed_entry_id, reclaimed_job_id = reclaimed
    assert reclaimed_job_id == job_id
    assert reclaimed_entry_id == entry_id

    # The payload is still there, so the replacement worker can actually re-run it.
    assert await queue.load_payload(job_id) is not None
    assert (await queue.mark_running(job_id)) == 2


async def test_heartbeat_keeps_a_live_job_from_being_reclaimed(queue, monkeypatch):
    """A long job must not be stolen while its worker is still alive."""
    monkeypatch.setattr(settings, "RECLAIM_IDLE_MS", 200)

    job_id = await _enqueue(queue)
    entry_id, _ = await queue.claim_next("worker-alive", block_ms=100)
    await queue.mark_running(job_id)

    # Heartbeat across more than one reclaim window.
    for _ in range(4):
        await asyncio.sleep(0.1)
        await queue.heartbeat(entry_id, "worker-alive", job_id, processed_docs=1)

    assert await queue.claim_next("worker-thief", block_ms=100) is None

    status = await queue.get_status(job_id)
    assert status["status"] == JobStatus.running.value
    assert status["processed_docs"] == 1
    assert status["attempts"] == 1


async def test_list_statuses_is_newest_first_and_prunes_expired(queue):
    first = await _enqueue(queue)
    second = await _enqueue(queue)

    statuses = await queue.list_statuses()
    assert [s["job_id"] for s in statuses] == [second, first]

    # Simulate the first job's hash expiring out from under the index.
    await queue.redis.delete(_job_key(first))
    statuses = await queue.list_statuses()
    assert [s["job_id"] for s in statuses] == [second]
    assert await queue.redis.zcard(f"{settings.KEY_PREFIX}:jobs:index") == 1


async def test_missing_result_blob_reads_as_not_found(queue):
    """A hash that outlived its result must 404, not return an empty graph."""
    job_id = await _enqueue(queue)
    entry_id, _ = await queue.claim_next("worker-1", block_ms=100)
    await queue.complete(entry_id, job_id, json.dumps({"entities": {}}), {}, 2)

    await queue.redis.delete(_result_key(job_id))
    with pytest.raises(JobNotFound):
        await queue.get_result(job_id)
