"""Background worker: consumes KG-generation jobs from the Redis stream.

Run one or more of these alongside the API::

    python -m app.worker

The API process no longer runs generation itself when ``KGGEN_REDIS_URL`` is set;
it validates the upload, stores the payload, and publishes a stream entry. Each
worker claims entries, runs generation, and writes status/results back -- so an
API restart, a worker restart, or a worker crash all leave the job recoverable
(see the module docstring in ``app.job_queue`` for the reclaim mechanics).

Sizing note: each concurrent job holds its corpus and intermediate graph in RAM
on top of the ~600 MB shared embedding/ML baseline, so prefer more replicas with
``KGGEN_WORKER_CONCURRENCY=1`` over one process running many jobs.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import uuid
from typing import Any

from fastapi import HTTPException

from app import settings
from app.generation import (
    execute_generation,
    kg_gen_from_params,
    prepare_generation_inputs,
)
from app.job_queue import RedisJobQueue
from app.kggen_logger import kggen_logger
from app.schemas import GenerationMetadata


class Worker:
    def __init__(self, queue: RedisJobQueue, concurrency: int):
        self._queue = queue
        self._concurrency = concurrency
        self._slots = asyncio.Semaphore(concurrency)
        self._stopping = asyncio.Event()
        self._in_flight: set[asyncio.Task] = set()
        # Distinct per process AND per restart: a consumer name that collided
        # with a live worker's would let one steal the other's pending entries.
        self.name = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"

    def request_stop(self) -> None:
        kggen_logger.info(
            f"[worker {self.name}] stop requested; will finish "
            f"{len(self._in_flight)} in-flight job(s) and exit"
        )
        self._stopping.set()

    async def run(self) -> None:
        await self._queue.ensure_group()
        kggen_logger.info(
            f"[worker {self.name}] listening on {settings.STREAM_KEY!r} "
            f"(group={settings.CONSUMER_GROUP!r}, concurrency={self._concurrency})"
        )

        while not self._stopping.is_set():
            await self._slots.acquire()
            if self._stopping.is_set():
                self._slots.release()
                break

            try:
                claimed = await self._queue.claim_next(self.name)
            except Exception as exc:
                # Redis unreachable: back off rather than spin.
                kggen_logger.warning(f"[worker {self.name}] claim failed: {exc}")
                self._slots.release()
                await asyncio.sleep(2)
                continue

            if claimed is None:
                self._slots.release()
                continue

            entry_id, job_id = claimed
            task = asyncio.create_task(self._run_job(entry_id, job_id))
            self._in_flight.add(task)
            task.add_done_callback(self._in_flight.discard)
            task.add_done_callback(lambda _t: self._slots.release())

        if self._in_flight:
            await asyncio.gather(*self._in_flight, return_exceptions=True)
        kggen_logger.info(f"[worker {self.name}] stopped")

    async def _run_job(self, entry_id: str, job_id: str) -> None:
        attempt = await self._queue.mark_running(job_id)
        last_attempt = attempt >= settings.MAX_ATTEMPTS

        payload = await self._queue.load_payload(job_id)
        if payload is None:
            # Payload gone (TTL, or a completed job's entry redelivered). Nothing
            # to run and nothing to retry -- ack it so it stops circulating.
            await self._queue.fail(
                entry_id,
                job_id,
                "Job payload is no longer available",
                410,
                retryable=False,
            )
            return

        progress = {"processed": 0}

        def on_progress(processed: int, _total: int) -> None:
            # Called synchronously from inside kg_gen.generate. Buffered rather
            # than written through, so per-document progress costs no Redis
            # round-trip; the heartbeat loop flushes it.
            progress["processed"] = processed

        heartbeat = asyncio.create_task(self._heartbeat(entry_id, job_id, progress))
        try:
            kggen_logger.info(
                f"[job {job_id}] starting (attempt {attempt}/{settings.MAX_ATTEMPTS}) "
                f"on worker {self.name}"
            )
            graph_json, headers, total_docs = await self._generate(payload, on_progress)
            await self._queue.complete(
                entry_id, job_id, graph_json, headers, total_docs
            )
        except asyncio.CancelledError:
            # Shutdown mid-job: leave the entry unacked so another worker (or
            # this one after restart) picks it up once it goes idle.
            kggen_logger.warning(f"[job {job_id}] cancelled; left for redelivery")
            raise
        except HTTPException as exc:
            # A 4xx here is bad input, which no amount of retrying fixes.
            retryable = exc.status_code >= 500 and not last_attempt
            await self._queue.fail(
                entry_id,
                job_id,
                str(exc.detail),
                exc.status_code,
                retryable=retryable,
            )
        except Exception as exc:
            kggen_logger.exception(f"[job {job_id}] generation failed")
            await self._queue.fail(
                entry_id,
                job_id,
                f"KGGen failed: {exc}",
                500,
                retryable=not last_attempt,
            )
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _generate(
        self, payload: dict[str, Any], on_progress
    ) -> tuple[str, dict[str, str], int]:
        params = GenerationMetadata.model_validate(payload["generation_params"])
        inputs, onto, rdflib_onto, types, predicates = prepare_generation_inputs(
            payload["corpus_bytes"], payload["ontology_bytes"]
        )
        if not inputs:
            raise HTTPException(status_code=400, detail="Corpus contains no documents")

        kg_gen = kg_gen_from_params(params, payload["api_key"])
        kg, headers = await execute_generation(
            kg_gen,
            inputs,
            onto,
            rdflib_onto,
            types,
            predicates,
            params,
            progress_callback=on_progress,
        )
        # Serialise here (not in job_queue) so Pydantic's own encoder handles the
        # KnowledgeGraph model rather than a generic json.dumps default.
        return kg.model_dump_json(), headers, len(inputs)

    async def _heartbeat(
        self, entry_id: str, job_id: str, progress: dict[str, int]
    ) -> None:
        while True:
            await asyncio.sleep(settings.HEARTBEAT_SECONDS)
            await self._queue.heartbeat(
                entry_id, self.name, job_id, progress["processed"]
            )


async def main() -> None:
    if not settings.redis_enabled():
        raise SystemExit(
            "KGGEN_REDIS_URL is not set -- the worker has no queue to consume. "
            "Set it to the same Redis the API uses."
        )

    queue = RedisJobQueue(settings.REDIS_URL)  # type: ignore[arg-type]
    if not await queue.ping():
        raise SystemExit(f"Cannot reach Redis at {settings.REDIS_URL}")

    worker = Worker(queue, settings.WORKER_CONCURRENCY)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, worker.request_stop)

    try:
        await worker.run()
    finally:
        await queue.close()


def run() -> None:
    """Console entrypoint (``python -m app.worker``)."""
    import logging.config
    from pathlib import Path

    import yaml

    config_path = Path(__file__).parent / "logging.yaml"
    if config_path.exists():
        with open(config_path) as f:
            logging.config.dictConfig(yaml.safe_load(f.read()))

    asyncio.run(main())


if __name__ == "__main__":
    run()
