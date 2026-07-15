"""In-memory store for background KG-generation jobs.

The service runs as a single uvicorn process (see docker-compose.yml), so an
in-memory store is sufficient: background work scheduled with
``asyncio.create_task`` keeps running on the event loop after the HTTP response
is sent, independent of whether the client stays connected. The trade-off is
that jobs (and their results) do not survive a process restart.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.pending
    total_docs: int = 0
    processed_docs: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Populated on success: the serialisable KnowledgeGraph and stat headers.
    result: Optional[Any] = None
    headers: Optional[dict[str, str]] = None
    # Populated on failure.
    error: Optional[str] = None
    status_code: int = 500
    task: Optional[asyncio.Task] = field(default=None, repr=False)

    def update_progress(self, processed: int, total: int) -> None:
        self.processed_docs = processed
        self.total_docs = total

    def _eta_seconds(self) -> Optional[float]:
        if self.started_at is None or self.processed_docs <= 0:
            return None
        reference_end = self.finished_at or time.time()
        elapsed = reference_end - self.started_at
        rate = self.processed_docs / elapsed if elapsed > 0 else 0.0
        if rate <= 0:
            return None
        remaining = max(self.total_docs - self.processed_docs, 0)
        return remaining / rate

    def to_status_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot of progress (no heavy result payload)."""
        now = self.finished_at or time.time()
        elapsed = now - self.started_at if self.started_at is not None else None
        pct = 100.0 * self.processed_docs / self.total_docs if self.total_docs else None
        return {
            "job_id": self.id,
            "status": self.status.value,
            "processed_docs": self.processed_docs,
            "total_docs": self.total_docs,
            "percent_complete": round(pct, 1) if pct is not None else None,
            "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
            "eta_seconds": (
                round(eta, 1) if (eta := self._eta_seconds()) is not None else None
            ),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class JobStore:
    """Bounded store of jobs, evicting oldest finished jobs when full."""

    def __init__(self, max_jobs: int = 128):
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._max_jobs = max_jobs

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job
        self._evict()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """All tracked jobs, newest first."""
        return list(reversed(self._jobs.values()))

    def _evict(self) -> None:
        # Drop the oldest finished jobs first; never evict a running/pending job.
        while len(self._jobs) > self._max_jobs:
            for jid, job in self._jobs.items():
                if job.status in (JobStatus.completed, JobStatus.failed):
                    del self._jobs[jid]
                    break
            else:
                # Nothing finished to evict; leave the store slightly oversized.
                break


job_store = JobStore()
