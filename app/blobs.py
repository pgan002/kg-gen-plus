"""Short-lived storage for data an agent wants to hand to a tool by reference.

An MCP tool's arguments are produced by the calling model, so anything passed
inline is paid for in that model's output tokens: on a real run, a 702-entity
list came to ~32,000 tokens, and a large one can exceed the model's output limit
outright. The tools already accept a *file path* instead, but a path only works
when the server shares a filesystem with the agent -- a remote server answers
``No such file or directory``.

A blob closes that gap without needing MCP to grow an upload primitive it does
not have (tools, resources, prompts and sampling are all either the wrong
direction or the same token-priced channel). The agent uploads with an ordinary
HTTP request, so the bytes travel disk -> network and never enter its context,
then passes back a handle of a couple of dozen characters.

Blobs live in Redis when it is configured, so every worker and API replica sees
them, and in process memory otherwise -- which is fine for the single-process
local server the MCP path is usually pointed at, but means a multi-replica
deployment without Redis can miss a blob it just stored.
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

from app import settings
from app.kggen_logger import kggen_logger

HANDLE_PREFIX = "blob:"

# In-process fallback: {blob_id: (expires_at, data)}
_memory: dict[str, tuple[float, bytes]] = {}


def _key(blob_id: str) -> str:
    return f"{settings.KEY_PREFIX}:blob:{blob_id}"


def _blob_id(data: bytes) -> str:
    """Content-addressed, so re-uploading identical bytes is idempotent."""
    return hashlib.sha256(data).hexdigest()[:16]


def _expire_memory(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    for blob_id in [k for k, (expires, _) in _memory.items() if expires <= now]:
        del _memory[blob_id]


async def store(data: bytes) -> str:
    """Store ``data`` and return its handle (``blob:<id>``)."""
    blob_id = _blob_id(data)
    ttl = settings.BLOB_TTL_SECONDS
    if settings.redis_enabled():
        from redis.asyncio import Redis

        redis = Redis.from_url(settings.REDIS_URL)
        try:
            await redis.set(_key(blob_id), data, ex=ttl)
        finally:
            await redis.aclose()
    else:
        _expire_memory()
        _memory[blob_id] = (time.time() + ttl, data)
    kggen_logger.info(f"stored blob {blob_id} ({len(data)} bytes, ttl {ttl}s)")
    return f"{HANDLE_PREFIX}{blob_id}"


def load(handle: str) -> bytes:
    """Return the bytes for a handle, or raise ``KeyError`` if it is gone.

    Synchronous because the MCP tools that consume handles are synchronous
    functions; the read is a single key fetch of data the caller just uploaded.

    Expiry and a typo are indistinguishable here on purpose: either way the
    caller has to re-upload, and distinguishing them would mean keeping a record
    of every handle ever issued.
    """
    blob_id = handle[len(HANDLE_PREFIX) :] if is_handle(handle) else handle
    if settings.redis_enabled():
        from redis import Redis

        redis = Redis.from_url(settings.REDIS_URL)
        try:
            data = redis.get(_key(blob_id))
        finally:
            redis.close()
        if data is None:
            raise KeyError(handle)
        return data

    _expire_memory()
    entry = _memory.get(blob_id)
    if entry is None:
        raise KeyError(handle)
    return entry[1]


def is_handle(value: object) -> bool:
    """Whether ``value`` is a blob handle rather than inline data or a path."""
    return isinstance(value, str) and value.startswith(HANDLE_PREFIX)
