# Durable jobs: the Redis queue + worker split

Background generation jobs used to live in a Python dict inside the API process
and run via `asyncio.create_task`. That meant a restart -- deploy, crash, OOM --
lost every in-flight job and every result. This describes the durable
replacement: a Redis stream the API publishes to and separate worker processes
consume from.

## Why a queue is not enough on its own

An MQ gives you two things:

* **durable work** -- the request survives an API or worker restart, and
* **redelivery** -- an entry a worker never acked is handed to another worker,
  which is what makes a crashed job restart automatically.

It does *not* give you `GET /api/jobs/{id}`. A stream is consumed, not randomly
queried, so there is nowhere to look up "what is job X doing" or "give me job
X's graph". Durable jobs therefore need a queue **and** a keyed state store.
Redis provides both, which is why it is one component here rather than a broker
plus a database.

## Shape

```
   ┌──────────────┐                  ┌──────────────────────────┐
   │   API        │  XADD            │         Redis            │
   │  (uvicorn)   ├─────────────────▶│  kggen:jobs      stream  │
   │              │                  │  kggen:job:{id}  hash    │
   │  poll status │◀────HGETALL──────┤  kggen:result:{id} str   │
   │  poll result │◀────GET──────────┤  kggen:payload:{id} hash │
   └──────────────┘                  └────────┬─────────────────┘
                                              │ XREADGROUP / XAUTOCLAIM
                                     ┌────────┴─────────┐
                                     │  worker × N      │
                                     │ python -m        │
                                     │   app.worker     │
                                     └──────────────────┘
```

`/api/generate_async` validates and parses the upload as before -- so malformed
input still gets a synchronous 4xx -- then stores the **raw bytes** and publishes
a stream entry. The worker re-parses those bytes. Validating twice is cheaper
than serialising rdflib graphs and Pydantic models across a process boundary,
and it keeps the API's error contract unchanged.

`/api/generate` (the synchronous endpoint) is untouched and needs no Redis.

## Enabling it

Set `KGGEN_REDIS_URL` on both the API and the workers, and run at least one
worker. `docker-compose.yml` does this already:

```bash
docker compose up -d                          # api + 1 worker + redis
docker compose up -d --scale worker=4         # more throughput
```

**With `KGGEN_REDIS_URL` unset, nothing changes**: `/generate_async` runs jobs on
the API event loop exactly as before, using the in-memory store. That keeps local
development and the test suite runnable without Redis, and is why
`tests/test_job_queue.py` skips unless `KGGEN_TEST_REDIS_URL` is set.

## Crash recovery, concretely

A worker `XREADGROUP`s an entry and acks it only when the job finishes. While
working it periodically re-`XCLAIM`s its own entry, which resets that entry's
idle timer. So:

* **worker crashes / is killed** -- it stops refreshing, the entry goes idle past
  `KGGEN_RECLAIM_IDLE_MS`, and the next worker's `XAUTOCLAIM` picks it up and
  re-runs it.
* **worker shuts down gracefully** (SIGTERM) -- it stops taking new jobs and
  finishes what it holds. A job cancelled mid-flight is left unacked, so it is
  redelivered.
* **API restarts** -- irrelevant to running jobs; it holds no job state.
* **long-running job** -- not mistaken for a dead one, because a live worker keeps
  resetting the idle timer. This is why `RECLAIM_IDLE_MS` only has to exceed the
  *heartbeat interval*, not the job's runtime. (Getting this wrong is the classic
  visibility-timeout bug: a 13-hour job with a 1-hour timeout gets run
  repeatedly, forever.)

Delivery is **at-least-once**, so a redelivered job **restarts from scratch** and
overwrites its own result. Generation is a pure function of the payload, so
re-running is safe; there is no partial-progress resume. A job that keeps failing
is dead-lettered after `KGGEN_MAX_ATTEMPTS` to `kggen:jobs:dead` rather than
looping through every worker forever -- which is what a deterministic failure
(say, a `max_tokens` larger than the model's context window) would otherwise do.

Failures are only retried when retrying could plausibly help: a 5xx is
retryable, a 4xx (bad input) is terminal immediately.

## Persistence, and what lands on disk

Redis runs **with persistence off by default** (`--save "" --appendonly no`):

| | in-memory (default) | persistent |
|---|---|---|
| API/worker restart or crash | jobs survive | jobs survive |
| Redis restart | jobs lost | jobs survive |
| written to disk | nothing | job payloads + results |

Switch by setting `REDIS_PERSISTENCE_ARGS=--appendonly yes`.

⚠️ **Before enabling persistence**, note what a payload contains: the document
text being processed, and any `X-API-Key` the client supplied (the worker needs
it for every LM call, including after a redelivery). Persistence puts both on
Redis's disk. If you need durability across a Redis restart *and* care about data
at rest, use an encrypted volume and restrict Redis network access.

`maxmemory-policy` is deliberately `noeviction`. With an LRU policy Redis would
silently evict job state under memory pressure -- losing jobs that clients are
still polling. `noeviction` makes writes fail loudly instead.

Results are gzipped before storage (a 15-document corpus produced a ~7 MB graph
in testing) and both the status hash and the result expire after
`KGGEN_RESULT_TTL_SECONDS`. Corpora over `KGGEN_MAX_PAYLOAD_BYTES` are rejected
with 413 rather than allowed to consume an unbounded slice of Redis memory.

## The DSPy cache is a separate disk consumer

Independent of jobs, DSPy writes an on-disk cache of LM **prompts and
completions** -- i.e. document text and extracted content -- to
`DSPY_CACHEDIR`. It is bounded, but DSPy's own default limit is **30 GB**.
It is a `diskcache.FanoutCache`, so it evicts LRU once over the limit rather
than growing forever.

* `DSPY_CACHE_LIMIT=<bytes>` -- set the ceiling. `docker-compose.yml` now sets
  1 GB rather than inheriting the 30 GB default.
* To turn the disk cache off entirely, call
  `dspy.configure_cache(enable_disk_cache=False)` at startup, or construct
  `KGGen(disable_cache=True)`.

One gotcha: the in-**memory** half of that cache defaults to `memory_max_entries
= 1_000_000` and is not settable by environment variable -- only via an explicit
`dspy.configure_cache(...)` call. Worth lowering if worker RSS grows over a long
run.

## Sizing workers

Roughly:

```
worker RSS ≈ 600 MB baseline + (concurrent jobs × per-job graph/corpus)
```

The ~600 MB baseline is measured, and is mostly `torch` +
`sentence-transformers` (~450 MB) loaded solely to embed entities for
deduplication; the embedding model itself is ~23 MB on top. It is shared
process-wide, so it is paid once per worker process, not once per job. Per-job
cost scales roughly linearly at ~4-5 KB per entity (Pydantic graph objects plus
dedup structures) -- ~1.2 GB peak at 50k entities, and well under 100 MB on top
of the baseline at realistic corpus sizes.

Because the baseline is per-process and the per-job cost is not, prefer
**more replicas at `KGGEN_WORKER_CONCURRENCY=1`** over one process running many
jobs: concurrency multiplies the volatile part while replicas give predictable,
independently-limitable memory.

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `KGGEN_REDIS_URL` | *(unset)* | Enables the durable path. Unset = in-process jobs. |
| `KGGEN_KEY_PREFIX` | `kggen` | Key namespace, so one Redis can host several environments. |
| `KGGEN_CONSUMER_GROUP` | `workers` | Consumer group name. |
| `KGGEN_WORKER_CONCURRENCY` | `1` | Jobs in flight per worker process. |
| `KGGEN_HEARTBEAT_SECONDS` | `30` | Claim-refresh and progress-flush interval. |
| `KGGEN_RECLAIM_IDLE_MS` | `4 × heartbeat` | Idle time after which an entry is treated as abandoned. |
| `KGGEN_MAX_ATTEMPTS` | `3` | Deliveries before a job is dead-lettered. |
| `KGGEN_RESULT_TTL_SECONDS` | `86400` | Expiry for status hash and result blob. |
| `KGGEN_MAX_PAYLOAD_BYTES` | `67108864` | Reject larger corpora with 413. |
| `KGGEN_STREAM_MAXLEN` | `10000` | Backstop cap on stream length. |
| `KGGEN_CLAIM_BLOCK_MS` | `5000` | How long an idle worker blocks per poll. |
| `KGGEN_SOCKET_TIMEOUT_SECONDS` | `block + 10` | Redis socket read timeout; must exceed the blocking read. |
| `REDIS_PERSISTENCE_ARGS` | `--save "" --appendonly no` | Redis persistence mode. |
| `REDIS_MAXMEMORY` | `2gb` | Redis memory ceiling (`noeviction`). |
| `DSPY_CACHE_LIMIT` | `1073741824` | On-disk LM cache ceiling (DSPy's own default is 30 GB). |

## Operating notes

```bash
# queue depth and un-acked work
redis-cli XLEN kggen:jobs
redis-cli XPENDING kggen:jobs workers

# jobs that gave up
redis-cli XRANGE kggen:jobs:dead - +

# one job's state
redis-cli HGETALL kggen:job:<id>
```

A job's `attempts` field is exposed in `GET /api/jobs/{id}`, so a client can see
it was retried.

## Testing

```bash
docker run -d --rm -p 63790:6379 redis:7-alpine \
    redis-server --save "" --appendonly no
KGGEN_TEST_REDIS_URL=redis://localhost:63790/0 pytest tests/test_job_queue.py
```

`tests/test_job_queue.py` covers enqueue/claim/complete, the gzipped result
round-trip, terminal vs. retryable failure, dead-lettering, index pruning, and
the two recovery guarantees that matter: a dead worker's job **is** reclaimed,
and a live worker's job **is not** stolen while it heartbeats.
