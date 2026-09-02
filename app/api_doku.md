# KG Explorer App

This application provides three main functionalities:

1.  **Web APIs (at `/api`)**: A set of endpoints to generate graphs from text, and to parse and convert ontologies.
2.  **Graphical User UI (at `/ui`)**: A user interface to visually investigate the generated graphs.
3.  **MCP server (at `/mcp`)**: A Model Context Protocol server exposing the deterministic building blocks (ontology parsing, predicate suggestion, conformance/schema validation, graph serialization, deduplication) for agent-driven extraction. See the "MCP server" section below.

## How to use the application

### 1. Generate a graph

Use the web APIs at `/api` to generate a graph from your text or ontology.

**Note:** Knowledge Graph generation is an LLM-based process. While the system attempts to follow the provided ontology, conformance (types, domains, ranges) is best-effort and not strictly guaranteed.

The model is shown ontology terms by label and answers with labels, so labels
are matched back by their normalized form: `birth_date`, `birthDate` and
`Birth Date` all resolve to a term labelled `birth date`. Nothing is guessed
beyond that — a misspelling stays unmatched, and whatever stays unmatched is
reported under `ontology_extensions` (with a `null` URI when the model did not
invent one) rather than being dropped. `skos:prefLabel`, `skos:altLabel` and
`skos:hiddenLabel` are honoured: they are shown to the model alongside the
term's description and accepted when matching its answer, which is the
supported way to teach it that e.g. `State` means your `Province` class.

There are two ways to generate a graph from a `.jsonl` corpus:

- **`POST /api/generate` (synchronous)**: the request blocks until generation
  finishes and returns the `KnowledgeGraph` in the response body. Simple, but the
  connection must stay open for the whole run — best for small corpora.
- **`POST /api/generate_async` (background job)**: recommended for large
  corpora. The uploads are validated and read up front (so malformed input still
  returns a `4xx` immediately), then generation runs in the background and keeps
  going even if the client disconnects. Returns `202 Accepted` with a `job_id`.

Both endpoints accept the same `corpus_file` / `ontology_file` uploads and
`GenerationMetadata` query parameters, and progress is logged server-side in a
tqdm-like form (`[generate] 45/200 docs (22.5%) elapsed=12.3s ETA=41.9s`).

#### Stat headers

Successful responses carry a few scalar `X-KG-Gen-*` headers: `Time`,
`Input-Tokens`, `Output-Tokens`, `Total-Tokens`, `Failed-Documents` (a count),
and `Dedup-Stats` when deduplication ran. They are deliberately small — the full
per-step breakdown, class and predicate usage, and the per-document failure
records are in the response body under `KnowledgeGraph.stats`. A stat header that
would exceed 1 KB or is not latin-1 encodable is logged and dropped rather than
being allowed to break the response.

#### Background jobs

- **`POST /api/generate_async`** → `{ "job_id", "status", "total_docs", "status_url", "result_url" }`
- **`GET /api/jobs`** → list of all tracked jobs (newest first).
- **`GET /api/jobs/{job_id}`** → status and progress:
  `status` (`pending` / `running` / `completed` / `failed`), `processed_docs`,
  `total_docs`, `percent_complete`, `elapsed_seconds`, `eta_seconds`.
- **`GET /api/jobs/{job_id}/result`** → the generated `KnowledgeGraph` (with the
  same `X-KG-Gen-*` stat headers as `/generate`). Returns `404` for an unknown
  job, `409` while it is still `pending`/`running`, and re-raises the original
  error status if the job failed.

Typical flow: `POST /api/generate_async`, then poll `GET /api/jobs/{job_id}`
until `status` is `completed`, then `GET /api/jobs/{job_id}/result`.

Where jobs are stored depends on whether `KGGEN_REDIS_URL` is configured.

**With Redis (recommended, and the `docker-compose.yml` default).** Jobs go to a
durable Redis stream and are executed by separate worker processes. Jobs and
results survive an API or worker restart; a worker that dies mid-job causes the
job to be redelivered and re-run automatically; and job state is shared across
every API replica and worker. A job that fails repeatedly is given up on after
`KGGEN_MAX_ATTEMPTS` (reported as `attempts` in the status response). Status
hashes and results expire after `KGGEN_RESULT_TTL_SECONDS` (24 h by default), so
still fetch results reasonably promptly. See
[docs/durable-jobs.md](../docs/durable-jobs.md).

> **⚠️ Caveat — without `KGGEN_REDIS_URL`, jobs are stored in memory.** The job
> store lives in the server process. As a consequence: (1) all jobs and their
> results are **lost on a server restart**; (2) the store is **bounded** — once
> full, the oldest *finished* jobs are evicted; and (3) it will **not work across
> multiple workers/replicas** (a job created on one worker is invisible to the
> others).

### 2. Visualize the graph

To investigate a graph visually, you first have to generate a graph and then call `/ui/add_graph` to add the graph to the visualizer.

After adding the graph, go to the `/ui` endpoint. In the user interface, you can open the graph by using the "Open Existing Graph" functionality.

## MCP server

A Model Context Protocol (MCP) server is mounted **in-process** at `/mcp`
(streamable HTTP transport), so it starts and stops together with this app — no
separate process to run. It offers an alternative, **agent-in-the-loop** path:
the agent performs the cognitive work (NER, entity typing, relation extraction),
while the MCP tools handle the deterministic parts.

Tools exposed: `parse_ontology`, `list_target_types`, `suggest_predicates`,
`validate_conformance`, `validate_graph_schema`, `serialize_graph`,
`convert_ontology`, `suggest_clusters`, `apply_clusters`. The server is
**stateless** — pass the ontology Turtle to each tool that needs it.

Deduplication is split across two tools so the agent reviews merges instead of
trusting embeddings blindly: `suggest_clusters` proposes candidate duplicate
entity clusters from local embeddings (predicates are not clustered — they
already come from the ontology's controlled vocabulary), the agent inspects
and edits the proposal, then `apply_clusters` merges the reviewed clusters
into the graph.

Point any MCP client at `<base-url>/mcp`. The accompanying `kg-extract` skill
(`skills/kg-extract/`) describes the extraction workflow for the agent. The MCP
server can also be run standalone with `fastmcp run mcp/server.py`.

> Requires the optional `mcp` extra (`pip install 'kg-gen[mcp]'`). If it is not
> installed, the `/mcp` endpoint is simply not mounted and the rest of the app
> works normally.
