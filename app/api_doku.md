# KG Explorer App

This application provides three main functionalities:

1.  **Web APIs (at `/api`)**: A set of endpoints to generate graphs from text, and to parse and convert ontologies.
2.  **Graphical User UI (at `/ui`)**: A user interface to visually investigate the generated graphs.
3.  **MCP server (at `/mcp`)**: A Model Context Protocol server exposing the deterministic building blocks (ontology parsing, predicate suggestion, conformance/schema validation, graph serialization, deduplication) for agent-driven extraction. See the "MCP server" section below.

## How to use the application

### 1. Generate a graph

Use the web APIs at `/api` to generate a graph from your text or ontology.

**Note:** Knowledge Graph generation is an LLM-based process. While the system attempts to follow the provided ontology, conformance (types, domains, ranges) is best-effort and not strictly guaranteed.

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

> **⚠️ Caveat — jobs are stored in memory.** The job store lives in the server
> process, which is correct for the current single-worker deployment. As a
> consequence: (1) all jobs and their results are **lost on a server restart**;
> (2) the store is **bounded** — once full, the oldest *finished* jobs are
> evicted, so fetch results reasonably promptly; and (3) it will **not work
> across multiple workers/replicas** (a job created on one worker is invisible
> to the others). If the service is scaled out, this needs a shared backend
> (e.g. Redis or a database).

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
clusters (entities and edges) from local embeddings, the agent inspects and
edits them, then `apply_clusters` merges the reviewed clusters into the graph.

Point any MCP client at `<base-url>/mcp`. The accompanying `kg-extract` skill
(`skills/kg-extract/`) describes the extraction workflow for the agent. The MCP
server can also be run standalone with `fastmcp run mcp/server.py`.

> Requires the optional `mcp` extra (`pip install 'kg-gen[mcp]'`). If it is not
> installed, the `/mcp` endpoint is simply not mounted and the rest of the app
> works normally.
