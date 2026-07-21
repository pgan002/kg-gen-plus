# Converting kg-gen to Skill + MCP Server

> **Status:** Proposal for discussion
> **Author:** Artem Revenko
> **Date:** 2026-07-08

## Goal

Offer an **interactive, agent-in-the-loop** path for knowledge-graph extraction alongside
the existing bulk HTTP service. The agent (an LLM) performs the *cognitive* work — NER,
entity typing, relation extraction — guided by a **Skill**, while an **MCP server** exposes
the *deterministic* scaffolding — ontology processing, target-type derivation, predicate
suggestion, and schema/conformance validation.

## The key insight: split on determinism

The current pipeline mixes two very different kinds of work in one call. The re-architecture
splits them cleanly:

| Kind | Examples | New home |
| :--- | :--- | :--- |
| **Cognitive / generative** (needs an LLM) | NER, entity typing, relation extraction | **Skill** — the agent does this, guided by skill prose |
| **Deterministic** (no LLM, or embeddings only) | ontology parsing, domain/range filtering, conformance scoring, TTL serialization, URI reconciliation, datatype guessing, embedding dedup | **MCP server** — thin, testable tools |

We **invert control**: instead of the extraction engine orchestrating its own LLM calls, the
agent orchestrates, calling deterministic MCP tools at each step. The extraction prompts that
today live buried inside the engine become skill instructions the agent can see and adapt.

## Scope decisions

- **Add alongside, don't replace.** The bulk HTTP service stays for parallel corpus ingestion
  (its core value: async fan-out, a-posteriori dedup, "infinite scale"). The skill+MCP is a
  **new interactive path** optimized for precision and ontology conformance on a handful of
  documents, and for interactive ontology authoring. Both sit on the same underlying package,
  so there is **no logic duplication**.
- **Deduplication is two MCP tools, not one.** Embedding-based cross-document merging stays
  available, but it no longer merges unattended: `suggest_clusters` proposes candidate clusters
  from embeddings, the agent reviews them, and `apply_clusters` merges only what was reviewed.
  The MCP server reuses a process-wide shared embedding model, so there is no extra memory cost
  beyond one instance.

## Proposed MCP tool inventory

Every tool is a **thin, deterministic wrapper** over logic that already exists (no LLM inside any
tool).

The server is **stateless** — it holds no parsed ontology between calls, so every tool that
needs the ontology takes the Turtle text (`ontology_ttl`) again. Simpler and safe for concurrent
use; the small cost is re-parsing the TTL per call.

| MCP tool | What it does | Requirement it serves |
| :--- | :--- | :--- |
| `parse_ontology(ontology_ttl) -> Ontology{classes, predicates}` | Parse a Turtle ontology into structured classes and predicates | Process the input ontology |
| `list_target_types(ontology_ttl) -> [EntityType]` | Return the entity types the agent should extract | Create target NE types to extract |
| `suggest_predicates(ontology_ttl, found_entity_types) -> [OntologyPredicate]` | Given the types the agent found, return the predicates whose domain **and** range are both satisfied (literal-range predicates always kept) | **Highest-value tool** — narrows the agent's relation-extraction search space |
| `validate_conformance(typed_entities, relations, ontology_ttl) -> {score, conformant, errors}` | Score how well entities/relations conform to the ontology and return human-readable errors | Validate output; drive the agent's self-correction loop |
| `validate_graph_schema(graph) -> {valid, errors}` | Structural validation of the output graph payload | Schema validation |
| `serialize_graph(typed_entities, relations, ontology_ttl) -> KnowledgeGraph` | Build the canonical output: URI reconciliation, literal-vs-object detection, datatype guessing, ontology-extension detection | Produce the final graph |
| `convert_ontology(classes, predicates) -> ttl` | Serialize classes/predicates (including discovered extensions) back to Turtle | Emit ontology / extensions as TTL |
| `suggest_clusters(typed_entities, entity_similarity_threshold, retrieval_model) -> {entity_clusters}` | Propose candidate duplicate-entity clusters from local embeddings; merges nothing. Predicates aren't clustered — they already come from the ontology's controlled vocabulary | First half of dedup — surfaces candidates for the agent to check |
| `apply_clusters(typed_entities, relations, entity_clusters, edge_clusters) -> KnowledgeGraph` | Merge the agent-reviewed clusters into the canonical graph, aggregating provenance | Second half of dedup — commits only what the agent approved |

## Proposed Skill workflow

A `SKILL.md` encoding the *methodology* and orchestration order (the extraction guidance plus the
conformance feedback loop, written as agent instructions):

1. Call `parse_ontology` on the user's TTL, then `list_target_types`.
2. **NER + typing** (agent): read the source text, extract entities thoroughly, assign each a type
   from the target types.
3. Call `suggest_predicates` with the entity types just found → get the compatible predicate shortlist.
4. **Relation extraction** (agent): extract subject–predicate–object triples using only the suggested
   predicates, respecting domain/range.
5. Call `validate_conformance`; if the score is below 1.0, feed the returned errors into a
   re-extraction pass. This replaces an opaque internal retry with an explicit, visible cycle the
   agent controls.
6. Call `serialize_graph` → optionally `suggest_clusters` → review the clusters →
   `apply_clusters` → optionally `convert_ontology` for extensions.

## End-to-end flow

```
                 ┌─────────────────────── Agent (LLM, driven by SKILL.md) ───────────────────────┐
                 │                                                                                 │
  TTL ontology ──┼─► parse_ontology ─► list_target_types ─►  [NER + typing]  ─► suggest_predicates │
                 │        (MCP)             (MCP)               (agent)             (MCP)           │
  source text ───┘                                                                     │           │
                 │                                                                     ▼           │
                 │   serialize_graph ◄─ (loop until conforms) validate_conformance ◄─ [relation    │
                 │        (MCP)                                    (MCP)               extraction]  │
                 │        │                                                             (agent)     │
                 │        ▼                                                                         │
                 │   suggest_clusters ─► [review clusters] ─► apply_clusters ─► convert_ontology    │
                 │        (MCP)              (agent)             (MCP)         (extensions, MCP)    │
                 │                                                    │                              │
                 │                                                    ▼                              │
                 │                                              KnowledgeGraph                       │
                 └─────────────────────────────────────────────────────────────────────────────┘
```

## Why this is a good fit

- **No logic duplication** — MCP tools reuse the exact logic the HTTP service already uses.
- **Fully testable surface** — every MCP tool is deterministic, so it can be unit-tested against
  fixtures (unlike the LLM extraction path).
- **Visible, agent-native conformance loop** — replaces an opaque internal retry with an explicit
  validate → read errors → re-extract cycle the agent controls.
- **Complementary, not disruptive** — bulk throughput via HTTP is preserved; the interactive path is
  additive.

## Use cases: what new does this enable?

The batch service is a single fire-and-forget endpoint: you hand it a corpus and an
ontology and get a graph back. Putting an agent in the loop with deterministic tools opens
up things that were hard or impossible before.

- **Conversational correction.** "You missed the acquisitions," "merge these two people,"
  "only keep Person–worksFor–Organization." The agent re-runs the relevant steps in place.
  The batch pipeline has no way to accept feedback short of a full re-run.

- **Entity linking to real knowledge bases.** Because the agent can call *other* MCP tools
  in the same session (Wikidata/SPARQL search, an internal catalog), it can assign genuine
  URIs instead of minting throwaway ones — directly addressing the known weakness that
  LLM-generated URIs are unreliable.

- **Extraction as one step in a larger task.** Read a PDF → extract a graph → query it →
  draft a briefing, all in one agent session. Extraction becomes composable with everything
  else the agent can do, rather than a standalone service call.

- **Explainable, auditable extraction.** The agent narrates why it typed an entity a certain
  way, shows the conformance errors it hit, and shows the fixes it made. Valuable in
  domains that require justification — legal, medical, compliance.

- **Context-aware incremental building.** Extending an existing graph with a new document,
  the agent can keep the current graph in context and reuse existing surface forms — softening
  the batch pipeline's "each chunk is extracted blind" limitation, without paying its
  quadratic-context cost at scale.

- **Standalone validation service.** The conformance and schema-validation tools are useful on
  their own — e.g. validating graphs produced by *other* pipelines, or a CI check that an
  ontology and its instance data still agree.

- **Zero-infra prototyping.** Analysts can try extraction on a snippet from their IDE or
  desktop agent host, with no FastAPI service to deploy and no server-side key management.

[//]: # (- **Interactive ontology bootstrapping.** Point the agent at a few sample documents with no)

[//]: # (  ontology &#40;or a partial one&#41;. It extracts, and everything it can't place shows up as)

[//]: # (  proposed *extensions* — candidate types and predicates. You review, keep what fits, and)

[//]: # (  converge on an ontology iteratively. Today you need the ontology up front.)

## Risks / open questions

- **Throughput vs. precision.** The agent path is serial and interactive; it is not a substitute for
  bulk ingestion. We need to be clear with users about which path to use when, and avoid the
  interactive path being reached for on large corpora.
- **Quality parity.** Does a general agent doing NER/RE from skill prose match the tuned extraction
  engine? This needs a side-by-side evaluation on our benchmarks before we recommend it broadly.
- **Cost and latency.** Multiple tool round-trips plus a possible re-extraction loop can be slower and
  more expensive per document than a single engine call. Worth measuring; the conformance loop may
  need a retry cap.
- **Ontology re-parsing (decided: stateless).** Predicate suggestion needs the parsed RDF graph (for
  class hierarchy), not just the flat type list. We chose to keep the server stateless and re-parse the
  Turtle on each call rather than hold session state — simpler and concurrency-safe. If TTL re-parsing
  ever shows up as a hotspot, add a small in-process parse cache keyed by ontology hash.
- **Identity/dedup semantics.** Types, predicates, and entities are identified by label/surface-form,
  so round-tripping through the tools silently merges same-label items. Usually desirable, but must be
  documented so results aren't surprising.
- **Conformance is best-effort.** Even with the loop, the agent may not reach full conformance. We
  should decide the fallback: return the best non-conforming result (as today) vs. surface the
  remaining violations to the user.
- **Model/host portability.** The skill assumes a capable agent host that supports MCP. Behavior may
  vary across hosts and models; we should define a minimum target.
- **Security/tenancy.** The MCP server accepts arbitrary TTL and graph JSON. We need limits on input
  size and parsing, and a story for multi-user isolation if it is shared.
