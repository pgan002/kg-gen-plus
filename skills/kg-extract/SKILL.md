---
name: kg-extract
description: >-
  Extract an ontology-conforming knowledge graph from text. Use when the user
  wants to turn source text into typed entities and subject-predicate-object
  relations, optionally guided by a Turtle (.ttl) ontology. You perform NER,
  entity typing, and relation extraction yourself, using the kg-gen MCP tools
  for ontology processing, predicate suggestion, and validation.
---

# Knowledge Graph Extraction

You extract a knowledge graph from text. **You** do the reading and extraction
(entities, types, relations). The **kg-gen MCP tools** handle everything
deterministic: parsing the ontology, suggesting valid predicates, validating
your output, and serializing the final graph.

The MCP server is **stateless** — it never remembers the ontology between calls.
Pass the ontology Turtle text to every tool that needs it (`parse_ontology`,
`list_target_types`, `suggest_predicates`, `validate_conformance`,
`serialize_graph`). Keep the original `.ttl` around and hand it over each time.

## Inputs

- **Source text** — the document(s) to extract from.
- **Ontology** (optional) — a Turtle (`.ttl`) file describing the target classes
  and predicates. If none is provided, use your own judgment for types and
  predicates, and the output will list them as ontology *extensions*.

## What the ontology file should look like

The ontology is plain RDF in Turtle. Only these constructs are read — anything
else is ignored:

| Construct | Meaning |
| :--- | :--- |
| `a owl:Class` | Declares an entity type (class) |
| `a owl:ObjectProperty` | Declares a predicate whose object is another entity |
| `a owl:DatatypeProperty` | Declares a predicate whose object is a literal value |
| `rdfs:label` | Human-readable name of a class/predicate (used for matching) |
| `rdfs:comment` | Optional description |
| `rdfs:domain` | The allowed **subject** type(s) of a predicate |
| `rdfs:range` | The allowed **object** type(s): a class, or an XSD datatype for literals |
| `rdfs:subClassOf` | Class hierarchy (a predicate on a superclass applies to subclasses) |

Example:

```turtle
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix ex:   <http://example.org/> .

ex:Person       a owl:Class ; rdfs:label "Person" ; rdfs:comment "A human being" .
ex:Organization a owl:Class ; rdfs:label "Organization" .

ex:works_for a owl:ObjectProperty ;
    rdfs:label  "works_for" ;
    rdfs:domain ex:Person ;
    rdfs:range  ex:Organization .

ex:birth_date a owl:DatatypeProperty ;
    rdfs:label  "birth_date" ;
    rdfs:domain ex:Person ;
    rdfs:range  xsd:date .
```

Notes:
- Classes, predicates, and entities are matched by their **label / surface form**,
  so items with the same label are treated as the same thing.
- Give every class and predicate a `uri` and an `rdfs:label`; predicates without
  a domain/range are treated as unconstrained.

## Workflow

1. **Process the ontology.** If a `.ttl` ontology was provided, call
   `parse_ontology`, then `list_target_types` to get the entity types to extract.

2. **Extract and type entities (you do this).** Read the source text and list the
   entities thoroughly. Assign each a type. If target types were provided, use
   only those; otherwise choose sensible types yourself. Stay faithful to the
   text — do not invent entities.

3. **Get suggested predicates.** Call `suggest_predicates` with the ontology and
   **all** the entity types you found. It returns the predicates whose domain and
   range are both satisfied by those types (predicates with a literal/XSD range
   are always included). Because a predicate needs *both* endpoints present, pass
   the complete set of types you extracted, not one at a time.

4. **Extract relations (you do this).** Extract subject–predicate–object triples
   from the text. Use only the suggested predicates, and respect their domain
   (subject type) and range (object type). Be thorough and faithful to the text.

5. **Validate and self-correct.** Call `validate_conformance` with your entities,
   relations, and the ontology. If the score is below 1.0, read the returned
   errors, fix the offending entities/relations, and re-run the check. Stop after
   a couple of passes even if not perfect — return your best result and note any
   remaining violations.

6. **Produce the final graph.** Call `serialize_graph` to build the canonical
   knowledge graph (with URI reconciliation and literal detection). Then
   optionally:
   - `suggest_clusters` + `apply_clusters` to merge near-duplicate entities/edges
     (see below).
   - `convert_ontology` to emit any newly discovered types/predicates as Turtle.
   - `validate_graph_schema` if you want to double-check the final payload.

## Deduplication: propose, review, apply

Deduplication is two tools, not one, because embeddings alone should not decide
what gets merged — **you** review every proposed cluster before anything changes.

1. **`suggest_clusters`** — proposes candidate duplicate clusters for entities
   and edges (predicates) using local sentence embeddings (no LLM). Nothing is
   merged yet; it returns `entity_clusters` and `edge_clusters`, each a list of
   `{members, representative}`.
2. **You review the clusters.** For each proposed cluster:
   - Does every member really mean the same thing (e.g. "USA" and "United
     States")? Drop the ones that don't belong.
   - If a cluster lumps together distinct things, split it into more clusters.
   - Is `representative` the best canonical surface form? Swap it for a
     different member, or a new label, if not.
   - It's fine to discard a cluster entirely (pass it through unchanged, or
     just omit it) if you disagree with the proposal.
3. **`apply_clusters`** — takes your reviewed `entity_clusters` and
   `edge_clusters` (plus the original `typed_entities`/`relations`) and merges
   them into the canonical graph, aggregating provenance. Pass an empty list
   for either kind to skip that merge.

Tune `suggest_clusters` with:

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `entity_similarity_threshold` | `0.8` | Cosine similarity above which two **entities** are proposed as a cluster. Higher = stricter (fewer candidates); lower = more aggressive. |
| `edge_similarity_threshold` | `0.9` | Same, for **predicates/edges**. Keep this higher than the entity threshold so distinct relations are not proposed together. |
| `retrieval_model` | `sentence-transformers/all-MiniLM-L6-v2` | The embedding model. `null` falls back to the library's built-in encoder. |

Guidance: start with the defaults. If distinct things keep showing up as
candidates, **raise** the thresholds; if obvious duplicates aren't proposed at
all, **lower** them — but the threshold only affects what gets *proposed*, you
still decide what actually merges.

## Notes

- Prefer types and predicates from the provided ontology. When you must go beyond
  it, the extra types/predicates appear as ontology *extensions* in the output —
  that's expected, not an error.
- This interactive path is for precision on a handful of documents. For bulk
  corpora, use the batch HTTP service instead.
