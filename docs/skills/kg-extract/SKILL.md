---
name: kg-extract
description: >-
  Extract an ontology-conforming knowledge graph from text. Use when the user
  wants to turn source text into typed entities and subject-predicate-object
  relations, optionally guided by a Turtle (.ttl) ontology. Performs NER,
  entity typing, and relation extraction yourself, using the kg-gen MCP tools
  for ontology processing, predicate suggestion, and validation.
---

# Knowledge Graph Extraction

You extract a knowledge graph from text. **You** do the reading and extraction
(entities, types, relations). The **MCP tools** handle everything deterministic:
parsing the ontology, suggesting valid predicates, and validating your output.

## Inputs

- **Source text** — the document(s) to extract from.
- **Ontology** (optional) — a Turtle (`.ttl`) file describing the target classes
  and predicates. If none is provided, use your own judgment for types and predicates.

## Workflow

1. **Process the ontology.** If a `.ttl` ontology was provided, call
   `parse_ontology`, then `list_target_types` to get the entity types to extract.

2. **Extract and type entities (you do this).** Read the source text and list the
   entities thoroughly. Assign each a type. If target types were provided, use only
   those types; otherwise choose sensible types yourself. Stay faithful to the text —
   do not invent entities.

3. **Get suggested predicates.** Call `suggest_predicates` with the entity types you
   found. It returns only the predicates whose domain/range fit those types — use this
   shortlist to focus relation extraction.

4. **Extract relations (you do this).** Extract subject–predicate–object triples from
   the text. Use only the suggested predicates, and respect their domain (subject type)
   and range (object type). Be thorough and faithful to the text.

5. **Validate and self-correct.** Call `validate_conformance` with your entities and
   relations. If the score is below 1.0, read the returned errors, fix the offending
   entities/relations, and re-run this check. Stop after a couple of passes even if not
   perfect — return your best result and note any remaining violations.

6. **Produce the final graph.** Call `serialize_graph` to build the canonical
   knowledge graph. Then optionally:
   - `deduplicate` to merge near-duplicate entities/edges across documents.
   - `convert_ontology` to emit any newly discovered types/predicates as Turtle.

## Notes

- Types, predicates, and entities are matched by label / surface form, so items with
  the same label are treated as the same thing.
- Prefer types and predicates from the provided ontology. When you must go beyond it,
  the extra types/predicates appear as ontology *extensions* in the output — that's
  expected, not an error.
- This interactive path is for precision on a handful of documents. For bulk corpora,
  use the batch HTTP service instead.
