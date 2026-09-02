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

3. **Get the predicates each chunk can support.** Build a mapping from chunk id
   to the entity types you found **in that chunk**, and pass it to
   `suggest_predicates_batch` in a single call. It returns each predicate's
   description once in `legend`, the distinct answers in `predicate_sets`, and
   the index of the applicable set per chunk in `group_predicate_set`.

   A predicate is kept for a chunk only when that chunk contains an entity
   satisfying its domain *and* one satisfying its range (a literal/XSD range
   needs no object entity, but the domain must still match). So narrowing is
   per chunk by construction: on an 18-class/17-predicate ontology a single
   chunk yields a median of 2 types and therefore **4 candidate predicates**,
   where the union of types over 200 chunks yields 17 of 17 — no narrowing at
   all. Do **not** aggregate the types first; that hands the model the whole
   property set for every chunk.

   Because narrowing follows your typing, **step 2 completeness now decides
   step 4's ceiling**: a relation whose subject or object you failed to type is
   not merely unsuggested, it is unreachable. If a chunk comes back with no
   candidate predicates, re-read it for an entity you missed before accepting
   that it carries no relations.

4. **Extract relations (you do this).** For each chunk, use only the predicates
   in *that chunk's* set, and respect their domain (subject type) and range
   (object type). Be thorough and faithful to the text.

5. **Self-check the extraction before validating.** `validate_conformance`
   checks your output against the *ontology*; it cannot tell you whether your
   output matches the *text*. Two script checks cover that, and both have caught
   real errors here — on one 200-document slice they found three misattributed
   provenance entries and three entities that runs without the checks had missed
   entirely:

   - **Provenance is real and complete.** Every provenance id must be an id that
     exists in the source. Then count the chunks with no entity at all and
     re-read those: an empty chunk is usually an extraction gap, not an empty
     chunk. (If you narrowed predicates per chunk, this is also where
     over-narrowing shows up — a chunk with no candidate predicates is a chunk
     whose entities you may have failed to type.)
   - **Every surface form appears in its chunk's text.** Normalise both sides
     (strip accents, lowercase, collapse non-alphanumerics) and confirm each
     entity's surface form — ignoring any parenthetical disambiguator you added
     — occurs in the text of at least one of its provenance chunks. A miss is
     either a hallucination or a wrong id.

   Fix what they flag and re-run them before moving on.

6. **Validate and self-correct.** Call `validate_conformance` with your entities,
   relations, and the ontology. If the score is below 1.0, read the returned
   errors, fix the offending entities/relations, and re-run the check. Stop after
   a couple of passes even if not perfect — return your best result and note any
   remaining violations.

7. **Produce the final graph.** Call `serialize_graph` to build the canonical
   knowledge graph (with URI reconciliation and literal detection). Then
   optionally:
   - `suggest_clusters` + `apply_clusters` to merge near-duplicate entities
     (see below).
   - `convert_ontology` to emit any newly discovered types/predicates as Turtle.
   - `validate_graph_schema` if you want to double-check the final payload.

## Deduplication: propose, review, apply

Deduplication is two tools, not one, because embeddings alone should not decide
what gets merged — **you** review every proposed cluster before anything changes.

1. **`suggest_clusters`** — proposes candidate duplicate clusters for
   **entities only**, using local sentence embeddings (no LLM). Predicates are
   not clustered here: they already come from the ontology's controlled
   vocabulary (via `suggest_predicates`), so they're canonical by construction
   and don't need semantic deduplication. Nothing is merged yet; it returns
   `entity_clusters`, a list of `{members, representative}`.
2. **You review the clusters.** For each proposed cluster:
   - Does every member really mean the same thing (e.g. "USA" and "United
     States")? Drop the ones that don't belong.
   - If a cluster lumps together distinct things, split it into more clusters.
   - Is `representative` the best canonical surface form? Swap it for a
     different member, or a new label, if not.
   - It's fine to discard a cluster entirely (pass it through unchanged, or
     just omit it) if you disagree with the proposal.
3. **`apply_clusters`** — takes your reviewed `entity_clusters` (plus the
   original `typed_entities`/`relations`) and merges them into the canonical
   graph, aggregating provenance. It also accepts `edge_clusters` if you want
   to manually merge specific predicates yourself; pass an empty list for
   either kind to skip that merge.

Tune `suggest_clusters` with:

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `entity_similarity_threshold` | `0.8` | Cosine similarity above which two **entities** are proposed as a cluster. Higher = stricter (fewer candidates); lower = more aggressive. |
| `retrieval_model` | `sentence-transformers/all-MiniLM-L6-v2` | The embedding model. `null` falls back to the library's built-in encoder. |

Guidance: start with the defaults. If distinct things keep showing up as
candidates, **raise** the threshold; if obvious duplicates aren't proposed at
all, **lower** it — but the threshold only affects what gets *proposed*, you
still decide what actually merges.

When a run's entity count has to be comparable with one from the batch HTTP
service, match that run's parameters rather than these defaults: pass its
`retrieval_model` and set `entity_similarity_threshold` to its
`entity_threshold`. Entity counts are only comparable after the same merge
policy has been applied to both.

At corpus scale, pass `output_file` to `suggest_clusters`. A few thousand
clusters is more than one context can review, and the proposal is the one large
result that cannot be summarised away — reviewing it *is* the step. Written to
disk it can be read in slices and reviewed by several agents in parallel, each
writing back the clusters it approved; concatenate those and hand the result to
`apply_clusters` (which also takes a path or blob handle).

## Large inputs: use file paths, not inline data

`validate_conformance`, `serialize_graph`, `suggest_clusters`, and
`apply_clusters` all accept `typed_entities`/`relations`/`entity_clusters`/
`edge_clusters` either as the list directly, or as a **string that's a path to
a JSON file** containing that list — the tool reads the file itself instead of
requiring you to retype the data as tool-call output. `serialize_graph` and
`apply_clusters` also accept an `output_file` path: if given, the merged/
serialized graph is written there instead of being returned inline, and you
get back a small summary (`num_entities`/`num_relations`/`output_file`) instead
of the full graph.

**Write the list with a generator script, not as literal JSON.** Whatever you
type is charged to your output tokens, and JSON spends most of them on syntax.
A script with one `add("<surface form>", "<type label>", ["<chunk id>", ...])`
call per entity, dumping the JSON at the end, measured 70,086 bytes against
102,805 for the same 610 entities as JSON — about 8,000 output tokens saved on
a single 200-document slice.

**Read the source text once.** Re-printing chunks to re-examine them is the
most expensive habit available: a 200-document slice is ~22k tokens, so one
extra dump of it costs more than the entire extraction it was meant to check.
Scroll back to what you already read instead.

**Use file paths once a list would run into the hundreds of items** (a long
document, or especially merging across several documents at once). Retyping
a large `typed_entities`/`relations` list as literal tool-call output is not
just slow and expensive — it can silently or loudly fail outright. On a real
multi-document merge, an `apply_clusters` call carrying ~300 relations inline
exceeded the output-token limit; a stronger model hit a hard API error trying
to route around it via a subagent, while a weaker model just quietly
submitted an empty relations list instead. Neither is a model problem — it's
a hard ceiling no amount of retyping avoids.

**A path only works when the server shares your filesystem.** It is resolved
by the server, so with a local server (`http://localhost:...`) any path you can
write works, while a **remote** server answers `No such file or directory` {EM}
and it is tempting to conclude, wrongly, that the optimization does not apply
to it.

**Use a blob handle instead; it works everywhere.** Upload the data with an
ordinary HTTP request and pass back the handle:

```bash
curl -sF file=@entities.json http://HOST/api/blobs     # {EM}> {"blob":"blob:f3c2...","bytes":126855}
curl -s --data-binary @entities.json http://HOST/api/blobs
```

Then pass `blob:f3c2...` wherever the list would have gone. The bytes travel
disk {EM}> network {EM}> server, so they never enter your context, and the handle
is about twenty characters. Handles are content-addressed, so re-uploading the
same data is free, and they expire after a few hours {EM} if one has gone, the
error says so and you re-upload.

**`ontology_ttl` takes a handle too**, and this is the larger saving: the server
is stateless, so the Turtle rides along on *every* call that needs it. Upload
the ontology once at the start and refer to the handle for the rest of the job.

Measured on a 200-document run, passing the entity list, the relation list and
the ontology all by handle:

| | inline | by handle |
|---|---|---|
| argument payload | ~41,500 tokens | **~16 tokens** |

MCP itself has no upload primitive {EM} tools are the only channel from you to
the server, and their arguments are your output tokens; resources, prompts and
sampling all run the other way, and `roots` shares path names, not contents.
`POST /api/blobs` exists precisely to give that missing direction a home.

Practical pattern for a large job: write each stage's output to a file with
the Write tool (or let `output_file` do it for you), then pass that path into
the next tool call rather than the data itself. For a single normal-sized
document this doesn't matter and inline data is simpler — reach for file
paths specifically when a list is large enough that reproducing it verbatim
would be a real amount of your own output.

## The other half of the cost: the ontology on every call

File paths fix the data you send *up*. The ontology and the tool results are
the rest of the bill, and the server being stateless means the Turtle rides
along on every call that needs it. Measured on an 18-class/17-predicate
ontology: the Turtle is ~4,700 tokens per call and `list_target_types` returns
~2,200. Against a 90-token document that scaffolding is the entire cost.
Upload the ontology once as a blob (above) and the per-call charge for it
disappears.

**Batch `suggest_predicates` — do not aggregate the types.** Its filtering
depends only on the type set you pass, never on the text, which makes it
tempting to call it once for the union of every type in the corpus and reuse
that. Don't: the answer is not identical, it is *unfiltered*. Over enough text
the union of types is the whole ontology, so the union's answer is every
predicate, and the narrowing the tool exists to provide is gone.

Use `suggest_predicates_batch` instead, keyed per chunk. Two collapses make one
call enough: chunks sharing a type signature share an answer, and distinct
signatures still land on far fewer distinct answers — measured on a 200-chunk
MuSiQue slice, 77 distinct type signatures collapsed onto 18 distinct predicate
sets. One batch call costs ~4,600 tokens against ~74,300 for 77 singular calls
(93.7% less) and returns the same per-chunk answers. Pass the mapping itself by
file path or blob handle, since on a real slice it is the bulky argument.

**Read the ontology file directly instead of calling `list_target_types`**, when
you have the file. You need the class descriptions in context either way to type
entities well; the raw Turtle carries them once, whereas the tool result repeats
them inside every predicate. Use `parse_ontology`/`list_target_types` when the
ontology arrives as text you have not read, or to confirm it parses.

## Extraction pitfalls

Each of these cost a real conformance failure or a wrong triple on a
200-document run.

**Disambiguate homonyms in the surface form.** Entities are keyed by surface
form throughout, so two different things sharing a name collapse into one and
take whichever type was assigned last. A corpus containing both the Antarctic
feature "Labyrinth" and the 1984 video game "Labyrinth" produced
`Labyrinth -[publisher]-> Acornsoft: Subject type Place not in domain
['Creative Work']`. Give one a distinguishing surface form
(`Labyrinth (Wright Valley)`) rather than dropping the relation.

**Type entities faithfully, even when it blocks a predicate.** Where the text
says someone "established the Kingdom of Saudi Arabia", `founder` does not
apply: its domain is Organization and a kingdom is a Country. Retyping the
country as an Organization to make the predicate fit yields a conforming graph
that misdescribes the world. Leave the relation out and let the gap show — an
ontology that cannot express a common fact is a finding worth reporting, not an
obstacle to route around.

**Do not assert what the text denies.** Sources correct themselves: "it is
sometimes asserted that Umm Ubays was the daughter of Al-Nahdiah ... however
Ibn Ishaq makes it clear that [they] were two different people" states no
parent relation. Extract the conclusion, not the claim being refuted.

## Notes

- Prefer types and predicates from the provided ontology. When you must go beyond
  it, the extra types/predicates appear as ontology *extensions* in the output —
  that's expected, not an error.
- This interactive path buys conformance, not coverage. On the first 200
  documents of a MuSiQue corpus it reached a conformance score of 1.0 with no
  invented types or predicates, but extracted 0.43 relations per document —
  against 0.59 for the same slice through the batch HTTP service, and 0.40 for a
  strong hosted model through that service. Reach for it when correctness per
  document matters; for bulk corpora, use the batch service.
