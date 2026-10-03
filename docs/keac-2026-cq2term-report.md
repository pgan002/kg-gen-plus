# KGGen+ for CQ2Term: Domain Graph Construction and Canonical Term Reassignment

**Philip Ganchev**  
Graphwise  
philip.ganchev@graphwise.ai

## Abstract

We describe the Graphwise submission to the Knowledge Engineering Automation Challenge 2026 CQ2Term sub-task. The system uses KGGen+ to construct a domain knowledge graph from all competency questions, rather than predicting an isolated term list from each question. Entities and relations are converted into candidate ontology classes and properties while retaining question provenance. String graph deduplication, lexical and morphological label normalization, and conservative semantic clustering produce a canonical domain vocabulary. A final semantic reassignment stage maps canonical terms back to the questions that require them. The submitted configuration uses the open 25.2B-parameter Gemma 4 26B A4B model and adds at most five terms of each kind per question at a cosine threshold of 0.35. Across six repeated public-data runs, reassignment leaves global term precision, recall, and F1 unchanged while improving mean question-conditioned coverage from 0.5901 to 0.7087. The submitted artifact is the run closest to the six-run mean. Source code, exact commands, and version-pinned documentation are public.

**Keywords:** competency questions; ontology engineering; knowledge graphs; large language models; term extraction; semantic clustering

## 1. Introduction

Competency questions (CQs) describe the information that an ontology should represent and answer. The CQ2Term task isolates an early ontology-engineering step: identifying the explicit classes and properties required by each CQ. A useful system must recover the domain vocabulary while also assigning each term to the correct question. The latter requirement is important because a term may be present in the aggregate domain output but absent from the CQ that needs it.

Our submission builds on KGGen+, an ontology-guided extension of KGGen. Instead of treating each CQ as an independent zero-shot term-prediction prompt, we first construct a graph over the complete CQ set of a domain. This creates a shared vocabulary and preserves evidence linking entities and relations to source questions. We then normalize and semantically consolidate that vocabulary before reassigning canonical terms to individual questions.

The submitted configuration was selected according to the CQ2Term leaderboard's question-conditioned mean coverage criterion. It intentionally does not include later experimental precision filters or predicate-induction stages, because those improved global F1 while reducing question-conditioned coverage.

## 2. Method

### 2.1 Domain-wide graph construction

For each of the six domains, all public competency questions are passed to KGGen+ as separately identified input documents. The language model extracts typed entities and subject--predicate--object relations from every CQ. Each extracted item carries the identifier of the source question. Per-question extraction is parallelized, after which the partial graphs are aggregated into one domain graph.

The generation prompt treats common nouns, roles, states, events, and kinds as candidate ontology concepts. Bracketed placeholders are interpreted as references to classes rather than named individuals. Relations required to represent the CQ are extracted as candidate properties. Generic fallback predicates are discouraged.

The submitted graphs use deterministic string-based graph deduplication after aggregation. Embedding-based graph deduplication is disabled because separate public-data experiments showed that it could remove true terms and reduce recall.

### 2.2 Class and property projection

CQ2Term classes are projected from entity surface forms. Properties are projected from relation predicates. Provenance identifiers determine the initial CQ-to-term assignments. Class and property vocabularies remain separate throughout export.

The exporter applies Unicode and separator normalization, CamelCase splitting, whitespace normalization, conservative class singularization, and first-verb lemmatization for properties. Examples include `PlantParts` to `plant part` and `uses platform` to `use platform`.

### 2.3 Semantic clustering

Normalized class labels and property labels are clustered separately with `sentence-transformers/all-MiniLM-L6-v2`. Complete-link clustering at cosine similarity 0.9 prevents a term from joining a cluster unless it is sufficiently similar to every cluster member. Additional guards preserve negation and prevent broader/narrower pairs such as `plant` and `plant part` from being merged. The most frequent concise cluster member becomes the canonical label.

Canonicalization is provenance-preserving. Runtime invariants assert that every CQ that contributed an alias still receives the resulting canonical term and that no unsupported CQ assignment is introduced.

### 2.4 CQ-to-canonical-term reassignment

Extraction and clustering preserve explicit provenance, but CQs containing pronouns, ellipsis, or alternative wording may fail to receive a term already discovered elsewhere in the domain. We therefore perform bounded semantic reassignment.

For each CQ and term kind, the system embeds a query containing the CQ text and its current assignments. Candidate terms are restricted to the canonical vocabulary already extracted from that domain; reassignment cannot invent a new term. Candidates with cosine similarity at least 0.35 are ranked, and at most five additional classes and five additional properties are added to the CQ.

This stage changes CQ-local attribution but not the set of unique domain terms. It is therefore expected to improve CQ-conditioned coverage without changing global term-level precision, recall, or F1.

## 3. Experimental Setup

The base model is `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, an Apache-2.0-licensed quantization of Gemma 4 26B A4B. The model has 25.2B total and 3.8B active parameters. It was served through vLLM under the OpenAI-compatible model name `gemma4`.

Generation used temperature 0.7, a 4096-token completion limit, ten parallel CQ documents, and six repeated runs. Each repeated run processed all six domains: AWO, ODRL, SWO, VGO, Water, and Wine. Evaluation used the unmodified CQ4OE evaluator distributed for the challenge.

The main ablation varied CQ reassignment:

- no reassignment;
- threshold 0.55 with at most two additions (`t055-k2`);
- threshold 0.45 with at most three additions (`t045-k3`);
- threshold 0.35 with at most five additions (`t035-k5`).

The submitted `t035-k5` configuration had the highest public-data CQ-Mean. Among its six repeated runs, run 03 had CQ-Mean 0.7079, closest to the six-run mean of 0.7087, and was selected as a representative run rather than by maximum score.

## 4. Results

Table 1 reports six-run macro means for the submitted configuration. Class and property F1 are global term-recovery metrics; CQ-Any, CQ-Mean, and CQ-Full evaluate whether required terms occur under the correct question.

| Domain | Class F1 | Property F1 | CQ-Any | CQ-Mean | CQ-Full |
|---|---:|---:|---:|---:|---:|
| AWO | 0.9139 | 0.6667 | 1.0000 | 1.0000 | 1.0000 |
| ODRL | 0.4722 | 0.4252 | 0.6316 | 0.3952 | 0.2632 |
| SWO | 0.4538 | 0.5257 | 0.9679 | 0.6191 | 0.2564 |
| VGO | 0.4159 | 0.4989 | 0.9545 | 0.8030 | 0.5682 |
| Water | 0.5613 | 0.7346 | 1.0000 | 0.6879 | 0.1083 |
| Wine | 0.7871 | 0.3849 | 1.0000 | 0.7472 | 0.1000 |
| **Macro mean** | **0.6007** | **0.5393** | **0.9257** | **0.7087** | **0.3827** |

Without reassignment, the corresponding CQ-Any, CQ-Mean, and CQ-Full values were 0.8813, 0.5901, and 0.2749. Global class and property metrics were unchanged because reassignment only redistributes an existing canonical vocabulary. Relative to no reassignment, `t035-k5` improved CQ-Any by 0.0444, CQ-Mean by 0.1187, and CQ-Full by 0.1077. Each coverage improvement had the same direction in all six repeated runs; the exact two-sided sign-flip p-value was 0.03125, the minimum non-zero value available with six pairs.

The threshold ablation was monotonic. CQ-Mean increased from 0.6631 for `t055-k2`, to 0.6821 for `t045-k3`, and to 0.7087 for `t035-k5`. CQ-Full similarly increased from 0.3310 to 0.3559 and 0.3827. This indicates that the more permissive policy recovered relevant globally known terms without affecting the global vocabulary.

## 5. Error Analysis and Limitations

Performance differs substantially by domain. AWO has a small, lexically direct vocabulary, while ODRL and SWO contain many ontology-specific property names and implicit references. VGO achieves high question coverage but low class precision because many plausible surface concepts are extracted beyond the reference vocabulary. Water has the strongest property F1 but low CQ-Full, indicating that properties are often discovered globally yet incompletely assigned within individual questions.

Reassignment cannot recover a term that KGGen+ failed to extract anywhere in the domain. It also cannot resolve a natural-language relation to an ontology-specific identifier when no equivalent candidate exists. Conversely, a low threshold may assign plausible but unnecessary terms to additional questions. The cap limits this risk, but the current selection emphasizes CQ-Mean rather than global F1.

Generation is stochastic. Temperature 0.7 was used to study repeated-run variation, and byte-identical reproduction is not expected without reproducing the complete model-serving environment and random-number state. The submitted run was selected by closeness to the condition mean to reduce public-data cherry-picking.

Finally, the method constructs a domain graph from the complete CQ set. This is appropriate for shared ontology conceptualization but means that CQ outputs are not independent. The approach intentionally exploits cross-question domain context.

## 6. Reproducibility

KGGen+ source code is available under the MIT license at commit `a7f0965`:

- https://github.com/pgan002/kg-gen-plus/tree/a7f0965

Exact installation, vLLM serving, generation, export, evaluation, and packaging commands are provided at:

- https://github.com/pgan002/kg-gen-plus/blob/d5dd9df/docs/keac-2026-cq2term.md

The submitted model is publicly available under Apache 2.0:

- https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit

The challenge submission and result files are available in pull request 8:

- https://codeberg.org/ke-automation-challenge/challenge-catalog/pulls/8

## 7. Conclusion

KGGen+ frames CQ2Term as domain vocabulary construction followed by provenance-aware question assignment. Canonical CQ reassignment substantially improves question-conditioned coverage without altering global term-recovery metrics. The strongest tested reassignment policy increases CQ-Mean by almost twelve percentage points over the same extraction pipeline without reassignment. Future work should improve ontology-specific predicate canonicalization and conservative class selection while preserving the coverage gains of domain-wide reassignment.

## References

1. B. Mo et al. KGGen: Extracting knowledge graphs from plain text with language models. Source repository: https://github.com/stair-lab/kg-gen.
2. OEG-UPM. CQ4OE: A benchmark for assessing LLM-assisted ontology generation from competency questions. https://github.com/oeg-upm/cq4oe-benchmark.
3. Knowledge Engineering Automation Challenge 2026. https://ke-automation-challenge.codeberg.page/.
4. N. Reimers and I. Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP*, 2019.
5. Google DeepMind. Gemma model family. https://ai.google.dev/gemma.
