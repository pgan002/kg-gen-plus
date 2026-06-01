# KGGen: Knowledge Graph Extraction Approach

## 1. Executive Summary

**KGGen** is a high-performance framework designed to extract structured Knowledge Graphs (KG) from unstructured text files. By leveraging asynchronous processing and large language models (LLMs), it transforms raw text into typed entities and relations, supporting both cloud-based APIs and locally deployed models.

KGGen enables rapid, scalable knowledge extraction, designed for throughput. Its core strength lies in its ability to handle high-volume text ingestion while allowing for flexible model configuration, making it suitable for both enterprise-grade deployments and local development environments.

---

## 2. Core Functionality

KGGen provides a streamlined interface for Knowledge Graph construction.

### 2.1 Asynchronous Extraction
The system is built for speed, utilizing `asyncio` to process large corpora concurrently. By optimizing the ingestion of text chunks, KGGen maintains high throughput, significantly reducing the wall-clock time required to process large document sets.

### 2.2 LLM Flexibility & Deployment
KGGen is provider-agnostic. Whether utilizing proprietary models via API (e.g., GPT-4o) or running open-weights models locally via **Ollama** or **vLLM**, the architecture remains consistent. This allows users to balance performance, cost, and data privacy requirements by swapping the underlying engine without changing the core logic.

### 2.3 Knowledge Graph Lifecycle
The framework supports the full KG lifecycle, including:
*   **Entity & Relation Extraction:** Automated identification of typed entities and relations from text.
*   **Type Resolution:** Mapping identified terms to specific types to ensure structural consistency.
*   **Flexible Aggregation & Deduplication Ordering:** Independently processed graphs can be merged and deduplicated in different orders depending on requirements. In practical, large-scale workflows, it is often optimal to **aggregate first** and then run a single, unified global deduplication pass.

---

## 3. Operational Methodology & A-Posteriori Deduplication

KGGen follows a structured workflow designed to turn raw text into a queryable graph:

1.  **Extraction:** The system processes text input to extract entities and relations. It utilizes context-aware prompting to ensure that the extraction aligns with domain-specific requirements or predefined ontologies.
2.  **Enrichment:** Extracted entities are typed and mapped, allowing for rich semantic relationships rather than simple, flat triples.
3.  **Aggregation:** Merges multiple subgraphs together while preserving provenance and merging identical nodes/edges.
4.  **Deduplication & Clustering:** Using local embedding models (e.g., `all-mpnet-base-v2` or `all-MiniLM-L6-v2`), KGGen computes semantic similarity across nodes and edges, collapsing near-duplicates and synonyms to refine the graph's topology.

### 3.1 A-Posteriori Deduplication: Trade-offs
Unlike systems that incrementally pass a growing, stateful partial Knowledge Graph back into the LLM context during the extraction of subsequent text chunks, KGGen performs deduplication **a-posteriori** on already constructed graphs.

*   **Disadvantages:**
    *   *Lack of Incremental Context:* Because chunks are processed completely independently, the extraction step cannot benefit from the context of already extracted entities. The LLM cannot be explicitly guided to reuse an exact surface form from previous chunks, which can result in more synonym variation that must be resolved later.
*   **Advantages:**
    *   *Full Parallelization:* Eliminating stateful, sequential dependencies between chunks allows the pipeline to process all text chunks concurrently, maximizing the speed of async runtimes and local inference clusters (vLLM).
    *   *Infinite Scale:* Since a growing partial graph is never injected into the LLM context window during extraction, the system can process arbitrary-sized text corpora without running into context window limits.
    *   *Massive Token Savings:* Not pushing a cumulative, partial KG into the prompt for each new chunk prevents quadratic growth in token usage, dramatically decreasing both inference latency and API costs.

### 3.2 Key Features for Scaling
*   **Token Usage Tracking:** The system monitors token consumption per step, ensuring observability during high-volume processing.
*   **Context Management:** Allows injection of entity and relation context, improving accuracy when extracting from technical or niche domains.
*   **Provenance Preservation:** Merges and retains source IDs and context through both aggregation and deduplication, ensuring every edge can be traced back to its origin text.

---

## 4. Technical Dependencies

The KGGen framework relies on a robust set of open-source libraries to coordinate LLM tasks, handle graph operations, and perform vector similarity operations:

*   **LLM Orchestration (`dspy` & `langchain-core`):** Powers programmatic, structured, and retry-aware interactions with various language models.
*   **Graph Manipulation (`networkx`):** Serves as the core data structure to represent, merge, traverse, and export the extracted knowledge graphs.
*   **Semantic Embeddings (`sentence-transformers`):** Used to generate high-quality vector representations of entities and relationships for semantic similarity clustering.
*   **Data Validation (`pydantic`):** Enforces strict data models, typed constraints, and schemas for entities, relations, graphs, and stats.
*   **Scientific Computing (`numpy`, `scikit-learn`):** Drives vector computation, cosine similarity, and thresholding during the clustering phase of deduplication.
*   **Optional MCP Support (`fastmcp` & `mcp`):** Enables seamless deployment of the tool suite as Model Context Protocol servers to integrate with MCP-compatible IDE clients or agents.

---

## 5. Pros and Cons of the Approach

### Pros
*   **Exceptional Throughput:** The combination of non-blocking `asyncio` execution and a-posteriori deduplication allows the pipeline to ingest massive amounts of text in parallel.
*   **Cost & Privacy Control:** Native compatibility with open-weights models run locally via **vLLM** or **Ollama** eliminates per-token API costs and keeps sensitive data on-premises.
*   **Ontology Grounding:** Support for predefined entity types and domain-range predicate constraints ensures that generated graphs remain structured and meaningful.
*   **Tracing and Auditability:** Complete provenance tracking allows users to inspect exactly which source chunk generated any node or relationship.

### Cons
*   **Post-processing Complexity:** Performing deduplication after extraction relies heavily on setting appropriate semantic thresholds; mismatched thresholds can lead to either concepts merging incorrectly or remaining overly fragmented.
*   **Heavy Local Compute Requirements:** Local deployment demands significant GPU memory to host both the extraction LLM and the local embedding models.
*   **Sensitivity to Model Capabilities:** Smaller open-weights models may struggle with structural prompt schemas, necessitating more robust reasoning models for complex, multi-typed domains.

---

## 6. Technical Flexibility

The design of KGGen emphasizes decoupling:

*   **Configuration:** Parameters such as models, temperatures, and similarity thresholds are configurable, supporting diverse deployment environments.
*   **Extensibility:** The modular `generate` method allows for the insertion of custom logic at each stage—entity extraction, typing, and relation extraction—without modifying the underlying `KGGen` core.
