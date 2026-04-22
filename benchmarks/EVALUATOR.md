# Knowledge Graph Evaluator

A standalone Python module for evaluating generated Knowledge Graphs (KGs) against gold-standard (reference) graphs.

The evaluator computes standard metrics (Precision, Recall, and F1-score) by comparing triples. It supports both **exact URI matching** and **fuzzy text matching** to handle minor variations in entity and relation surface forms.

## Dependencies

This module requires `rapidfuzz` for fast fuzzy string matching:

```bash
pip install rapidfuzz
```

## Data Structures

To keep the evaluator decoupled from any specific Knowledge Graph generation pipeline, it uses its own set of standard `dataclasses`. You need to map your generated and gold-standard data into these structures before evaluation.

### 1. `EvalEntity`
Represents a node (Subject/Object) or an edge (Predicate) in the graph.
- `surface_form` (str): The raw text representation of the entity or relation (e.g., "Barack Obama", "born in").
- `uri` (Optional[str]): An optional unique identifier (e.g., "Q43274", "P19"). **If provided for predicates, the evaluator will strictly require URIs to match.**

### 2. `EvalTriple`
Represents a single fact in the graph.
- `subject` (EvalEntity)
- `predicate` (EvalEntity)
- `object` (EvalEntity)

### 3. `EvalGraph`
Represents an entire Knowledge Graph.
- `triples` (List[EvalTriple]): A list of `EvalTriple` objects.

---

## How Matching Works

1. **Text Normalization**: Before comparison, all `surface_form` strings are normalized. HTML entities are unescaped, URLs are unquoted, underscores are replaced with spaces, and Unicode is standardized to NFKC. Everything is lowercased and stripped.
2. **Predicate Evaluation**:
    - **By URI**: If both the generated and gold predicates contain a `uri`, they **must** match exactly.
    - **By Text (Fuzzy)**: If URIs are missing, predicates are compared using fuzzy string matching via `rapidfuzz`.
    - **Class/Type Fallback**: Special predicates (`instance of`, `is a`, `type`, `has type`, `class`) are treated as identical to one another.
3. **Subject/Object Evaluation**: Subjects and Objects are compared using fuzzy string matching.
4. **Thresholding**: For a generated triple to "match" a gold triple, the Subject, Predicate, and Object similarity scores must *all* meet or exceed the `match_threshold` (default is `90.0` out of 100).

---

## Usage Example

Here is a complete example demonstrating how to structure your results and run the evaluator.

```python
from evaluator import EvalEntity, EvalTriple, EvalGraph, GraphEvaluator

# 1. Structure your Gold Standard Data
gold_triples = [
    EvalTriple(
        subject=EvalEntity(surface_form="Albert Einstein"),
        predicate=EvalEntity(surface_form="born in", uri="P19"),
        object=EvalEntity(surface_form="Ulm")
    ),
    EvalTriple(
        subject=EvalEntity(surface_form="Albert Einstein"),
        predicate=EvalEntity(surface_form="is a", uri="P31"),
        object=EvalEntity(surface_form="Physicist")
    )
]
gold_graph = EvalGraph(triples=gold_triples)

# 2. Structure your Generated Data
gen_triples = [
    # This will match (Fuzzy text match >= 90.0 & exact URI match)
    EvalTriple(
        subject=EvalEntity(surface_form="Albert_Einstein"),
        predicate=EvalEntity(surface_form="place of birth", uri="P19"),
        object=EvalEntity(surface_form="Ulm, Germany")
    ),
    # This will match (Fallback text logic for "type" == "is a")
    EvalTriple(
        subject=EvalEntity(surface_form="Albert Einstein"),
        predicate=EvalEntity(surface_form="type"),
        object=EvalEntity(surface_form="Physicist")
    ),
    # This is a false positive
    EvalTriple(
        subject=EvalEntity(surface_form="Albert Einstein"),
        predicate=EvalEntity(surface_form="studied at"),
        object=EvalEntity(surface_form="University of Zurich")
    )
]
generated_graph = EvalGraph(triples=gen_triples)

# 3. Initialize the Evaluator
# match_threshold defines the strictness of the fuzzy text match (0-100)
evaluator = GraphEvaluator(match_threshold=90.0)

# 4. Evaluate (You would typically do this in a loop over your dataset)
per_graph_metrics = evaluator.evaluate(generated_graph, gold_graph)

print(f"Graph F1 Score: {per_graph_metrics['f1_score']:.2f}")
print(f"True Positives: {per_graph_metrics['true_positives']}")
print(f"False Positives: {per_graph_metrics['false_positives']}")

# 5. Get Final Aggregated Results
aggregated_metrics = evaluator.get_aggregated_results()

print("\n--- Final Aggregated Dataset Metrics ---")
print(f"Micro F1: {aggregated_metrics['micro_f1']:.4f}")
print(f"Macro F1: {aggregated_metrics['macro_f1']:.4f}")
```

---

## Outputs and Metrics

### Per-Graph Metrics (`evaluate`)
Calling `evaluate()` returns a dictionary for that specific graph pair containing:
- `precision`, `recall`, `f1_score`
- `true_positives`, `false_positives`, `false_negatives` (integer counts)
- `true_positive_triples` (List of matched triples)
- `false_positive_triples` (List of hallucinated/incorrect generated triples)
- `false_negative_triples` (List of missed gold triples)

### Aggregated Metrics (`get_aggregated_results`)
After evaluating a stream or list of graphs, calling `get_aggregated_results()` returns:

- **Micro Metrics** (`micro_precision`, `micro_recall`, `micro_f1`):
  Calculated globally. It sums up all True Positives, False Positives, and False Negatives across the entire dataset *first*, and then calculates the metrics. Better for understanding overall token/triple-level performance.

- **Macro Metrics** (`macro_precision`, `macro_recall`, `macro_f1`):
  Calculated locally. It calculates the Precision/Recall/F1 for each individual graph, and then averages those scores. Better for understanding performance on an "average document" level.

- **Totals**:
  `total_graphs_evaluated`, `total_true_positives`, `total_false_positives`, `total_false_negatives`.
