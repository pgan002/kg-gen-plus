import json
import os
import time
from datetime import datetime
import requests
from pathlib import Path

from benchmarks.ga_pubmed.pubmed_config import (
    KG_GEN_URL,
    KG_GENERATION_PARAMS,
    KG_GENERATION_HEADERS,
    corpus_path,
    ontology_path,
    output_dir,
    LIMIT,
)


def main():
    # Merge params
    params = KG_GENERATION_PARAMS.copy()
    headers = KG_GENERATION_HEADERS.copy()

    print(f"Connecting to {KG_GEN_URL}...")
    print(f"Parameters: {json.dumps(params, indent=2)}")

    if not corpus_path.exists():
        print(f"Error: Corpus file not found at {corpus_path}")
        return

    if not ontology_path.exists():
        print(f"Error: Ontology file not found at {ontology_path}")
        return

    # Load corpus
    corpus_items = []
    with open(corpus_path, "r") as f:
        for line in f:
            corpus_items.append(json.loads(line))
            if LIMIT and len(corpus_items) >= LIMIT:
                break

    print(f"Loaded {len(corpus_items)} documents.")

    # Create intermediate corpus file with requested number of docs
    os.makedirs(output_dir, exist_ok=True)
    temp_corpus_path = output_dir / f"intermediate_corpus_{len(corpus_items)}.jsonl"
    with open(temp_corpus_path, "w") as f:
        for item in corpus_items:
            f.write(json.dumps(item) + "\n")

    print(f"Created intermediate corpus at {temp_corpus_path}")

    results = []
    print("Processing entire corpus in one request...")
    res = call_api(KG_GEN_URL, params, temp_corpus_path, ontology_path, headers=headers)
    if res:
        results.append(res)

    # Cleanup intermediate file
    if temp_corpus_path.exists():
        os.remove(temp_corpus_path)

    if not results:
        print("No results obtained.")
        return

    # Aggregate metrics
    summary = aggregate_results(results, params, output_dir)
    print_summary(summary)


def call_api(url, params, corpus_path, ontology_path, headers):
    corpus_path = Path(corpus_path)
    total_chars = 0
    with open(corpus_path, "r") as f:
        for line in f:
            total_chars += len(line)

    with open(corpus_path, "rb") as corpus_f, open(ontology_path, "rb") as onto_f:
        files = {
            "corpus_file": (corpus_path.name, corpus_f, "application/jsonl"),
            "ontology_file": (ontology_path.name, onto_f, "text/turtle"),
        }

        start_time = time.time()
        try:
            response = requests.post(url, params=params, files=files)
            total_time = time.time() - start_time
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to the service at {url}.")
            return None

    if response.status_code != 200:
        print(f"Request failed with status {response.status_code}")
        print(response.text)
        return None

    result_graph = response.json()
    headers = response.headers

    stats = {}
    if "X-KG-Gen-Stats" in headers:
        try:
            stats = json.loads(headers["X-KG-Gen-Stats"])
        except json.JSONDecodeError:
            pass

    dedup_stats = {}
    if "X-KG-Gen-Dedup-Stats" in headers:
        try:
            dedup_stats = json.loads(headers["X-KG-Gen-Dedup-Stats"])
        except json.JSONDecodeError:
            pass

    gen_time = float(headers.get("X-KG-Gen-Time", 0))
    input_tokens = int(headers.get("X-KG-Gen-Input-Tokens", 0))
    output_tokens = int(headers.get("X-KG-Gen-Output-Tokens", 0))

    num_entities = len(result_graph.get("entities", {}) or {})
    num_relations = len(result_graph.get("relations", []) or [])

    return {
        "graph": result_graph,
        "metrics": {
            "wall_time": total_time,
            "api_time": gen_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "num_entities": num_entities,
            "num_relations": num_relations,
            "input_chars": total_chars,
        },
        "stats": stats,
        "dedup_stats": dedup_stats,
    }


def aggregate_results(results, params, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_metrics = {
        "wall_time": 0,
        "api_time": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "num_entities": 0,
        "num_relations": 0,
        "input_chars": 0,
    }

    all_stats = []

    for res in results:
        m = res["metrics"]
        for k in total_metrics:
            total_metrics[k] += m[k]
        all_stats.append(res["stats"])

    # Calculate throughputs
    if total_metrics["api_time"] > 0:
        total_metrics["tokens_per_sec_api"] = (
            total_metrics["total_tokens"] / total_metrics["api_time"]
        )

    if total_metrics["wall_time"] > 0:
        total_metrics["tokens_per_sec_wall"] = (
            total_metrics["total_tokens"] / total_metrics["wall_time"]
        )

    model_name = params.get("model", "unknown").replace("/", "_")
    base_name = f"pubmed_{model_name}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Save combined graph
    if len(results) == 1:
        combined_graph = results[0]["graph"]
    else:
        # Simple merging for multiple results if ever needed
        combined_graph = {
            "entities": {},
            "relations": [],
            "ontology_extensions": {"classes": [], "predicates": []},
        }
        entity_clusters = {}
        edge_clusters = {}

        for res in results:
            g = res["graph"]
            combined_graph["entities"].update(g.get("entities", {}))
            combined_graph["relations"].extend(g.get("relations", []))

            if "ontology_extensions" in g and g["ontology_extensions"]:
                combined_graph["ontology_extensions"]["classes"].extend(
                    g["ontology_extensions"].get("classes", [])
                )
                combined_graph["ontology_extensions"]["predicates"].extend(
                    g["ontology_extensions"].get("predicates", [])
                )

            if "clusters" in g and g["clusters"]:
                c = g["clusters"]
                if "entities" in c and c["entities"]:
                    entity_clusters.update(c["entities"])
                if "edges" in c and c["edges"]:
                    edge_clusters.update(c["edges"])

        if entity_clusters or edge_clusters:
            combined_graph["clusters"] = {}
            if entity_clusters:
                combined_graph["clusters"]["entities"] = entity_clusters
            if edge_clusters:
                combined_graph["clusters"]["edges"] = edge_clusters

        # Cleanup empty optional fields
        if "ontology_extensions" in combined_graph:
            if (
                not combined_graph["ontology_extensions"]["classes"]
                and not combined_graph["ontology_extensions"]["predicates"]
            ):
                del combined_graph["ontology_extensions"]

    graph_path = os.path.join(output_dir, f"{base_name}_graph.json")
    with open(graph_path, "w") as f:
        json.dump(combined_graph, f, indent=2)

    summary = {
        "timestamp": timestamp,
        "params": params,
        "total_metrics": total_metrics,
        "per_doc_results": [
            {"doc_id": r.get("doc_id"), "metrics": r["metrics"]} for r in results
        ]
        if len(results) > 1
        else None,
        "all_stats": all_stats,
    }

    results_path = os.path.join(output_dir, f"{base_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    summary["graph_path"] = graph_path
    summary["results_path"] = results_path
    return summary


def print_summary(summary):
    m = summary["total_metrics"]
    print("\nBenchmark Summary:")
    print(f"  Total Wall Time: {m['wall_time']:.2f}s")
    print(f"  Total API Time: {m['api_time']:.2f}s")
    print(f"  Total Input Chars: {m['input_chars']}")
    print(
        f"  Total Tokens: {m['input_tokens']} in / {m['output_tokens']} out (Total: {m['total_tokens']})"
    )
    print(
        f"  Total Graph: {m['num_entities']} entities, {m['num_relations']} relations"
    )
    print(f"  Throughput (API): {m.get('tokens_per_sec_api', 0):.2f} tokens/s")
    print(f"  Throughput (Wall): {m.get('tokens_per_sec_wall', 0):.2f} tokens/s")

    # If multiple docs, print averages
    if summary["per_doc_results"]:
        num_docs = len(summary["per_doc_results"])
        print(f"\nAverages per document ({num_docs} docs):")
        print(f"  Avg Wall Time: {m['wall_time'] / num_docs:.2f}s")
        print(f"  Avg Tokens: {m['total_tokens'] / num_docs:.2f}")
        print(f"  Avg Entities: {m['num_entities'] / num_docs:.2f}")

    print(f"\n  Results saved to: {summary['results_path']}")


if __name__ == "__main__":
    main()
