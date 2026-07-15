import json
import os
import time
from datetime import datetime
import requests

try:
    import rdflib
    from rdflib import RDF, RDFS, OWL

    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False

# --- Configuration: modify these values directly ---
CONFIG = {
    "base_url": "http://dsx-gws-rai-docker-dmo-apl-n-01:8087",
    "corpus": "benchmarks/musk_example/data/test_corpus.jsonl",
    "ontology": "benchmarks/musk_example/data/test_ontology.ttl",
    "model": "openai/gpt-5.4-mini",
    "temperature": None,  # Set to a float value if needed
    "n_parallel": 10,
    # Conformance flags
    "enforce_type": True,
    "enforce_domain": True,
    "enforce_range": True,
    "enforce_predicate": True,
    "deduplicate": True,
    "output_dir": "benchmarks/musk_example/data/results",
    "entity_threshold": 0.9,
    "predicate_threshold": 0.9,
    "retrieval_model": "mixedbread-ai/mxbai-embed-large-v1",
}
# --------------------------------------------------


def main():
    url = f"{CONFIG['base_url']}/api/generate"

    # Parameters for the API call (query params)
    params = {
        "model": CONFIG["model"],
        "n_parallel": CONFIG["n_parallel"],
        "enforce_type_conformance": str(CONFIG["enforce_type"]).lower(),
        "enforce_domain_conformance": str(CONFIG["enforce_domain"]).lower(),
        "enforce_range_conformance": str(CONFIG["enforce_range"]).lower(),
        "enforce_predicate_conformance": str(CONFIG["enforce_predicate"]).lower(),
        "deduplicate": str(CONFIG["deduplicate"]).lower(),
        "entity_threshold": CONFIG["entity_threshold"],
        "predicate_threshold": CONFIG["predicate_threshold"],
        "retrieval_model": CONFIG["retrieval_model"],
    }
    if CONFIG["temperature"] is not None:
        params["temperature"] = CONFIG["temperature"]

    print(f"Connecting to {url}...")
    print(f"Parameters: {json.dumps(params, indent=2)}")

    corpus_path = CONFIG["corpus"]
    ontology_path = CONFIG["ontology"]

    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        # Try relative to script location if not found
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_corpus = os.path.join(script_dir, "data", "test_corpus.jsonl")
        if os.path.exists(alt_corpus):
            corpus_path = alt_corpus
            print(f"Found corpus at {corpus_path}")
        else:
            return

    if not os.path.exists(ontology_path):
        print(f"Error: Ontology file not found at {ontology_path}")
        # Try relative to script location if not found
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_ontology = os.path.join(script_dir, "data", "test_ontology.ttl")
        if os.path.exists(alt_ontology):
            ontology_path = alt_ontology
            print(f"Found ontology at {ontology_path}")
        else:
            return

    with open(corpus_path, "rb") as corpus_f, open(ontology_path, "rb") as onto_f:
        files = {
            "corpus_file": (
                os.path.basename(corpus_path),
                corpus_f,
                "application/jsonl",
            ),
            "ontology_file": (os.path.basename(ontology_path), onto_f, "text/turtle"),
        }

        start_time = time.time()
        try:
            response = requests.post(url, params=params, files=files)
            total_time = time.time() - start_time
        except requests.exceptions.ConnectionError:
            print(
                f"Error: Could not connect to the service at {CONFIG['base_url']}. Is it running?"
            )
            return

    if response.status_code != 200:
        print(f"Request failed with status {response.status_code}")
        print(response.text)
        return

    result_graph = response.json()
    headers = response.headers

    # Extract stats from headers
    stats = {}
    if "X-KG-Gen-Stats" in headers:
        try:
            stats = json.loads(headers["X-KG-Gen-Stats"])
        except json.JSONDecodeError:
            print("Warning: Could not parse X-KG-Gen-Stats header")

    dedup_stats = {}
    if "X-KG-Gen-Dedup-Stats" in headers:
        try:
            dedup_stats = json.loads(headers["X-KG-Gen-Dedup-Stats"])
        except json.JSONDecodeError:
            print("Warning: Could not parse X-KG-Gen-Dedup-Stats header")

    metrics = {
        "total_wall_time": total_time,
        "gen_time": float(headers.get("X-KG-Gen-Time", 0)),
        "input_tokens": int(headers.get("X-KG-Gen-Input-Tokens", 0)),
        "output_tokens": int(headers.get("X-KG-Gen-Output-Tokens", 0)),
        "num_entities": len(result_graph.get("typed_entities", []) or []),
        "num_relations": len(
            result_graph.get("relations_wo_class_assertions", []) or []
        ),
    }

    # Calculate derived metrics
    if metrics["gen_time"] > 0:
        metrics["tokens_per_sec"] = (
            metrics["input_tokens"] + metrics["output_tokens"]
        ) / metrics["gen_time"]
        metrics["relations_per_sec"] = metrics["num_relations"] / metrics["gen_time"]

    # Save results
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a unique filename based on params
    conf_str = f"t{int(CONFIG['enforce_type'])}d{int(CONFIG['enforce_domain'])}r{int(CONFIG['enforce_range'])}p{int(CONFIG['enforce_predicate'])}"
    base_name = f"musk_{CONFIG['model'].replace('/', '_')}_p{CONFIG['n_parallel']}_{conf_str}_{timestamp}"

    graph_path = os.path.join(CONFIG["output_dir"], f"{base_name}_graph.json")
    results_path = os.path.join(CONFIG["output_dir"], f"{base_name}_results.json")

    with open(graph_path, "w") as f:
        json.dump(result_graph, f, indent=2)

    summary = {
        "timestamp": timestamp,
        "config": CONFIG,
        "metrics": metrics,
        "stats": stats,
        "dedup_stats": dedup_stats,
    }

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBenchmark Summary:")
    print(f"  Total Wall Time: {metrics['total_wall_time']:.2f}s")
    print(f"  Generation Time (API): {metrics['gen_time']:.2f}s")
    print(f"  Tokens: {metrics['input_tokens']} in / {metrics['output_tokens']} out")
    print(
        f"  Graph: {metrics['num_entities']} entities, {metrics['num_relations']} relations"
    )

    if RDFLIB_AVAILABLE:
        analyze_conformance(result_graph, ontology_path)
    else:
        print("\nWarning: rdflib not installed. Skipping conformance analysis.")

    analyze_deduplication(result_graph, summary)

    print(f"\n  Results saved to: {CONFIG['output_dir']}")
    print(f"    Graph: {os.path.basename(graph_path)}")
    print(f"    Metrics: {os.path.basename(results_path)}")


def analyze_conformance(graph, ontology_path):
    print("\n--- Conformance Analysis ---")
    if not os.path.exists(ontology_path):
        print(f"  Ontology file not found at {ontology_path}")
        return

    g = rdflib.Graph()
    try:
        g.parse(ontology_path, format="turtle")
    except Exception as e:
        print(f"  Error parsing ontology: {e}")
        return

    classes = {str(c) for c in g.subjects(RDF.type, OWL.Class)}
    obj_props = {str(p) for p in g.subjects(RDF.type, OWL.ObjectProperty)}
    data_props = {str(p) for p in g.subjects(RDF.type, OWL.DatatypeProperty)}
    all_props = obj_props | data_props

    pred_info = {}
    for p in all_props:
        p_ref = rdflib.URIRef(p)
        domains = {str(d) for d in g.objects(p_ref, RDFS.domain)}
        ranges = {str(r) for r in g.objects(p_ref, RDFS.range)}
        pred_info[p] = {
            "domains": domains,
            "ranges": ranges,
            "is_data": p in data_props,
        }

    typed_entities = graph.get("typed_entities", [])
    total_entities = len(typed_entities)
    with_type = 0
    type_in_onto = 0
    type_not_in_onto = 0

    entity_types_map = {}  # surface_form -> set of type URIs

    for te in typed_entities:
        sf = te.get("surface_form")
        t = te.get("type")
        if t:
            with_type += 1
            t_uri = t.get("uri")
            if t_uri:
                entity_types_map.setdefault(sf, set()).add(t_uri)
                if t_uri in classes:
                    type_in_onto += 1
                else:
                    type_not_in_onto += 1
            else:
                # Check label if URI is missing
                t_label = t.get("label")
                if t_label:
                    # Very simple label match if no URI
                    found = False
                    for c_uri in classes:
                        if t_label.lower() in c_uri.lower():
                            type_in_onto += 1
                            found = True
                            break
                    if not found:
                        type_not_in_onto += 1
                else:
                    type_not_in_onto += 1
        else:
            entity_types_map.setdefault(sf, set())

    print(f"  Total Entities: {total_entities}")
    print(f"  Entities with type: {with_type}")
    print(f"  Types in ontology: {type_in_onto}")
    print(f"  Types NOT in ontology: {type_not_in_onto}")

    relations = graph.get("relations_wo_class_assertions", [])
    literal_objects = set()
    subjects = {rel.get("subject", {}).get("surface_form") for rel in relations}

    broken_domain = []
    broken_range = []

    for rel in relations:
        s_sf = rel.get("subject", {}).get("surface_form")
        p_uri = rel.get("predicate", {}).get("uri")
        o_sf = rel.get("object", {}).get("surface_form")

        # Check if it's a literal: object of a data property and not used as a subject
        if p_uri in data_props:
            if o_sf not in subjects:
                literal_objects.add(o_sf)

        if p_uri in pred_info:
            info = pred_info[p_uri]

            # Domain check
            if info["domains"]:
                s_types = entity_types_map.get(s_sf, set())
                if not (s_types & info["domains"]):
                    broken_domain.append(rel)

            # Range check
            if info["ranges"]:
                if info["is_data"]:
                    # Simple check: literals shouldn't be in entity_types_map with a type
                    if o_sf in entity_types_map and entity_types_map[o_sf]:
                        # If it has a type, it might not be a literal in the strict sense,
                        # but for musk example dates/years are often types-less.
                        pass
                else:
                    o_types = entity_types_map.get(o_sf, set())
                    if not (o_types & info["ranges"]):
                        broken_range.append(rel)

    print(f"  Literal-only objects: {len(literal_objects)}")
    print(f"  Relations with broken domain: {len(broken_domain)}")
    print(f"  Relations with broken range: {len(broken_range)}")

    if broken_domain:
        print("\n  Broken Domain Relations:")
        for rel in broken_domain:
            p_sf = rel["predicate"]["surface_form"]
            print(
                f"    {rel['subject']['surface_form']} --[{p_sf}]--> {rel['object']['surface_form']}"
            )

    if broken_range:
        print("\n  Broken Range Relations:")
        for rel in broken_range:
            p_sf = rel["predicate"]["surface_form"]
            print(
                f"    {rel['subject']['surface_form']} --[{p_sf}]--> {rel['object']['surface_form']}"
            )


def analyze_deduplication(graph, summary):
    print("\n--- Deduplication Analysis ---")
    entity_clusters = graph.get("entity_clusters") or {}

    total_in_clusters = sum(len(cluster) for cluster in entity_clusters.values())
    num_clusters = len(entity_clusters)
    num_merged = total_in_clusters - num_clusters

    print(f"  Entity Clusters: {num_clusters}")
    print(f"  Entities merged: {num_merged}")

    dedup_stats = summary.get("dedup_stats", {})
    if dedup_stats:
        print(f"  Deduplication Time: {dedup_stats.get('execution_time', 0):.2f}s")

    num_current_entities = len(graph.get("typed_entities", []))
    original_entity_count = num_current_entities + num_merged

    if original_entity_count > 0:
        compression = (num_merged / original_entity_count) * 100
        print(
            f"  Entity Compression: {compression:.1f}% ({original_entity_count} -> {num_current_entities})"
        )


if __name__ == "__main__":
    main()
