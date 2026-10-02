"""Export repeated KGGen runs as CQ4OE CQ2Term and/or CQ2Onto inputs.

This script only converts and stages predictions. It never reads gold data,
runs evaluators, or computes metrics.

Examples:
    python misc/export_cq4oe.py --task cq2term
    python misc/export_cq4oe.py --task cq2onto --mode mymodel
    python misc/export_cq4oe.py --task both --class-source surface-and-type
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from rdflib import RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL

from kg_gen.ontology.cq_term_assignment import (
    CanonicalCQTermAssigner,
    TermEncoder,
    reassign_canonical_terms,
)
from kg_gen.ontology.term_clustering import (
    OntologyTermClusterer,
    cluster_term_assignments,
)
from kg_gen.ontology.term_normalization import (
    OntologyTermKind,
    normalize_ontology_term,
)
from kg_gen.ontology.term_roles import filter_ontology_term_roles

PROV = Namespace("urn:kggen:provenance:")
CLASS_SOURCES = ("surface", "type", "surface-and-type")
TERM_ROLE_FILTERS = ("none", "ontology-conservative")


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage KGGen run outputs for the CQ4OE benchmark."
    )
    parser.add_argument(
        "--task",
        choices=("cq2term", "cq2onto", "both"),
        default="both",
        help="Outputs to stage (default: both).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Recursively discover domain run directories below this path.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=workspace_root() / "cq4oe-benchmark",
    )

    parser.add_argument(
        "--mode",
        default="mymodel",
        help="CQ2Onto generation mode; must be configured in its evaluator.",
    )
    parser.add_argument(
        "--class-source",
        choices=CLASS_SOURCES,
        default="surface-and-type",
        help="Which TypedEntity fields become CQ2Term class predictions.",
    )
    parser.add_argument(
        "--export-label",
        help=(
            "Optional label appended to staged CQ4OE run names for export "
            "ablations, such as surface or type. It does not alter raw results."
        ),
    )
    parser.add_argument(
        "--semantic-clustering",
        action="store_true",
        help="Cluster semantically equivalent CQ2Term labels within each run/domain.",
    )
    parser.add_argument(
        "--semantic-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for semantic clustering.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.9,
        help="Minimum cosine similarity for semantic clustering (default: 0.9).",
    )
    parser.add_argument(
        "--term-role-filter",
        choices=TERM_ROLE_FILTERS,
        default="none",
        help=(
            "Filter copular pseudo-properties, class/property overlap, and "
            "literal-like class candidates (default: none)."
        ),
    )
    parser.add_argument(
        "--cq-reassignment",
        action="store_true",
        help=(
            "Reassign the canonical domain class/property vocabulary to each CQ "
            "using bounded semantic similarity."
        ),
    )
    parser.add_argument(
        "--cq-reassignment-threshold",
        type=float,
        default=0.45,
        help="Minimum CQ-to-term cosine similarity (default: 0.45).",
    )
    parser.add_argument(
        "--cq-reassignment-max-additions",
        type=int,
        default=3,
        help="Maximum canonical terms added per CQ and term kind (default: 3).",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def ensure_writable(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {path}. Manually remove or rename the "
            f"output directory {path.parent.parent} before exporting again."
        )


def write_json(path: Path, value: Any) -> None:
    ensure_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def clean_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:")
    if not label or any(token in label for token in ('"', "{", "}", "[", "]")):
        return None
    if label.lower().startswith(("surface_form", "predicate", "subject", "object")):
        return None
    return label


def provenance_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def append_unique(
    terms: dict[str, dict[str, str]],
    cq_id: str,
    label: str,
    *,
    kind: OntologyTermKind,
) -> None:
    normalized = normalize_ontology_term(label, kind)
    if normalized:
        terms[cq_id].setdefault(normalized, normalized)


def cluster_terms(
    terms_by_cq: dict[str, dict[str, str]], clusterer: OntologyTermClusterer
) -> dict[str, dict[str, str]]:
    assignments = cluster_term_assignments(
        {cq_id: terms.keys() for cq_id, terms in terms_by_cq.items()}, clusterer
    )
    return defaultdict(
        dict,
        {
            cq_id: {term: term for term in terms}
            for cq_id, terms in assignments.items()
        },
    )


def cq2term_predictions(
    run_data: dict[str, Any],
    class_source: str,
    semantic_clusterer: OntologyTermClusterer | None = None,
    cq_assigner: CanonicalCQTermAssigner | None = None,
    term_role_filter: str = "none",
) -> list[dict[str, Any]]:
    questions = run_data.get("questions")
    payload = run_data.get("graph")
    if not isinstance(questions, list):
        raise TypeError("Run JSON has no list-valued 'questions' field")
    if not isinstance(payload, dict):
        raise TypeError("Run JSON has no object-valued 'graph' field")

    classes: dict[str, dict[str, str]] = defaultdict(dict)
    properties: dict[str, dict[str, str]] = defaultdict(dict)
    entities = payload.get("typed_entities", [])
    if not isinstance(entities, list):
        raise TypeError("graph.typed_entities must be a list")

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        surface = clean_label(entity.get("surface_form"))
        entity_type = entity.get("type")
        type_label = (
            clean_label(entity_type.get("label"))
            if isinstance(entity_type, dict)
            else None
        )
        labels = []
        if class_source in ("surface", "surface-and-type") and surface:
            labels.append(surface)
        if class_source in ("type", "surface-and-type") and type_label:
            labels.append(type_label)
        for cq_id in provenance_ids(entity.get("provenance_ids")):
            for label in labels:
                append_unique(classes, cq_id, label, kind="class")

    relations = payload.get("relations_wo_class_assertions", [])
    if not isinstance(relations, list):
        raise TypeError("graph.relations_wo_class_assertions must be a list")
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        predicate = relation.get("predicate")
        label = (
            clean_label(predicate.get("surface_form"))
            if isinstance(predicate, dict)
            else None
        )
        if not label:
            continue
        for cq_id in provenance_ids(relation.get("provenance_ids")):
            append_unique(properties, cq_id, label, kind="property")

    if semantic_clusterer is not None:
        classes = cluster_terms(classes, semantic_clusterer)
        properties = cluster_terms(properties, semantic_clusterer)

    if term_role_filter == "ontology-conservative":
        class_assignments, property_assignments, _ = filter_ontology_term_roles(
            {cq_id: set(terms) for cq_id, terms in classes.items()},
            {cq_id: set(terms) for cq_id, terms in properties.items()},
        )
        classes = defaultdict(
            dict,
            {
                cq_id: {term: term for term in terms}
                for cq_id, terms in class_assignments.items()
            },
        )
        properties = defaultdict(
            dict,
            {
                cq_id: {term: term for term in terms}
                for cq_id, terms in property_assignments.items()
            },
        )
    elif term_role_filter != "none":
        raise ValueError(f"Unknown term role filter: {term_role_filter}")

    if cq_assigner is not None:
        class_assignments = reassign_canonical_terms(
            questions,
            {cq_id: set(terms) for cq_id, terms in classes.items()},
            "class",
            cq_assigner,
        )
        property_assignments = reassign_canonical_terms(
            questions,
            {cq_id: set(terms) for cq_id, terms in properties.items()},
            "property",
            cq_assigner,
        )
        classes = defaultdict(
            dict,
            {
                cq_id: {term: term for term in terms}
                for cq_id, terms in class_assignments.items()
            },
        )
        properties = defaultdict(
            dict,
            {
                cq_id: {term: term for term in terms}
                for cq_id, terms in property_assignments.items()
            },
        )

    output = []
    for index, question in enumerate(questions):
        if not isinstance(question, dict) or not {"id", "value"} <= question.keys():
            raise ValueError(f"Invalid question at index {index}: {question!r}")
        cq_id = str(question["id"])
        output.append(
            {
                "id": cq_id,
                "value": question["value"],
                "class": sorted(classes[cq_id].values(), key=str.casefold),
                "property": sorted(properties[cq_id].values(), key=str.casefold),
            }
        )
    return output


def term_uri(namespace: Namespace, label: str) -> URIRef:
    return URIRef(str(namespace) + quote(label.replace(" ", "_"), safe="-_~."))


def add_class(graph: Graph, namespace: Namespace, label: str) -> URIRef:
    uri = term_uri(namespace, label)
    graph.add((uri, RDF.type, OWL.Class))
    graph.add((uri, RDFS.label, Literal(label, lang="en")))
    return uri


def add_property(graph: Graph, namespace: Namespace, label: str) -> URIRef:
    uri = term_uri(namespace, label)
    graph.add((uri, RDF.type, OWL.ObjectProperty))
    graph.add((uri, RDFS.label, Literal(label, lang="en")))
    return uri


def graph_json_to_owl(
    run_data: dict[str, Any], destination: Path, domain: str
) -> None:
    ensure_writable(destination)
    payload = run_data.get("graph")
    if not isinstance(payload, dict):
        raise TypeError("Run JSON has no object-valued 'graph' field")

    graph = Graph()
    namespace = Namespace(f"urn:kggen:{domain}:")
    graph.bind("kg", namespace)
    graph.bind("owl", OWL)
    graph.bind("prov", PROV)
    graph.add((URIRef(f"urn:kggen:{domain}:ontology"), RDF.type, OWL.Ontology))

    entities = payload.get("typed_entities", [])
    if not isinstance(entities, list):
        raise TypeError("graph.typed_entities must be a list")
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        label = clean_label(entity.get("surface_form"))
        if not label:
            continue
        class_uri = add_class(graph, namespace, label)
        entity_type = entity.get("type")
        type_label = (
            clean_label(entity_type.get("label"))
            if isinstance(entity_type, dict)
            else None
        )
        if type_label and type_label.casefold() != label.casefold():
            graph.add(
                (class_uri, RDFS.subClassOf, add_class(graph, namespace, type_label))
            )
        description = clean_label(entity.get("description"))
        if description:
            graph.add((class_uri, RDFS.comment, Literal(description, lang="en")))
        for cq_id in provenance_ids(entity.get("provenance_ids")):
            graph.add((class_uri, PROV.wasDerivedFrom, Literal(cq_id)))

    property_pairs: dict[URIRef, set[tuple[URIRef, URIRef]]] = defaultdict(set)
    relations = payload.get("relations_wo_class_assertions", [])
    if not isinstance(relations, list):
        raise TypeError("graph.relations_wo_class_assertions must be a list")
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        subject_data = relation.get("subject")
        predicate_data = relation.get("predicate")
        object_data = relation.get("object")
        if not isinstance(subject_data, dict):
            continue
        if not isinstance(predicate_data, dict):
            continue
        if not isinstance(object_data, dict):
            continue
        subject = clean_label(subject_data.get("surface_form"))
        predicate = clean_label(predicate_data.get("surface_form"))
        object_ = clean_label(object_data.get("surface_form"))
        if not subject or not predicate or not object_:
            continue

        subject_uri = add_class(graph, namespace, subject)
        object_uri = add_class(graph, namespace, object_)
        predicate_uri = add_property(graph, namespace, predicate)
        property_pairs[predicate_uri].add((subject_uri, object_uri))
        restriction = BNode()
        graph.add((subject_uri, RDFS.subClassOf, restriction))
        graph.add((restriction, RDF.type, OWL.Restriction))
        graph.add((restriction, OWL.onProperty, predicate_uri))
        graph.add((restriction, OWL.someValuesFrom, object_uri))
        for cq_id in provenance_ids(relation.get("provenance_ids")):
            graph.add((restriction, PROV.wasDerivedFrom, Literal(cq_id)))

    for predicate_uri, pairs in property_pairs.items():
        domains = {subject for subject, _ in pairs}
        ranges = {object_ for _, object_ in pairs}
        if len(domains) == 1:
            graph.add((predicate_uri, RDFS.domain, next(iter(domains))))
        if len(ranges) == 1:
            graph.add((predicate_uri, RDFS.range, next(iter(ranges))))

    destination.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=destination, format="xml")


def run_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot determine run number from {path.name}")
    return int(match.group(1))


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing required manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected an object in {manifest_path}")
    return manifest


def infer_domain(run_dir: Path) -> str:
    domain = load_manifest(run_dir).get("domain")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError(f"Missing required 'domain' in {run_dir / 'manifest.json'}")
    return domain.strip().lower()


def identifier(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_.-")


def infer_experiment(run_dir: Path) -> str:
    manifest = load_manifest(run_dir)
    manifest_path = run_dir / "manifest.json"
    required = {}
    for field in ("method", "model", "label", "experiment_id"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing required '{field}' in {manifest_path}")
        required[field] = value.strip()

    method = identifier(required["method"])
    llm = identifier(required["model"].rsplit("/", maxsplit=1)[-1])
    label = identifier(required["label"])
    expected_experiment = f"{llm}-{label}"
    if required["experiment_id"] != expected_experiment:
        raise ValueError(
            f"Inconsistent experiment_id in {manifest_path}: expected "
            f"'{expected_experiment}', got '{required['experiment_id']}'"
        )
    if run_dir.parent.name != expected_experiment:
        raise ValueError(
            f"Experiment directory mismatch for {manifest_path}: expected parent "
            f"'{expected_experiment}', got '{run_dir.parent.name}'"
        )
    return f"{method}-{expected_experiment}"


def discover_run_groups(results_dir: Path) -> list[tuple[Path, str, str, list[Path]]]:
    domain_dirs = sorted(path.parent for path in results_dir.rglob("manifest.json"))
    groups = []
    for domain_dir in domain_dirs:
        run_files = sorted((domain_dir / "runs").glob("*.json"), key=run_number)
        if not run_files:
            raise FileNotFoundError(
                f"No run JSON files found in {domain_dir / 'runs'}"
            )
        domain = infer_domain(domain_dir)
        experiment = infer_experiment(domain_dir)
        groups.append((domain_dir, domain, experiment, run_files))
    if not groups:
        raise FileNotFoundError(f"No manifests found below {results_dir}")
    return groups


def staged_manifest(
    *,
    task: str,
    run_name: str,
    run_number_value: int,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    source_run: Path,
    domain: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "task": task,
        "run_name": run_name,
        "run": run_number_value,
        "method": source_manifest["method"],
        "model": source_manifest["model"],
        "source_experiment_id": source_manifest["experiment_id"],
        "source_manifests": {domain: str(source_manifest_path.resolve())},
        "source_runs": {domain: str(source_run.resolve())},
        "domains": [domain],
        "generation": {
            key: source_manifest.get(key)
            for key in (
                "api_base",
                "temperature",
                "max_tokens",
                "n_parallel",
                "entity_context",
                "relation_context",
            )
        },
        "graph_processing": {
            key: source_manifest.get(key)
            for key in (
                "deduplicate",
                "deduplicate_with_embeddings",
                "retrieval_model",
                "postprocessing",
                "derived_from",
            )
        },
    }
    if task == "cq2term":
        manifest["term_processing"] = {
            "class_source": args.class_source,
            "lexical_normalization": True,
            "morphological_normalization": True,
            "semantic_clustering": args.semantic_clustering,
            "semantic_model": args.semantic_model
            if args.semantic_clustering
            else None,
            "semantic_threshold": args.semantic_threshold
            if args.semantic_clustering
            else None,
            "export_label": args.export_label,
            "term_role_filter": args.term_role_filter,
            "cq_reassignment": args.cq_reassignment,
            "cq_reassignment_model": args.semantic_model
            if args.cq_reassignment
            else None,
            "cq_reassignment_threshold": args.cq_reassignment_threshold
            if args.cq_reassignment
            else None,
            "cq_reassignment_max_additions": args.cq_reassignment_max_additions
            if args.cq_reassignment
            else None,
        }
    else:
        manifest["mode"] = args.mode
        manifest["ontology_export"] = {
            "format": "rdfxml",
            "file_extension": ".owl",
        }
    return manifest


def merge_staged_manifest(
    existing: dict[str, Any], incoming: dict[str, Any], domain: str
) -> None:
    for key in (
        "schema_version",
        "task",
        "run_name",
        "run",
        "method",
        "model",
        "source_experiment_id",
        "generation",
        "graph_processing",
        "term_processing",
        "mode",
        "ontology_export",
    ):
        if existing.get(key) != incoming.get(key):
            raise ValueError(
                f"Inconsistent staged manifest field {key!r} for "
                f"{existing['run_name']}: {existing.get(key)!r} versus "
                f"{incoming.get(key)!r}"
            )
    existing["source_manifests"][domain] = incoming["source_manifests"][domain]
    existing["source_runs"][domain] = incoming["source_runs"][domain]
    existing["domains"] = sorted(set(existing["domains"]) | {domain})


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    benchmark_root = args.benchmark_root.resolve()
    export_label = identifier(args.export_label) if args.export_label else None
    if args.export_label and not export_label:
        raise ValueError("--export-label must contain a letter or number")
    if not 0.0 <= args.semantic_threshold <= 1.0:
        raise ValueError("--semantic-threshold must be between 0 and 1")
    if not 0.0 <= args.cq_reassignment_threshold <= 1.0:
        raise ValueError("--cq-reassignment-threshold must be between 0 and 1")
    if args.cq_reassignment_max_additions < 1:
        raise ValueError("--cq-reassignment-max-additions must be at least 1")
    semantic_clusterer = (
        OntologyTermClusterer(args.semantic_model, args.semantic_threshold)
        if args.semantic_clustering
        else None
    )
    cq_assigner = (
        CanonicalCQTermAssigner(
            args.semantic_model,
            args.cq_reassignment_threshold,
            args.cq_reassignment_max_additions,
            encoder=(
                cast(TermEncoder, semantic_clusterer.encoder)
                if semantic_clusterer
                else None
            ),
        )
        if args.cq_reassignment
        else None
    )
    destinations: dict[Path, Path] = {}
    staged_manifests: dict[Path, dict[str, Any]] = {}

    for run_dir, domain, experiment, run_files in discover_run_groups(results_dir):
        staged_experiment = (
            f"{experiment}-{export_label}" if export_label else experiment
        )
        print(
            f"Discovered domain={domain}, experiment={experiment}, "
            f"staged_experiment={staged_experiment}, runs={len(run_files)} "
            f"in {run_dir}"
        )
        source_manifest_path = run_dir / "manifest.json"
        source_manifest = load_manifest(run_dir)
        for run_file in run_files:
            number = run_number(run_file)
            model_name = f"{staged_experiment}-run-{number:02d}"
            run_data = read_json(run_file)
            if not isinstance(run_data, dict):
                raise TypeError(f"Expected an object in {run_file}")

            if args.task in ("cq2term", "both"):
                destination = (
                    benchmark_root
                    / "CQ2Term"
                    / "01_predictions"
                    / model_name
                    / "terms"
                    / f"{domain}_cq2terms_terms.json"
                )
                previous = destinations.setdefault(destination, run_file)
                if previous != run_file:
                    raise ValueError(
                        f"Both {previous} and {run_file} map to {destination}"
                    )
                write_json(
                    destination,
                    cq2term_predictions(
                        run_data,
                        args.class_source,
                        semantic_clusterer,
                        cq_assigner,
                        args.term_role_filter,
                    ),
                )
                manifest_path = destination.parents[1] / "manifest.json"
                incoming = staged_manifest(
                    task="cq2term",
                    run_name=model_name,
                    run_number_value=number,
                    source_manifest=source_manifest,
                    source_manifest_path=source_manifest_path,
                    source_run=run_file,
                    domain=domain,
                    args=args,
                )
                if manifest_path in staged_manifests:
                    merge_staged_manifest(
                        staged_manifests[manifest_path], incoming, domain
                    )
                else:
                    staged_manifests[manifest_path] = incoming
                print(f"Staged CQ2Term {run_file.name} -> {destination}")

            if args.task in ("cq2onto", "both"):
                destination = (
                    benchmark_root
                    / "CQ2Onto"
                    / "01_predictions"
                    / args.mode
                    / model_name
                    / "ontology"
                    / f"{domain}_ontology.owl"
                )
                previous = destinations.setdefault(destination, run_file)
                if previous != run_file:
                    raise ValueError(
                        f"Both {previous} and {run_file} map to {destination}"
                    )
                graph_json_to_owl(run_data, destination, domain)
                manifest_path = destination.parents[1] / "manifest.json"
                incoming = staged_manifest(
                    task="cq2onto",
                    run_name=model_name,
                    run_number_value=number,
                    source_manifest=source_manifest,
                    source_manifest_path=source_manifest_path,
                    source_run=run_file,
                    domain=domain,
                    args=args,
                )
                if manifest_path in staged_manifests:
                    merge_staged_manifest(
                        staged_manifests[manifest_path], incoming, domain
                    )
                else:
                    staged_manifests[manifest_path] = incoming
                print(f"Staged CQ2Onto {run_file.name} -> {destination}")

    for manifest_path, manifest in sorted(staged_manifests.items()):
        write_json(manifest_path, manifest)
        print(f"Staged manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
