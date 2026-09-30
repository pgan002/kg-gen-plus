"""Apply KGGen graph deduplication to previously extracted run JSON files.

Examples:
    python misc/deduplicate_saved_runs.py \
        --source results/gemma4-dedup \
        --label dedup-embeddings-posthoc \
        --with-embeddings

    python misc/deduplicate_saved_runs.py \
        --source results/gemma4-baseline \
        --label post-string-dedup
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from kg_gen.kg_gen import KGGen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate saved KGGen graphs without calling the LLM."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source experiment directory containing domain manifests and runs/.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Label for the derived experiment, e.g. dedup-embeddings-posthoc.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Parent output directory (default: the source experiment's parent).",
    )
    parser.add_argument(
        "--domain",
        action="append",
        help="Domain or comma-separated domains to process. Default: all domains.",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Run embedding deduplication after KGGen's string grouping.",
    )
    parser.add_argument(
        "--retrieval-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model used when --with-embeddings is enabled.",
    )
    parser.add_argument("--entity-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.9)
    parser.add_argument(
        "--deduplicate-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate predicates as well as entities (default: enabled).",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def identifier(value: str, option: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-.")
    if not normalized:
        raise ValueError(f"{option} must contain a letter or number")
    return normalized


def selected_domains(source: Path, values: list[str] | None) -> list[str]:
    available = sorted(path.name for path in source.iterdir() if path.is_dir())
    if not values:
        return available
    requested = []
    for value in values:
        requested.extend(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown source domain(s): {unknown}; available: {available}")
    return list(dict.fromkeys(requested))


def experiment_identity(source: Path, label: str) -> tuple[str, str]:
    manifests = sorted(source.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No domain manifests found in {source}")
    models = {
        manifest.get("model")
        for path in manifests
        if isinstance((manifest := read_json(path)), dict)
    }
    if len(models) != 1:
        raise ValueError(f"Source manifests do not share one model: {models}")
    model = next(iter(models))
    if not isinstance(model, str) or not model:
        raise ValueError(f"Source manifests have an invalid model: {models}")
    model_id = identifier(model.rsplit("/", maxsplit=1)[-1], "manifest model")
    return model, f"{model_id}-{identifier(label, '--label')}"


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.entity_threshold <= 1.0:
        raise ValueError("--entity-threshold must be between 0 and 1")
    if not 0.0 <= args.edge_threshold <= 1.0:
        raise ValueError("--edge-threshold must be between 0 and 1")

    source = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source experiment does not exist: {source}")
    label = identifier(args.label, "--label")
    model, experiment_id = experiment_identity(source, label)
    output_root = (
        args.output_root.resolve() if args.output_root else source.parent
    )
    destination = output_root / experiment_id
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite {destination}. Manually remove or rename it, "
            "or choose another --label."
        )

    domains = selected_domains(source, args.domain)
    retrieval_model = args.retrieval_model if args.with_embeddings else None
    kg = KGGen(model="openai/unused", retrieval_model=retrieval_model)

    print(f"Source: {source}")
    print(f"Destination: {destination}")
    print(f"Domains: {', '.join(domains)}")
    for domain in domains:
        source_domain = source / domain
        source_manifest_path = source_domain / "manifest.json"
        manifest = read_json(source_manifest_path)
        if not isinstance(manifest, dict):
            raise TypeError(f"Expected an object in {source_manifest_path}")
        if manifest.get("model") != model:
            raise ValueError(f"Model mismatch in {source_manifest_path}")

        destination_domain = destination / domain
        destination_runs = destination_domain / "runs"
        destination_runs.mkdir(parents=True)
        manifest.update(
            {
                "label": label,
                "experiment_id": experiment_id,
                "deduplicate": True,
                "deduplicate_with_embeddings": args.with_embeddings,
                "retrieval_model": retrieval_model,
                "derived_from": str(source_domain.resolve()),
                "postprocessing": {
                    "source_deduplicate": manifest.get("deduplicate", False),
                    "source_deduplicate_with_embeddings": manifest.get(
                        "deduplicate_with_embeddings", False
                    ),
                    "entity_similarity_threshold": args.entity_threshold,
                    "edge_similarity_threshold": args.edge_threshold,
                    "deduplicate_edges": args.deduplicate_edges,
                },
            }
        )
        write_json(destination_domain / "manifest.json", manifest)

        for source_run in sorted((source_domain / "runs").glob("*.json")):
            run = read_json(source_run)
            if not isinstance(run, dict) or not isinstance(run.get("graph"), dict):
                raise TypeError(f"Expected a run object with graph in {source_run}")
            graph = kg.from_dict(run["graph"])
            deduplicated_graph, dedup_stats = kg.deduplicate(
                graph,
                entity_similarity_threshold=args.entity_threshold,
                edge_similarity_threshold=args.edge_threshold,
                deduplicate_edges=args.deduplicate_edges,
                use_embeddings=args.with_embeddings,
            )
            run.update(
                {
                    "label": label,
                    "experiment_id": experiment_id,
                    "derived_from": str(source_run.resolve()),
                    "graph": deduplicated_graph.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "postprocessing": {
                        "input_already_string_deduplicated": manifest[
                            "postprocessing"
                        ]["source_deduplicate"],
                        "deduplicate_with_embeddings": args.with_embeddings,
                        "retrieval_model": retrieval_model,
                        "entity_similarity_threshold": args.entity_threshold,
                        "edge_similarity_threshold": args.edge_threshold,
                        "deduplicate_edges": args.deduplicate_edges,
                        "stats": dedup_stats.model_dump(
                            mode="json", exclude_none=True
                        ),
                    },
                }
            )
            destination_run = destination_runs / source_run.name
            write_json(destination_run, run)
            print(f"Saved {destination_run}")


if __name__ == "__main__":
    main()
