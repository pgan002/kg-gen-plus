"""Run KGGen repeatedly over one or more KEAC CQ2Onto domains.

Examples:
    # All six domains, six runs each, under results/gemma4-baseline
    python misc/run_domain.py --model gemma4

    # A subset (comma-separated or repeated flags)
    python misc/run_domain.py --model cq2term-model --domain awo,wine
    python misc/run_domain.py --model cq2term-model --domain awo --domain wine
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_gen.kg_gen import KGGen
from kg_gen.models import InputData

DOMAINS = ("awo", "odrl", "swo", "vgo", "water", "wine")

ENTITY_CONTEXT = """
The source is one competency question for ontology engineering.
Extract all conceptual terms required to represent and answer the question.
Treat common nouns, roles, states, events, and kinds as candidate ontology
classes. Treat bracketed placeholders as references to classes, not named
individuals. Do not extract interrogative words or literal example values.
Use concise singular canonical labels.
""".strip()

RELATION_CONTEXT = """
Interpret relationships required by the competency question as candidate
ontology properties. Include clearly implied properties when necessary to
represent the question. Use concise canonical predicates. Do not output
generic predicates such as relatedTo unless explicitly required.
""".strip()


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_dir() -> Path:
    return (
        workspace_root()
        / "challenge-catalog"
        / "keac-2026"
        / "dataset"
        / "CQ2Onto"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KEAC CQ2Onto domains through KGGen repeatedly."
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "vLLM served model name. The LiteLLM 'openai/' prefix is added "
            "unless a supported provider prefix is already present."
        ),
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible vLLM base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY", "dummy"),
        help="vLLM API key (default: VLLM_API_KEY or 'dummy').",
    )
    parser.add_argument(
        "--domain",
        action="append",
        help=(
            "Domain or comma-separated domains. May be repeated. "
            f"Defaults to all: {','.join(DOMAINS)}."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Parent directory for the derived experiment directory.",
    )
    parser.add_argument(
        "--label",
        default="baseline",
        help=(
            "Short experiment variant used in output and export names "
            "(default: baseline; e.g. deduplicated or schema-prompt-v2)."
        ),
    )
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--n-parallel", type=int, default=10)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; use 0 for deterministic runs.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deduplicate the aggregated graph after extraction (default: disabled).",
    )
    return parser.parse_args()


def selected_domains(values: list[str] | None) -> list[str]:
    if not values:
        return list(DOMAINS)
    requested = []
    for value in values:
        requested.extend(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(requested) - set(DOMAINS))
    if unknown:
        raise ValueError(
            f"Unknown domain(s): {unknown}. Choose from: {', '.join(DOMAINS)}"
        )
    # Preserve the user's order while avoiding accidental duplicate runs.
    return list(dict.fromkeys(requested))


def load_questions(path: Path) -> tuple[list[dict[str, Any]], list[InputData]]:
    with path.open(encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list):
        raise TypeError(f"Expected a JSON list in {path}")

    inputs = []
    for index, question in enumerate(questions):
        if not isinstance(question, dict) or not {"id", "value"} <= question.keys():
            raise ValueError(f"Invalid question at index {index}: {question!r}")
        inputs.append(InputData(id=str(question["id"]), text=str(question["value"])))
    return questions, inputs


def litellm_model_name(model: str) -> str:
    """Add a LiteLLM provider without mistaking an org/model ID for one."""
    if model.startswith(("openai/", "hosted_vllm/")):
        return model
    return f"openai/{model}"


def openai_api_base(api_base: str) -> str:
    """Normalize a vLLM server URL to its OpenAI-compatible API base."""
    normalized = api_base.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def identifier(value: str, option: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "-." else "-"
        for character in value.strip()
    ).strip("-.")
    if not normalized:
        raise ValueError(f"{option} must contain a letter or number")
    return normalized


def experiment_id(model: str, label: str) -> str:
    model_id = identifier(model.rsplit("/", maxsplit=1)[-1], "--model")
    label_id = identifier(label, "--label")
    return f"{model_id}-{label_id}"


def write_json_atomic(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


async def run_domain(
    args: argparse.Namespace,
    domain: str,
    model: str,
    api_base: str,
) -> None:
    input_path = args.dataset_dir / f"{domain}_cq2onto_cqs.json"
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing CQ2Onto input for {domain}: {input_path}")
    questions, inputs = load_questions(input_path)
    output_dir = args.output_root / domain
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "domain": domain,
        "method": "kggen",
        "label": args.label,
        "experiment_id": args.experiment_id,
        "input": str(input_path.resolve()),
        "question_count": len(inputs),
        "runs_requested": args.runs,
        "model": model,
        "api_base": api_base,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "n_parallel": args.n_parallel,
        "deduplicate": args.deduplicate,
        "entity_context": ENTITY_CONTEXT,
        "relation_context": RELATION_CONTEXT,
        "completed_runs": [],
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    for run_number in range(1, args.runs + 1):
        print(
            f"Starting {domain} run {run_number}/{args.runs}",
            flush=True,
        )
        kg = KGGen(
            model=model,
            api_base=api_base,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retrieval_model=None,
            disable_cache=True,
        )
        graph, stats = await kg.generate(
            input_data=inputs,
            entity_context=ENTITY_CONTEXT,
            relation_context=RELATION_CONTEXT,
            n_parallel=args.n_parallel,
            deduplicate=args.deduplicate,
        )

        result_path = runs_dir / f"{run_number:02d}.json"
        result = {
            "domain": domain,
            "method": "kggen",
            "label": args.label,
            "experiment_id": args.experiment_id,
            "run": run_number,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "questions": questions,
            "graph": graph.model_dump(mode="json", exclude_none=True),
            "stats": stats.model_dump(mode="json", exclude_none=True),
        }
        write_json_atomic(result_path, result)
        manifest["completed_runs"].append(
            result_path.relative_to(output_dir).as_posix()
        )
        write_json_atomic(manifest_path, manifest)
        print(f"Saved {result_path}", flush=True)


async def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.n_parallel < 1:
        raise ValueError("--n-parallel must be at least 1")

    domains = selected_domains(args.domain)
    args.dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_root.resolve()
    model = litellm_model_name(args.model)
    api_base = openai_api_base(args.api_base)
    args.label = identifier(args.label, "--label")
    args.experiment_id = experiment_id(model, args.label)
    args.output_root = output_root / args.experiment_id
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty experiment directory "
            f"{args.output_root}. Manually remove or rename it, or choose "
            "another --label."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {args.experiment_id}", flush=True)
    print(f"Output: {args.output_root}", flush=True)
    print(f"Selected domains: {', '.join(domains)}", flush=True)
    for domain in domains:
        await run_domain(args, domain, model, api_base)


if __name__ == "__main__":
    asyncio.run(main())
