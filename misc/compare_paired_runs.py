"""Compare all metrics between two repeated-run aggregate conditions.

The input directories are produced by cq4oe-benchmark's
``scripts/aggregate_run_results.py`` and contain one ``<domain>.json`` file per
benchmark domain.

Example:
    python misc/compare_paired_runs.py \
        --baseline ../cq4oe-benchmark/CQ2Term/04_summary/kggen/gemma4/lex-morpho-sem \
        --comparison ../cq4oe-benchmark/CQ2Term/04_summary/kggen/gemma4/dedup-embeds-posthoc-lex-morph-sem \
        --output-json paired-comparison.json \
        --output-csv paired-comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

DEFAULT_DOMAINS = ("awo", "odrl", "swo", "vgo", "water", "wine")
RUN_NUMBER_PATTERN = re.compile(r"(?:-run-|_run_)(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare all numeric metrics in matched repeated runs. Positive "
            "differences mean the comparison condition scored higher."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Aggregate-summary directory for the baseline condition.",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        required=True,
        help="Aggregate-summary directory for the comparison condition.",
    )
    parser.add_argument(
        "--domains",
        help=(
            "Comma-separated domains. Default: use the standard domains that "
            "exist in both input directories."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Absolute difference treated as a tie (default: 1e-12).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("paired-comparison.json"),
        help="Full machine-readable output (default: paired-comparison.json).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("paired-comparison.csv"),
        help="Long-form paired observations (default: paired-comparison.csv).",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_number(run_name: str) -> int:
    match = RUN_NUMBER_PATTERN.search(run_name)
    if match is None:
        raise ValueError(f"Cannot determine run number from {run_name!r}")
    return int(match.group(1))


def load_runs(path: Path) -> dict[int, dict[str, float]]:
    document = read_json(path)
    rows = document.get("per_run") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise TypeError(f"{path} has no per_run list")

    runs = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"Invalid per_run row in {path}: {row!r}")
        name = row.get("run_name")
        if not isinstance(name, str):
            raise TypeError(f"Missing run_name in {path}: {row!r}")
        number = run_number(name)
        if number in runs:
            raise ValueError(f"Duplicate run number {number} in {path}")
        metrics = {
            key: float(value)
            for key, value in row.items()
            if key != "run_name"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
        if not metrics:
            raise ValueError(f"No numeric metrics for {name} in {path}")
        runs[number] = metrics
    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs


def selected_domains(
    baseline: Path, comparison: Path, requested: str | None
) -> list[str]:
    if requested:
        domains = [item.strip() for item in requested.split(",") if item.strip()]
    else:
        domains = [
            domain
            for domain in DEFAULT_DOMAINS
            if (baseline / f"{domain}.json").is_file()
            and (comparison / f"{domain}.json").is_file()
        ]
    if not domains:
        raise ValueError("No domains were selected")
    missing = [
        str(path)
        for domain in domains
        for path in (baseline / f"{domain}.json", comparison / f"{domain}.json")
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing domain aggregate file(s): {missing}")
    return list(dict.fromkeys(domains))


def summarize(values: list[float], tolerance: float) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "wins": sum(value > tolerance for value in values),
        "ties": sum(abs(value) <= tolerance for value in values),
        "losses": sum(value < -tolerance for value in values),
    }


def exact_sign_flip_pvalue(differences: list[float]) -> float | None:
    """Two-sided paired randomization test, exact for at most 20 run pairs."""
    if len(differences) > 20:
        return None
    observed = abs(statistics.fmean(differences))
    exceedances = 0
    permutations = 0
    for signs in itertools.product((-1, 1), repeat=len(differences)):
        permuted = abs(
            statistics.fmean(
                sign * value for sign, value in zip(signs, differences, strict=True)
            )
        )
        exceedances += permuted >= observed - 1e-15
        permutations += 1
    return exceedances / permutations


def bootstrap_ci(
    differences: list[float], samples: int, seed: int
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("--bootstrap-samples must be at least 1")
    generator = random.Random(seed)
    estimates = sorted(
        statistics.fmean(generator.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return lower, upper


def load_condition(
    root: Path, domains: list[str]
) -> dict[str, dict[int, dict[str, float]]]:
    return {domain: load_runs(root / f"{domain}.json") for domain in domains}


def common_metrics(
    baseline: dict[str, dict[int, dict[str, float]]],
    comparison: dict[str, dict[int, dict[str, float]]],
) -> list[str]:
    baseline_metrics = {
        metric
        for domain_runs in baseline.values()
        for metrics in domain_runs.values()
        for metric in metrics
    }
    comparison_metrics = {
        metric
        for domain_runs in comparison.values()
        for metrics in domain_runs.values()
        for metric in metrics
    }
    shared = sorted(baseline_metrics & comparison_metrics)
    if not shared:
        raise ValueError("The two conditions have no shared numeric metrics")

    for condition_name, condition in (
        ("baseline", baseline),
        ("comparison", comparison),
    ):
        missing = {
            (domain, run): sorted(set(shared) - set(metrics))
            for domain, domain_runs in condition.items()
            for run, metrics in domain_runs.items()
            if set(shared) - set(metrics)
        }
        if missing:
            raise ValueError(
                f"Shared metrics are missing from some {condition_name} rows: {missing}"
            )
    return shared


def compare_metric(
    metric: str,
    domains: list[str],
    baseline: dict[str, dict[int, dict[str, float]]],
    comparison: dict[str, dict[int, dict[str, float]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    observations = []
    domain_summaries = {}
    run_domain_differences: dict[int, list[float]] = {}

    for domain in domains:
        baseline_runs = baseline[domain]
        comparison_runs = comparison[domain]
        if baseline_runs.keys() != comparison_runs.keys():
            raise ValueError(
                f"Run mismatch for {domain}: {sorted(baseline_runs)} versus "
                f"{sorted(comparison_runs)}"
            )

        differences = []
        for number in sorted(baseline_runs):
            baseline_value = baseline_runs[number][metric]
            comparison_value = comparison_runs[number][metric]
            difference = comparison_value - baseline_value
            differences.append(difference)
            run_domain_differences.setdefault(number, []).append(difference)
            observations.append(
                {
                    "metric": metric,
                    "domain": domain,
                    "run": number,
                    "baseline": baseline_value,
                    "comparison": comparison_value,
                    "difference": difference,
                }
            )
        domain_summaries[domain] = summarize(differences, args.tolerance)

    expected_domain_count = len(domains)
    incomplete = {
        number: len(values)
        for number, values in run_domain_differences.items()
        if len(values) != expected_domain_count
    }
    if incomplete:
        raise ValueError(f"Runs do not cover every selected domain: {incomplete}")

    run_differences = {
        number: statistics.fmean(values)
        for number, values in sorted(run_domain_differences.items())
    }
    values = list(run_differences.values())
    lower, upper = bootstrap_ci(values, args.bootstrap_samples, args.seed)
    overall = summarize(values, args.tolerance)
    overall.update(
        {
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "bootstrap_ci_95": [lower, upper],
            "exact_sign_flip_pvalue_two_sided": exact_sign_flip_pvalue(values),
        }
    )
    return {
        "domain_summaries": domain_summaries,
        "run_mean_differences": {
            str(number): value for number, value in run_differences.items()
        },
        "overall_run_level": overall,
        "observations": observations,
    }


def build_comparison(args: argparse.Namespace) -> dict[str, Any]:
    baseline_root = args.baseline.resolve()
    comparison_root = args.comparison.resolve()
    if not baseline_root.is_dir():
        raise FileNotFoundError(
            f"Baseline directory does not exist: {baseline_root}"
        )
    if not comparison_root.is_dir():
        raise FileNotFoundError(
            f"Comparison directory does not exist: {comparison_root}"
        )
    if args.tolerance < 0:
        raise ValueError("--tolerance cannot be negative")

    domains = selected_domains(baseline_root, comparison_root, args.domains)
    baseline = load_condition(baseline_root, domains)
    comparison = load_condition(comparison_root, domains)
    metrics = common_metrics(baseline, comparison)
    results = {
        metric: compare_metric(metric, domains, baseline, comparison, args)
        for metric in metrics
    }
    return {
        "baseline": str(baseline_root),
        "comparison": str(comparison_root),
        "difference_direction": "comparison - baseline",
        "domains": domains,
        "metrics": metrics,
        "results": results,
    }


def print_report(result: dict[str, Any]) -> None:
    print("Difference: comparison - baseline")
    print(f"Domains: {', '.join(result['domains'])}")
    print()
    print(
        "Metric                              Mean Δ    Median Δ    "
        "95% CI                     p-value    W/T/L"
    )
    for metric in result["metrics"]:
        overall = result["results"][metric]["overall_run_level"]
        interval = overall["bootstrap_ci_95"]
        pvalue = overall["exact_sign_flip_pvalue_two_sided"]
        pvalue_text = f"{pvalue:.6f}" if pvalue is not None else "n/a"
        print(
            f"{metric:<35} {overall['mean']:+.6f}  "
            f"{overall['median']:+.6f}    "
            f"[{interval[0]:+.6f}, {interval[1]:+.6f}]  "
            f"{pvalue_text:>8}    "
            f"{overall['wins']}/{overall['ties']}/{overall['losses']}"
        )


def ensure_outputs_do_not_exist(args: argparse.Namespace) -> None:
    collisions = [
        path
        for path in (args.output_json, args.output_csv)
        if path is not None and path.exists()
    ]
    if collisions:
        joined = ", ".join(str(path) for path in collisions)
        raise FileExistsError(
            f"Refusing to overwrite output file(s): {joined}. Remove or rename "
            "them, or choose different output paths."
        )


def write_outputs(args: argparse.Namespace, result: dict[str, Any]) -> None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    observations = [
        observation
        for metric in result["metrics"]
        for observation in result["results"][metric]["observations"]
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "domain",
                "run",
                "baseline",
                "comparison",
                "difference",
            ],
        )
        writer.writeheader()
        writer.writerows(observations)


def main() -> None:
    args = parse_args()
    ensure_outputs_do_not_exist(args)
    result = build_comparison(args)
    print_report(result)
    write_outputs(args, result)


if __name__ == "__main__":
    main()
