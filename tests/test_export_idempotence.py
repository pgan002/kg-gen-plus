from __future__ import annotations

import json

import pytest

from misc.export_cq4oe import existing_export_matches, write_json_atomic


def manifest(domain: str = "awo"):
    return {
        "schema_version": 1,
        "task": "cq2term",
        "run_name": "experiment-run-01",
        "run": 1,
        "method": "kggen",
        "model": "gemma4",
        "source_experiment_id": "gemma4-dedup",
        "generation": {"temperature": 0.7},
        "graph_processing": {"deduplicate": True},
        "term_processing": {"predicate_induction": "llm-domain"},
        "mode": None,
        "ontology_export": None,
        "source_manifests": {domain: f"/source/{domain}/manifest.json"},
        "source_runs": {domain: f"/source/{domain}/runs/01.json"},
        "domains": [domain],
    }


def paths(tmp_path):
    run_dir = tmp_path / "experiment-run-01"
    destination = run_dir / "terms" / "awo_terms.json"
    trace = run_dir / "metadata" / "awo_predicate_induction.json"
    return run_dir, destination, trace


def write_complete_export(tmp_path):
    run_dir, destination, trace = paths(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_text("[]")
    trace.parent.mkdir(parents=True)
    trace.write_text("{}")
    (run_dir / "manifest.json").write_text(json.dumps(manifest()))
    return run_dir, destination, trace


def test_matching_complete_export_is_skipped(tmp_path):
    _, destination, trace = write_complete_export(tmp_path)

    assert existing_export_matches(
        destination, manifest(), "awo", [destination, trace], {}
    )


def test_configuration_mismatch_refuses_reuse(tmp_path):
    _, destination, trace = write_complete_export(tmp_path)
    incoming = manifest()
    incoming["term_processing"] = {"predicate_induction": "none"}

    with pytest.raises(FileExistsError, match="configuration differs"):
        existing_export_matches(
            destination, incoming, "awo", [destination, trace], {}
        )


def test_existing_directory_without_manifest_refuses_reuse(tmp_path):
    _, destination, trace = paths(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_text("[]")

    with pytest.raises(FileExistsError, match="missing manifest.json"):
        existing_export_matches(
            destination, manifest(), "awo", [destination, trace], {}
        )


def test_missing_required_trace_refuses_reuse(tmp_path):
    run_dir, destination, trace = paths(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_text("[]")
    (run_dir / "manifest.json").write_text(json.dumps(manifest()))

    with pytest.raises(FileExistsError, match="missing required files"):
        existing_export_matches(
            destination, manifest(), "awo", [destination, trace], {}
        )


def test_completed_manifest_resumes_a_domain_not_yet_recorded(tmp_path):
    run_dir, _, _ = paths(tmp_path)
    run_dir.mkdir(parents=True)
    existing = manifest("awo")
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(existing))
    destination = run_dir / "terms" / "odrl_terms.json"
    trace = run_dir / "metadata" / "odrl_predicate_induction.json"
    pending = {}

    assert not existing_export_matches(
        destination,
        manifest("odrl"),
        "odrl",
        [destination, trace],
        pending,
    )
    assert pending[manifest_path]["domains"] == ["awo"]


def test_atomic_manifest_write_replaces_existing_document(tmp_path):
    path = tmp_path / "manifest.json"
    write_json_atomic(path, {"domains": ["awo"]})
    write_json_atomic(path, {"domains": ["awo", "odrl"]})

    assert json.loads(path.read_text()) == {"domains": ["awo", "odrl"]}
    assert not path.with_suffix(".json.tmp").exists()


def test_pending_manifest_allows_other_domain_files_in_same_invocation(tmp_path):
    run_dir, destination, trace = paths(tmp_path)
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"

    assert not existing_export_matches(
        destination,
        manifest(),
        "awo",
        [destination, trace],
        {manifest_path: manifest()},
    )
