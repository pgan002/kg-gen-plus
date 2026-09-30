from typing import ClassVar

import numpy as np

from kg_gen.ontology.term_clustering import (
    OntologyTermClusterer,
    cluster_term_assignments,
    semantic_pair_allowed,
)


class FakeEncoder:
    vectors: ClassVar[dict[str, list[float]]] = {
        "plant part": [1.0, 0.0, 0.0],
        "part of plant": [0.99, 0.01, 0.0],
        "plant": [0.99, 0.01, 0.0],
        "event": [1.0, 0.0, 0.0],
        "occurrence": [1.0, 0.0, 0.0],
        "purchase game offering": [0.0, 1.0, 0.0],
        "buy game offering": [0.0, 0.99, 0.01],
    }

    def encode(self, sentences, *, normalize_embeddings):
        assert normalize_embeddings
        return np.asarray([self.vectors[term] for term in sentences])


def clusterer():
    return OntologyTermClusterer("unused", 0.9, encoder=FakeEncoder())


def test_semantic_guards_keep_broader_and_unrelated_terms_separate():
    assert semantic_pair_allowed("plant part", "part of plant")
    assert not semantic_pair_allowed("plant", "plant part")
    assert not semantic_pair_allowed("event", "occurrence")


def test_complete_link_clustering_does_not_bridge_blocked_pairs():
    counts = {
        "plant part": 1,
        "part of plant": 1,
        "plant": 1,
        "event": 1,
        "occurrence": 1,
        "purchase game offering": 1,
        "buy game offering": 1,
    }
    mapping = clusterer().canonical_map(counts)
    assert mapping["plant part"] == mapping["part of plant"]
    assert mapping["plant"] == "plant"
    assert mapping["event"] == "event"
    assert mapping["occurrence"] == "occurrence"
    assert mapping["purchase game offering"] == mapping["buy game offering"]


def test_clustering_retains_source_assignments_and_prefers_frequent_label():
    assignments = {
        "CQ1": {"part of plant"},
        "CQ2": {"plant part"},
        "CQ3": {"plant part"},
    }
    clustered = cluster_term_assignments(assignments, clusterer())
    assert clustered == {
        "CQ1": {"plant part"},
        "CQ2": {"plant part"},
        "CQ3": {"plant part"},
    }
