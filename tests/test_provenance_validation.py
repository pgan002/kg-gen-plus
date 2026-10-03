from __future__ import annotations

import pytest

from kg_gen.ontology.provenance_validation import (
    ProvenanceInvariantError,
    provenance_by_term,
    validate_canonical_assignment_provenance,
)


def test_provenance_by_term_unions_every_source():
    assert provenance_by_term(
        {"CQ1": {"plant part"}, "CQ2": {"plant part", "plant"}}
    ) == {"plant part": {"CQ1", "CQ2"}, "plant": {"CQ2"}}


def test_invariant_accepts_many_aliases_with_unioned_provenance():
    validate_canonical_assignment_provenance(
        {
            "CQ1": {"PlantParts"},
            "CQ2": {"plant_parts"},
            "CQ3": {"plant-parts"},
        },
        {
            "CQ1": {"plant part"},
            "CQ2": {"plant part"},
            "CQ3": {"plant part"},
        },
        {
            "PlantParts": "plant part",
            "plant_parts": "plant part",
            "plant-parts": "plant part",
        },
        stage="test normalization",
    )


def test_invariant_detects_lost_source_assignment():
    with pytest.raises(ProvenanceInvariantError, match="CQ2.*missing"):
        validate_canonical_assignment_provenance(
            {"CQ1": {"alias a"}, "CQ2": {"alias b"}},
            {"CQ1": {"canonical"}, "CQ2": set()},
            {"alias a": "canonical", "alias b": "canonical"},
            stage="test clustering",
        )


def test_invariant_detects_invented_source_assignment():
    with pytest.raises(ProvenanceInvariantError, match="CQ2.*unexpected"):
        validate_canonical_assignment_provenance(
            {"CQ1": {"alias"}, "CQ2": set()},
            {"CQ1": {"canonical"}, "CQ2": {"canonical"}},
            {"alias": "canonical"},
            stage="test clustering",
        )


def test_invariant_detects_unmapped_source_term():
    with pytest.raises(ProvenanceInvariantError, match="missing source terms"):
        validate_canonical_assignment_provenance(
            {"CQ1": {"alias"}},
            {"CQ1": set()},
            {},
            stage="test clustering",
        )
