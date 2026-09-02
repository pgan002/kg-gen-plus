"""Passing large tool arguments by reference instead of by value.

An MCP tool's arguments are written by the calling model, so anything inline is
paid for in that model's output tokens: the 702-entity list from a 200-document
run is ~32,000 tokens, and the ontology another ~4,700 on *every* call that
needs it. The tools already accepted a file path, but a path only works when the
server shares a filesystem with the caller -- a remote server cannot see it. A
blob handle closes that gap: the data goes up over plain HTTP, which the model
never has to type, and the handle is ~20 characters.
"""

import time

import pytest
from fastapi.testclient import TestClient

from app import blobs, settings
from app.server import app
from kg_gen.models import Relation, TypedEntity

client = TestClient(app)

ONTOLOGY_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <http://schema.org/> .
schema:Person a owl:Class ; rdfs:label "Person" .
schema:Place a owl:Class ; rdfs:label "Place" .
schema:birthPlace a owl:ObjectProperty ; rdfs:domain schema:Person ;
    rdfs:range schema:Place ; rdfs:label "birth place" ;
    rdfs:comment "Where a person was born." .
"""


@pytest.fixture(autouse=True)
def clear_blobs():
    blobs._memory.clear()
    yield
    blobs._memory.clear()


def upload(data: bytes) -> str:
    response = client.post("/api/blobs", content=data)
    assert response.status_code == 200, response.text
    return response.json()["blob"]


def test_round_trip():
    handle = upload(b'{"hello": "world"}')

    assert blobs.load(handle) == b'{"hello": "world"}'


def test_the_handle_is_content_addressed():
    """The same bytes twice must not accumulate two copies."""
    first = upload(b"same")
    second = upload(b"same")

    assert first == second
    assert len(blobs._memory) == 1


def test_multipart_and_raw_body_agree():
    raw = client.post("/api/blobs", content=b"payload").json()
    multipart = client.post(
        "/api/blobs", files={"file": ("x.json", b"payload", "application/json")}
    ).json()

    assert raw["blob"] == multipart["blob"]
    assert raw["bytes"] == multipart["bytes"] == len(b"payload")


def test_empty_upload_is_rejected():
    response = client.post("/api/blobs", content=b"")

    assert response.status_code == 400
    assert "No data to store" in response.json()["detail"]


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PAYLOAD_BYTES", 4)

    response = client.post("/api/blobs", content=b"more than four bytes")

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_a_blob_expires():
    handle = upload(b"transient")
    blob_id = handle.removeprefix(blobs.HANDLE_PREFIX)
    # Bring the expiry forward rather than sleeping out the real TTL.
    _, data = blobs._memory[blob_id]
    blobs._memory[blob_id] = (time.time() - 1, data)

    with pytest.raises(KeyError):
        blobs.load(handle)


def test_unknown_handle_says_what_to_do():
    from tools import _read_reference

    with pytest.raises(ValueError, match="Unknown or expired blob handle"):
        _read_reference("blob:nosuchthing")


class TestResolveList:
    def test_inline_still_passes_straight_through(self):
        """An already-materialised list is returned untouched -- the MCP layer
        has validated it against the declared model before the call."""
        from tools import _resolve_list

        inline = [TypedEntity(surface_form="Ada")]

        assert _resolve_list(inline, TypedEntity) is inline

    def test_from_a_path(self, tmp_path):
        from tools import _resolve_list

        path = tmp_path / "entities.json"
        path.write_text('[{"surface_form": "Ada"}]')

        resolved = _resolve_list(str(path), TypedEntity)

        assert [e.surface_form for e in resolved] == ["Ada"]

    def test_from_a_blob(self):
        from tools import _resolve_list

        handle = upload(b'[{"surface_form": "Ada"}]')

        resolved = _resolve_list(handle, TypedEntity)

        assert [e.surface_form for e in resolved] == ["Ada"]


class TestResolveOntology:
    def test_turtle_text_passes_through(self):
        from tools import resolve_ontology_ttl

        assert resolve_ontology_ttl(ONTOLOGY_TTL) == ONTOLOGY_TTL

    def test_none_passes_through(self):
        from tools import resolve_ontology_ttl

        assert resolve_ontology_ttl(None) is None

    def test_from_a_path(self, tmp_path):
        from tools import resolve_ontology_ttl

        path = tmp_path / "onto.ttl"
        path.write_text(ONTOLOGY_TTL)

        assert resolve_ontology_ttl(str(path)) == ONTOLOGY_TTL

    def test_from_a_blob(self):
        from tools import resolve_ontology_ttl

        handle = upload(ONTOLOGY_TTL.encode())

        assert resolve_ontology_ttl(handle) == ONTOLOGY_TTL

    def test_a_path_that_does_not_exist_is_treated_as_turtle(self):
        """Otherwise a typo'd path would fail as a missing file rather than as
        unparseable Turtle, which is the more informative error."""
        from tools import resolve_ontology_ttl

        assert resolve_ontology_ttl("/no/such/onto.ttl") == "/no/such/onto.ttl"


def test_validate_conformance_accepts_blobs_for_everything():
    import json

    import tools

    entities = [
        {
            "surface_form": "Ada",
            "type": {"label": "Person", "uri": "http://schema.org/Person"},
        },
        {
            "surface_form": "Bath",
            "type": {"label": "Place", "uri": "http://schema.org/Place"},
        },
    ]
    relations = [
        {
            "subject": {"surface_form": "Ada"},
            "predicate": {"surface_form": "birth place"},
            "object": {"surface_form": "Bath"},
        }
    ]

    report = tools.validate_conformance(
        typed_entities=upload(json.dumps(entities).encode()),
        relations=upload(json.dumps(relations).encode()),
        ontology_ttl=upload(ONTOLOGY_TTL.encode()),
        enforce_type_conformance=True,
        enforce_predicate_conformance=True,
    )

    assert report.score == 1.0 and report.conformant
    assert Relation  # the models the tool validated against


def test_suggest_predicates_drops_duplicated_class_descriptions():
    import tools
    from kg_gen.models import EntityType

    suggested = tools.suggest_predicates(
        ONTOLOGY_TTL,
        [
            EntityType(label="Person", uri="http://schema.org/Person"),
            EntityType(label="Place", uri="http://schema.org/Place"),
        ],
    )

    assert suggested, "expected birth place to be suggested"
    predicate = suggested[0]
    # The predicate keeps its own comment -- it is unique and states direction.
    assert predicate.description
    # Its domain/range classes do not repeat theirs.
    for entity_type in list(predicate.domain) + list(predicate.range):
        assert entity_type.description is None
        assert entity_type.label and entity_type.uri
