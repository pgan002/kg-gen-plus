import json
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from app.server import app
from kg_gen.models import Graph, KGGenStats, Relation, TypedEntity, StepStats

client = TestClient(app)


@pytest.fixture
def mock_kg_gen():
    # Patched in both modules: the generation helpers live in app.generation
    # (shared with the background worker), while /aggregate_and_deduplicate still
    # calls get_kg_gen directly from the router module.
    with (
        patch("app.generation.get_kg_gen") as mocked_get,
        patch("app.apis.kg_construct.get_kg_gen", new=mocked_get),
    ):
        kg_gen = MagicMock()
        kg_gen.generate = AsyncMock()
        kg_gen.aggregate = MagicMock()
        kg_gen.deduplicate = MagicMock()
        mocked_get.return_value = kg_gen
        yield kg_gen


def test_generate_graph_parallel(mock_kg_gen):
    # Prepare mock data
    doc1 = {"id": "doc1", "text": "text1", "terms": []}
    doc2 = {"id": "doc2", "text": "text2", "terms": []}
    corpus_content = json.dumps(doc1) + "\n" + json.dumps(doc2) + "\n"

    # Setup the mock return values
    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="e1", type={"label": "T1"}),
            TypedEntity(surface_form="e3", type={"label": "T2"}),
        },
        relations_wo_class_assertions=[
            Relation(
                subject={"surface_form": "e1"},
                predicate={"surface_form": "p1"},
                object={"surface_form": "e2"},
            )
        ],
    )
    stats = KGGenStats()
    stats.deduplicate = StepStats(execution_time=0.1)

    mock_kg_gen.generate.return_value = (graph, stats)

    response = client.post(
        "/api/generate",
        params={"model": "test-model", "n_parallel": 5},
        files={
            "corpus_file": (
                "corpus.jsonl",
                corpus_content.encode("utf-8"),
                "application/jsonl",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["entities"]) == 2
    assert len(data["relations"]) == 1

    # Check structure
    for entity_id, entity_data in data["entities"].items():
        assert "surface_form" in entity_data
        assert "type" in entity_data

    for relation in data["relations"]:
        assert "subject_id" in relation
        assert "predicate" in relation
        assert "is_literal" in relation

    assert mock_kg_gen.generate.called
    # Stat headers are scalars only; the detail lives in the body under
    # `stats` (see tests/test_stat_headers.py for why).
    assert "X-KG-Gen-Stats" not in response.headers
    assert response.headers["X-KG-Gen-Time"] == "0.1"  # the mock's dedup time
    assert response.headers["X-KG-Gen-Failed-Documents"] == "0"
    assert "X-KG-Gen-Dedup-Stats" in response.headers


def test_generate_graph_invalid_jsonl():
    corpus_content = "invalid json\n"
    corpus_file = io.BytesIO(corpus_content.encode("utf-8"))

    response = client.post(
        "/api/generate",
        params={"model": "test-model"},
        files={"corpus_file": ("corpus.jsonl", corpus_file, "application/jsonl")},
    )

    assert response.status_code == 422
    assert "Invalid document format" in response.json()["detail"]


def test_generate_graph_wrong_extension():
    corpus_file = io.BytesIO(b"{}")
    response = client.post(
        "/api/generate",
        params={"model": "test-model"},
        files={"corpus_file": ("corpus.txt", corpus_file, "text/plain")},
    )
    assert response.status_code == 400
    assert "Corpus must be a .jsonl file" in response.json()["detail"]


def test_convert_ontology():
    input_data = {
        "classes": [
            {
                "uri": "http://example.org/Person",
                "label": "Person",
                "description": "A human",
            }
        ],
        "predicates": [
            {
                "uri": "http://example.org/knows",
                "label": "knows",
                "domain": [{"uri": "http://example.org/Person", "label": "Person"}],
                "range": [{"uri": "http://example.org/Person", "label": "Person"}],
            }
        ],
    }
    response = client.post("/api/convert_ontology", json=input_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-turtle"
    content = response.content.decode("utf-8")
    assert "example.org/Person" in content
    assert "example.org/knows" in content
    assert "owl:Class" in content
    assert "owl:ObjectProperty" in content
