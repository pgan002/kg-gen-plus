"""The embedding model must not load unless something needs it.

It costs ~613 MB of RSS and ~10 s of startup, and its only consumer is the
*embedding* pass of deduplication -- which is opt-in. It used to load in
`KGGen.__init__` whenever a model name was configured, so a `deduplicate=false`
run paid for it in full. That matters against the worker's container memory
limit, especially once `n_parallel` threads are added on top.
"""

from unittest.mock import patch

import pytest

from kg_gen.kg_gen import KGGen
from kg_gen.models import Graph, TypedEntity


@pytest.fixture
def loader():
    """Patch the shared loader so no real model is downloaded."""
    with patch("kg_gen.kg_gen._get_shared_sentence_transformer") as mocked:
        yield mocked


def _kg_gen(**kwargs):
    return KGGen(model="openai/none", retrieval_model="some/embedding-model", **kwargs)


def test_constructing_does_not_load_the_model(loader):
    kg_gen = _kg_gen()

    loader.assert_not_called()
    assert kg_gen.retrieval_model is None
    assert kg_gen.retrieval_model_name == "some/embedding-model"


def test_string_only_deduplication_does_not_load_the_model(loader):
    kg_gen = _kg_gen()
    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="Cats"),
            TypedEntity(surface_form="Cat"),
        },
        relations_wo_class_assertions=[],
    )

    deduplicated, _ = kg_gen.deduplicate(graph, use_embeddings=False)

    loader.assert_not_called()
    # The string pass still did its job without any model.
    assert len(deduplicated.entities) == 1


def test_the_embedding_pass_does_load_it(loader):
    kg_gen = _kg_gen()

    model = kg_gen._embedding_model()

    loader.assert_called_once_with("some/embedding-model")
    assert model is loader.return_value


def test_it_loads_only_once_per_instance(loader):
    kg_gen = _kg_gen()

    first = kg_gen._embedding_model()
    second = kg_gen._embedding_model()

    assert first is second
    loader.assert_called_once()


def test_no_configured_model_still_raises(loader):
    kg_gen = KGGen(model="openai/none", retrieval_model=None)

    with pytest.raises(ValueError, match="No retrieval model provided"):
        kg_gen._embedding_model()

    loader.assert_not_called()


def test_an_explicitly_passed_model_is_used_as_is(loader):
    kg_gen = _kg_gen()
    sentinel = object()

    assert kg_gen._parse_embedding_model(sentinel) is sentinel
    loader.assert_not_called()
