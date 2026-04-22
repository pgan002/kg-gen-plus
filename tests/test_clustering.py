from unittest.mock import patch, MagicMock
import pytest

from kg_gen.steps._3_deduplicate import DeduplicateMethod
from kg_gen.kg_gen import KGGen
from kg_gen.models import Graph, Entity, Relation, TypedEntity
from kg_gen.config import settings


def test_empty_graph_clustering(kg: KGGen):
    # Test with empty graph
    empty_graph = Graph(typed_entities=set(), relations_wo_class_assertions=[])
    clustered, stats = kg.deduplicate(empty_graph)

    assert len(clustered.entities) == 0
    assert len(clustered.edges) == 0
    assert len(clustered.relations) == 0
    assert clustered.entity_clusters is None
    assert clustered.edge_clusters is None


@pytest.mark.skipif(not settings.llm_api_key, reason="LLM API key not set")
def test_single_item_clustering(kg: KGGen):
    # Test with single items
    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="person"),
            TypedEntity(surface_form="home"),
        },
        relations_wo_class_assertions=[
            Relation(
                subject=TypedEntity(surface_form="person"),
                predicate=Entity(surface_form="walks"),
                object=TypedEntity(surface_form="home"),
            )
        ],
    )

    clustered, _ = kg.deduplicate(graph)

    # Check that relations are preserved
    assert len(clustered.relations) == len(graph.relations)

    # Validate cluster mappings exist
    assert clustered.entity_clusters is not None
    assert clustered.edge_clusters is not None

    # Check that each entity appears in some cluster
    for entity in graph.entities:
        found = False
        for cluster in clustered.entity_clusters.values():
            if entity in cluster:
                found = True
                break
        assert found, f"Entity {entity} not found in any cluster"

    # Check that each edge appears in some cluster
    for edge in graph.edges:
        found = False
        for cluster in clustered.edge_clusters.values():
            if edge.surface_form in cluster:
                found = True
                break
        assert found, f"Edge {edge} not found in any cluster"


def test_semhash_deduplication(kg: KGGen):
    """
    Test SEMHASH deduplication method.
    SEMHASH should catch:
    - Plurals vs singulars (cat/cats, dog/dogs)
    - Case variations (Person/person/PERSON)
    - Very similar strings through normalization

    SEMHASH should NOT catch:
    - True synonyms (CEO/Chief Executive Officer)
    - Semantic equivalents (joyful/happy)
    """
    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="cat"),
            TypedEntity(surface_form="cats"),
            TypedEntity(surface_form="kitten"),  # Plurals - should be caught
            TypedEntity(surface_form="dog"),
            TypedEntity(surface_form="dogs"),  # Plurals - should be caught
            TypedEntity(surface_form="Person"),
            TypedEntity(surface_form="person"),
            TypedEntity(surface_form="PERSON"),  # Case variations - should be caught
            TypedEntity(surface_form="CEO"),
            TypedEntity(
                surface_form="Chief Executive Officer"
            ),  # Synonyms - should NOT be caught by semhash
            TypedEntity(surface_form="happy"),
            TypedEntity(
                surface_form="joyful"
            ),  # Synonyms - should NOT be caught by semhash
        },
        relations_wo_class_assertions=[
            Relation(
                subject=TypedEntity(surface_form="cat"),
                predicate=Entity(surface_form="likes"),
                object=TypedEntity(surface_form="dog"),
            ),
            Relation(
                subject=TypedEntity(surface_form="cats"),
                predicate=Entity(surface_form="like"),
                object=TypedEntity(surface_form="dogs"),
            ),
            Relation(
                subject=TypedEntity(surface_form="Person"),
                predicate=Entity(surface_form="manages"),
                object=TypedEntity(surface_form="CEO"),
            ),
            Relation(
                subject=TypedEntity(surface_form="person"),
                predicate=Entity(surface_form="Manages"),
                object=TypedEntity(surface_form="Chief Executive Officer"),
            ),
            Relation(
                subject=TypedEntity(surface_form="CEO"),
                predicate=Entity(surface_form="supervises"),
                object=TypedEntity(surface_form="happy"),
            ),
            Relation(
                subject=TypedEntity(surface_form="Chief Executive Officer"),
                predicate=Entity(surface_form="oversees"),
                object=TypedEntity(surface_form="joyful"),
            ),
        ],
    )

    deduplicated, stats = kg.deduplicate(
        graph=graph,
        method=DeduplicateMethod.SEMHASH,
        semhash_similarity_threshold=0.95,
    )

    # SEMHASH should merge plurals
    assert (
        TypedEntity(surface_form="cat") in deduplicated.typed_entities
        or TypedEntity(surface_form="cats") in deduplicated.typed_entities
    )
    assert not (
        TypedEntity(surface_form="cat") in deduplicated.typed_entities
        and TypedEntity(surface_form="cats") in deduplicated.typed_entities
    )

    # SEMHASH should merge case variations
    deduped_person = [
        e for e in deduplicated.entities if e.surface_form.lower() == "person"
    ]
    assert len(deduped_person) == 1, "Case variations should be merged to one"

    # SEMHASH should NOT merge true synonyms (they're different words)
    # Both CEO and Chief Executive Officer should still exist
    ceo_variants = [
        e
        for e in deduplicated.entities
        if "ceo" in e.surface_form.lower()
        or "chief executive" in e.surface_form.lower()
    ]
    assert len(ceo_variants) == 2, (
        "SEMHASH should not merge CEO and Chief Executive Officer"
    )

    # SEMHASH should NOT merge happy and joyful (different words)
    emotion_variants = [
        e
        for e in deduplicated.entities
        if e.surface_form.lower() in ["happy", "joyful"]
    ]
    assert len(emotion_variants) == 2, "SEMHASH should not merge happy and joyful"

    # Check edges
    assert Entity(surface_form="supervises") in deduplicated.edges
    assert Entity(surface_form="oversees") in deduplicated.edges
    assert Entity(surface_form="supervises") != Entity(surface_form="oversees"), (
        "SEMHASH should not merge synonym edges"
    )

    print(
        f"Original entities: {len(graph.entities)}, Deduplicated: {len(deduplicated.entities)}"
    )
    print(
        f"Original edges: {len(graph.edges)}, Deduplicated: {len(deduplicated.edges)}"
    )


def test_lm_based_deduplication(kg: KGGen):
    """
    Test LM_BASED deduplication method with mocks.
    """

    def mock_side_effect(item, item_set):
        mock_response = MagicMock()
        if item.surface_form in ["CEO", "Chief Executive Officer"]:
            mock_response.alias = TypedEntity(surface_form="CEO")
            mock_response.duplicates = [
                e for e in item_set if e.surface_form == "Chief Executive Officer"
            ]
        elif item.surface_form in ["USA", "United States of America"]:
            mock_response.alias = TypedEntity(surface_form="United States of America")
            mock_response.duplicates = [e for e in item_set if e.surface_form == "USA"]
        elif item.surface_form in ["happy", "joyful", "glad"]:
            mock_response.alias = TypedEntity(surface_form="happy")
            mock_response.duplicates = [
                e for e in item_set if e.surface_form in ["joyful", "glad"]
            ]
        elif item.surface_form in ["big", "large"]:
            mock_response.alias = TypedEntity(surface_form="large")
            mock_response.duplicates = [e for e in item_set if e.surface_form == "big"]
        elif item.surface_form in ["automobile", "car", "vehicle"]:
            mock_response.alias = TypedEntity(surface_form="car")
            mock_response.duplicates = [
                e for e in item_set if e.surface_form in ["automobile", "vehicle"]
            ]
        elif item.surface_form in ["manages", "oversees", "supervises"]:
            mock_response.alias = Entity(surface_form="manages")
            mock_response.duplicates = [
                e for e in item_set if e.surface_form in ["oversees", "supervises"]
            ]
        elif item.surface_form in ["running", "runs", "run"]:
            mock_response.alias = Entity(surface_form="run")
            mock_response.duplicates = [
                e for e in item_set if e.surface_form in ["running", "runs"]
            ]
        elif item.surface_form in ["possesses", "owns", "has"]:
            mock_response.alias = Entity(surface_form="owns")
            mock_response.duplicates = [
                e for e in item_set if e.surface_form in ["possesses", "has"]
            ]
        else:
            mock_response.alias = item
            mock_response.duplicates = []
        return mock_response

    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="CEO"),
            TypedEntity(surface_form="Chief Executive Officer"),
            TypedEntity(surface_form="USA"),
            TypedEntity(surface_form="United States of America"),
            TypedEntity(surface_form="happy"),
            TypedEntity(surface_form="joyful"),
            TypedEntity(surface_form="glad"),
            TypedEntity(surface_form="big"),
            TypedEntity(surface_form="large"),
            TypedEntity(surface_form="automobile"),
            TypedEntity(surface_form="car"),
            TypedEntity(surface_form="vehicle"),
        },
        relations_wo_class_assertions=[
            Relation(
                subject=TypedEntity(surface_form="CEO"),
                predicate=Entity(surface_form="manages"),
                object=TypedEntity(surface_form="USA"),
            ),
            Relation(
                subject=TypedEntity(surface_form="Chief Executive Officer"),
                predicate=Entity(surface_form="oversees"),
                object=TypedEntity(surface_form="United States of America"),
            ),
            Relation(
                subject=TypedEntity(surface_form="happy"),
                predicate=Entity(surface_form="running"),
                object=TypedEntity(surface_form="big"),
            ),
            Relation(
                subject=TypedEntity(surface_form="joyful"),
                predicate=Entity(surface_form="runs"),
                object=TypedEntity(surface_form="large"),
            ),
            Relation(
                subject=TypedEntity(surface_form="automobile"),
                predicate=Entity(surface_form="possesses"),
                object=TypedEntity(surface_form="USA"),
            ),
            Relation(
                subject=TypedEntity(surface_form="car"),
                predicate=Entity(surface_form="owns"),
                object=TypedEntity(surface_form="United States of America"),
            ),
        ],
    )

    with patch("dspy.Predict") as mock_predict_class:
        mock_predict_instance = MagicMock()
        mock_predict_instance.side_effect = mock_side_effect
        mock_predict_class.return_value = mock_predict_instance
        deduplicated, _ = kg.deduplicate(
            graph=graph,
            method=DeduplicateMethod.LM_BASED,
        )

    ceo_count = sum(
        1
        for e in deduplicated.entities
        if "ceo" in e.surface_form.lower() or "chief" in e.surface_form.lower()
    )
    assert ceo_count == 1, "LM_BASED should merge CEO and Chief Executive Officer"

    usa_count = sum(
        1
        for e in deduplicated.entities
        if "usa" in e.surface_form.lower() or "united states" in e.surface_form.lower()
    )
    assert usa_count == 1, "LM_BASED should merge USA and United States of America"

    happy_emotions = [
        e
        for e in deduplicated.entities
        if e.surface_form.lower() in ["happy", "joyful", "glad"]
    ]
    assert len(happy_emotions) == 1, (
        f"LM_BASED should merge happy/joyful/glad, but got: {happy_emotions}"
    )

    size_words = [
        e for e in deduplicated.entities if e.surface_form.lower() in ["big", "large"]
    ]
    assert len(size_words) == 1, (
        f"LM_BASED should merge big/large, but got: {size_words}"
    )

    manage_edges = [
        e
        for e in deduplicated.edges
        if e.surface_form.lower() in ["manages", "oversees", "supervises"]
    ]
    assert len(manage_edges) == 1, (
        f"LM_BASED should merge management synonyms, but got: {manage_edges}"
    )

    run_edges = [e for e in deduplicated.edges if "run" in e.surface_form.lower()]
    assert len(run_edges) == 1, (
        f"LM_BASED should merge run tense variations, but got: {run_edges}"
    )

    print(
        f"Original entities: {len(graph.entities)}, Deduplicated: {len(deduplicated.entities)}"
    )
    print(
        f"Original edges: {len(graph.edges)}, Deduplicated: {len(deduplicated.edges)}"
    )
    print(f"Deduplicated entities: {deduplicated.entities}")
    print(f"Deduplicated edges: {deduplicated.edges}")


@pytest.mark.skipif(not settings.llm_api_key, reason="LLM API key not set")
def test_full_deduplication_comprehensive(kg: KGGen):
    """
    Test FULL deduplication method.
    """
    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="cat"),
            TypedEntity(surface_form="cats"),
            TypedEntity(surface_form="Cat"),
            TypedEntity(surface_form="CATS"),
            TypedEntity(surface_form="CEO"),
            TypedEntity(surface_form="Chief Executive Officer"),
            TypedEntity(surface_form="USA"),
            TypedEntity(surface_form="United States of America"),
        },
        relations_wo_class_assertions=[
            Relation(
                subject=TypedEntity(surface_form="cat"),
                predicate=Entity(surface_form="likes"),
                object=TypedEntity(surface_form="CEO"),
            ),
            Relation(
                subject=TypedEntity(surface_form="cats"),
                predicate=Entity(surface_form="like"),
                object=TypedEntity(surface_form="Chief Executive Officer"),
            ),
            Relation(
                subject=TypedEntity(surface_form="Cat"),
                predicate=Entity(surface_form="Manages"),
                object=TypedEntity(surface_form="USA"),
            ),
            Relation(
                subject=TypedEntity(surface_form="CATS"),
                predicate=Entity(surface_form="manages"),
                object=TypedEntity(surface_form="United States of America"),
            ),
            Relation(
                subject=TypedEntity(surface_form="CEO"),
                predicate=Entity(surface_form="supervises"),
                object=TypedEntity(surface_form="cat"),
            ),
            Relation(
                subject=TypedEntity(surface_form="Chief Executive Officer"),
                predicate=Entity(surface_form="oversees"),
                object=TypedEntity(surface_form="cats"),
            ),
        ],
    )

    deduplicated, _ = kg.deduplicate(
        graph=graph,
        method=DeduplicateMethod.FULL,
        semhash_similarity_threshold=0.95,
    )

    cat_variants = [e for e in deduplicated.entities if "cat" in e.surface_form.lower()]
    assert len(cat_variants) == 1, (
        f"FULL should merge all cat variations, but got: {cat_variants}"
    )

    ceo_count = sum(
        1
        for e in deduplicated.entities
        if "ceo" in e.surface_form.lower() or "chief" in e.surface_form.lower()
    )
    assert ceo_count == 1, "FULL should merge CEO and Chief Executive Officer"

    usa_count = sum(
        1
        for e in deduplicated.entities
        if "usa" in e.surface_form.lower() or "united states" in e.surface_form.lower()
    )
    assert usa_count == 1, "FULL should merge USA and United States of America"

    like_edges = [e for e in deduplicated.edges if "like" in e.surface_form.lower()]
    assert len(like_edges) == 1, (
        f"FULL should merge like variations, but got: {like_edges}"
    )

    manage_edges = [e for e in deduplicated.edges if "manage" in e.surface_form.lower()]
    assert len(manage_edges) == 1, (
        f"FULL should merge manages variations, but got: {manage_edges}"
    )

    supervise_edges = [
        e
        for e in deduplicated.edges
        if e.surface_form.lower() in ["supervises", "oversees"]
    ]
    assert len(supervise_edges) <= 2, (
        f"Got unexpected supervise edges: {supervise_edges}"
    )

    assert len(deduplicated.entities) <= 5, (
        f"FULL should significantly reduce entities, but got {len(deduplicated.entities)}"
    )
    assert len(deduplicated.edges) <= 4, (
        f"FULL should significantly reduce edges, but got {len(deduplicated.edges)}"
    )

    print(
        f"Original entities: {len(graph.entities)}, Deduplicated: {len(deduplicated.entities)}"
    )
    print(
        f"Original edges: {len(graph.edges)}, Deduplicated: {len(deduplicated.edges)}"
    )
    print(f"Final entities: {deduplicated.entities}")
    print(f"Final edges: {deduplicated.edges}")
