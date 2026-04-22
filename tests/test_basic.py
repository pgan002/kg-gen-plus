import pytest

from src.kg_gen.kg_gen import KGGen
from src.kg_gen.config import settings
from src.kg_gen.models import TypedEntity


def match_subset(set1: set[TypedEntity], set2: set[TypedEntity]) -> bool:
    """
    Check if set1 is a subset of set2, with fuzzy matching for similar names.
    set1, must be the expected set, and should contain less words for an entity, so fuzzy matching works in case.
    """
    diff = set1 - set2

    if len(diff) == 0:
        return True

    # Check for fuzzy matches for remaining entities
    for entity1 in diff:
        found_match = False
        for entity2 in set2:
            if (
                entity1.surface_form.lower() in entity2.surface_form.lower()
                or entity2.surface_form.lower() in entity1.surface_form.lower()
            ):
                found_match = True
                break
        if not found_match:
            return False

    return True


@pytest.mark.skipif(not settings.llm_api_key, reason="LLM API key not set")
def test_basic(kg: KGGen):
    # Generate a simple graph
    text = "Harry has two parents - his dad James Potter and his mom Lily Potter. Harry and his wife Ginny have three kids together: their oldest son James Sirius, their other son Albus, and their daughter Lily Luna."

    graph, _ = kg.generate(input_data=text)

    expected_entities = {
        TypedEntity(surface_form="Harry"),
        TypedEntity(surface_form="James Potter"),
        TypedEntity(surface_form="Lily Potter"),
        TypedEntity(surface_form="Ginny"),
        TypedEntity(surface_form="James Sirius"),
        TypedEntity(surface_form="Albus"),
        TypedEntity(surface_form="Lily Luna"),
    }
    print(graph)
    assert match_subset(expected_entities, graph.entities)


@pytest.mark.skipif(not settings.llm_api_key, reason="LLM API key not set")
def test_clustered(kg: KGGen):
    # Test texts
    text1 = "Linda is Joshua's mother. Ben is Josh's brother. Andrew is Josh's father."
    text2 = "Judy is Andrew's sister. Josh is Judy's nephew. Judy is Josh's aunt. Josh also goes by Joshua."

    # Generate individual graphs
    graph1, _ = kg.generate(
        input_data=text1,
        entity_context="Family relationships",
    )

    graph2, _ = kg.generate(
        input_data=text2,
        entity_context="Family relationships",
    )

    # # Aggregate the graphs
    combined_graph = kg.aggregate([graph1, graph2])

    # Cluster the combined graph
    clustered_graph, _ = kg.deduplicate(
        combined_graph,
    )
    expected_entities = {
        TypedEntity(surface_form="Linda"),
        TypedEntity(surface_form="Joshua"),
        TypedEntity(surface_form="Josh"),
        TypedEntity(surface_form="Ben"),
        TypedEntity(surface_form="Andrew"),
        TypedEntity(surface_form="Judy"),
    }
    expected_edges = {
        TypedEntity(surface_form="is mother of"),
        TypedEntity(surface_form="is brother of"),
        TypedEntity(surface_form="is father of"),
        TypedEntity(surface_form="is sister of"),
        TypedEntity(surface_form="is nephew of"),
        TypedEntity(surface_form="is aunt of"),
    }
    # TODO: with gpt-5 temperature 1.0, it makes tests not deterministic, thus `is brother of` could be `is isbling of`.
    # print(clustered_graph)
    print("entities:", clustered_graph.entities)
    print("edges:", clustered_graph.edges)
    assert match_subset(expected_entities, clustered_graph.entities)
    assert match_subset(expected_edges, clustered_graph.edges)

    print("\nGraph 1:")
    print("Entities:", graph1.entities)
    print("Relations:", graph1.relations)
    print("Edges:", graph1.edges)

    print("\nGraph 2:")
    print("Entities:", graph2.entities)
    print("Relations:", graph2.relations)
    print("Edges:", graph2.edges)

    print("\nCombined Graph:")
    print("Entities:", combined_graph.entities)
    print("Relations:", combined_graph.relations)
    print("Edges:", combined_graph.edges)

    print("\nClustered Combined Graph:")
    print("Entities:", clustered_graph.entities)
    print("Relations:", clustered_graph.relations)
    print("Edges:", clustered_graph.edges)
    print("Entity Clusters:", clustered_graph.entity_clusters)
    print("Edge Clusters:", clustered_graph.edge_clusters)


@pytest.mark.skipif(not settings.llm_api_key, reason="LLM API key not set")
def test_conversation(kg: KGGen):
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]

    graph, _ = kg.generate(
        input_data=str(messages),
    )
    expected_entities = {
        TypedEntity(surface_form="France"),
        TypedEntity(surface_form="Paris"),
    }
    assert match_subset(expected_entities, graph.entities)
    print(graph)
