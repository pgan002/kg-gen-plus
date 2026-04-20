import logging

from kg_gen.models import Graph


def test_generate_with_stats(mock_kg_gen):
    input_data = "This is a test."
    graph, stats = mock_kg_gen.generate(input_data, deduplication_method=None)

    # Check the generated graph
    assert isinstance(graph, Graph)
    assert "entity1" in graph.entities
    assert "entity2" in graph.entities
    assert len(graph.relations) == 1
    assert graph.relations[0].subject == "entity1"

    # Check the statistics
    logging.warning(f"{stats = }")

    assert stats.get_entities
    assert stats.get_entities.lm_usage.total_tokens == 30
    assert stats.get_entities.execution_time > 0

    assert stats.type_terms.lm_usage.total_tokens == 40
    assert stats.type_terms.execution_time > 0

    assert stats.get_relations_typed.lm_usage.total_tokens == 50
    assert stats.get_relations_typed.execution_time > 0
