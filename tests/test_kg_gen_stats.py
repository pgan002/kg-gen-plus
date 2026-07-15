import pytest

from kg_gen.models import Graph, InputData


@pytest.mark.asyncio
async def test_generate_with_stats(mock_kg_gen):
    input_data = "This is a test."
    graph, stats = await mock_kg_gen.generate(
        InputData(text=input_data, id="test"), deduplicate=False
    )

    # Check the generated graph
    assert isinstance(graph, Graph)
    # entities is a computed field returning set[Entity]
    entity_names = {e.surface_form for e in graph.entities}
    assert "entity1" in entity_names
    assert "entity2" in entity_names
    assert len(graph.relations) == 3
    assert graph.relations[0].subject.surface_form == "entity1"

    # Check the statistics
    assert stats.extract_entities
    assert stats.extract_entities.lm_usage.total_tokens == 34
    assert stats.extract_entities.execution_time > 0

    assert stats.type_terms.lm_usage.total_tokens == 0
    assert stats.type_terms.execution_time == 0

    # assert stats.get_relations_typed.lm_usage.total_tokens == 50
    assert stats.get_relations_typed.execution_time > 0
