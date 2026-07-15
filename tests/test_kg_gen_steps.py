import pytest
from kg_gen.models import InputData, TypedEntity


@pytest.mark.asyncio
async def test_generate_with_terms_only(mock_kg_gen):
    # Case: terms provided as strings (no types)
    input_data = InputData(
        text="Alice works at Google.", id="test_terms", terms=["Alice", "Google"]
    )

    graph, stats = await mock_kg_gen.generate(input_data)

    # Check that extract_entities was NOT called
    assert stats.extract_entities.lm_usage.total_tokens == 0
    # Check that type_terms WAS called
    assert stats.type_terms.lm_usage.total_tokens > 0

    # Check graph content
    entity_names = {e.surface_form for e in graph.entities}
    assert (
        "Alice" in entity_names or "entity1" in entity_names
    )  # MockLM returns entity1/entity2
    # Wait, MockLM in conftest.py returns entity1/entity2 regardless of input if it matches "predict their type/class"


@pytest.mark.asyncio
async def test_generate_with_already_typed_terms(mock_kg_gen):
    # Case: terms provided with types
    input_data = InputData(
        text="Alice works at Google.",
        id="test_already_typed",
        terms=[
            TypedEntity(surface_form="Alice", type={"label": "Person"}),
            TypedEntity(surface_form="Google", type={"label": "Organization"}),
        ],
    )

    graph, stats = await mock_kg_gen.generate(input_data)

    # Check that extract_entities was NOT called
    assert stats.extract_entities.lm_usage.total_tokens == 0
    # Check that type_terms was NOT called (all already typed)
    assert stats.type_terms.lm_usage.total_tokens == 0

    entity_names = {e.surface_form for e in graph.entities}
    assert "Alice" in entity_names
    assert "Google" in entity_names
