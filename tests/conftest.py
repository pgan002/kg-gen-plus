from unittest.mock import patch

import dspy
import pytest

from kg_gen.config import settings
from kg_gen.kg_gen import KGGen


@pytest.fixture(scope="session")
def kg():
    assert not settings.llm_api_key, (
        f"LLM_API_KEY environment variable is set. {settings.llm_api_key = }"
    )
    return KGGen(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        retrieval_model=settings.retrieval_model,
    )


class MockLM(dspy.LM):
    def __init__(self, *args, **kwargs):
        super().__init__("mock-model")
        self.history = []

    def __call__(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        # Simulate different responses based on the prompt content
        if "Extract key entities" in prompt:
            response_content = '[[ ## entities ## ]]\n[{"surface_form": "entity1", "uri": "Q1"}, {"surface_form": "entity2", "uri": "Q2"}]\n[[ ## completed ## ]]'
            usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        elif "predict their type/class" in prompt:
            response_content = (
                "[[ ## typed_entities ## ]]\n"
                '[{"surface_form": "entity1", "uri": "Q1", "type": {"label": "type1", "uri": "U1"}}, '
                '{"surface_form": "entity2", "uri": "Q2", "type": {"label": "type2", "uri": "U2"}}]\n[[ ## completed ## ]]'
            )
            usage = {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
        elif "Extract subject-predicate-object triples" in prompt:
            response_content = '[[ ## relations ## ]]\n[{"subject": {"surface_form": "entity1", "uri": "Q1"}, "predicate": {"surface_form": "related_to", "uri": "P1"}, "object": {"surface_form": "entity2", "uri": "Q2"}}]\n[[ ## completed ## ]]'
            usage = {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}
        else:
            response_content = "{}"
            usage = {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}

        response = {
            "model": "mock-model",
            "choices": [{"message": {"content": response_content}}],
            "usage": usage,
        }
        self.history.append(
            {
                "prompt": prompt,
                "response": response,
                "kwargs": kwargs,
                "usage": usage,
                "messages": kwargs["messages"],
            }
        )
        return [{"text": response_content}]


@pytest.fixture(scope="session")
def mock_kg_gen():
    with patch("dspy.LM", new=MockLM):
        dspy.settings.configure(track_usage=True)
        kg_gen = KGGen(model="mock-model", disable_cache=True)
        return kg_gen
