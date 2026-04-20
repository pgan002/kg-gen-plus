from unittest.mock import patch

import dspy
import pytest

from kg_gen.config import settings
from src.kg_gen import KGGen


@pytest.fixture
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
            response_content = (
                '[[ ## entities ## ]]\n["entity1", "entity2"]\n[[ ## completed ## ]]'
            )
            usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        elif "predict their type/class" in prompt:
            response_content = '[[ ## typed_entities ## ]]\n[{"entity": "entity1", "type": "type1"}, {"entity": "entity2", "type": "type2"}]\n[[ ## completed ## ]]'
            usage = {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
        elif "Extract subject-predicate-object triples" in prompt:
            response_content = '[[ ## relations ## ]]\n[{"subject": "entity1", "predicate": "related_to", "object": "entity2"}]\n[[ ## completed ## ]]'
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


@pytest.fixture
def mock_kg_gen():
    with patch("dspy.LM", new=MockLM):
        dspy.settings.configure(track_usage=True)
        kg_gen = KGGen(model="mock-model", disable_cache=True)
        return kg_gen
