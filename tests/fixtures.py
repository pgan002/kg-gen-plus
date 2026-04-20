from src.kg_gen import KGGen
from kg_gen.config import settings
import pytest


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
