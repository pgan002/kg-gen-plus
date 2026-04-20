from typing import List, Optional
from pathlib import Path
import dspy
import litellm

from kg_gen.models import TextEntities, ConversationEntities, EntitiesResponse, TypedEntity, TypedEntities


def _load_entities_prompt() -> str:
    """Load the entities prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "entities.txt"
    return prompt_path.read_text()


def _get_entities_litellm(
    input_data: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
) -> List[str]:
    prompt_template = _load_entities_prompt()
    user_prompt = f"""
Here is the text to extract entities from:

<article>
{input_data}
</article>
    """

    # Build schema with additionalProperties: false (required by OpenAI)
    schema = EntitiesResponse.model_json_schema()
    schema["additionalProperties"] = False

    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "entities_response",
                "schema": schema,
                "strict": True,
            }
        },
    }

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.responses(**kwargs)
    # print(response.model_dump_json(indent=2))
    parsed = EntitiesResponse.model_validate_json(response.output[-1].content[0].text)
    return parsed.entities


def get_entities(
    input_data: str,
    temperature: float = 0.0,
    types: list[str] = None,
    context: str = None
) -> List[str]:
    extract = dspy.Predict(TextEntities, temperature=temperature)
    if types is None:
        result = extract(source_text=input_data, context=context)
    else:
        result = extract(source_text=input_data, context=context, types_to_extract=types)
    return result.entities


def type_terms(input_data: str,
               terms: list[str],
               types: Optional[list[str]] = None,
               temperature: float = 0.0,
               context: Optional[str] = None
               ) -> List[TypedEntity]:
    predict_type = dspy.Predict(TypedEntities, temperature=temperature)
    result = predict_type(entities=terms, types=types, source_text=input_data, context=context)
    return result.typed_entities
