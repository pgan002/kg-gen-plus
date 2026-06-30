from typing import Optional

from kg_gen.kg_gen import KGGen

_kg_gen_cache = {}


def get_kg_gen(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    retrieval_model: Optional[str] = None,
    model: str = "openai/gpt-5.4-mini",
    enforce_type_conformance: bool = False,
    enforce_domain_conformance: bool = True,
    enforce_range_conformance: bool = True,
    enforce_predicate_conformance: bool = False,
) -> KGGen:
    """
    Dependency to get a KGGen instance.
    It creates a new instance if the parameters are different from the cached one.
    This allows for dynamic model selection per API call.
    """
    cache_key = f"{model}-{api_base}-{api_key}-{enforce_type_conformance}-{enforce_domain_conformance}-{enforce_range_conformance}-{enforce_predicate_conformance}"

    if cache_key not in _kg_gen_cache:
        _kg_gen_cache[cache_key] = KGGen(
            model=model,
            api_base=api_base,
            api_key=api_key,
            retrieval_model=retrieval_model,
            enforce_type_conformance=enforce_type_conformance,
            enforce_domain_conformance=enforce_domain_conformance,
            enforce_range_conformance=enforce_range_conformance,
            enforce_predicate_conformance=enforce_predicate_conformance,
        )

    return _kg_gen_cache[cache_key]
