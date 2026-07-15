from collections import OrderedDict
from typing import Optional

from kg_gen.kg_gen import KGGen

# Bounded LRU cache of KGGen instances. Bounding it prevents unbounded growth
# (and indefinite retention of API keys in memory) when many distinct
# model/api_base/api_key combinations are seen. Embedding models are shared
# process-wide (see kg_gen._get_shared_sentence_transformer), so cached KGGen
# instances themselves are lightweight.
_KG_GEN_CACHE_MAXSIZE = 32
_kg_gen_cache: "OrderedDict[str, KGGen]" = OrderedDict()


def get_kg_gen(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    retrieval_model: Optional[str] = None,
    model: str = "openai/gpt-5.4-mini",
    max_tokens: int = 16000,
    temperature: float = 0.0,
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
    cache_key = f"{model}-{max_tokens}-{temperature}-{api_base}-{api_key}-{enforce_type_conformance}-{enforce_domain_conformance}-{enforce_range_conformance}-{enforce_predicate_conformance}"

    cached = _kg_gen_cache.get(cache_key)
    if cached is not None:
        _kg_gen_cache.move_to_end(cache_key)
        return cached

    instance = KGGen(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_base=api_base,
        api_key=api_key,
        retrieval_model=retrieval_model,
        enforce_type_conformance=enforce_type_conformance,
        enforce_domain_conformance=enforce_domain_conformance,
        enforce_range_conformance=enforce_range_conformance,
        enforce_predicate_conformance=enforce_predicate_conformance,
    )
    _kg_gen_cache[cache_key] = instance
    if len(_kg_gen_cache) > _KG_GEN_CACHE_MAXSIZE:
        _kg_gen_cache.popitem(last=False)

    return instance
