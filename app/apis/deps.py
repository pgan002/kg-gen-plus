from fastapi import Depends
from typing import Annotated

from kg_gen.kg_gen import KGGen

_kg_gen = None


def get_kg_gen() -> KGGen:
    global _kg_gen
    if _kg_gen is None:
        _kg_gen = KGGen()
    return _kg_gen


CurrentKGGen = Annotated[KGGen, Depends(get_kg_gen)]
