import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field


class TripleNode(BaseModel):
    surfaceform: str
    uri: str


class EntityMention(TripleNode):
    mention_start_index: int | None


class Triple(BaseModel):
    subject: TripleNode
    predicate: TripleNode
    object: TripleNode


class SynthieTestItem(BaseModel):
    id_: int = Field(..., alias="id")
    text: str
    triplets: list[Triple]
    target_dict: dict[str, Any]
    num_tokens_dict: dict[str, int]
    entities: list[EntityMention]
    relations: list[TripleNode]


def iter_synthie_jsonl(file_path: str | Path) -> Iterator[SynthieTestItem]:
    with open(file_path) as f:
        for i, line in enumerate(f):
            data_line = json.loads(line)
            synth_item = SynthieTestItem(**data_line)
            yield synth_item
