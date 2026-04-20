import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from kg_gen import KGGen
from kg_gen.steps._3_deduplicate import DeduplicateMethod


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


if __name__ == "__main__":
    synthie_base_data_path = Path(__file__).parent / "data" / "synthie"
    davinci2_test_small_path = synthie_base_data_path / "test_small_ordered.jsonl"

    # keycloak_token = get_keycloak_token()
    kg = KGGen(
        # Use 'openai/' prefix to force standard HTTP client with Bearer token auth
        # model="openai/gpt-oss:120b",
        model="openai/gpt-5.4-mini",
        temperature=1.0,
        # api_base="https://ollama.dev.memorise.sdu.dk/v1",
        # api_key=keycloak_token
        retrieval_model="sentence-transformers/all-mpnet-base-v2",
    )

    gs = []
    for i, item in enumerate(iter_synthie_jsonl(davinci2_test_small_path), start=1):
        print(f"\n\n{i = }, {item.id_ = }\n{item.text = }\n")
        item_entities = [e.surfaceform for e in item.entities]
        g = kg.generate(
            input_data=item.text,
            relation_context="Use predicates from Wikidata for the extracted relations. "
            "Provide the Wikidata identifiers for the extracted relations, "
            'for example, "operator (P137)".',
            terms=item_entities,
            types=["Pick the types from Wikidata, add Wikidata identifier in brackets"],
            output_folder=str(synthie_base_data_path),
            deduplication_method=None,
        )
        print(f"{item.triplets = }")
        print(f"{g = }")
        gs.append(g)
        if i > 15:
            break
    agg_g = kg.aggregate(gs)
    agg_g = kg.deduplicate(
        graph=agg_g,
        # method=DeduplicateMethod.FULL
        # method=DeduplicateMethod.SEMHASH,
        # semhash_similarity_threshold=0.5
        method=DeduplicateMethod.LM_BASED,
    )
    kg.export_graph(graph=agg_g, output_path=str(synthie_base_data_path / "graph.json"))
    kg.visualize(agg_g, str(synthie_base_data_path / "graph.html"), True)
