import json
from pathlib import Path
from typing import Iterator, Optional

import httpx
import rdflib
from pydantic import BaseModel, Field

from kg_gen.models import Ontology, EntityType, OntologyPredicate, TypedEntity


class MusiqueChunk(BaseModel):
    chunk_id: str
    source_doc: str
    content: str
    terms: Optional[list[TypedEntity]] = Field(default_factory=list)


class ExtractedTerm(BaseModel):
    term: str
    pref_label: str
    alt_labels: list[str]
    definition: str
    lang: str = "en"
    rank: int = 0


def iter_musique_chunks_jsonl(
    file_path: str | Path,
) -> Iterator[MusiqueChunk]:
    with open(file_path) as f:
        for i, line in enumerate(f):
            data_line = json.loads(line)
            yield MusiqueChunk(**data_line)


def parse_ontology(onto_path: Path | str) -> Ontology:
    """
    Load the ontology from a file.
    """
    g = rdflib.Graph()
    g.parse(onto_path)

    ontology = Ontology()
    class_map = {}

    # First pass: identify all classes
    for class_uri in g.subjects(predicate=rdflib.RDF.type, object=rdflib.OWL.Class):
        if isinstance(class_uri, rdflib.URIRef):
            label = g.value(subject=class_uri, predicate=rdflib.RDFS.label)
            if label:
                label = str(label)
            else:
                label = class_uri.split("/")[-1].split("#")[-1]

            entity_type = EntityType(label=label, uri=str(class_uri))
            ontology.classes.add(entity_type)
            class_map[class_uri] = entity_type

    # Second pass: identify all predicates and their domain/range
    properties = set(
        g.subjects(predicate=rdflib.RDF.type, object=rdflib.OWL.ObjectProperty)
    )
    properties.update(
        g.subjects(predicate=rdflib.RDF.type, object=rdflib.OWL.DatatypeProperty)
    )

    for pred_uri in properties:
        if isinstance(pred_uri, rdflib.URIRef):
            label = g.value(subject=pred_uri, predicate=rdflib.RDFS.label)
            if label:
                label = str(label)
            else:
                label = pred_uri.split("/")[-1].split("#")[-1]

            description = g.value(subject=pred_uri, predicate=rdflib.RDFS.comment)
            predicate = OntologyPredicate(
                label=label,
                uri=str(pred_uri),
                description=str(description) if description else None,
            )
            # Get domain
            for domain_uri in g.objects(subject=pred_uri, predicate=rdflib.RDFS.domain):
                if domain_uri in class_map:
                    predicate.domain.add(class_map[domain_uri])
            # Get range
            for range_uri in g.objects(subject=pred_uri, predicate=rdflib.RDFS.range):
                if range_uri in class_map:
                    predicate.range.add(class_map[range_uri])

            ontology.predicates.add(predicate)

    return ontology


async def extract_terms_for_text(text: str) -> list[ExtractedTerm]:
    """Extracts terms for a single piece of text by sending to the external service async."""
    url = "http://dsx-gws-rai-docker-dmo-apl-n-01:8089/extract_from_text"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    params = {
        # "categories": "",
        # "questions": "",
        "model": "gpt-mini-4o",
        "window_size": "24000",
        "window_overlap_size": "1000",
    }
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=text, params=params, headers=headers)
    response.raise_for_status()
    out = [ExtractedTerm(**item) for item in response.json()]
    return out
