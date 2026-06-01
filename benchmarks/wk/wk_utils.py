import json
from pathlib import Path
from typing import Iterator

import rdflib
from pydantic import BaseModel

from kg_gen.models import Ontology, EntityType, OntologyPredicate, InputData


class Triple(BaseModel):
    sub: str
    rel: str
    obj: str


def iter_wk_chunked_jsonl(file_path: str | Path) -> Iterator[InputData]:
    with open(file_path) as f:
        for i, line in enumerate(f):
            data_line = json.loads(line)
            wk_item = InputData(**data_line)

            yield wk_item


def parse_ontology(onto_path: Path | str = "ont_7_space.ttl"):
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
    for pred_uri in g.subjects(
        predicate=rdflib.RDF.type, object=rdflib.OWL.ObjectProperty
    ):
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
