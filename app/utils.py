from __future__ import annotations

from pathlib import Path

from typing import TextIO, BinaryIO

import rdflib
from pydantic import BaseModel
from rdflib.namespace import XSD

from kg_gen.models import Ontology, EntityType, OntologyPredicate

APP_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = (
    APP_DIR.parent / "src" / "kg_gen" / "utils" / "template.html"
).resolve()
DATA_ROOT = APP_DIR / "examples"
STATIC_DIR = APP_DIR / "static"


class ExampleGraph(BaseModel):
    slug: str
    title: str
    path: Path
    wiki_url: str | None = None


# EXAMPLE_GRAPHS: tuple[ExampleGraph, ...] = ()
# EXAMPLE_INDEX: dict[str, ExampleGraph] = dict()


def refresh_examples():
    # global EXAMPLE_GRAPHS, EXAMPLE_INDEX
    EXAMPLE_GRAPHS = ()
    for filename in DATA_ROOT.glob("*.json"):
        EXAMPLE_GRAPHS += (
            ExampleGraph(
                slug=filename.stem,
                title=filename.stem,
                path=filename,
                # wiki_url=f"https://en.wikipedia.org/wiki/{file.stem}",
            ),
        )
    EXAMPLE_INDEX = {
        example.slug: example for example in EXAMPLE_GRAPHS if example.path.exists()
    }
    # kggen_logger.warning(f'examples refreshed {EXAMPLE_INDEX = }')
    return EXAMPLE_INDEX


if not TEMPLATE_PATH.exists():
    raise RuntimeError(f"Template not found at {TEMPLATE_PATH}")


xsd_to_python_type = {
    XSD.string: EntityType(label="str", uri=str(XSD.string)),
    XSD.integer: EntityType(label="int", uri=str(XSD.integer)),
    XSD.decimal: EntityType(label="float", uri=str(XSD.decimal)),
    XSD.double: EntityType(label="float", uri=str(XSD.double)),
    XSD.boolean: EntityType(label="bool", uri=str(XSD.boolean)),
    XSD.date: EntityType(label="date", uri=str(XSD.date)),
    XSD.dateTime: EntityType(label="datetime", uri=str(XSD.dateTime)),
    XSD.time: EntityType(label="time", uri=str(XSD.time)),
    XSD.gYear: EntityType(label="gYear", uri=str(XSD.gYear)),
}
python_type_to_xsd = {v: k for k, v in xsd_to_python_type.items()}


def parse_ontology(onto_file: TextIO | BinaryIO) -> tuple[Ontology, rdflib.Graph]:
    """
    Load the ontology from a file. The file is expected to have a proper file extension or Media Type, for more info
    see rdflib documentation.

    The ontology is parsed using the following predicates:
    - rdf:type (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty)
    - rdfs:label
    - rdfs:comment
    - rdfs:domain
    - rdfs:range
    """
    g = rdflib.Graph()
    g.parse(onto_file)

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
            description = g.value(subject=class_uri, predicate=rdflib.RDFS.comment)

            entity_type = EntityType(
                label=label,
                uri=str(class_uri),
                description=str(description) if description else None,
            )
            ontology.classes.add(entity_type)
            class_map[class_uri] = entity_type

    # Second pass: identify all predicates and their domain/range
    object_properties = set(
        g.subjects(predicate=rdflib.RDF.type, object=rdflib.OWL.ObjectProperty)
    )
    datatype_properties = set(
        g.subjects(predicate=rdflib.RDF.type, object=rdflib.OWL.DatatypeProperty)
    )
    all_properties = object_properties | datatype_properties

    for pred_uri in all_properties:
        if isinstance(pred_uri, rdflib.URIRef):
            label = g.value(subject=pred_uri, predicate=rdflib.RDFS.label)
            if label:
                label = str(label)
            else:
                label = pred_uri.split("/")[-1].split("#")[-1]

            description = g.value(subject=pred_uri, predicate=rdflib.RDFS.comment)
            prop_type = (
                "owl:DatatypeProperty"
                if pred_uri in datatype_properties
                else "owl:ObjectProperty"
            )
            predicate = OntologyPredicate(
                label=label,
                uri=str(pred_uri),
                description=str(description) if description else None,
                property_type=prop_type,
            )
            # Get domain
            for domain_uri in g.objects(subject=pred_uri, predicate=rdflib.RDFS.domain):
                if domain_uri in class_map:
                    predicate.domain.add(class_map[domain_uri])
            # Get range
            for range_uri in g.objects(subject=pred_uri, predicate=rdflib.RDFS.range):
                if range_uri in class_map:
                    predicate.range.add(class_map[range_uri])
                elif range_uri in xsd_to_python_type:
                    predicate.range.add(xsd_to_python_type[range_uri])

            ontology.predicates.add(predicate)

    return ontology, g
