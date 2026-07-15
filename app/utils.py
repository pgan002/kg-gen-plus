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


ONTOLOGY_PREDICATES_DOC = """The ontology is parsed using the following predicates:
- rdf:type (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty)
- rdfs:label
- rdfs:comment
- rdfs:domain
- rdfs:range"""


def parse_ontology(onto_file: TextIO | BinaryIO) -> tuple[Ontology, rdflib.Graph]:
    """
    Load the ontology from a file. The file is expected to have a proper file extension or Media Type, for more info
    see rdflib documentation.

    {predicates}
    """.format(predicates=ONTOLOGY_PREDICATES_DOC)
    g = rdflib.Graph()
    g.parse(onto_file)
    return _extract_ontology(g), g


def parse_ontology_from_string(
    ttl: str, fmt: str = "turtle"
) -> tuple[Ontology, rdflib.Graph]:
    """
    Load an ontology from an in-memory string (Turtle by default).

    Same parsing rules as :func:`parse_ontology`; use this when the ontology is
    passed as text (e.g. through an MCP tool) rather than a file.
    """
    g = rdflib.Graph()
    g.parse(data=ttl, format=fmt)
    return _extract_ontology(g), g


def _extract_ontology(g: rdflib.Graph) -> Ontology:
    """Extract the structured :class:`Ontology` (classes + predicates) from a
    parsed RDF graph. See :data:`ONTOLOGY_PREDICATES_DOC` for the predicates used."""
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

    return ontology


def serialize_ontology_to_ttl(
    classes: list[EntityType], predicates: list[OntologyPredicate]
) -> str:
    """Serialize a set of classes and predicates into a Turtle (.ttl) ontology.

    Produces a standards-compliant RDF ontology: classes become ``owl:Class``,
    predicates become ``owl:ObjectProperty`` or ``owl:DatatypeProperty`` (the
    latter when declared as such or when every range URI is an XSD datatype),
    with ``rdfs:label``, ``rdfs:comment``, ``rdfs:domain`` and ``rdfs:range``.
    """
    from rdflib import Graph as RDFGraph, RDFS, OWL, XSD, URIRef, RDF, Literal

    g = RDFGraph()
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)

    for entity_type in classes:
        if entity_type.uri:
            class_uri = URIRef(entity_type.uri)
            g.add((class_uri, RDF.type, OWL.Class))
            if entity_type.label:
                g.add((class_uri, RDFS.label, Literal(entity_type.label)))
            if entity_type.description:
                g.add((class_uri, RDFS.comment, Literal(entity_type.description)))

    def is_datatype(entity_type: EntityType) -> bool:
        if entity_type.uri:
            return entity_type.uri.startswith(str(XSD))
        return False

    for predicate in predicates:
        if predicate.uri:
            prop_uri = URIRef(predicate.uri)

            is_data_prop = False
            if predicate.property_type == "owl:DatatypeProperty":
                is_data_prop = True
            elif predicate.range:
                if all(is_datatype(r) for r in predicate.range):
                    is_data_prop = True

            prop_type = OWL.DatatypeProperty if is_data_prop else OWL.ObjectProperty
            g.add((prop_uri, RDF.type, prop_type))

            if predicate.label:
                g.add((prop_uri, RDFS.label, Literal(predicate.label)))
            if predicate.description:
                g.add((prop_uri, RDFS.comment, Literal(predicate.description)))

            for domain in predicate.domain:
                if domain.uri:
                    g.add((prop_uri, RDFS.domain, URIRef(domain.uri)))
            for range_ in predicate.range:
                if range_.uri:
                    g.add((prop_uri, RDFS.range, URIRef(range_.uri)))

    return g.serialize(format="turtle")
