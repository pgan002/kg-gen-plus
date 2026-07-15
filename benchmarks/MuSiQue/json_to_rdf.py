import json
import re
import sys
from typing import Dict, Any, List, Optional

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

from benchmarks.MuSiQue.musique_config import musique_onto_path
from benchmarks.MuSiQue.musique_utils import iter_musique_chunks_jsonl

# Define namespaces
M = Namespace("http://example.org/ontology#")


def load_ontology(ontology_file: str) -> Dict[str, str]:
    """Load ontology and create a mapping from labels to URIs."""
    g = Graph()
    g.parse(ontology_file, format="turtle")
    label_to_uri: Dict[str, str] = {}
    for s, p, o in g.triples((None, RDFS.label, None)):
        if isinstance(o, Literal):
            label_to_uri[o.value.lower()] = str(s)
    return label_to_uri


def slugify(text: str) -> str:
    """Convert text to a slug suitable for URIs."""
    if not text:
        return "unknown"
    slug = re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "_")
    return slug


def resolve_uri(
    keys: List[Optional[str]],
    suggested_uri: Optional[str],
    ontology_mapping: Dict[str, str],
    g: Graph,
    base_prefix: Namespace = M,
) -> str:
    """Resolve a URI, checking the ontology (by ``keys``) first.

    ``keys`` are candidate labels/surface forms to look up in the ontology
    label->URI mapping. If none match, an explicit ``suggested_uri`` is used,
    otherwise a URI is minted from the first key.
    """
    for key in keys:
        if key and key.lower() in ontology_mapping:
            return ontology_mapping[key.lower()]

    if suggested_uri:
        try:
            return g.namespace_manager.expand_curie(suggested_uri)
        except ValueError:
            return suggested_uri

    minted = next((k for k in keys if k), None)
    return str(base_prefix[slugify(minted or "unknown")])


def literal_datatype(datatype: Optional[str]) -> Optional[URIRef]:
    """Map an ``xsd:<type>`` shorthand to its XSD URIRef, if recognised."""
    if not datatype:
        return None
    local = datatype.split(":", 1)[-1].split("#")[-1]
    return getattr(XSD, local, None)


def convert_to_turtle(
    json_data: Dict[str, Any],
    ontology_mapping: Dict[str, str],
    source_texts: Dict[str, str],
) -> str:
    """Convert a kg-gen KnowledgeGraph to Turtle with RDF-star provenance.

    The KnowledgeGraph format is an ``entities`` map (id -> entity) plus a
    ``relations`` list where each relation references ``subject_id`` and either
    ``object_id`` (entity object) or ``object_value`` (literal object).
    """
    g = Graph()
    g.bind("", M)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    entities: Dict[str, Any] = json_data.get("entities", {})

    # Resolve a URIRef for every entity (by its own surface form) up front, and
    # emit its rdf:type from the entity's type (these class assertions live on
    # the entity in the KnowledgeGraph format, not among the relations).
    entity_uris: Dict[str, URIRef] = {}
    for entity_id, entity in entities.items():
        entity_uri = URIRef(
            resolve_uri(
                keys=[entity.get("surface_form")],
                suggested_uri=entity.get("uri"),
                ontology_mapping=ontology_mapping,
                g=g,
            )
        )
        entity_uris[entity_id] = entity_uri

        type_ref = entity.get("type") or {}
        if type_ref.get("label") or type_ref.get("uri"):
            type_uri = URIRef(
                resolve_uri(
                    keys=[type_ref.get("label")],
                    suggested_uri=type_ref.get("uri"),
                    ontology_mapping=ontology_mapping,
                    g=g,
                )
            )
            g.add((entity_uri, RDF.type, type_uri))

    seen_entities = set()
    rdf_star_annotations: List[str] = []

    for rel in json_data.get("relations", []):
        subj_id = rel.get("subject_id")
        subj_uri = entity_uris.get(subj_id)
        if subj_uri is None:
            continue

        pred = rel.get("predicate", {})
        pred_uri = URIRef(
            resolve_uri(
                keys=[pred.get("label")],
                suggested_uri=pred.get("uri"),
                ontology_mapping=ontology_mapping,
                g=g,
            )
        )

        prov_ids = rel.get("provenance_ids", [])

        if rel.get("is_literal"):
            obj_node: Any = Literal(
                rel.get("object_value"),
                datatype=literal_datatype(rel.get("object_datatype")),
            )
        else:
            obj_node = entity_uris.get(rel.get("object_id"))
            if obj_node is None:
                continue

        g.add((subj_uri, pred_uri, obj_node))

        # Manually create RDF-star provenance annotation strings.
        obj_repr = obj_node.n3() if isinstance(obj_node, Literal) else f"<{obj_node}>"
        for prov_id in prov_ids:
            if prov_id in source_texts:
                source_literal = Literal(source_texts[prov_id]).n3()
                rdf_star_annotations.append(
                    f"<< <{subj_uri}> <{pred_uri}> {obj_repr} >> "
                    f"<{M.sourceText}> {source_literal} ."
                )

        # Emit label/description for entity nodes (skip literal objects).
        node_ids = (
            [subj_id] if rel.get("is_literal") else [subj_id, rel.get("object_id")]
        )
        for node_id in node_ids:
            uri = entity_uris.get(node_id)
            if uri is None or uri in seen_entities:
                continue
            seen_entities.add(uri)
            entity = entities.get(node_id, {})
            if entity.get("surface_form"):
                g.add((uri, RDFS.label, Literal(entity["surface_form"])))
            if entity.get("description"):
                g.add((uri, RDFS.comment, Literal(entity["description"])))

    # Serialize the graph that rdflib understands
    base_turtle = g.serialize(format="turtle")

    # Append the manually generated RDF-star annotations
    if rdf_star_annotations:
        return base_turtle + "\n" + "\n".join(rdf_star_annotations)
    else:
        return base_turtle


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python json_to_rdf.py <input_json_file> <source_text_file> [OPTIONAL output_file]"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    source_text_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_texts = {
        chunk.chunk_id: chunk.content
        for chunk in iter_musique_chunks_jsonl(source_text_file)
    }

    ontology_mapping = load_ontology(musique_onto_path)
    turtle_str = convert_to_turtle(data, ontology_mapping, source_texts)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(turtle_str)
    else:
        print(turtle_str)
