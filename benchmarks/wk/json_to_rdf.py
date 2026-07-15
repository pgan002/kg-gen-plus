import json
import hashlib
import re
import sys
from typing import Dict, Any

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, DCTERMS

from benchmarks.wk.wk_config import ontology_file

# Define namespaces
WK = Namespace("https://kg.wolterskluwer.com/")
# WK_NEW = Namespace("https://kg.wolterskluwer.com/new/")
WK_LEGAL = Namespace("https://kg.wolterskluwer.com/ontology/legal/")


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


def get_uri(
    entity: Dict[str, Any],
    ontology_mapping: Dict[str, str],
    g: Graph,
    base_prefix: Namespace = WK,
    new_prefix: Namespace = WK_LEGAL,
    is_predicate: bool = False,
) -> str:
    """Get URI for an entity, checking ontology first."""
    surface_form = entity.get("surface_form", "").lower()
    if surface_form in ontology_mapping:
        return ontology_mapping[surface_form]

    suggested_uri = entity.get("uri", None)
    if suggested_uri is not None:
        try:
            return g.namespace_manager.expand_curie(suggested_uri)
        except ValueError:
            pass
        # else:
        #     out_uri = suggested_uri.strip('<').strip('>')
        # # if "rdf:type" in suggested_uri:
        # #     input(f"{suggested_uri = }, {out_uri = }")
        # return out_uri

    # Fallback to generating a new URI
    slug = slugify(entity.get("surface_form", "unknown"))
    if is_predicate:
        return str(new_prefix[slug])
    else:
        return str(base_prefix[slug])


def generate_statement_id(subject_uri: str, predicate_uri: str, object_uri: str) -> str:
    """Generate a unique ID for a statement."""
    content = f"{subject_uri}{predicate_uri}{object_uri}"
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    return str(WK[f"s_{h}"])


def convert_to_turtle(
    json_data: Dict[str, Any], ontology_mapping: Dict[str, str]
) -> str:
    """Convert JSON data to Turtle RDF, using an ontology for URI resolution."""
    g = Graph()
    g.bind("wk", WK)
    g.bind("wk-legal", WK_LEGAL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("dcterms", DCTERMS)

    seen_entities = set()

    for rel in json_data.get("relations", []):
        subj = rel.get("subject", {})
        pred = rel.get("predicate", {})
        obj = rel.get("object", {})
        prov_ids = rel.get("provenance_ids", [])

        subj_uri = get_uri(subj, ontology_mapping, g=g)
        pred_uri = get_uri(pred, ontology_mapping, g=g, is_predicate=True)
        obj_uri = get_uri(obj, ontology_mapping, g=g)

        stmt_id = generate_statement_id(subj_uri, pred_uri, obj_uri)
        stmt_ref = URIRef(stmt_id)

        g.add((stmt_ref, RDF.type, RDF.Statement))
        g.add((stmt_ref, RDF.subject, URIRef(subj_uri)))
        g.add((stmt_ref, RDF.predicate, URIRef(pred_uri)))
        g.add((stmt_ref, RDF.object, URIRef(obj_uri)))

        for prov_id in prov_ids:
            g.add((stmt_ref, DCTERMS.source, URIRef(prov_id)))

        for entity, uri in [(subj, subj_uri), (obj, obj_uri)]:
            if uri not in seen_entities:
                seen_entities.add(uri)
                entity_ref = URIRef(uri)
                label = entity.get("surface_form")
                description = entity.get("description")
                if label:
                    g.add((entity_ref, RDFS.label, Literal(label, lang="de")))
                if description:
                    g.add((entity_ref, RDFS.comment, Literal(description, lang="en")))

    return g.serialize(format="turtle")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_rdf.py <input_json_file> [OPTIONAL output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ontology_mapping = load_ontology(ontology_file)
    turtle_str = convert_to_turtle(data, ontology_mapping)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(turtle_str)
    else:
        print(turtle_str)
