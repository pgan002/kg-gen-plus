import json
import hashlib
import re
import sys
from typing import Dict, Any


def slugify(text: str) -> str:
    """Convert text to a slug suitable for URIs."""
    if not text:
        return "unknown"
    # Replace non-word characters with underscores
    slug = re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "_")
    return slug


def get_uri(entity: Dict[str, Any], base_prefix: str = "wk:") -> str:
    """Get URI for an entity, using its 'uri' field or generating one from 'surface_form'."""
    uri = entity.get("uri")
    if uri:
        # If it's already a full URI or a prefixed one, return it
        if uri.startswith("http") or ":" in uri:
            return f"<{uri}>" if uri.startswith("http") else uri
        return f"{base_prefix}{uri}"

    # Generate from surface_form
    slug = slugify(entity.get("surface_form", "unknown"))
    return f"{base_prefix}{slug}"


def generate_statement_id(subject_uri: str, predicate_uri: str, object_uri: str) -> str:
    """Generate a unique ID for a statement based on its components."""
    content = f"{subject_uri}{predicate_uri}{object_uri}"
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"wk:s_{h}"


def convert_to_turtle(json_data: Dict[str, Any]) -> str:
    relations = json_data.get("relations", [])

    prefixes = [
        "@prefix wk: <https://kg.wolterskluwer.com/> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "",
    ]

    lines = prefixes

    # To avoid repeating entity definitions
    seen_entities = set()

    for rel in relations:
        subj = rel.get("subject", {})
        pred = rel.get("predicate", {})
        obj = rel.get("object", {})
        prov_ids = rel.get("provenance_ids", [])

        subj_uri = get_uri(subj)
        pred_uri = get_uri(pred)
        obj_uri = get_uri(obj)

        stmt_id = generate_statement_id(subj_uri, pred_uri, obj_uri)

        # Statement definition
        lines.append(f"{stmt_id} a rdf:Statement ;")
        lines.append(f"  rdf:subject {subj_uri} ;")
        lines.append(f"  rdf:predicate {pred_uri} ;")
        lines.append(f"  rdf:object {obj_uri} .")

        for prov_id in prov_ids:
            lines.append(f"{stmt_id} dcterms:source <{prov_id}> .")

        lines.append("")

        # Entity metadata
        for entity, uri in [(subj, subj_uri), (pred, pred_uri), (obj, obj_uri)]:
            if uri not in seen_entities:
                seen_entities.add(uri)

                label = entity.get("surface_form")
                description = entity.get("description")

                if label or description:
                    # We use a simplified representation here
                    parts = []
                    if label:
                        parts.append(f'rdfs:label "{label}"')
                    if description:
                        # Escape quotes and backslashes in description
                        desc_escaped = description.replace("\\", "\\\\").replace(
                            '"', '\\"'
                        )
                        parts.append(f'rdfs:comment "{desc_escaped}"')

                    if parts:
                        lines.append(f"{uri} " + " ;\n  ".join(parts) + " .")
                        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_rdf.py <input_json_file> [OPTIONAL output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    turtle_str = convert_to_turtle(data)

    if len(sys.argv) > 1:
        output_file = sys.argv[2]
        with open(output_file, "w") as f:
            f.write(turtle_str)
    else:
        print(turtle_str)
