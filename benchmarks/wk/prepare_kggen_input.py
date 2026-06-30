import json
import logging
import rdflib

from benchmarks.wk.wk_config import wk_base_data_path, chunked_path, ontology_file
from benchmarks.wk.wk_utils import iter_wk_chunked_jsonl
from kg_gen.models import TypedEntity, InputData

logging.basicConfig(level=logging.INFO)


def main():
    terms_cache_dir = wk_base_data_path / "terms_cache"
    output_path = wk_base_data_path / "kggen_input.jsonl"

    if not terms_cache_dir.exists():
        logging.error(f"Terms cache directory not found at {terms_cache_dir}")
        return

    # Load ontology to map labels to URIs
    onto = rdflib.Graph()
    onto.parse(ontology_file)
    # label_to_uri = {
    #     str(label): str(s)
    #     for s, p, o in onto.triples((None, rdflib.RDFS.label, None))
    #     for label in [o]
    # }

    with open(output_path, "w") as f_out:
        for chunk in iter_wk_chunked_jsonl(chunked_path):
            chunk_id = chunk.id
            cache_file = terms_cache_dir / f"terms_{chunk_id}.json"

            if not cache_file.exists():
                logging.warning(f"Cache file not found for chunk {chunk_id}, skipping.")
                continue

            with open(cache_file, "r") as f_cache:
                terms_data = json.load(f_cache)

            kg_input_terms = []
            for term in terms_data:
                categories = term.get("categories", [])
                term_description = term.get("definition")
                if categories:
                    term_description += f"\nPredicted types: {categories}."
                if term.get("alt_labels"):
                    alt_labels_str = ", ".join(term["alt_labels"])
                    term_description += f"\nAlternative labels: {alt_labels_str}."

                kg_input_terms.append(
                    TypedEntity(
                        surface_form=term["pref_label"],
                        description=term_description,
                        type=None,
                    )
                )

            input_data = InputData(
                text=chunk.text,
                id=chunk_id,
                terms=[term.model_dump() for term in kg_input_terms],
            )
            f_out.write(input_data.model_dump_json() + "\n")

    logging.info(f"Successfully created {output_path}")


if __name__ == "__main__":
    main()
