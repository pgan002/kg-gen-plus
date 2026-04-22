import logging
import io
import requests
import zipfile
from pathlib import Path


from benchmarks.evaluator import EvalGraph, EvalTriple, EvalEntity, GraphEvaluator
from benchmarks.synthie_utils import iter_synthie_jsonl
from kg_gen.kg_gen import KGGen


def extract_terms_for_text(text: str, item_id: str) -> list[str]:
    """Extracts terms for a single piece of text by sending an in-memory zip file to the external service."""
    url = "http://dsx-gws-rai-docker-dmo-apl-n-01:8089/extract"
    headers = {"accept": "application/json"}
    data = {
        "categories": "",
        "questions": "",
        "model": "gpt-mini-4o",
        "window_size": "24000",
        "window_overlap_size": "1000",
    }

    # Create an in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(f"{item_id}.txt", text.encode("utf-8"))

    zip_buffer.seek(0)

    files = {"zip_file_docs": (f"{item_id}.zip", zip_buffer, "application/zip")}
    response = requests.post(url, headers=headers, files=files, data=data)

    response.raise_for_status()
    return [item["term"] for item in response.json() if "term" in item]


if __name__ == "__main__":
    # dspy.configure_cache(
    #     enable_disk_cache=False,
    #     enable_memory_cache=False,
    # )
    logging.basicConfig(level=logging.INFO)

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

    i_start = 1
    i_end = 100

    gs = []
    total_usage = None
    evaluator = GraphEvaluator()
    for i, item in enumerate(iter_synthie_jsonl(davinci2_test_small_path), start=1):
        if i < i_start:
            continue
        if i > i_end:
            break
        logging.debug(
            f"\n\n{i = }, {item.id_ = }, {len(item.triplets) = }\n{item.text = }"
        )

        logging.info(f"Extracting terms for item {item.id_}...")
        extracted_terms = extract_terms_for_text(item.text, str(item.id_))
        logging.info(f"Extracted {len(extracted_terms)} terms.")

        g, usage = kg.generate(
            input_data=item.text,
            relation_context="Use predicates from Wikidata for the extracted relations. "
            "Provide the Wikidata identifiers for the extracted relations, "
            'for example, "{surface_form: operator, uri: P137}".',
            terms=extracted_terms,
            types="Pick the types from Wikidata.",
            output_folder=str(synthie_base_data_path),
            deduplication_method=None,
        )
        if total_usage is not None:
            total_usage += usage
        else:
            total_usage = usage
        ### Evaluate

        # Convert generated graph to EvalGraph
        generated_eval_triples = [
            EvalTriple(
                subject=EvalEntity(
                    surface_form=r.subject.surface_form, uri=r.subject.uri
                ),
                predicate=EvalEntity(
                    surface_form=r.predicate.surface_form, uri=r.predicate.uri
                ),
                object=EvalEntity(surface_form=r.object.surface_form, uri=r.object.uri),
            )
            for r in g.relations
        ]
        generated_eval_graph = EvalGraph(triples=generated_eval_triples)

        # Convert gold graph to EvalGraph
        gold_eval_triples = [
            EvalTriple(
                subject=EvalEntity(
                    surface_form=t["subject"]["surfaceform"],
                    uri=t["subject"].get("uri"),
                ),
                predicate=EvalEntity(
                    surface_form=t["predicate"]["surfaceform"],
                    uri=t["predicate"].get("uri"),
                ),
                object=EvalEntity(
                    surface_form=t["object"]["surfaceform"], uri=t["object"].get("uri")
                ),
            )
            for t in item.model_dump()["triplets"]
        ]
        gold_eval_graph = EvalGraph(triples=gold_eval_triples)

        metrics = evaluator.evaluate(
            generated_graph=generated_eval_graph, gold_graph=gold_eval_graph
        )
        logging.info(
            f"{i = }, {item.id_ = }, {len(item.triplets) = }\n"
            f"Precision: {metrics['precision']}\n"
            f"Recall: {metrics['recall']}\n"
            f"F1: {metrics['f1_score']}\n"
            f"False positives: {metrics['false_positive_triples']}\n"
            f"False negatives: {metrics['false_negative_triples']}\n"
        )
        logging.info(f"{usage = }\n")
        gs.append(g)
    logging.info(f"Total usage: {total_usage}")

    agg_results = evaluator.get_aggregated_results()
    if agg_results:
        logging.info("--- Overall Results ---")
        logging.info(f"Macro Precision: {agg_results['macro_precision']:.4f}")
        logging.info(f"Macro Recall: {agg_results['macro_recall']:.4f}")
        logging.info(f"Macro F1 Score: {agg_results['macro_f1']:.4f}")
        logging.info(f"Micro Precision: {agg_results['micro_precision']:.4f}")
        logging.info(f"Micro Recall: {agg_results['micro_recall']:.4f}")
        logging.info(f"Micro F1 Score: {agg_results['micro_f1']:.4f}")
        logging.info(f"Total Evaluated: {agg_results['total_graphs_evaluated']}")
        logging.info("-----------------------")
    # agg_g = kg.aggregate(gs)
    # agg_g, dedup_stats = kg.deduplicate(
    #     graph=agg_g,
    #     method=DeduplicateMethod.FULL
    #     # method=DeduplicateMethod.SEMHASH,
    #     # semhash_similarity_threshold=0.5
    #     # method=DeduplicateMethod.LM_BASED,
    # )
    # print(f"{dedup_stats = }")
    # kg.export_graph(graph=agg_g, output_path=str(synthie_base_data_path / "graph.json"))
    # kg.visualize(agg_g, str(synthie_base_data_path / "graph.html"), True)
