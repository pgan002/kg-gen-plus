import logging

from benchmarks.evaluation.evaluator import (
    EvalGraph,
    EvalTriple,
    EvalEntity,
    GraphEvaluator,
)
from benchmarks.synthie.config import (
    llm_model,
    retrieval_model,
    test_path,
    synthie_base_data_path,
    i_start,
    i_end,
)
from benchmarks.synthie.synthie_utils import iter_synthie_jsonl
from kg_gen.kg_gen import KGGen
from kg_gen.models import Graph, InputData

if __name__ == "__main__":
    # keycloak_token = get_keycloak_token()
    kg = KGGen(
        # Use 'openai/' prefix to force standard HTTP client with Bearer token auth
        # model="openai/gpt-oss:120b",
        model=llm_model,
        temperature=1.0,
        # api_base="https://ollama.dev.memorise.sdu.dk/v1",
        # api_key=keycloak_token
        retrieval_model=retrieval_model,
    )

    gs: list[Graph] = []  # to aggregate the output graphs
    prev_usage = None
    evaluator = GraphEvaluator()
    for i, item in enumerate(iter_synthie_jsonl(test_path), start=1):
        if i < i_start:
            continue
        if i > i_end:
            break
        logging.info(
            f"\n\n{i = }, {item.id_ = }, {len(item.triplets) = }\n{item.text = }"
        )
        item_entities = [e.surfaceform for e in item.entities]
        g, usage = kg.generate(
            input_data=InputData(text=item.text, id=str(item.id_), terms=item_entities),
            relation_context="Use predicates from Wikidata for the extracted relations. "
            "Provide the Wikidata identifiers for the extracted relations, "
            'for example, "{surface_form: operator, uri: P137}".',
            # terms=item_entities,
            entity_context="Pick the types from Wikidata.",
            output_folder=str(synthie_base_data_path),
            deduplication_method=None,
        )
        if prev_usage is not None:
            prev_usage += usage
        else:
            prev_usage = usage
        logging.info(f"{usage = }\n")
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
            f"Precision: {metrics['precision']}\n"
            f"Recall: {metrics['recall']}\n"
            f"F1: {metrics['f1_score']}\n"
            f"False positives: {metrics['false_positive_triples']}\n"
            f"False negatives: {metrics['false_negative_triples']}\n"
        )
        gs.append(g)
    logging.info(f"{prev_usage = }")

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
    # logging.info(f"{dedup_stats = }")
    # kg.export_graph(graph=agg_g, output_path=str(synthie_base_data_path / "graph.json"))
    # kg.visualize(agg_g, str(synthie_base_data_path / "graph.html"), True)
