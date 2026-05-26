import logging
from pathlib import Path

from benchmarks.evaluation.evaluator import (
    GraphEvaluator,
    EvalTriple,
    EvalEntity,
    EvalGraph,
)
from benchmarks.synthie.run_synthie_w_termext import extract_terms_for_text
from benchmarks.text2kg_bench.text2kgbench_config import (
    llm_model,
    retrieval_model,
    i_start,
    i_end,
    text2kgbench_base_data_path,
    use_gold_entities,
    clean_triples,
    configure_logging,
    ground_truth_dir,
    ontologies_dir,
)
from benchmarks.text2kg_bench.text2kgbench_utils import (
    iter_text2kgbench_jsonl,
    parse_ontology,
)
from kg_gen.kg_gen import KGGen
from kg_gen.models import Graph, InputData


def process_file(test_path: Path, ontology_path: Path, kg: KGGen):
    text2kgbench_base_results_path = (
        text2kgbench_base_data_path / "results" / f"{ontology_path.stem}_graph"
    )
    configure_logging(log_file_path=f"{text2kgbench_base_results_path}.log")
    # extract types and relations from ontology
    onto = parse_ontology(onto_path=ontology_path)
    logging.info(
        f"Extracted {len(onto.classes)} classes and {len(onto.predicates)} relations."
    )

    gs: list[Graph] = []  # to aggregate the output graphs
    total_usage = None
    evaluator = GraphEvaluator()
    for i, item in enumerate(
        iter_text2kgbench_jsonl(test_path, clean_triples=clean_triples), start=1
    ):
        if i < i_start:
            continue
        if i > i_end:
            break
        logging.info(
            f"\n\n{i = }, {item.id_ = }, {len(item.triples) = }\n{item.sent = }"
        )

        if use_gold_entities:
            # For Text2KgBench, the gold entities are not directly provided in the item
            # We need to extract them from the triples
            item_entities = []
            for triple in item.triples:
                item_entities.append(triple.sub)
                item_entities.append(triple.obj)
            # Remove duplicates and convert to surface forms
            item_entities = list(set([e for e in item_entities]))
        else:
            # Use term extraction from synthie
            logging.info(f"Extracting terms for item {item.id_}...")
            item_entities = extract_terms_for_text(item.sent, str(item.id_))
            logging.info(f"Extracted {len(item_entities)} terms.")

        g, usage = kg.generate(
            input_data=InputData(text=item.sent, id=item.id_, terms=item_entities),
            types=onto.classes,
            output_folder=str(text2kgbench_base_data_path),
            predicate_domain_range=onto.predicates,
            deduplication_method=None,
        )
        if total_usage is not None:
            total_usage += usage
        else:
            total_usage = usage
        logging.info(f"{usage = }\n")
        ### Evaluate
        g.output_class_assertions = False
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
                subject=EvalEntity(surface_form=t["sub"]),
                predicate=EvalEntity(surface_form=t["rel"]),
                object=EvalEntity(surface_form=t["obj"]),
            )
            for t in item.model_dump()["triples"]
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
    logging.info(f"{total_usage = }")

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

    agg_g = kg.aggregate(gs)
    # agg_g, dedup_stats = kg.deduplicate(
    #     graph=agg_g,
    #     method=DeduplicateMethod.FULL
    #     # method=DeduplicateMethod.SEMHASH,
    #     # semhash_similarity_threshold=0.5
    #     # method=DeduplicateMethod.LM_BASED,
    # )
    # print(f"{dedup_stats = }")
    kg.export_graph(graph=agg_g, output_path=f"{text2kgbench_base_results_path}.json")
    kg.visualize(agg_g, f"{text2kgbench_base_results_path}.html", False)


if __name__ == "__main__":
    # keycloak_token = get_keycloak_token()
    kg_gen = KGGen(
        # Use 'openai/' prefix to force standard HTTP client with Bearer token auth
        # model="openai/gpt-oss:120b",
        model=llm_model,
        temperature=1.0,
        # api_base="https://ollama.dev.memorise.sdu.dk/v1",
        # api_key=keycloak_token
        retrieval_model=retrieval_model,
    )

    for ground_truth_file in ground_truth_dir.glob("*.jsonl"):
        ontology_name = ground_truth_file.stem.replace("_ground_truth", "")
        ontology_file = ontologies_dir / f"{ontology_name}.ttl"

        if ontology_file.exists():
            logging.info(
                f"Processing {ground_truth_file.name} with {ontology_file.name}"
            )
            process_file(
                test_path=ground_truth_file,
                ontology_path=ontology_file,
                kg=kg_gen,
            )
        else:
            logging.warning(f"Ontology file not found for {ground_truth_file.name}")
