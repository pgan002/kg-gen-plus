import argparse
import json


def normalize_gt_triple(triple):
    sub = triple["sub"].replace("_", " ").lower()
    rel = triple["rel"].lower()
    obj = triple["obj"].strip('"').lower()
    return (sub, rel, obj)


def normalize_extracted_relation(relation):
    sub = relation["subject"]["surface_form"].lower()
    pred = relation["predicate"]["surface_form"].lower()
    obj = relation["object"]["surface_form"].lower()
    return (sub, pred, obj)


def find_offsets(sentence, text):
    try:
        start = sentence.index(text)
        end = start + len(text)
        return start, end
    except ValueError:
        return None, None


def get_ontology_name(gt_id):
    return "_".join(gt_id.split("_")[:-2])


def convert(results_path, ground_truth_path, output_path):
    with open(ground_truth_path, "r") as f:
        ground_truth_data = [json.loads(line) for line in f]

    with open(results_path, "r") as f:
        results_data = json.load(f)

    relations_by_provenance = {}
    for relation in results_data.get("relations_wo_class_assertions", []):
        for prov_id in relation.get("provenance_ids", []):
            if prov_id not in relations_by_provenance:
                relations_by_provenance[prov_id] = []
            relations_by_provenance[prov_id].append(relation)

    output_data = []
    for gt_item in ground_truth_data:
        gt_id = gt_item["id"]
        sentence = gt_item["sent"]
        # ontology_name = get_ontology_name(gt_id)
        extracted_relations_for_id = relations_by_provenance.get(gt_id, [])

        output_relations = []
        for rel in extracted_relations_for_id:
            start_sub, end_sub = find_offsets(sentence, rel["subject"]["surface_form"])
            start_obj, end_obj = find_offsets(sentence, rel["object"]["surface_form"])

            output_rel = {
                "label": rel["predicate"]["surface_form"].lower(),
                "uri": rel["predicate"]["uri"],
                "domain argument": {
                    "text": rel["subject"]["surface_form"],
                    "start": start_sub,
                    "end": end_sub,
                    "label": None,
                    "uri": None,
                },
                "range argument": {
                    "text": rel["object"]["surface_form"],
                    "start": start_obj,
                    "end": end_obj,
                    "label": None,
                    "uri": None,
                },
            }
            output_relations.append(output_rel)

        expected_triples = {normalize_gt_triple(t) for t in gt_item["triples"]}
        actual_triples = {
            normalize_extracted_relation(r) for r in extracted_relations_for_id
        }

        tp_set = actual_triples.intersection(expected_triples)
        fp_set = actual_triples.difference(expected_triples)
        fn_set = expected_triples.difference(actual_triples)

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        gt_map = {normalize_gt_triple(t): t for t in gt_item["triples"]}
        ext_map = {
            normalize_extracted_relation(r): r for r in extracted_relations_for_id
        }

        matched_triples_out = []
        for s, p, o in sorted(list(tp_set)):
            actual_rel = ext_map[(s, p, o)]
            expected_triple = gt_map[(s, p, o)]
            matched_triples_out.append(
                {
                    "actual": {
                        "subject": {
                            "text": actual_rel["subject"]["surface_form"],
                            "normalized": s,
                            "uri": None,
                        },
                        "predicate": {
                            "text": actual_rel["predicate"]["surface_form"],
                            "normalized": p,
                            "uri": None,
                        },
                        "object": {
                            "text": actual_rel["object"]["surface_form"],
                            "normalized": o,
                            "uri": None,
                        },
                    },
                    "expected": {
                        "subject": {
                            "text": expected_triple["sub"].replace("_", " "),
                            "normalized": s,
                            "uri": None,
                        },
                        "predicate": {
                            "text": expected_triple["rel"],
                            "normalized": p,
                            "uri": None,
                        },
                        "object": {
                            "text": expected_triple["obj"].strip('"'),
                            "normalized": o,
                            "uri": None,
                        },
                    },
                    "match_score": 100,
                }
            )

        fp_triples_out = []
        for s, p, o in sorted(list(fp_set)):
            actual_rel = ext_map[(s, p, o)]
            fp_triples_out.append(
                {
                    "subject": {
                        "text": actual_rel["subject"]["surface_form"],
                        "normalized": s,
                        "uri": None,
                    },
                    "predicate": {
                        "text": actual_rel["predicate"]["surface_form"],
                        "normalized": p,
                        "uri": None,
                    },
                    "object": {
                        "text": actual_rel["object"]["surface_form"],
                        "normalized": o,
                        "uri": None,
                    },
                }
            )

        fn_triples_out = []
        for s, p, o in sorted(list(fn_set)):
            expected_triple = gt_map[(s, p, o)]
            fn_triples_out.append(
                {
                    "subject": {
                        "text": expected_triple["sub"].replace("_", " "),
                        "normalized": s,
                        "uri": None,
                    },
                    "predicate": {
                        "text": expected_triple["rel"],
                        "normalized": p,
                        "uri": None,
                    },
                    "object": {
                        "text": expected_triple["obj"].strip('"'),
                        "normalized": o,
                        "uri": None,
                    },
                }
            )

        output_item = {
            "id": gt_id,
            "sent": sentence,
            "relations": output_relations,
            "scoring": {
                "metrics": {
                    "p": precision,
                    "r": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                },
                "match_threshold": 90,
                "matched_triples": matched_triples_out,
                "fp_triples": fp_triples_out,
                "fn_triples": fn_triples_out,
            },
            "extracted_relation_count": len(extracted_relations_for_id),
            "generated_eval_triple_count": len(actual_triples),
            "expected_triple_count": len(expected_triples),
        }
        output_data.append(output_item)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KG extraction results.")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Filepath to the KG extraction results file",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Filepath to the ground truth file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for the new .json structure",
    )
    args = parser.parse_args()

    convert(args.results, args.ground_truth, args.output)
