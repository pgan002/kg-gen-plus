import argparse
import json
import re


def parse_logs(log_path):
    with open(log_path, "r") as f:
        log_content = f.read()

    log_entries = {}

    # Regex to capture each item's log block
    item_blocks = re.split(r"i = \d+, item.id_ = '([^']*)'", log_content)

    item_ids = item_blocks[1::2]
    log_datas = item_blocks[2::2]

    for i in range(len(item_ids)):
        item_id = item_ids[i]
        block = log_datas[i]

        sent_match = re.search(r"item.sent = [\"\'](.*)[\"\']", block)
        assert sent_match, (item_id, block)

        precision_match = re.search(r"Precision: ([\d.]+)", block)
        recall_match = re.search(r"Recall: ([\d.]+)", block)
        f1_match = re.search(r"F1: ([\d.]+)", block)

        fp_match = re.search(r"False positives: (\[.*\])", block)
        fn_match = re.search(r"False negatives: (\[.*\])", block)

        usage_match = re.search(r"usage = (KGGenStats\(.*\))", block, re.DOTALL)

        if sent_match:
            log_entries[item_id] = {
                "id": item_id,
                "sent": sent_match.group(1),
                "precision": float(precision_match.group(1))
                if precision_match
                else 0.0,
                "recall": float(recall_match.group(1)) if recall_match else 0.0,
                "f1": float(f1_match.group(1)) if f1_match else 0.0,
                "fp": eval(fp_match.group(1)) if fp_match else [],
                "fn": eval(fn_match.group(1)) if fn_match else [],
                "usage": usage_match.group(1) if usage_match else "",
            }

    return log_entries


def convert_from_logs(log_path, ground_truth_path, output_path):
    log_data = parse_logs(log_path)

    with open(ground_truth_path, "r") as f:
        ground_truth_data = [json.loads(line) for line in f]

    i = 0
    with open(output_path, "w") as f:
        pass
    for gt_item in ground_truth_data:
        gt_id = gt_item["id"]
        if gt_id not in log_data:
            print(f"{gt_id = } not in log data")
        else:
            log_entry = log_data[gt_id]

            # Reconstruct triples from logs and ground truth
            expected_triples_norm = {
                (
                    t["sub"].replace("_", " ").lower(),
                    t["rel"].lower(),
                    t["obj"].strip('"').lower(),
                )
                for t in gt_item["triples"]
            }

            fn_triples_norm = {
                (s.lower(), p.lower(), o.lower()) for s, p, o in log_entry["fn"]
            }
            fp_triples_norm = {
                (s.lower(), p.lower(), o.lower()) for s, p, o in log_entry["fp"]
            }

            tp_triples_norm = expected_triples_norm - fn_triples_norm

            # This is an approximation of actual triples
            actual_triples_norm = tp_triples_norm.union(fp_triples_norm)

            def find_offsets(sentence, text):
                try:
                    start = sentence.index(text)
                    end = start + len(text)
                    return start, end
                except ValueError:
                    return None, None

            relations = []
            for sub, pred, obj in actual_triples_norm:
                start_sub, end_sub = find_offsets(log_entry["sent"], sub)
                start_obj, end_obj = find_offsets(log_entry["sent"], obj)
                relations.append(
                    {
                        "label": pred,
                        "uri": None,
                        "domain argument": {
                            "text": sub,
                            "start": start_sub,
                            "end": end_sub,
                            "label": None,
                            "uri": None,
                        },
                        "range argument": {
                            "text": obj,
                            "start": start_obj,
                            "end": end_obj,
                            "label": None,
                            "uri": None,
                        },
                    }
                )

            output_item = {
                "id": gt_id,
                "sent": log_entry["sent"],
                "relations": relations,
                "scoring": {
                    "metrics": {
                        "p": log_entry["precision"],
                        "r": log_entry["recall"],
                        "f1": log_entry["f1"],
                        "tp": len(tp_triples_norm),
                        "fp": len(fp_triples_norm),
                        "fn": len(fn_triples_norm),
                    },
                    "match_threshold": 90,
                    "matched_triples": [
                        {
                            "actual": {
                                "subject": {"text": s, "normalized": s, "uri": None},
                                "predicate": {"text": p, "normalized": p, "uri": None},
                                "object": {"text": o, "normalized": o, "uri": None},
                            },
                            "expected": {
                                "subject": {"text": s, "normalized": s, "uri": None},
                                "predicate": {"text": p, "normalized": p, "uri": None},
                                "object": {"text": o, "normalized": o, "uri": None},
                            },
                            "match_score": 100,
                        }
                        for s, p, o in tp_triples_norm
                    ],
                    "fp_triples": [
                        {
                            "subject": {"text": s, "normalized": s, "uri": None},
                            "predicate": {"text": p, "normalized": p, "uri": None},
                            "object": {"text": o, "normalized": o, "uri": None},
                        }
                        for s, p, o in fp_triples_norm
                    ],
                    "fn_triples": [
                        {
                            "subject": {"text": s, "normalized": s, "uri": None},
                            "predicate": {"text": p, "normalized": p, "uri": None},
                            "object": {"text": o, "normalized": o, "uri": None},
                        }
                        for s, p, o in fn_triples_norm
                    ],
                },
                "extracted_relation_count": len(actual_triples_norm),
                "generated_eval_triple_count": len(actual_triples_norm),
                "expected_triple_count": len(expected_triples_norm),
            }
            with open(output_path, "a") as f:
                json.dump(output_item, f)
                f.write("\n")
            i += 1
    print(f"total items {i}")
    #         output_data.append(output_item)
    #
    # with open(output_path, 'w') as f:
    #     json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KG extraction logs to JSON.")
    parser.add_argument(
        "--logs", type=str, required=True, help="Filepath to the log file"
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

    convert_from_logs(args.logs, args.ground_truth, args.output)
