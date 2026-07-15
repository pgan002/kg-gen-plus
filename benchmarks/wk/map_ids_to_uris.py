import argparse
import json
import csv
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def map_ids_to_uris(uris_path, texts_path, output_path, csv_path=None):
    """
    Maps IDs from a texts file to URIs from a uris file based on text containment,
    and optionally merges mappings from a CSV file. Logs warnings for unmapped IDs.

    Args:
        uris_path (str): Path to the .jsonl file with URIs and texts.
        texts_path (str): Path to the .jsonl file with IDs and texts.
        output_path (str): Path to the output .json file for the mapping.
        csv_path (str, optional): Path to the .csv file for additional mappings.
    """
    with open(uris_path, "r", encoding="utf-8") as f:
        uris_data = [json.loads(line) for line in f]

    with open(texts_path, "r", encoding="utf-8") as f:
        texts_data = [json.loads(line) for line in f]

    mapping = {}

    # Text containment mapping
    for text_item in texts_data:
        text_id = text_item["id"]

        if not text_id.startswith("0000_"):
            continue

        main_text = text_item["text"]
        found_uris = []

        for uri_item in uris_data:
            uri_id = uri_item["id"]
            sub_text = uri_item["text"]

            if sub_text in main_text:
                found_uris.append(uri_id)

        if found_uris:
            mapping[text_id] = found_uris

    # CSV-based mapping
    if csv_path:
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            # csv_uris = []
            # for row in csv_reader:
            #     csv_uri = row[0]
            #     csv_uri = csv_uri.replace("wk-decision:", "https://kg.wolterskluwer.com/decision/")
            #     csv_uris.append(csv_uri)
            csv_uris = [
                row[0].replace("wk-decision:", "https://kg.wolterskluwer.com/decision/")
                for row in csv_reader
            ]

        for text_item in texts_data:
            text_id = text_item["id"]
            try:
                row_num_str = text_id.split("_")[0]
                row_num = int(row_num_str)

                if 1 <= row_num <= len(csv_uris):
                    uri_from_csv = csv_uris[row_num - 1]
                    if text_id in mapping:
                        if uri_from_csv not in mapping[text_id]:
                            mapping[text_id].append(uri_from_csv)
                    else:
                        mapping[text_id] = [uri_from_csv]
            except (ValueError, IndexError):
                continue

    # Check for unmapped IDs and log a warning
    for text_item in texts_data:
        text_id = text_item["id"]
        if text_id not in mapping:
            logging.warning(f"ID '{text_id}' was not mapped to any URI.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    logging.info(f"Mapping saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map IDs to URIs based on text containment and CSV."
    )
    parser.add_argument(
        "--uris",
        type=str,
        required=True,
        help="Path to the .jsonl file with URIs and texts.",
    )
    parser.add_argument(
        "--texts",
        type=str,
        required=True,
        help="Path to the .jsonl file with IDs and texts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output .json file for the mapping.",
    )
    parser.add_argument(
        "--csv", type=str, help="Optional path to a .csv file for additional mappings."
    )

    args = parser.parse_args()

    map_ids_to_uris(args.uris, args.texts, args.output, args.csv)
