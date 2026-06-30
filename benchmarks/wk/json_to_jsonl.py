import json
from pathlib import Path


def convert_sparql_json_to_jsonl(input_filepath, output_filepath):
    """
    Converts a JSON file containing a 'chunks' array into a JSONL file.
    """
    try:
        with open(input_filepath, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        tgt_nested = data["results"]["bindings"]

        with open(output_filepath, "w", encoding="utf-8") as outfile:
            for chunk in tgt_nested:
                uri = chunk["s"]["value"]
                text = chunk["text"]["value"]
                json_string = json.dumps(
                    {"id": uri, "text": text}, separators=(",", ":"), ensure_ascii=False
                )
                outfile.write(json_string + "\n")

        print(f"Successfully converted {len(tgt_nested)} chunks to {output_filepath}")

    except FileNotFoundError:
        print(f"Error: The file {input_filepath} was not found.")
    except json.JSONDecodeError:
        print("Error: The input file contains invalid JSON.")


def convert_json_to_jsonl(input_filepath, output_filepath):
    """
    Converts a JSON file containing a 'chunks' array into a JSONL file.
    """
    try:
        with open(input_filepath, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        tgt_nested = data["chunks"]

        with open(output_filepath, "w", encoding="utf-8") as outfile:
            for chunk in tgt_nested:
                uri = chunk["chunk_id"]
                text = chunk["content"]
                json_string = json.dumps(
                    {"id": uri, "text": text}, separators=(",", ":"), ensure_ascii=False
                )
                outfile.write(json_string + "\n")

        print(f"Successfully converted {len(tgt_nested)} chunks to {output_filepath}")

    except FileNotFoundError:
        print(f"Error: The file {input_filepath} was not found.")
    except json.JSONDecodeError:
        print("Error: The input file contains invalid JSON.")


# --- Execution ---
if __name__ == "__main__":
    # Specify your input and output file names here
    INPUT_FILE = Path(__file__).parent / "data" / "wkg_chunks_v2.json"
    OUTPUT_FILE = Path(__file__).parent / "data" / "wkg_chunks_v2.jsonl"

    convert_json_to_jsonl(INPUT_FILE, OUTPUT_FILE)
