import json
from pathlib import Path


def convert_json_to_jsonl(input_filepath, output_filepath):
    """
    Converts a JSON file containing a 'chunks' array into a JSONL file.
    """
    try:
        # 1. Read the JSON file
        with open(input_filepath, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        # 2. Extract the list of chunks
        chunks = data.get("chunks", [])
        if not chunks:
            print("Warning: No 'chunks' found in the input JSON.")
            return

        # 3. Write to the JSONL file
        with open(output_filepath, "w", encoding="utf-8") as outfile:
            for chunk in chunks:
                # json.dumps converts the Python dictionary back to a JSON string
                # Separators ensure there are no trailing spaces, keeping it strictly on one line
                json_string = json.dumps(
                    chunk, separators=(",", ":"), ensure_ascii=False
                )
                outfile.write(json_string + "\n")

        print(f"Successfully converted {len(chunks)} chunks to {output_filepath}")

    except FileNotFoundError:
        print(f"Error: The file {input_filepath} was not found.")
    except json.JSONDecodeError:
        print("Error: The input file contains invalid JSON.")


# --- Execution ---
if __name__ == "__main__":
    # Specify your input and output file names here
    INPUT_FILE = Path(__file__).parent / "data" / "musique_chunks.json"
    OUTPUT_FILE = Path(__file__).parent / "data" / "musique_chunks.jsonl"

    convert_json_to_jsonl(INPUT_FILE, OUTPUT_FILE)
