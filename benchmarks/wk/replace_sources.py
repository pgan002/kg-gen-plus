import argparse
import json
import re


def replace_sources_in_ttl(input_ttl_path, mappings_path, output_ttl_path):
    """
    Replaces dcterms:source objects in a TTL file based on a JSON mapping.

    Args:
        input_ttl_path (str): Path to the input .ttl file.
        mappings_path (str): Path to the .json file with id-to-URI mappings.
        output_ttl_path (str): Path to the output .ttl file.
    """
    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    with open(input_ttl_path, "r", encoding="utf-8") as f:
        ttl_content = f.read()

    def replace_match(match):
        # source_line = match.group(0)
        source_ids_str = match.group(1)

        # Split the source IDs, handling potential variations in spacing and commas
        source_ids = [item.strip() for item in re.split(r",\s*", source_ids_str)]

        new_sources = []
        for source_id in source_ids:
            # Clean the ID by removing angle brackets if they exist
            clean_id = source_id.strip("<>")

            # Check if the cleaned ID is in the mappings
            if clean_id in mappings:
                # Replace with the new URIs from the mapping
                new_uris = mappings[clean_id]
                new_sources.extend([f"<{uri}>" for uri in new_uris])
            else:
                # Keep the original source ID if no mapping is found
                new_sources.append(source_id)

        # Reconstruct the source line
        return "    dcterms:source " + ",\n        ".join(new_sources) + ";"

    # Regex to find the dcterms:source line and capture all its objects
    # This handles multi-line source declarations
    # pattern = re.compile(r"dcterms:source\s+((?:<[^>]+>|\"[^\"]+\")(?:,\s*\n?\s*(?:<[^>]+>|\"[^\"]+\"))*)\s*;", re.MULTILINE)

    # A more robust regex that handles the provided snippet format
    pattern = re.compile(r"dcterms:source\s+([^;]+);", re.DOTALL)

    modified_ttl_content = pattern.sub(replace_match, ttl_content)

    with open(output_ttl_path, "w", encoding="utf-8") as f:
        f.write(modified_ttl_content)

    print(f"TTL file with replaced sources saved to {output_ttl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace dcterms:source in a TTL file."
    )
    parser.add_argument(
        "--input-ttl", type=str, required=True, help="Path to the input .ttl file."
    )
    parser.add_argument(
        "--mappings",
        type=str,
        required=True,
        help="Path to the JSON file with id-to-URI mappings.",
    )
    parser.add_argument(
        "--output-ttl", type=str, required=True, help="Path to the output .ttl file."
    )

    args = parser.parse_args()

    replace_sources_in_ttl(args.input_ttl, args.mappings, args.output_ttl)
