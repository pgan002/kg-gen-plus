import json
from pathlib import Path
import argparse


def prepare_corpus(dataset_dir, output_file):
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / "metadata.jsonl"

    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found")
        return

    with open(metadata_path, "r") as f_meta, open(output_file, "w") as f_out:
        for line in f_meta:
            meta = json.loads(line)
            pmcid = meta["pmcid"]
            text_file_rel = meta["textFile"]
            text_file_path = dataset_path / text_file_rel

            if text_file_path.exists():
                with open(text_file_path, "r") as f_text:
                    text = f_text.read()

                corpus_item = {"id": pmcid, "text": text}
                f_out.write(json.dumps(corpus_item) + "\n")
            else:
                print(f"Warning: Text file {text_file_path} not found for {pmcid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PubMed corpus for KGGen")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory (e.g., benchmarks/ga_pubmed/data/dataset_small)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output JSONL corpus file"
    )

    args = parser.parse_args()
    prepare_corpus(args.dataset, args.output)
