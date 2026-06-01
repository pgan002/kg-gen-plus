import os
import re
from pathlib import Path


def extract_first_item_from_log(file_path):
    """
    Extracts the log content for the first item (i = 1) from a log file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    start_index = -1
    end_index = -1

    # Find the start of the 'i = 1' block
    for i, line in enumerate(lines):
        if "i = 1," in line:
            start_index = i
            break

    if start_index == -1:
        return None  # No 'i = 1' block found

    # Find the start of the next block ('i = 2') to determine the end of the 'i = 1' block
    for i, line in enumerate(lines[start_index + 1 :], start=start_index + 1):
        if re.search(r"i = \d+,", line):
            end_index = i
            break

    # If no next block is found, take everything until the end of the file from start_index
    if end_index == -1:
        return "".join(lines[start_index:])
    else:
        return "".join(lines[start_index:end_index])


def extract_logs(log_dir, output_file):
    """
    Extracts the first item from all .log files in a directory and saves to an output file.
    """
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".log")])

    with open(output_file, "w") as outfile:
        for log_file in log_files:
            outfile.write(f"--- Results from {log_file} ---\n")
            try:
                extracted_content = extract_first_item_from_log(log_dir / log_file)
                if extracted_content:
                    outfile.write(extracted_content)
                    outfile.write("\n\n")
            except Exception as e:
                error_message = f"Error processing file {log_file}: {e}\n\n"
                outfile.write(error_message)
                print(error_message)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Extract first item logs from all .log files in a directory.")
    # parser.add_argument("log_dir", type=str, help="The directory containing the log files.")
    # parser.add_argument("output_file", type=str, help="The path to the output file to save extracted logs.")
    # args = parser.parse_args()
    #
    # log_dir = Path(args.log_dir)
    # output_file = Path(args.output_file)

    log_dir = Path(__file__).parent / "data" / "wikidata_tekgen" / "results" / "run1"
    output_file = Path(__file__).parent / "data" / "first_items_output_wikidata.txt"

    if log_dir.exists():
        extract_logs(log_dir, output_file)
        print(f"Extraction complete. Results saved to {output_file}")
    else:
        print(f"Log directory not found: {log_dir.resolve()}")
