import os
import pandas as pd
import re
from pathlib import Path


def process_log_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    results = {}

    recall_match = re.search(r"Macro Recall: (\d+\.\d+)", content)
    results["Recall"] = float(recall_match.group(1)) if recall_match else None

    f1_match = re.search(r"Macro F1 Score: (\d+\.\d+)", content)
    results["F1"] = float(f1_match.group(1)) if f1_match else None

    total_usage_match = re.search(
        r"total_usage = .*overall_usage=StepStats\(lm_usage=LMUsage\(prompt_tokens=(\d+), completion_tokens=(\d+), total_tokens=\d+\), execution_time=([\d.]+)\)",
        content,
    )
    if total_usage_match:
        results["Input Tokens"] = int(total_usage_match.group(1))
        results["Output Tokens"] = int(total_usage_match.group(2))
        results["Exec Time"] = float(total_usage_match.group(3))
    else:
        results["Input Tokens"] = None
        results["Output Tokens"] = None
        results["Exec Time"] = None

    total_evaluated_match = re.search(r"Total Evaluated: (\d+)", content)
    results["Total Items"] = (
        int(total_evaluated_match.group(1)) if total_evaluated_match else None
    )

    return results


def process_logs(log_dir):
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]

    data = []
    for log_file in log_files:
        file_path = os.path.join(log_dir, log_file)
        try:
            processed_data = process_log_file(file_path)
            if processed_data:
                processed_data["file"] = log_file
                data.append(processed_data)
        except Exception as e:
            print(f"Error processing file {log_file}: {e}")

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    log_dir = Path(__file__).parent / "data" / "dbpedia_webnlg" / "results"
    output_path = log_dir / "all_results.csv"
    if log_dir.exists():
        df = process_logs(log_dir)

        if not df.empty:
            print("--- Individual File Results ---")
            print(df)

            summary_rows = [
                {},  # empty row as a separator
                {"file": "--- SUMMARY ---"},
                {"file": "Recall Avg", "Recall": df["Recall"].mean()},
                {"file": "Recall Std", "Recall": df["Recall"].std()},
                {"file": "F1 Avg", "F1": df["F1"].mean()},
                {"file": "F1 Std", "F1": df["F1"].std()},
                {
                    "file": "Total Input Tokens",
                    "Input Tokens": df["Input Tokens"].sum(),
                },
                {
                    "file": "Total Output Tokens",
                    "Output Tokens": df["Output Tokens"].sum(),
                },
                {"file": "Total Exec Time", "Exec Time": df["Exec Time"].sum()},
                {"file": "Total Items", "Total Items": df["Total Items"].sum()},
            ]
            summary_df = pd.DataFrame(summary_rows)

            df_combined = pd.concat([df, summary_df], ignore_index=True)

            df_combined.to_csv(output_path, index=False)
            print(f"\nResults and summary saved to {output_path}")
        else:
            print("No log files processed.")
    else:
        print(f"Log directory not found: {log_dir.resolve()}")
