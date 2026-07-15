"""Poll a submitted MuSiQue KG generation job and download its result (step 2 of 2).

Run this after ``run_musique.py`` has submitted a job. It reads the job id from
the record file written at submission time, polls until the job completes, then
saves the generated knowledge graph to the path recorded at submission.
"""

import json
import logging
from pathlib import Path

import requests

from benchmarks.MuSiQue.musique_config import job_record_path
from benchmarks.MuSiQue.run_musique import wait_for_job, fetch_job_result


def load_job_record() -> dict:
    with open(job_record_path) as f:
        return json.load(f)


def main() -> None:
    if not job_record_path.exists():
        logging.error(
            f"No job record found at {job_record_path}. "
            "Run run_musique.py first to submit a job."
        )
        return

    record = load_job_record()
    job_id = record["job_id"]
    output_path = Path(record["output_path"])
    logging.info(
        f"Polling job {job_id} ({record.get('num_docs')} docs); "
        f"result will be saved to {output_path}"
    )

    try:
        wait_for_job(job_id)
        final_graph = fetch_job_result(job_id)
    except requests.RequestException as exc:
        logging.error(f"Failed to fetch job result: {exc}")
        return
    except RuntimeError as exc:
        logging.error(str(exc))
        return

    if not final_graph:
        logging.warning("No graph was returned.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_graph, f, indent=2)
    logging.info(f"Final knowledge graph saved to {output_path}")


if __name__ == "__main__":
    main()
