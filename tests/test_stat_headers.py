"""Stat headers must never be able to break a response.

The regression these cover: ``build_stats_headers`` used to dump the whole
``KGGenStats`` JSON into ``X-KG-Gen-Stats``. That JSON embeds one record per
failed document, each carrying the document's full LM response, so on a
16,131-document run with 14 failures it reached 142 KB and contained an en dash.
Assigning it to ``response.headers`` raised ``UnicodeEncodeError``, which escaped
as a bare 500 -- and since the graph had already been generated and stored, every
attempt to fetch that job's result failed permanently.
"""

import pytest
from fastapi import Response

from app.generation import (
    MAX_STAT_HEADER_BYTES,
    apply_stat_headers,
    build_stats_headers,
)
from kg_gen.models import KGGenStats, LMUsage, StepStats


def _stats_with_failures(count: int, error: str) -> KGGenStats:
    stats = KGGenStats()
    stats.extract_entities = StepStats(
        lm_usage=LMUsage(prompt_tokens=7, completion_tokens=11, total_tokens=18),
        execution_time=1.5,
    )
    stats.failed_documents = [{"id": f"doc{i}", "error": error} for i in range(count)]
    return stats


def test_headers_stay_small_and_sendable_with_many_failures():
    # An en dash is what actually broke production; the long error text is what
    # made the old header 142 KB.
    stats = _stats_with_failures(
        14, "Adapter JSONAdapter failed to parse the LM response – " + "x" * 5000
    )

    headers = build_stats_headers(stats)

    for key, value in headers.items():
        value.encode("latin-1")  # would raise on the old X-KG-Gen-Stats
        assert len(value.encode("latin-1")) <= MAX_STAT_HEADER_BYTES, key

    # Applying them to a real response must not raise.
    response = Response()
    apply_stat_headers(response, headers)
    assert response.headers["X-KG-Gen-Failed-Documents"] == "14"


def test_overall_usage_is_reported():
    stats = _stats_with_failures(0, "")

    headers = build_stats_headers(stats)

    assert headers["X-KG-Gen-Time"] == "1.5"
    assert headers["X-KG-Gen-Input-Tokens"] == "7"
    assert headers["X-KG-Gen-Output-Tokens"] == "11"
    assert headers["X-KG-Gen-Total-Tokens"] == "18"
    assert headers["X-KG-Gen-Failed-Documents"] == "0"


def test_dedup_stats_reported_only_when_dedup_ran():
    stats = _stats_with_failures(0, "")
    assert "X-KG-Gen-Dedup-Stats" not in build_stats_headers(stats)

    stats.deduplicate = StepStats(execution_time=0.25)
    assert "X-KG-Gen-Dedup-Stats" in build_stats_headers(stats)


@pytest.mark.parametrize(
    "value",
    [
        "en dash – is not latin-1",
        "x" * (MAX_STAT_HEADER_BYTES + 1),
    ],
    ids=["not-latin-1", "oversized"],
)
def test_unsendable_values_are_skipped_not_raised(value):
    response = Response()

    apply_stat_headers(response, {"X-KG-Gen-Time": "1.0", "X-KG-Gen-Bad": value})

    assert "X-KG-Gen-Bad" not in response.headers
    assert response.headers["X-KG-Gen-Time"] == "1.0"


def test_apply_tolerates_no_headers():
    response = Response()

    apply_stat_headers(response, None)

    assert not any(key.startswith("x-kg-gen") for key in response.headers)
