"""One document's extraction raising must not discard every other document's
results in the same batch -- see the ``asyncio.gather(..., return_exceptions=True)``
change in ``KGGen.generate()``.
"""

import pytest
from unittest.mock import patch

from kg_gen.models import InputData
from kg_gen.steps._1_get_entities import extract_entities as real_extract_entities


@pytest.mark.asyncio
async def test_generate_tolerates_one_document_failing(mock_kg_gen):
    good = InputData(text="Alice works at Google.", id="doc_good")
    bad = InputData(text="Bob works at Amazon.", id="doc_bad")

    async def flaky_extract_entities(*args, **kwargs):
        if kwargs.get("provenance_ids") == ["doc_bad"]:
            raise RuntimeError("Simulated JSONAdapter parse failure")
        return await real_extract_entities(*args, **kwargs)

    with patch("kg_gen.kg_gen.extract_entities", side_effect=flaky_extract_entities):
        graph, stats = await mock_kg_gen.generate(
            [good, bad], deduplicate=False, n_parallel=2
        )

    # The good document's entities/relations still made it into the result --
    # the bad document's failure did not take down the whole batch.
    assert len(graph.entities) > 0
    assert len(graph.relations) > 0

    # The failure is recorded, not silently swallowed.
    assert len(stats.failed_documents) == 1
    assert stats.failed_documents[0]["id"] == "doc_bad"
    assert "Simulated JSONAdapter parse failure" in stats.failed_documents[0]["error"]


@pytest.mark.asyncio
async def test_generate_raises_if_every_document_fails(mock_kg_gen):
    bad1 = InputData(text="x", id="doc_bad_1")
    bad2 = InputData(text="y", id="doc_bad_2")

    async def always_fail(*args, **kwargs):
        raise RuntimeError("boom")

    with patch("kg_gen.kg_gen.extract_entities", side_effect=always_fail):
        with pytest.raises(RuntimeError) as exc_info:
            await mock_kg_gen.generate([bad1, bad2], deduplicate=False, n_parallel=2)

    assert "doc_bad_1" in str(exc_info.value)
    assert "doc_bad_2" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_progress_callback_reaches_total_despite_failure(mock_kg_gen):
    good = InputData(text="Alice works at Google.", id="doc_good")
    bad = InputData(text="Bob works at Amazon.", id="doc_bad")
    progress_calls = []

    async def flaky_extract_entities(*args, **kwargs):
        if kwargs.get("provenance_ids") == ["doc_bad"]:
            raise RuntimeError("Simulated JSONAdapter parse failure")
        return await real_extract_entities(*args, **kwargs)

    with patch("kg_gen.kg_gen.extract_entities", side_effect=flaky_extract_entities):
        await mock_kg_gen.generate(
            [good, bad],
            deduplicate=False,
            n_parallel=2,
            progress_callback=lambda done, total: progress_calls.append((done, total)),
        )

    assert progress_calls[-1] == (2, 2)
