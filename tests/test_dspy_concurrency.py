"""n_parallel must not be silently throttled by dspy.asyncify.

Entity extraction goes through `dspy.asyncify`, which acquires a process-global
`anyio.CapacityLimiter` sized by `dspy.settings.async_max_workers` -- 8 by
default. The semaphore in `kg_gen.generate` therefore does not decide how many
documents reach the LLM at once: the limiter does. Measured on the full MuSiQue
corpus, n_parallel=20 and n_parallel=50 both gave ~1,150 docs/h with the
endpoint reporting 0 queued requests.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import dspy
import pytest
from dspy.utils.asyncify import get_limiter

from app.generation import configure_dspy_concurrency, execute_generation
from app.schemas import GenerationMetadata
from kg_gen.models import Graph, KGGenStats


@pytest.fixture(autouse=True)
def restore_async_max_workers():
    original = dspy.settings.get("async_max_workers")
    yield
    dspy.settings.configure(async_max_workers=original)


def test_limiter_is_raised_to_n_parallel():
    dspy.settings.configure(async_max_workers=8)

    configure_dspy_concurrency(50)

    assert dspy.settings.get("async_max_workers") == 50
    # The limiter is what asyncify actually acquires, so assert on it too.
    assert get_limiter().total_tokens == 50


def test_the_default_would_have_throttled_us():
    """Guard the premise: 8 really is the default the fix exists to raise."""
    assert dspy.settings.get("async_max_workers") == 8


@pytest.mark.parametrize("n_parallel", [1, 4, 8])
def test_a_smaller_n_parallel_does_not_lower_the_limiter(n_parallel):
    dspy.settings.configure(async_max_workers=32)

    configure_dspy_concurrency(n_parallel)

    # Another job may be running in this process with a higher setting, and
    # there is only one global limiter to share.
    assert dspy.settings.get("async_max_workers") == 32
    assert get_limiter().total_tokens == 32


def test_generation_actually_applies_it():
    """The wiring, not just the helper: a request for 50 must raise the limiter."""
    dspy.settings.configure(async_max_workers=8)
    kg_gen = MagicMock()
    kg_gen.generate = AsyncMock(
        return_value=(
            Graph(typed_entities=set(), relations_wo_class_assertions=[]),
            KGGenStats(),
        )
    )

    asyncio.run(
        execute_generation(
            kg_gen,
            inputs=[],
            onto=None,
            rdflib_onto=None,
            types=None,
            predicates=None,
            generation_params=GenerationMetadata(n_parallel=50),
        )
    )

    assert get_limiter().total_tokens == 50
    assert kg_gen.generate.await_args.kwargs["n_parallel"] == 50
