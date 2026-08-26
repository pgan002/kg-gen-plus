"""The real LM error must reach the operator, not an AttributeError about None."""

import pytest

from kg_gen.steps._1_get_entities import _refine_or_surface_error


class _Boom:
    """Stands in for the dspy module that dspy.Refine wraps."""

    def __init__(self):
        self.calls = 0

    async def acall(self, **kwargs):
        self.calls += 1
        raise RuntimeError("unsupported value: 'temperature' does not support 0.4")


@pytest.mark.asyncio
async def test_real_error_surfaces_when_every_refine_attempt_fails():
    """dspy.Refine swallows the cause and returns None; we must not hide behind that.

    Without this, callers touch ``.entities`` on the None and the operator sees
    ``'NoneType' object has no attribute 'entities'`` -- which says nothing about
    the authentication failure, unsupported parameter or rate limit that actually
    stopped the run.
    """
    module = _Boom()

    def refine_that_gives_up(**kwargs):
        return None

    with pytest.raises(RuntimeError, match="temperature"):
        await _refine_or_surface_error(refine_that_gives_up, module, source_text="x")

    assert module.calls == 1, "the bare module should be retried exactly once"


@pytest.mark.asyncio
async def test_successful_refine_does_not_re_call_the_module():
    """The extra call happens only on the already-failed path, never in the happy one."""
    module = _Boom()
    sentinel = object()

    def refine_that_succeeds(**kwargs):
        return sentinel

    result = await _refine_or_surface_error(
        refine_that_succeeds, module, source_text="x"
    )
    assert result is sentinel
    assert module.calls == 0
