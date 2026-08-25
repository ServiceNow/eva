import pytest

from eva.user_simulator.cascade.phrase_cache import PhraseCache


@pytest.fixture(autouse=True)
def _isolate_phrase_cache():
    """Clear the process-global phrase audio between tests.

    The cache is shared across conversations on purpose — that is what stops every
    record re-rendering the same "mm-hmm" — but tests are conversations too, so
    without this one test's renders satisfy the next one's prerender and call
    counts come out wrong.
    """
    PhraseCache.clear()
    yield
    PhraseCache.clear()
