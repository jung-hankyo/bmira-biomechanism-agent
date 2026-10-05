"""Fixtures shared by the test modules. Tests are offline: no keys, no network."""
import pytest

from bmira.graph import run
from bmira.offline import offline_runtime


@pytest.fixture(scope="module")
def offline():
    rt, sc = offline_runtime()
    return rt, run(sc["question"], rt)



@pytest.fixture
def fake_chat_openai(monkeypatch):
    """Replaces langchain_openai for one test. ChatOpenAI(**kw) records kw and returns it,
    so tests can read the parameters a model client was built with."""
    import sys
    import types
    seen = []
    fake = types.ModuleType("langchain_openai")
    fake.ChatOpenAI = lambda **kw: seen.append(kw) or kw
    monkeypatch.setitem(sys.modules, "langchain_openai", fake)
    return seen
