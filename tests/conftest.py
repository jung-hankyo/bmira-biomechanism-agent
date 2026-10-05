"""Fixtures shared by the test modules. Tests are offline: no keys, no network."""
import pytest

from bmira.graph import run
from bmira.offline import offline_runtime


@pytest.fixture(scope="module")
def offline():
    rt, sc = offline_runtime()
    return rt, run(sc["question"], rt)

