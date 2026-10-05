"""The LLM client: parameters, model choice, error classes (bmira.llm). Offline: no keys, no network."""
from bmira.config import Settings


def test_temperature_is_not_sent_by_default(fake_chat_openai):
    """Some reasoning models reject any temperature but their default."""
    from bmira.llm import LangChainLLM
    seen = fake_chat_openai
    LangChainLLM(Settings(), api_key="k")._model("reasoning")
    assert "temperature" not in seen[-1]
    LangChainLLM(Settings(temperature=0.0), api_key="k")._model("cheap")
    assert seen[-1]["temperature"] == 0.0


def test_error_classification():
    from bmira.llm import is_fatal

    class E(Exception):
        status_code = None
    assert is_fatal(E("Error code: 429 - insufficient_quota credit_balance_exhausted"))
    assert is_fatal(E("Unsupported value: 'temperature' ... unsupported_value"))
    assert is_fatal(E("model_not_found"))
    assert not is_fatal(E("Error code: 429 - slow_down"))
    assert not is_fatal(E("Error code: 503 - server_is_overloaded"))


def test_effort_retries_and_cost(fake_chat_openai):
    from bmira.llm import LangChainLLM
    from bmira.telemetry import estimate_cost
    seen = fake_chat_openai
    llm = LangChainLLM(Settings(), api_key="k")
    llm._for("screen", "cheap")
    assert seen[-1]["reasoning_effort"] == "low" and seen[-1]["max_retries"] == 6
    llm._for("extract", "reasoning")
    assert seen[-1]["reasoning_effort"] == "medium" and llm.model_of["extract"] == Settings().models["openai"]["reasoning"]
    assert estimate_cost("gpt-6-sol", 1_000_000, 100_000, Settings().prices) == 3.0
    assert estimate_cost("unknown-model", 10, 10, Settings().prices) is None


def test_classification_tasks_use_cheap_model(fake_chat_openai):
    from bmira.llm import LangChainLLM
    llm, m = LangChainLLM(Settings(), api_key="k"), Settings().models["openai"]
    assert llm._for("pair", "reasoning")["model"] == m["cheap"]
    assert llm._for("extract", "reasoning")["model"] == m["reasoning"]


def test_one_bad_screening_reply_does_not_discard_the_paid_batch():
    from types import SimpleNamespace
    from bmira.llm import LangChainLLM
    from bmira.schemas import Screen
    llm = LangChainLLM(Settings(), api_key="x")
    ok = Screen(relevant=True, relevance_score=80, reason="r", study_type="animal")
    batch = lambda msgs, config, return_exceptions: [{"parsed": ok, "raw": None}, ValueError("bad json"),
                                                     {"parsed": None, "raw": None, "parsing_error": ValueError()}]
    llm._for = lambda task, role: SimpleNamespace(with_structured_output=lambda *a, **k: SimpleNamespace(batch=batch))
    assert llm.structured_many("screen", Screen, "s", ["a", "b", "c"]) == [ok, None, None]
    assert llm.failures["screen"] == 2

