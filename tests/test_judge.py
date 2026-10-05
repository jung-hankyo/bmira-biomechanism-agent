"""The decision model (bmira.judge, bmira.questions) and its shadow touchpoints (bmira.shadow).
HTTP is replaced; offline runs use SurrogateJudge. Offline: no keys, no network."""
import json

import pytest
import requests

import bmira.judge as jd
from bmira import questions as Q
from bmira.config import Settings
from bmira.offline import offline_runtime
from bmira.shadow import agreement, main as shadow_main, typed_relation, _reconstructed
from bmira.telemetry import execute, preflight, summarize

from helpers import make_claim, run_session

NOUL = {"ok": {"type": "noul", "instructions": "Is `word` the word ping?"}}


class Reply:
    def __init__(self, status=200, body=None, bad_json=False):
        self.status_code, self.body, self.bad_json = status, body, bad_json

    def json(self):
        if self.bad_json:
            raise ValueError("not JSON")
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def http(monkeypatch):
    """Scripted POST replies (a list consumed in order) and recorded request bodies; no real waiting."""
    script = {"replies": [], "sent": []}

    def post(url, json=None, **kw):
        script["sent"].append(json)
        r = script["replies"].pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(jd.requests, "post", post)
    monkeypatch.setattr(jd.time, "sleep", lambda s: None)
    return script


def _judge(tmp_path, **kw):
    return jd.JevJudge(Settings(judge_provider="jev", cache_dir=str(tmp_path), **kw), api_key="k")


OK = Reply(200, {"answers": {"ok": {"noul": 0.97}}, "usage": {"input_tokens": 40}})


def test_make_judge_refuses_act_mode_aliases_and_unknown_providers(monkeypatch):
    assert jd.make_judge(Settings()) is None                                      # off by default
    with pytest.raises(ValueError, match="calibration"):
        jd.make_judge(Settings(judge_provider="jev", judge_mode="act"))
    with pytest.raises(ValueError, match="moving alias"):
        jd.make_judge(Settings(judge_provider="jev", judge_model="jev-latest"))
    with pytest.raises(ValueError, match="unknown judge_provider"):
        jd.make_judge(Settings(judge_provider="gpt"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        jd.make_judge(Settings(judge_provider="jev"))
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert isinstance(jd.make_judge(Settings(judge_provider="jev")), jd.JevJudge)


def test_ask_counts_caches_and_persists_answers(tmp_path, http):
    http["replies"] = [OK]
    j = _judge(tmp_path)
    assert j.ask("t", {"word": "ping"}, NOUL) == {"ok": {"noul": 0.97}}
    assert http["sent"][0]["model"] == "jev-1.13.0" and http["sent"][0]["questions"] == NOUL
    assert j.ask("t", {"word": "ping"}, NOUL)["ok"]["noul"] == 0.97                # second time: no request
    assert (j.calls["t"], j.cache_hits["t"], j.tokens_in["t"]) == (1, 1, 40)
    j.save()
    again = _judge(tmp_path)                                                       # a replay: free and identical
    assert again.ask("t", {"word": "ping"}, NOUL) == {"ok": {"noul": 0.97}} and again.calls["t"] == 0
    assert not (tmp_path / "judge_jev-1.13.0.tmp").exists()                        # written atomically
    other = _judge(tmp_path, judge_model="jev-1.14.0")                             # a new version is a new model
    assert other.cache == {}


def test_transient_errors_are_retried_and_exhaustion_is_one_failure(tmp_path, http):
    http["replies"] = [Reply(429), requests.ConnectionError("reset"), Reply(529), OK]
    j = _judge(tmp_path)
    assert j.ask("t", {"word": "ping"}, NOUL)["ok"]["noul"] == 0.97 and j.failures["t"] == 0
    http["replies"] = [Reply(503)] * 6
    with pytest.raises(jd.JudgeUnavailable, match="retries exhausted"):
        j.ask("t", {"word": "other"}, NOUL)
    assert j.failures["t"] == 1 and not http["replies"]


@pytest.mark.parametrize("reply", [Reply(200, {"usage": {}}), Reply(200, {"answers": ["x"]}),
                                   Reply(200, bad_json=True), Reply(400)])
def test_malformed_replies_cost_one_item(tmp_path, http, reply):
    http["replies"] = [reply]
    j = _judge(tmp_path)
    with pytest.raises(jd.JudgeUnavailable):
        j.ask("t", {"word": "ping"}, NOUL)
    assert j.failures["t"] == 1 and j.cache == {}


def test_a_bad_key_switches_the_judge_off_for_the_run(tmp_path, http):
    http["replies"] = [Reply(401)]
    j = _judge(tmp_path)
    assert j.ask_many("t", [({"word": "a"}, NOUL), ({"word": "b"}, NOUL), ({"word": "c"}, NOUL)]) == [None] * 3
    assert j.disabled == "HTTP 401" and len(http["sent"]) == 1                     # one request, not three


def test_ask_many_isolates_failures(tmp_path, http):
    j = _judge(tmp_path, judge_workers=1)                                          # one worker: replies in order
    http["replies"] = [OK, Reply(200, bad_json=True), OK]
    out = j.ask_many("t", [({"word": "a"}, NOUL), ({"word": "b"}, NOUL), ({"word": "c"}, NOUL, {"current": {}})])
    assert out[0] and out[1] is None and out[2] and j.failures["t"] == 1
    assert j.ask_many("t", []) == []


def test_an_unreadable_cache_starts_empty(tmp_path, capsys):
    (tmp_path / "judge_jev-1.13.0.json").write_text("{not json", encoding="utf-8")
    assert _judge(tmp_path).cache == {} and "unreadable" in capsys.readouterr().out


@pytest.mark.parametrize("answer, expected", [
    ({"probabilities": [0.1, 0.2, 0.3, 0.4]}, [0.1, 0.2, 0.3, 0.4]),
    ({"probabilities": {"0": 1, "1": 1, "2": 1, "3": 1}}, [0.25] * 4),               # 0-based keys, normalized
    ({"probabilities": {"1": 0.0, "2": 0.0, "3": 0.5, "4": 0.5}}, [0, 0, 0.5, 0.5]),  # 1-based keys
    ({"probabilities": {c: (1.0 if i == 2 else 0.0) for i, c in enumerate(Q.SCREEN["relevance"]["criteria"])}},
     [0, 0, 1, 0]),                                                                   # keyed by criterion text
    ({"score": 3}, [0, 0, 0, 1]), ({"score": 4}, [0, 0, 0, 1]), ({"score": 0.6}, [0, 1, 0, 0]),
    ({"probabilities": {"a": 1}}, None), ({"probabilities": [0, 0, 0, 0]}, None), ({}, None), (None, None)])
def test_score_answers_become_level_probabilities(answer, expected):
    got = jd.level_probs(answer, Q.SCREEN["relevance"]["criteria"])
    assert got == (None if expected is None else pytest.approx(expected))


def test_answer_readers_tolerate_missing_and_malformed_answers():
    assert jd.p_yes(None) is None and jd.p_yes({"noul": "high"}) is None and jd.p_yes({"noul": 1}) == 1.0
    assert jd.top(None) == (None, 0.0, {}) and jd.top({"probabilities": {}}) == (None, 0.0, {})
    assert jd.top({"choice": "x", "confidence": None})[:2] == ("x", 0.0)
    assert jd.expected_level(None) is None and jd.expected_level([0, 0.5, 0.5]) == 1.5


def test_reversed_options_only_reorders_choices():
    rev = Q.reversed_options(Q.STANCE["stance"])
    assert list(rev["criteria"]) == ["says_nothing", "contradicts", "supports"]
    assert Q.reversed_options(Q.SUBJECT_LOST["subject_lost"]) is Q.SUBJECT_LOST["subject_lost"]
    assert list(Q.RELATION["relation"]["criteria"])[-1] == "unresolved"            # kept last (Appendix B)


def _verdicts(final):
    return [(h.id, h.status, h.score, h.reason) for h in final["hypotheses"]]


def test_shadow_mode_changes_no_decision():
    """The judge asks and logs; every verdict, score and the report are those of a run without it."""
    rt0, sc = offline_runtime()
    off, _ = execute(sc["question"], rt0, echo=False)
    rt1, _ = offline_runtime(judge_provider="jev", judge_workers=2)
    rt1.judge = jd.SurrogateJudge(p=0.95, disagree={"comparator", "stance", "strength"})   # a judge that disagrees
    on, info = execute(sc["question"], rt1, echo=False)
    assert _verdicts(on) == _verdicts(off) and on["report"] == off["report"]
    assert [c.model_dump() for c in on["claims"]] == [c.model_dump() for c in off["claims"]]
    assert not off.get("judge_log") and on["judge_log"]
    m = summarize(on, rt1, info)["judge"]
    assert m["agreement"]["methods.comparator"]["rate"] == 0.0 and m["agreement"]["methods.rescue"]["rate"] == 1.0
    assert m["agreement"]["methods.comparator"]["disagreements"]
    tasks = {e["task"] for e in on["judge_log"]}
    assert {"screen.include", "screen.original_data", "stance", "methods.perturbation", "relation",
            "report.strength", "report.overclaim", "report.entailment"} <= tasks


def test_a_broken_judge_never_costs_the_run(capsys):
    class Broken(jd.SurrogateJudge):
        def ask_many(self, task, items):
            raise RuntimeError("judge exploded")
    rt, sc = offline_runtime(judge_provider="jev")
    rt.judge = Broken()
    final, info = execute(sc["question"], rt, echo=False)
    assert info["status"] == "completed" and final["verification"]["passed"] and not final.get("judge_log")
    assert "[shadow][WARN]" in "\n".join(info["log"])


def test_judge_off_leaves_the_summary_unchanged_and_session_records_the_setting(tmp_path):
    session = run_session(tmp_path, "one\n")
    assert session["runs"][0]["judge"] == {"provider": "off"} and session["session"]["judge"]["provider"] == "off"
    with_judge = run_session(tmp_path, "one\n", "--judge", "jev", out="j.json")
    run = with_judge["runs"][0]
    assert run["judge"]["judgments"] > 0 and run["judge"]["mode"] == "shadow"
    assert any(c["check"].startswith("Judge") and not c["required"] for c in with_judge["session"]["preflight"])


def test_preflight_reports_a_failing_judge_without_blocking():
    rt, _ = offline_runtime(judge_provider="jev")

    def down(*a, **k):
        raise jd.JudgeUnavailable("down")
    rt.judge.ask = down
    judge_check = next(c for c in preflight(rt) if c["check"].startswith("Judge"))
    assert not judge_check["ok"] and not judge_check["required"]


def test_shadow_replays_a_saved_run_offline(tmp_path, capsys):
    run_session(tmp_path, "only question\n")
    out = tmp_path / "log.jsonl"
    log = shadow_main([str(tmp_path / "s_q1.state.json"), "--offline", "--out", str(out),
                       "--cache-dir", str(tmp_path / "cache")])
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert rows and len(rows) == len(log) and all(r["state"].endswith("s_q1.state.json") for r in rows)
    table = agreement(log)
    assert table["stance"]["rate"] == 1.0 and table["relation"]["compared"] > 0
    assert "| screen.include |" in capsys.readouterr().out


def test_saved_claims_are_read_back_as_extracted():
    c = make_claim("c", "p", "decreases", subject_lost=True)
    c.subject, c.object, c.relation_norm = "Gpr109a", "CD103+ DC", "increases"     # restated after the loss
    assert typed_relation(c) == "decreases"
    c.subject_lost, c.subject = False, "Tet2 loss"
    assert typed_relation(c) == "decreases"
    c.subject = "Tet2"
    assert typed_relation(c) == "increases"                                          # nothing was restated
    c.method_checks = ["comparator_present not evidenced", "perturbation 'knockout' not evidenced"]
    c.comparator_present, c.perturbation_class = False, "none"
    before = _reconstructed(c)
    assert before.comparator_present and before.perturbation_class == "knockout" and not c.comparator_present


def test_a_reply_that_is_not_an_object_costs_one_item(tmp_path, http):
    """Review finding: '["oops"]' raised TypeError out of ask_many and lost the whole batch."""
    j = _judge(tmp_path, judge_workers=1)
    http["replies"] = [OK, Reply(200, ["oops"]), Reply(200, {"answers": {"ok": {"noul": 0.5}}, "usage": "n/a"})]
    out = j.ask_many("t", [({"word": "a"}, NOUL), ({"word": "b"}, NOUL), ({"word": "c"}, NOUL)])
    assert out[0] and out[1] is None and out[2] == {"ok": {"noul": 0.5}} and j.failures["t"] == 1


def test_an_unreachable_service_is_switched_off_without_waiting_after_the_last_try(tmp_path, monkeypatch, http):
    """Review finding: a hanging endpoint cost ~4 minutes per item inside pipeline nodes."""
    waits = []
    monkeypatch.setattr(jd.time, "sleep", waits.append)
    j = _judge(tmp_path, judge_workers=1, judge_max_retries=2)
    http["replies"] = [requests.Timeout("slow")] * 6
    assert j.ask_many("t", [({"word": w}, NOUL) for w in "abcde"]) == [None] * 5
    assert j.disabled.startswith("unreachable") and len(http["sent"]) == 6        # 3 items x 2 tries, then off
    assert waits == [1, 1, 1]                                                      # one wait per item, none after the last try
    http["replies"] = [requests.Timeout("slow"), OK]                               # a success resets the count
    j2 = _judge(tmp_path / "x", judge_workers=1, judge_max_retries=1)
    j2.ask_many("t", [({"word": "a"}, NOUL), ({"word": "b"}, NOUL)])
    assert j2.disabled == "" and j2._lost_in_a_row == 0
