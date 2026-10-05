"""Session runner, saved runs, failure handling and run summaries (bmira.experiments, bmira.telemetry). Offline: no keys, no network."""
from bmira.offline import offline_runtime

from helpers import fail_on_call


# Telemetry: one session file, rewritten per run, with metrics and revision signals.
def test_experiment_runner_writes_one_session_file(tmp_path):
    import json
    from bmira.experiments import main
    qs = tmp_path / "q.txt"
    qs.write_text("# comment\nfirst question\nsecond question\n", encoding="utf-8")
    out = main(["--offline", "--quiet", "--questions", str(qs), "--out", str(tmp_path / "s.json")])
    session = json.loads(out.read_text(encoding="utf-8"))   # the file is UTF-8; Windows defaults differ
    assert len(session["runs"]) == 2 and session["session"]["mode"] == "offline"
    run = session["runs"][0]
    for section in ("llm", "retrieval", "papers", "extraction", "normalization", "grading",
                    "comparison", "steps", "pathways", "verification", "signals"):
        assert section in run
    assert run["extraction"]["drop_reasons"]["null claim but the quote reports an effect"] == 1
    assert run["pathways"]["ranked"][0]["verdict"] == "Supported"
    assert all({"signal", "value", "threshold", "look_at"} <= set(x) for x in run["signals"])


# H1-H3: failure-safe telemetry, fatal errors, preflight, budget, effort and cost.
def test_fatal_error_aborts_session_but_keeps_partial_run(tmp_path, monkeypatch):
    import json
    from bmira.experiments import main
    from bmira.llm import FatalLLMError
    fail_on_call(monkeypatch, "plan", FatalLLMError("plan: insufficient_quota"), on_call=2)
    qs = tmp_path / "q.txt"
    qs.write_text("one\ntwo\n", encoding="utf-8")
    out = main(["--offline", "--quiet", "--questions", str(qs), "--out", str(tmp_path / "s.json")])
    session = json.loads(out.read_text(encoding="utf-8"))
    assert len(session["runs"]) == 1 and "fatal" in session["session"]["aborted"]
    run = session["runs"][0]
    assert run["status"] == "aborted" and run["failed_node"] == "plan"
    assert run["papers"]["retrieved"] > 0 and run["extraction"]["claims_kept"] > 0   # round 1 kept
    assert run["llm"]["total_tokens_in"] > 0 and "run did not finish" in [x["signal"] for x in run["signals"]]


def test_nonfatal_error_fails_one_run_only(tmp_path, monkeypatch):
    import json
    from bmira.experiments import main
    fail_on_call(monkeypatch, "seed", RuntimeError("boom"), on_call=1)
    qs = tmp_path / "q.txt"
    qs.write_text("one\ntwo\n", encoding="utf-8")
    session = json.loads(main(["--offline", "--quiet", "--questions", str(qs), "--out",
                               str(tmp_path / "s.json")]).read_text(encoding="utf-8"))
    assert len(session["runs"]) == 2 and "aborted" not in session["session"]   # seeding degrades, run completes


def test_uncaught_nonfatal_error_fails_run_and_session_continues(tmp_path, monkeypatch):
    import json
    from bmira.experiments import main
    fail_on_call(monkeypatch, "plan", RuntimeError("upstream hiccup"), on_call=2)
    qs = tmp_path / "q.txt"
    qs.write_text("one\ntwo\n", encoding="utf-8")
    session = json.loads(main(["--offline", "--quiet", "--questions", str(qs), "--out",
                               str(tmp_path / "s.json")]).read_text(encoding="utf-8"))
    first, second = session["runs"]
    assert first["status"] == "failed" and first["failed_node"] == "plan" and first["papers"]["retrieved"] > 0
    assert second["status"] == "completed"


def test_preflight_blocks_spending(tmp_path, monkeypatch):
    import json
    from bmira.experiments import main
    from bmira.llm import FatalLLMError
    qs = tmp_path / "q.txt"
    qs.write_text("one\n", encoding="utf-8")
    ok = json.loads(main(["--offline", "--quiet", "--questions", str(qs), "--out",
                          str(tmp_path / "a.json")]).read_text(encoding="utf-8"))
    assert all(c["ok"] for c in ok["session"]["preflight"]) and len(ok["runs"]) == 1
    fail_on_call(monkeypatch, "preflight", FatalLLMError("preflight: invalid_api_key"))
    bad = json.loads(main(["--offline", "--quiet", "--questions", str(qs), "--out",
                           str(tmp_path / "b.json")]).read_text(encoding="utf-8"))
    assert bad["runs"] == [] and "preflight failed" in bad["session"]["aborted"]


def test_budget_stops_search_and_still_reports():
    from bmira.telemetry import execute, summarize
    rt, sc = offline_runtime(budget_tokens=1)
    final, info = execute(sc["question"], rt, echo=False)
    m = summarize(final, rt, info)
    assert final["gate"] == "BUDGET" and final["round_idx"] == 1 and final["report"]
    assert m["run"]["stop_reason"] == "token budget reached" and any("budget" in w for w in final["warnings"])


# K1-K10: fixes from the pilot3 live run (butyrate -> Treg).
def test_summary_hides_keys_and_lists_claims_and_links():
    import json
    from bmira.telemetry import execute, signals, summarize
    rt, sc = offline_runtime()
    rt.settings.ncbi_api_key, rt.settings.ncbi_email = "SECRET-KEY", "me@example.org"
    final, info = execute(sc["question"], rt, echo=False)
    m = summarize(final, rt, info)
    blob = json.dumps(m, default=str)
    assert "SECRET-KEY" not in blob and "me@example.org" not in blob
    assert len(m["claims"]) == len(final["claims"])
    assert {"id", "pmid", "subject", "relation", "object", "grade", "limiting_axis"} <= set(m["claims"][0])
    assert {"key", "status", "reason", "n_studies", "grade"} <= set(m["links"][0])
    assert m["normalization"]["duplicate_labels"] == [] and m["comparison"]["pairs_asked"] == len(rt.pair_cache)
    fired = {s["signal"] for s in signals({"normalization": {"duplicate_labels": ["regulatory t cell"]},
                                           "comparison": {"pairs_asked": 227, "pairs_judged": 0}})}
    assert {"one label, several concepts", "pair verdicts lost"} <= fired


# M1-M5: exposure direction ("NAD+ decline", "TET2 loss") and randomized-trial grading.
def test_saved_run_replays_without_searching_or_rejudging(tmp_path):
    """W1: a live run leaves a state file; a replay re-runs normalize -> verify on its claims
    with the current code, never searches, and reuses the pair verdicts the run paid for."""
    import json
    from bmira.experiments import main
    qs = tmp_path / "q.txt"
    qs.write_text("only question\n", encoding="utf-8")
    first = main(["--offline", "--quiet", "--questions", str(qs), "--out", str(tmp_path / "s.json")])
    state = tmp_path / "s_q1.state.json"
    assert state.exists()
    assert all(not p["source_text"] for p in json.loads(state.read_text(encoding="utf-8"))["state"]["papers"])
    again = main(["--offline", "--quiet", "--replay", str(state), "--out", str(tmp_path / "r.json")])
    a = json.loads(first.read_text(encoding="utf-8"))["runs"][0]
    session = json.loads(again.read_text(encoding="utf-8"))
    b = session["runs"][0]
    assert session["session"]["mode"] == "replay" and b["status"] == "completed" and b["replay_of"] == str(state)
    assert not {"parse", "plan", "screen", "extract", "pair"} & set(b["llm"]["per_task"])
    assert b["papers"] == a["papers"] and b["extraction"]["claims_kept"] == a["extraction"]["claims_kept"]
    assert b["run"]["rounds"] == a["run"]["rounds"]
    assert [p["verdict"] for p in b["pathways"]["ranked"]] == [p["verdict"] for p in a["pathways"]["ranked"]]
    assert b["verification"]["passed"]


# P1-P6: fixes from the pilot5 live run (butyrate -> Treg).
def test_split_exposure_and_off_portfolio_support_are_signalled():
    """Pilot4's real failure left no signal: 25 claims on butyrate variants and Supported steps
    on no pathway, with fragmentation reading a healthy 0.32."""
    from bmira.telemetry import execute, signals, summarize
    rt, sc = offline_runtime()
    final, info = execute(sc["question"], rt, echo=False)
    names = lambda m: {x["signal"] for x in signals(m)}
    ok = summarize(final, rt, info)
    assert ok["normalization"]["exposure_variants"]["claims"] == 0
    assert "exposure split" not in names(ok) and "supported off-portfolio" not in names(ok)
    exposure = rt.resolver.concepts[final["exposure"]].label
    claim = next(c for c in final["claims"] if c.subject_concept == final["exposure"])
    claim.subject_concept, claim.subject_label = "LOCAL:variant", f"sodium {exposure}"
    split = summarize(final, rt, info)
    assert split["normalization"]["exposure_variants"]["labels"] == {f"sodium {exposure}": 1}
    assert "exposure split" in names(split)
    off = summarize({**final, "hypotheses": []}, rt, info)          # no pathway uses any Supported step
    n = ok["steps"]["verdicts"]["Supported"]
    assert n >= 2 and len(off["steps"]["supported_off_portfolio"]) == n
    assert "supported off-portfolio" in names(off)

