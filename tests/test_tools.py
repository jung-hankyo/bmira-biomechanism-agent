"""Command-line tools around the pipeline: the question probe (bmira.probe) and the extraction
A/B harness (bmira.ab_extract), plus repository metadata. Offline: no keys, no network."""
import json
import re
from pathlib import Path

import pytest

import bmira
from bmira import ab_extract, probe
from bmira.offline import offline_runtime

from helpers import make_claim

ROOT = Path(__file__).resolve().parent.parent


def test_probe_shows_the_parse_and_round1_hits_offline(capsys):
    probe.main(["--offline"])
    out = capsys.readouterr().out
    assert "exposure  'lactate'" in out and "outcome   'CD8 T cell effector function'" in out
    hits = re.findall(r"^  (\w+)\s+(\d+) hits", out, flags=re.M)
    assert len(hits) == 4 and all(int(n) > 0 for _, n in hits)      # one line per round-1 query
    assert re.search(r"tokens in/out: \d+/\d+", out)


def test_probe_stops_at_an_out_of_scope_question(capsys):
    rt, sc = offline_runtime()
    sc["parsed"] = {**sc["parsed"], "in_scope": False, "scope_note": "Not a biomedical question."}
    probe.probe("What is the capital of France?", rt)
    out = capsys.readouterr().out
    assert "OUT OF SCOPE: Not a biomedical question." in out and "hits" not in out
    assert rt.llm.calls["plan"] == 0


def test_ab_extract_summary_and_overlap():
    kept = [make_claim("a", "p", "increases"), make_claim("b", "p", "no_effect", perturbation_class="knockout")]
    arm = [{"claims": kept, "dropped": [kept[0]], "out": 10, "reason": 4, "cost": 0.5},
           {"claims": [], "dropped": [], "out": 6, "reason": 0, "cost": 0.25}]
    s = ab_extract.summarize(arm)
    assert s["papers_x_reps"] == 2 and s["claims_per_paper"] == 1.0 and s["dropped_per_paper"] == 0.5
    assert s["null_claims_per_paper"] == 0.5 and s["knockout_claims_per_paper"] == 0.5
    assert (s["tokens_out"], s["tokens_reasoning"], s["est_cost_usd"]) == (16, 4, 0.75)
    assert ab_extract.summarize([])["claims_per_paper"] == 0
    assert ab_extract.pairs({"claims": kept}) == {"a|b"}
    assert ab_extract.jaccard({"x", "y"}, {"y", "z"}) == pytest.approx(1 / 3)
    assert ab_extract.jaccard(set(), set()) == 1.0


def _states(tmp_path, *pmid_sets):
    paths = []
    for i, pmids in enumerate(pmid_sets):
        p = tmp_path / f"run{i}.state.json"
        p.write_text(json.dumps({"state": {"question": f"q{i}", "claims": [{"pmid": x} for x in pmids]}}),
                     encoding="utf-8")
        paths.append(p)
    return paths


def test_ab_extract_loads_shared_papers_once_and_then_from_cache(tmp_path, monkeypatch):
    rt, _ = offline_runtime()
    fetched = []

    class Source:                                          # the fixture corpus stands in for PubMed
        def __init__(self, settings):
            pass

        def fetch(self, pmids):
            fetched.append(list(pmids))
            return rt.source.fetch(pmids)

        def fulltext(self, paper):
            return rt.source.fulltext(paper)
    monkeypatch.setattr("bmira.sources.PubMedSource", Source)
    states, cache = _states(tmp_path, ["S001", "S011", "S002"], ["S011", "S001", "S003"]), tmp_path / "texts.json"
    question, papers = ab_extract.load(states, rt.settings, cache)
    assert question == "q1" and [p.pmid for p in papers] == ["S001", "S011"]          # papers read in both runs
    assert fetched == [["S001", "S011"]] and papers[1].text_access == "full_text" and cache.exists()
    again = ab_extract.load(states, rt.settings, cache)[1]
    assert fetched == [["S001", "S011"]] and [p.model_dump() for p in again] == [p.model_dump() for p in papers]


def test_ab_extract_runs_one_extraction_and_surfaces_failures():
    rt, sc = offline_runtime()
    paper = rt.source.fetch(["S001"])[0]
    kept, dropped = ab_extract.run_once(rt, paper, sc["question"])
    assert kept and all(c.pmid == "S001" for c in kept + dropped)

    def broken(*a, **k):
        raise ValueError("unparseable")
    rt.llm.structured = broken
    with pytest.raises(RuntimeError, match="failed"):        # extract() swallows the error; run_once must not
        ab_extract.run_once(rt, paper, sc["question"])


def test_version_is_the_same_everywhere():
    cff = re.search(r"^version: (\S+)$", (ROOT / "CITATION.cff").read_text(encoding="utf-8"), flags=re.M).group(1)
    assert cff == bmira.__version__
    assert f"**v{bmira.__version__}**" in (ROOT / "README.md").read_text(encoding="utf-8")   # a versioning entry


def test_no_new_q1_literals_in_the_package():
    """EV-7: question-specific fixes do not generalize. 37 lines of bmira/*.py named butyrate or Treg at
    v2.6.0; new fixes route a misreading to a judge or the gold set instead of adding literals."""
    lines = [ln for f in sorted((ROOT / "bmira").glob("*.py"))
             for ln in f.read_text(encoding="utf-8").splitlines() if re.search(r"butyrate|treg", ln, re.I)]
    assert len(lines) <= 37, f"{len(lines)} lines name butyrate or Treg (limit 37)"


# Distinguishing terms of the development questions Q2-Q8 (experiments/questions.txt). Q1, the pilots' own question,
# is capped by the test above. A prompt example that names one of these turns that question's probe into recall.
DEV_TERMS = ("lactate", "metformin", "tet2", "nad+", "inflammaging", "atheroscler", "sglt2", "gliflozin",
             "heart failure", "natriuretic", "probnp", "vitamin d", "autoimmune", "gpr81", "hcar1", "cd8")


def test_prompts_do_not_quote_the_development_questions():
    """EV-8: the pilot7 parse prompt taught 'NAD+ decline drives inflammaging', 'SGLT2 inhibitors' and 'heart
    failure hospitalization' (Q5, Q6) and the extract prompt a lactate/Gpr81 blocking test (Q2, Q8), so the
    probe's good reading of those questions was partly recall. Examples come from outside the question file."""
    from bmira.llm import PROMPTS
    text = " ".join(PROMPTS.values()).lower()
    assert [term for term in DEV_TERMS if term in text] == []


def test_ab_extract_reads_chosen_papers_and_lists_blocking_tests(tmp_path, monkeypatch):
    """EM-1/2 check on real text: the census found blocking tests in disguise in pilots 5-7 (e.g. 'Gpr109a
    was essential for butyrate-mediated induction of IL-18'); --pmids re-reads exactly those papers."""
    rt, _ = offline_runtime()

    class Source:
        def __init__(self, settings):
            pass

        def fetch(self, pmids):
            return rt.source.fetch(pmids)

        def fulltext(self, paper):
            return rt.source.fulltext(paper)
    monkeypatch.setattr("bmira.sources.PubMedSource", Source)
    states = _states(tmp_path, ["S001", "S002"])
    question, papers = ab_extract.load(states, rt.settings, tmp_path / "t.json", pmids=["S011", "S003"])
    assert question == "q0" and [p.pmid for p in papers] == ["S003", "S011"]          # not the state's papers

    block = make_claim("a", "p", "required_for", effect_exposure="butyrate", effect_result="abolished")
    lost = make_claim("b", "p", "required_for", effect_exposure="butyrate", effect_result="abolished")
    lost.drop_reason = "blocking-test treatment not named in quote"
    arm = [{"rep": 0, "claims": [block, make_claim("c", "p", "increases")], "dropped": [lost],
            "out": 1, "reason": 0, "cost": 0.0}]
    s = ab_extract.summarize(arm)
    assert s["blocking_tests_per_paper"] == 1.0 and s["blocking_tests_dropped_per_paper"] == 1.0
    rows = ab_extract.blocking_tests(arm)
    assert [(r["kept"], r["effect_result"]) for r in rows] == [(True, "abolished"), (False, "abolished")]
    assert rows[1]["drop_reason"].startswith("blocking-test treatment")
    every = ab_extract.listing(arm)            # a missed blocking test is read here, among the ordinary claims
    assert [(r["pmid"], r["kept"]) for r in every] == [("p", True), ("p", True), ("p", False)]
