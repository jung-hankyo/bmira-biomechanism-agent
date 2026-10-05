"""Offline runs of the whole graph, and single graph nodes (bmira.graph). Offline: no keys, no network."""
import pytest

from bmira import portfolio as pf
from bmira.config import Settings
from bmira.graph import build_agent
from bmira.normalize import EntityResolver
from bmira.offline import offline_runtime

from helpers import StubLLM, fake_ols, make_claim, parsed_question


# P0: the whole graph runs offline and the report passes its own verification.
def test_end_to_end(offline):
    rt, final = offline
    assert final["verification"]["passed"], final["report"]
    assert "Pathway portfolio (computed)" in final["report"]


# P1: retracted papers yield nothing; stitched spans are dropped; nothing is judged twice.
def test_hardening(offline):
    rt, final = offline
    pmids = {c.pmid for c in final["claims"]}
    assert "S008" not in pmids                               # retracted
    assert any(c.pmid == "S009" for c in final["dropped_claims"])   # fabricated span
    llm_rel = sum(c.relation_source == "llm" for c in final["claims"])
    assert rt.llm.items["relation"] == llm_rel               # each relation sent once
    assert rt.llm.items["pair"] == len(rt.pair_cache)        # each pair judged once


# P2: alias merges survive re-normalization from raw surface text.
def test_aliases_in_run(offline):
    _, final = offline
    subj = {c.pmid: c.subject_concept for c in final["claims"]}
    assert subj["S001"] == subj["S002"]                      # lactate == lactic acid


# P3: the LLM's favourite pathway is wrong; the unseeded one must finish leading.
def test_portfolio_escapes_lock_in(offline):
    _, final = offline
    hyps = {h.name: h for h in final["hypotheses"]}
    assert final["hypotheses"][0].name == "Redox-metabolic route"
    assert hyps["Redox-metabolic route"].status == "supported"
    assert hyps["Redox-metabolic route"].origin == "llm_expansion"
    assert hyps["Chromatin route"].status == "contradicted"
    receptor = hyps["Receptor route"]
    assert receptor.status == "insufficient" and "no study found" in receptor.reason
    assert not receptor.open                                  # searched out: no more budget


def test_run_applies_evidence_rules(offline):
    _, final = offline
    dropped = {c.pmid: c.drop_reason for c in final["dropped_claims"]}
    assert dropped["S018"] == "null claim but the quote reports an effect"
    nad = next(ln for ln in final["links"].values()
               if ln.subject_label == "Lactate" and ln.object_label == "NAD+" and ln.relation == "decreases")
    assert nad.n_studies == 2 and any("review" in r for r in nad.uncounted.values())
    for p in final["papers"]:                                  # R7: every targeted hit was read for its step
        if p.screen_status == "included":
            assert set(p.retrieved_for) <= set(p.read_for)


def test_interactive_review_drops_pathway():
    from langgraph.types import Command
    rt, sc = offline_runtime(interactive=True)
    agent = build_agent(rt)
    cfg = {"configurable": {"thread_id": "t1"}, "recursion_limit": 250}
    agent.invoke({"question": sc["question"]}, cfg)
    final = agent.invoke(Command(resume={"drop": ["H2"]}), cfg)
    assert "H2" not in {h.id for h in final["hypotheses"]}


# Chat: follow-ups are answered from the run and pass the same verifier as the report.
def test_chat_followup(offline):
    from bmira.chat import answer, portfolio_rows
    rt, final = offline
    reply, issues = answer(final, rt, "What supports the redox route?", [])
    assert "Redox-metabolic route: Supported" in reply and not issues
    assert portfolio_rows(final)[0]["verdict"] == "Supported"


def test_streamlit_app_offline():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    at.chat_input[0].set_value("anything").run()            # offline: runs the demo scenario
    assert not at.exception and "Leading pathway" in at.chat_message[1].markdown[0].value
    at.chat_input[0].set_value("Why is the chromatin route contradicted?").run()
    assert not at.exception and "Contradicted" in at.chat_message[3].markdown[0].value


def test_screening_cutoff_and_targeted_retmax():
    from bmira.telemetry import execute
    rt, sc = offline_runtime()
    next(p for p in rt.llm.s["papers"] if p["pmid"] == "S007")["relevance"] = 30
    asked = []
    real = rt.source.search
    rt.source.search = lambda q, n: asked.append(n) or real(q, n)
    final, _ = execute(sc["question"], rt, echo=False)
    s007 = next(p for p in final["papers"] if p.pmid == "S007")
    assert s007.screen_status == "excluded" and "cut-off" in s007.relevance_reason
    assert asked[:4] == [20] * 4 and set(asked[4:]) == {5}                # coverage 20, targeted 5


def test_parse_strips_exposure_direction_and_keeps_it():
    from types import SimpleNamespace
    from bmira.graph import parse
    llm = StubLLM(parsed_question(population_model="macrophages", exposure="age-related NAD+ decline",
                                  comparator="young", outcome="inflammaging", expected_direction="up"))
    rt = SimpleNamespace(llm=llm, resolver=EntityResolver(Settings(ontology_provider="off")))
    out = parse({"question": "q"}, rt)
    assert out["parsed"].exposure == "NAD+" and out["parsed"].exposure_change == "down"
    assert out["exposure"] == rt.resolver.resolve("NAD+").id


def test_loss_of_function_claims_are_restated_for_the_bare_entity():
    from bmira.graph import normalize
    rt, _ = offline_runtime()
    c = make_claim("c1", "p", "")
    c.subject, c.object, c.relation, c.relation_raw = "Tet2 loss", "IL-6", "increased", "increased"
    parsed = parsed_question(exposure="Tet2")
    out = normalize({"claims": [c], "parsed": parsed}, rt)["claims"]
    assert out[0].subject_label == "Tet2" and out[0].relation_norm == "decreases"      # loss increases = Tet2 decreases
    again = normalize({"claims": out, "parsed": parsed}, rt)["claims"]
    assert again[0].relation_norm == "decreases"                                       # flipped once, not every round


def test_plan_matches_target_ids_leniently_and_says_when_it_drops_queries(capsys):
    """Pilot5 ran 3 queries for 3 targets (pilot4: 12) and nothing said why."""
    from bmira.graph import plan
    from bmira.schemas import QueryPlan, SearchQuery
    rt, _ = offline_runtime()
    key = "A|increases|B"
    ln = pf.LinkEvidence(key=key, subject="A", relation="increases", object="B", subject_label="a", object_label="b")

    def LLM(targets):
        return StubLLM(QueryPlan(queries=[SearchQuery(query=f"q{i}", intent="gap_positive", target=t)
                                          for i, t in enumerate(targets)]))
    parsed = parsed_question(outcome="b")
    state = {"parsed": parsed, "question": "q", "links": {key: ln}, "targets": [key]}
    rt.llm = LLM(["T1", "[T1]", "1", f"[{key}]", "T2", "nonsense"])
    out = plan(state, rt)["queries"]
    assert [q.target for q in out[:4]] == [key] * 4 and len(out) == 5           # 4 matched + 1 built by code
    assert "[T1]" in rt.llm.prompt and key not in rt.llm.prompt                  # a number, nothing to echo
    assert "[plan][WARN] 2 of 6 model queries named no known target" in capsys.readouterr().out
    rt.llm = LLM([key])
    plan(state, rt)
    assert "WARN" not in capsys.readouterr().out


def test_long_form_replaces_an_abbreviation_only_when_it_resolves_to_nothing():
    """Pilot5: SB stayed LOCAL:sb. GPR109A and iTreg already resolve and must not move; a
    long form that is no better (the abbreviation's own definition garbled) is ignored."""
    from bmira.graph import _long_forms, normalize
    from bmira.schemas import Paper
    rt, _ = offline_runtime()
    rt.resolver.settings.ontology_provider = "ols"
    rt.resolver._ols = fake_ols({"sodium butyrate": ("CHEBI:64103", "sodium butyrate"),
                                 "gpr109a": ("PR:000001629", "hydroxycarboxylic acid receptor 2"),
                                 "butyrate": ("CHEBI:17968", "butyrate")})
    rt.resolver.llm = None
    abstract = "Mice received sodium butyrate (SB). GPR109A (also G protein-coupled receptor 109A) was measured."
    paper = Paper(pmid="p", title="t", abstract=abstract, source_text=abstract)
    sb = make_claim("c1", "p", "increases"); sb.subject, sb.object, sb.span = "SB", "IL-10", "SB increased IL-10."
    gp = make_claim("c2", "p", "increases"); gp.subject, gp.object, gp.span = "GPR109A", "IL-10", "GPR109A rose."
    forms = _long_forms({"papers": [paper]})
    assert forms["p"]["sb"] == "sodium butyrate"
    parsed = parsed_question(exposure="butyrate")
    out = normalize({"claims": [sb, gp], "parsed": parsed, "papers": [paper]}, rt)["claims"]
    assert out[0].subject_concept == "CHEBI:17968"                  # SB -> sodium butyrate -> butyrate (N1)
    assert out[1].subject_concept == "PR:000001629" and out[1].subject_label == "hydroxycarboxylic acid receptor 2"


def test_a_lost_subject_is_restated_to_its_normal_role():
    """Pilot7: 'Mice lacking GPR109A showed fewer CD103+ DCs' was stored as 'GPR109A decreases CD103+ DCs'
    (about 8 of 19 loss-of-function claims in pilots 6-7 had the sign inverted)."""
    from bmira.evidence import verify_methods
    from bmira.graph import normalize
    rt, _ = offline_runtime()
    parsed = parsed_question(exposure="butyrate")
    span = "Mice lacking GPR43 or GPR109A, receptors for SCFAs, showed exacerbated food allergy and fewer CD103(+) DCs."
    c = make_claim("c1", "p", "")
    c.subject, c.object, c.relation, c.relation_raw, c.span, c.subject_lost = "GPR109A", "CD103+ DCs", "reduced", "reduced", span, True
    assert verify_methods(c, span).subject_lost                                  # 'lacking' is the wording that earns the flag
    out = normalize({"claims": [c], "parsed": parsed}, rt)["claims"]
    assert out[0].relation_norm == "increases"                                   # GPR109A promotes CD103+ DCs
    assert normalize({"claims": out, "parsed": parsed}, rt)["claims"][0].relation_norm == "increases"   # once

    both = make_claim("c2", "p", "")                       # 'Tet2 loss' AND the flag: still one flip
    both.subject, both.object, both.relation, both.relation_raw, both.subject_lost = "Tet2 loss", "IL-6", "increased", "increased", True
    assert normalize({"claims": [both], "parsed": parsed}, rt)["claims"][0].relation_norm == "decreases"

    plain = make_claim("c3", "p", "")                      # a flag with no loss wording in the quote is not believed
    plain.subject, plain.relation_raw, plain.span, plain.subject_lost = "GPR109A", "increased", "GPR109A increased colonic Tregs in mice.", True
    assert not verify_methods(plain, plain.span).subject_lost and "subject_lost not evidenced" in plain.method_checks


def test_out_of_scope_question_stops_before_any_search():
    rt, sc = offline_runtime()
    sc["parsed"] = {**sc["parsed"], "in_scope": False, "scope_note": "It asks for a personal treatment decision."}
    from bmira.telemetry import execute, summarize
    final, info = execute("Should I take metformin tonight?", rt, echo=False)
    assert "not investigated" in final["report"] and "personal treatment decision" in final["report"]
    assert rt.llm.calls["plan"] == 0 and not final.get("papers") and info["status"] == "completed"
    assert summarize(final, rt, info)["report"] == final["report"]          # the session file still gets a row


def test_exposure_members_share_the_exposure_node():
    """A question about 'SGLT2 inhibitors' meets papers about 'empagliflozin'."""
    from types import SimpleNamespace
    from bmira.graph import parse
    llm = StubLLM(parsed_question(population_model="patients", exposure="SGLT2 inhibitors", comparator="placebo",
                                  outcome="heart failure hospitalization",
                                  exposure_members=["empagliflozin", "dapagliflozin"]))
    rt = SimpleNamespace(llm=llm, resolver=EntityResolver(Settings(ontology_provider="off")))
    out = parse({"question": "q"}, rt)
    assert rt.resolver.resolve("empagliflozin").id == out["exposure"] == rt.resolver.resolve("dapagliflozin").id

