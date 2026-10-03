"""One test per plan exit gate. Offline: no keys, no network."""
import pytest

from bmira import portfolio as pf
from bmira.config import Settings
from bmira.evidence import sentence_tier, verify_text
from bmira.graph import build_agent, run
from bmira.normalize import EntityResolver, span_is_anchored
from bmira.offline import HashingEmbedder, offline_runtime
from bmira.schemas import Claim, Hypothesis
from bmira.semantic import candidate_pairs


@pytest.fixture(scope="module")
def offline():
    rt, sc = offline_runtime()
    return rt, run(sc["question"], rt)


def _claim(i, pmid, rel, grade="moderate", subj="LOCAL:a", obj="LOCAL:b", ctx="cd8",
           system="animal_cells", study_type="animal", **kw):
    return Claim(id=i, pmid=pmid, claim_type="observation", subject="a", relation=rel, object="b",
                 span="x" * 30, study_type=study_type, text_access="abstract_only", system=system,
                 subject_concept=subj, object_concept=obj, relation_norm=rel, grade=grade,
                 context_cell_type=ctx, comparator_present=True, **kw)


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
def test_alias_registry():
    r = EntityResolver(Settings(ontology_provider="off"))
    a, b = r.resolve("Lactic acid"), r.resolve("lactate")
    r.merge([a.id, b.id])
    assert r.resolve("lactic acid").id == r.resolve("Lactate").id


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


def test_logic_checks():
    links = {}
    keys = [pf.link_key("A", "increases", "B"), pf.link_key("B", "increases", "C")]
    assert "sign_mismatch" in pf.logic_check(keys, links, {}, "down")[1]
    assert "disconnected" in pf.logic_check([keys[0], pf.link_key("X", "increases", "C")], links, {}, "up")[1]
    assert "cycle" in pf.logic_check([keys[0], pf.link_key("B", "increases", "A")], links, {}, "up")[1]


def test_allocation_explores():
    s = Settings(targets_per_round=2, exploration_slots=1)
    links = {k: pf.LinkEvidence(key=k, subject=k[0], relation="increases", object=k[-1],
                                completeness=c)
             for k, c in [("A|increases|B", 0.3), ("B|increases|C", 0.3), ("A|increases|Z", 0.0)]}
    hyps = [Hypothesis(id="H1", name="lead", origin="llm_seed", links=["A|increases|B", "B|increases|C"], score=0.3),
            Hypothesis(id="H2", name="alt", origin="llm_seed", links=["A|increases|Z"], score=0.0)]
    assert "A|increases|Z" in pf.allocate(hyps, links, s, 1)


def test_allocation_prefers_the_step_nearest_the_exposure():
    """Pilot4: equal-priority steps were ordered by key string, so CHEBI-keyed links beat the
    HDAC route's NCIT and LOCAL steps. The later step here sorts first alphabetically."""
    s = Settings(targets_per_round=1, exploration_slots=0)
    first, later = "Z|increases|Y", "Y|increases|A"
    links = {k: pf.LinkEvidence(key=k, subject=k[0], relation="increases", object=k[-1]) for k in (first, later)}
    hyps = [Hypothesis(id="H1", name="chain", origin="llm_seed", links=[first, later])]
    assert pf.allocate(hyps, links, s, 1) == [first]
    # progress along a route, not its weakest step, sets its weight: a started route beats an unstarted one
    started = pf.LinkEvidence(key="S|increases|T", subject="S", relation="increases", object="T", completeness=0.4)
    idle = pf.LinkEvidence(key="I|increases|J", subject="I", relation="increases", object="J")
    two = {**links, started.key: started, idle.key: idle}
    hyps = [Hypothesis(id="H2", name="started", origin="llm_seed", links=[started.key, later]),
            Hypothesis(id="H3", name="idle", origin="llm_seed", links=[idle.key, first])]
    assert pf.allocate(hyps, two, s, 1) == [started.key]     # by min-step score both routes weigh 0 and idle wins


# P4: more evidence never lowers a link; counted contradiction blocks support.
def test_link_aggregation():
    s, labels = Settings(), {}
    strong = [_claim("c1", "p1", "increases", "strong"), _claim("c2", "p2", "increases", "strong")]
    k = pf.link_key("LOCAL:a", "increases", "LOCAL:b")
    g1 = pf.build_links(strong, {}, {}, labels, s)[k].grade
    g2 = pf.build_links(strong + [_claim("c3", "p3", "increases", "weak")], {}, {}, labels, s)[k].grade
    assert g1 == g2 == "strong"
    contra = strong[:1] + [_claim("c4", "p4", "decreases", "strong"), _claim("c5", "p5", "no_effect", "strong")]
    assert pf.build_links(contra, {}, {}, labels, s)[k].status == "contradicted"
    # R6: triaged 'context-dependent' sets opposing claims aside only if contexts truly differ...
    tri = {frozenset(("c1", "c4")), frozenset(("c1", "c5"))}
    assert pf.build_links(contra, {}, {}, labels, s, discounted=tri)[k].status == "contradicted"
    other = strong[:1] + [_claim("c4", "p4", "decreases", "strong", ctx="tumor"),
                          _claim("c5", "p5", "no_effect", "strong", ctx="tumor")]
    ln = pf.build_links(other, {}, {}, labels, s, discounted=tri)[k]
    assert set(ln.uncounted) == {"c4", "c5"} and ln.status != "contradicted"
    # ...and never when the opposing evidence is closer to humans than the support
    human = strong[:1] + [_claim("c4", "p4", "decreases", "strong", ctx="tumor", system="human_primary_cells"),
                          _claim("c5", "p5", "no_effect", "strong", ctx="tumor", system="human_primary_cells")]
    assert pf.build_links(human, {}, {}, labels, s, discounted=tri)[k].status == "contradicted"


def test_evidence_floor_and_null_asymmetry():
    s, k = Settings(), pf.link_key("LOCAL:a", "increases", "LOCAL:b")
    weak = [_claim("c1", "p1", "increases", "weak"), _claim("c2", "p2", "increases", "weak")]
    ln = pf.build_links(weak, {}, {}, {}, s)[k]
    assert ln.status == "insufficient" and "only weak" in ln.reason            # R8
    strong = [_claim("c1", "p1", "increases", "strong"), _claim("c2", "p2", "increases", "strong")]
    nulls = [_claim("c3", "p3", "no_effect", "moderate"), _claim("c4", "p4", "no_effect", "moderate")]
    ln = pf.build_links(strong + nulls, {}, {}, {}, s)[k]
    assert ln.status == "supported" and set(ln.uncounted) == {"c3", "c4"}     # R5
    review = [_claim("c5", "p5", "increases", "weak", study_type="review")]
    ln = pf.build_links(strong[:1] + review, {}, {}, {}, s)[k]
    assert ln.n_studies == 1 and "c5" in ln.uncounted                          # R1


# P5: negation-aware verbs; tags required.
def test_verification():
    assert sentence_tier("Lactylation does not induce IFNG [C1].") == 1
    c = _claim("C1", "p1", "associated_with", "moderate")
    v = verify_text("Lactate drives IFNG loss in T cells [C1].", [c], ["H1"])
    assert v["overclaims"] and v["missing_tags"] == ["H1"]


def test_claim_checks():
    from bmira.normalize import check_claim
    src = "Lactate increased GPR81 expression in tumor cells compared with controls."
    c = _claim("c", "p", "")
    c.span, c.subject, c.object = src, "lactate", "GPR81 expression"
    c.relation = "did not change"
    assert check_claim(c, src)[0] == "null claim but the quote reports an effect"
    c.relation, c.object = "increased", "IFNG"
    assert check_claim(c, src)[0] == "object not named in quote"
    neg = "Lactate did not increase GPR81 expression in tumor cells compared with controls."
    c.span, c.object = neg, "GPR81"
    assert check_claim(c, neg)[0] == "quote negates the claimed effect"
    fabricated = "Lactate had no effect on GPR81 expression in tumor cells compared with controls."
    assert not span_is_anchored(fabricated, fabricated.replace("no effect", "an effect"))


def test_method_and_design_rules():
    from bmira.evidence import claim_study_type, grade_claim, verify_methods
    from bmira.sources import study_type_from_pubtypes
    c = _claim("c", "p", "increases", perturbation_class="knockout", rescue_arm=True)
    c.span = "Lactate increased IFNG in mouse T cells."               # no knockout, rescue or control
    c.comparator_present = True
    verify_methods(c, c.span)
    assert c.perturbation_class == "none" and not c.rescue_arm and not c.comparator_present
    assert claim_study_type("animal", None, "human_primary_cells") == "human_primary"   # mixed paper
    assert claim_study_type("human_primary", None, "animal_in_vivo") == "animal"
    assert study_type_from_pubtypes(["Systematic Review"]) == "review"
    # design forced to 3 so only the indirectness rule can cap this mouse in vivo claim
    strong = _claim("s", "p", "increases", system="animal_in_vivo", study_type="human_primary",
                    perturbation_class="knockout", rescue_arm=True)
    strong.text_access = "full_text"
    assert grade_claim(strong, "human").grade == "moderate" and "indirect_system" in strong.grade_detail["caps"]


def test_attribute_split_and_sections():
    import xml.etree.ElementTree as ET
    from bmira.normalize import split_attribute
    from bmira.sources import sectioned_text
    assert split_attribute("IFNG expression") == ("IFNG", "expression")
    assert split_attribute("NAD+ levels") == ("NAD+", "amount")
    assert split_attribute("T cell activation") == ("T cell activation", "none")
    body = ET.fromstring("<body><sec><title>Methods</title><p>M</p></sec><sec><title>Results</title>"
                         "<p>R</p><fig><caption>Fig 1 legend</caption></fig></sec></body>")
    text = sectioned_text("abs", body, 1000)
    assert text.index("RESULTS: R") < text.index("FIGURE LEGENDS") < text.index("METHODS: M")


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


def test_top_k_cap():
    claims = [_claim(f"c{i}", f"p{i}", "increases") for i in range(20)]
    for c in claims:
        c.subject_label, c.object_label = "lactate", "nad"
    pairs = candidate_pairs(claims, HashingEmbedder(), 0.0, k=3)
    count = {}
    for a, b, _ in pairs:
        count[a] = count.get(a, 0) + 1
        count[b] = count.get(b, 0) + 1
    assert max(count.values()) <= 3


def test_anchor_rejects_stitched_span():
    src = "Lactate increased histone lactylation in T cells. Many unrelated words follow here. IFNG fell."
    assert span_is_anchored("Lactate increased histone lactylation in T cells.", src)
    assert not span_is_anchored("Lactate increased histone lactylation and IFNG fell sharply.", src)


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


def test_temperature_is_not_sent_by_default(monkeypatch):
    """Some reasoning models reject any temperature but their default."""
    import sys
    import types
    from bmira.llm import LangChainLLM
    seen = []
    fake = types.ModuleType("langchain_openai")
    fake.ChatOpenAI = lambda **kw: seen.append(kw) or object()
    monkeypatch.setitem(sys.modules, "langchain_openai", fake)
    LangChainLLM(Settings(), api_key="k")._model("reasoning")
    assert "temperature" not in seen[-1]
    LangChainLLM(Settings(temperature=0.0), api_key="k")._model("cheap")
    assert seen[-1]["temperature"] == 0.0



# H1-H3: failure-safe telemetry, fatal errors, preflight, budget, effort and cost.
def _failing_runtime(monkeypatch, task, exc, on_call=1):
    """Offline runtime whose surrogate raises `exc` on the n-th call of `task`."""
    import bmira.offline as off
    real, seen = off.offline_runtime, {"n": 0}          # shared: the failure happens once per test

    def patched(**kw):
        rt, sc = real(**kw)
        original = rt.llm.structured

        def structured(t, *args, **kwargs):
            if t == task:
                seen["n"] += 1
                if seen["n"] == on_call:
                    raise exc
            return original(t, *args, **kwargs)
        rt.llm.structured = structured
        return rt, sc
    monkeypatch.setattr(off, "offline_runtime", patched)


def test_fatal_error_aborts_session_but_keeps_partial_run(tmp_path, monkeypatch):
    import json
    from bmira.experiments import main
    from bmira.llm import FatalLLMError
    _failing_runtime(monkeypatch, "plan", FatalLLMError("plan: insufficient_quota"), on_call=2)
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
    _failing_runtime(monkeypatch, "seed", RuntimeError("boom"), on_call=1)
    qs = tmp_path / "q.txt"
    qs.write_text("one\ntwo\n", encoding="utf-8")
    session = json.loads(main(["--offline", "--quiet", "--questions", str(qs), "--out",
                               str(tmp_path / "s.json")]).read_text(encoding="utf-8"))
    assert len(session["runs"]) == 2 and "aborted" not in session["session"]   # seeding degrades, run completes



def test_uncaught_nonfatal_error_fails_run_and_session_continues(tmp_path, monkeypatch):
    import json
    from bmira.experiments import main
    _failing_runtime(monkeypatch, "plan", RuntimeError("upstream hiccup"), on_call=2)
    qs = tmp_path / "q.txt"
    qs.write_text("one\ntwo\n", encoding="utf-8")
    session = json.loads(main(["--offline", "--quiet", "--questions", str(qs), "--out",
                               str(tmp_path / "s.json")]).read_text(encoding="utf-8"))
    first, second = session["runs"]
    assert first["status"] == "failed" and first["failed_node"] == "plan" and first["papers"]["retrieved"] > 0
    assert second["status"] == "completed"


def test_error_classification():
    from bmira.llm import is_fatal

    class E(Exception):
        status_code = None
    assert is_fatal(E("Error code: 429 - insufficient_quota credit_balance_exhausted"))
    assert is_fatal(E("Unsupported value: 'temperature' ... unsupported_value"))
    assert is_fatal(E("model_not_found"))
    assert not is_fatal(E("Error code: 429 - slow_down"))
    assert not is_fatal(E("Error code: 503 - server_is_overloaded"))


def test_preflight_blocks_spending(tmp_path, monkeypatch):
    import json
    from bmira.experiments import main
    from bmira.llm import FatalLLMError
    qs = tmp_path / "q.txt"
    qs.write_text("one\n", encoding="utf-8")
    ok = json.loads(main(["--offline", "--quiet", "--questions", str(qs), "--out",
                          str(tmp_path / "a.json")]).read_text(encoding="utf-8"))
    assert all(c["ok"] for c in ok["session"]["preflight"]) and len(ok["runs"]) == 1
    _failing_runtime(monkeypatch, "preflight", FatalLLMError("preflight: invalid_api_key"))
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


def test_effort_retries_and_cost():
    import sys
    import types
    from bmira.llm import LangChainLLM
    from bmira.telemetry import estimate_cost
    seen = []
    fake = types.ModuleType("langchain_openai")
    fake.ChatOpenAI = lambda **kw: seen.append(kw) or object()
    sys.modules["langchain_openai"] = fake
    try:
        llm = LangChainLLM(Settings(), api_key="k")
        llm._for("screen", "cheap")
        assert seen[-1]["reasoning_effort"] == "low" and seen[-1]["max_retries"] == 6
        llm._for("extract", "reasoning")
        assert seen[-1]["reasoning_effort"] == "medium" and llm.model_of["extract"] == Settings().models["openai"]["reasoning"]
    finally:
        del sys.modules["langchain_openai"]
    assert estimate_cost("gpt-6-sol", 1_000_000, 100_000, Settings().prices) == 3.0
    assert estimate_cost("unknown-model", 10, 10, Settings().prices) is None


# J1-J5: entity granularity, species-aware ontology, mention check, retrieval balance, throughput.
def test_entity_variants_collapse_to_one_node():
    from bmira.normalize import entity_of, entity_parts, lookup_key, singular
    variants = ["Induction of colonic regulatory T cells", "regulatory T-cell frequency",
                "Bone marrow, splenic and Peyer’s patch regulatory T cells",
                "Regulatory T cells in pancreatic lymph nodes", "butyrate-induced regulatory T cells"]
    assert {lookup_key(singular(entity_of(v)[0])) for v in variants} == {"regulatory t cell"}
    assert entity_of("Induction of colonic regulatory T cells")[1:] == ("differentiation", "colon")
    assert entity_of("T cell activation") == ("T cell activation", "none", "")      # phenotype kept
    assert entity_parts("NFAT1 and SMAD3") == (["NFAT1", "SMAD3"], "")
    assert entity_parts("signal transducer and activator of transcription 3")[0] == [
        "signal transducer and activator of transcription 3"]


def test_mention_check_accepts_abbreviations_and_previous_sentence():
    from bmira.normalize import abbreviations, check_claim
    src = ("Mice received sodium butyrate (NaB) in drinking water. NaB increased Foxp3 expression in "
           "colonic T cells compared with controls. This metabolite also increased IL-10 in the same cells.")
    c = _claim("c", "p", "increases")
    c.subject, c.object, c.relation = "sodium butyrate", "Foxp3", "increased"
    c.span = "NaB increased Foxp3 expression in colonic T cells compared with controls."
    assert check_claim(c, src, abbreviations(src)) == ("", [])
    c.object, c.span = "IL-10", "This metabolite also increased IL-10 in the same cells."
    assert check_claim(c, src, abbreviations(src))[0] == ""             # named one sentence earlier
    c.subject = "not specified"
    assert check_claim(c, src)[0] == "subject not specified"


def test_ontology_prefers_species_agnostic_and_rejects_other_species(monkeypatch):
    import bmira.normalize as nz

    def docs(*items):
        class R:
            def json(self):
                return {"response": {"docs": [{"obo_id": i, "label": lab, "synonym": ["IL-10"],
                                               "ontology_name": "pr", "iri": ""} for i, lab in items]}}
        return lambda *a, **k: R()
    r = nz.EntityResolver(Settings(ontology_provider="ols"))
    monkeypatch.setattr(nz.requests, "get", docs(("PR:1", "interleukin-10 (chicken)")))
    assert r._ols("IL-10") is None
    monkeypatch.setattr(nz.requests, "get", docs(("PR:1", "interleukin-10 (chicken)"),
                                                 ("PR:2", "interleukin-10 (mouse)"), ("PR:3", "interleukin-10")))
    assert r._ols("IL-10").id == "PR:3"
    monkeypatch.setattr(nz.requests, "get", docs(("PR:2", "interleukin-10 (mouse)"), ("PR:4", "interleukin-10 (human)")))
    assert r._ols("IL-10").id == "PR:4"


def test_batched_resolution_and_disk_cache(tmp_path):
    from bmira.normalize import EntityResolver
    from bmira.offline import SurrogateLLM, load_scenario
    s = Settings(ontology_provider="llm", cache_dir=str(tmp_path))
    llm = SurrogateLLM(load_scenario())
    r = EntityResolver(s, llm)
    r.resolve_many(["lactate", "NAD+", "glycolytic flux", "lactate"])
    assert llm.calls["entities"] == 1 and llm.items["entities"] == 3     # one call for three names
    r.save()
    llm2 = SurrogateLLM(load_scenario())
    r2 = EntityResolver(s, llm2)
    r2.resolve_many(["lactate", "NAD+"])
    assert llm2.calls["entities"] == 0 and r2.disk_hits == 2              # reused across runs


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


def test_classification_tasks_use_cheap_model():
    import sys
    import types
    from bmira.llm import LangChainLLM
    fake = types.ModuleType("langchain_openai")
    fake.ChatOpenAI = lambda **kw: kw
    sys.modules["langchain_openai"] = fake
    try:
        llm, m = LangChainLLM(Settings(), api_key="k"), Settings().models["openai"]
        assert llm._for("pair", "reasoning")["model"] == m["cheap"]
        assert llm._for("extract", "reasoning")["model"] == m["reasoning"]
    finally:
        del sys.modules["langchain_openai"]


# K1-K10: fixes from the pilot3 live run (butyrate -> Treg).
class _FakeLLM:
    """Answers 'entity'/'entities' with one fixed label and 'alias' with 'same'."""
    def __init__(self, label="regulatory T cell"):
        self.label = label

    def structured(self, task, schema, system, user, role="reasoning", ctx=None, n_items=1):
        from bmira.schemas import AliasBatch, AliasVerdict, EntityBatch, EntityItem, EntityResolution
        if task == "entities":
            return EntityBatch(items=[EntityItem(surface=s, normalized_label=self.label, category="cell_type",
                                                 confidence=0.9) for s in ctx["surfaces"]])
        if task == "alias":
            return AliasBatch(verdicts=[AliasVerdict(label_a=a, label_b=b, same_entity=True) for a, b in ctx["pairs"]])
        return EntityResolution(normalized_label=self.label, category="cell_type", confidence=0.9)


def test_llm_label_is_looked_up_before_going_local():
    import bmira.normalize as nz
    cl = nz.Concept("CL:1", "regulatory T cell", "cell_type", "ols", 0.9)
    r = nz.EntityResolver(Settings(ontology_provider="hybrid"), _FakeLLM())
    r._ols = lambda name: cl if nz.lookup_key(name) == "regulatory t cell" else None
    r.resolve_many(["Treg cell", "regulatory T cell"])                  # batched path
    assert r.resolve("Treg cell").id == "CL:1"
    assert r.resolve("Foxp3 Treg ratio").id == "CL:1"                   # single path


def test_ols_skips_allele_terms(monkeypatch):
    import bmira.normalize as nz

    class R:
        def json(self):
            return {"response": {"docs": [{"obo_id": "NCIT:C102493", "label": "HDAC9 wt Allele",
                                           "synonym": ["HDAC"], "ontology_name": "ncit", "iri": ""}]}}
    monkeypatch.setattr(nz.requests, "get", lambda *a, **k: R())
    assert nz.EntityResolver(Settings(ontology_provider="ols"))._ols("HDAC") is None


def test_clean_drops_abbreviation_parentheses():
    from bmira.normalize import clean, lookup_key
    assert clean("histone deacetylase (HDAC)") == "histone deacetylase"
    assert clean("Interleukin-2 receptor subunit alpha (CD25") == "Interleukin-2 receptor subunit alpha"
    assert clean("(CD25)") == "CD25" and clean("CD25") == "CD25"
    assert clean("interleukin-10 (mouse)") == "interleukin-10 (mouse)"      # not an abbreviation
    assert lookup_key("histone deacetylase (HDAC)") == lookup_key("histone deacetylase")


def test_pathway_nodes_join_alias_consolidation():
    import bmira.normalize as nz
    r = nz.EntityResolver(Settings(ontology_provider="off"))
    a, b = r.resolve("histone H3 lysine 9 acetylation"), r.resolve("acetylated histone H3 lysine 9")
    claims = [_claim("c1", "p", "increases", subj=a.id, obj="LOCAL:b")]       # b is only in a pathway
    assert nz.consolidate_aliases(claims, r, _FakeLLM(), {}, extra_ids=[b.id]) == 1
    assert r.canonical(a).id == r.canonical(b).id


def test_pair_verdicts_are_matched_by_position_not_echoed_ids():
    from bmira.schemas import PairAdjudication, PairBatch
    from bmira.semantic import adjudicate

    class LLM:
        def structured(self, task, schema, system, user, role="reasoning", ctx=None, n_items=1):
            assert "[PAIR 1]" in user and "[PAIR 2]" in user
            return PairBatch(pairs=[PairAdjudication(pair=i + 1, same_finding=True, same_context=True)
                                    for i in range(n_items)])
    claims = [_claim(i, "p", "increases") for i in ("C1_0", "C2_0", "C3_0")]
    cache = {}
    failed = adjudicate(claims, [("C1_0", "C2_0", .9), ("C1_0", "C3_0", .8)], LLM(), cache)
    assert failed == 0 and set(cache) == {frozenset(("C1_0", "C2_0")), frozenset(("C1_0", "C3_0"))}


def test_method_cues_cover_real_wording():
    from bmira.evidence import verify_methods

    def run(span, **kw):
        c = _claim("c", "p", "increases")
        c.span = span
        c.comparator_present = kw.pop("comparator", False)
        for k, v in kw.items():
            setattr(c, k, v)
        return verify_methods(c, span)
    assert run("Provision of butyrate to mice increased Foxp3+ Treg cells in the colon.",
               perturbation_class="pharmacological").perturbation_class == "pharmacological"
    assert run("Butyrate at 0.25 mM enhanced Foxp3 expression in CD4+ T cells.",
               perturbation_class="pharmacological").perturbation_class == "pharmacological"
    assert run("Butyrate in the absence of TGF-b1 did not lead to Foxp3+ Treg conversion.",
               comparator=True).comparator_present
    assert run("While butyrate inhibited HDAC, acetate lacked this activity.", comparator=True).comparator_present
    assert run("Tbx21−/− CD4+ T cells made less IFN-g after butyrate.",
               perturbation_class="knockout").perturbation_class == "knockout"
    assert run("Treg frequency was higher in healthy donors.",
               perturbation_class="pharmacological").perturbation_class == "none"   # still guarded


def test_but_not_phrase_does_not_negate_the_claim():
    from bmira.normalize import check_claim
    src = ("We show that butyrate but not pentanoate exerts a concentration-dependent effect on "
           "Treg and Th17 differentiation.")
    c = _claim("c", "p", "")
    c.subject, c.object, c.relation, c.span = "butyrate", "Treg", "exerts a concentration-dependent effect on", src
    assert check_claim(c, src)[0] == ""
    src2 = "Treg generation was potentiated by propionate, an HDAC inhibitor, but not acetate."
    c.subject, c.object, c.relation, c.span = "acetate", "HDAC", "lacks", src2
    assert check_claim(c, src2) == ("", [])                              # 'lacks' is a null relation


def test_entity_names_and_negated_scope_do_not_trigger_overclaim():
    weak = [_claim("C1", "p", "increases", "weak")]
    ok = ["Butyrate is associated with more induced regulatory T cells [C1].",
          "These weak findings do not establish that **gut-produced** butyrate induces colonic Treg [NO_EVIDENCE].",
          "Experiments would need to test effects on induced [L15] and other regulatory T cells [L16][NO_EVIDENCE]."]
    for text in ok:
        assert verify_text(text, weak, [])["overclaims"] == [], text
    for text in ["Butyrate induced regulatory T cells in mice [C1].", "Butyrate induces IFNG in T cells [C1]."]:
        assert verify_text(text, weak, [])["overclaims"], text             # real overclaims still caught


def test_findings_on_subtypes_support_the_parent_link():
    parent = pf.link_key("LOCAL:a", "increases", "CL:parent")
    claims = [_claim("A", "p1", "increases", obj="CL:sub1"), _claim("B", "p2", "increases", obj="CL:sub2"),
              _claim("N", "p3", "no_effect", obj="CL:sub1")]
    anc = {"CL:sub1": ("CL:parent",), "CL:sub2": ("CL:parent",)}
    ln = pf.build_links(claims, {}, {}, {}, Settings(), extra={parent}, ancestors=anc)[parent]
    assert ln.status == "supported" and ln.n_studies == 2
    assert ln.n_contra_studies == 0                                        # a subtype null never refutes the parent
    assert pf.build_links(claims, {}, {}, {}, Settings(), extra={parent})[parent].status != "supported"


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


# L1-L5: problems the other seven questions would hit (found by probing the code, not yet seen live).
def test_greek_letters_and_charges_keep_entities_apart():
    from bmira.normalize import local_concept, lookup_key
    assert lookup_key("IFN-γ") != lookup_key("IFN-α")             # both used to be 'ifn'
    assert lookup_key("IL-1β") != lookup_key("IL-1α")
    assert lookup_key("TGF-β1") == lookup_key("TGF-beta1")
    assert lookup_key("NAD+") == lookup_key("NAD(+)") == lookup_key("NAD⁺")
    assert local_concept("IFN-γ").id != local_concept("IFN-α").id


def test_genotype_notation_is_not_split_into_two_entities():
    from bmira.normalize import entity_parts
    assert entity_parts("Tet2−/− bone marrow")[0] == ["Tet2−/− bone marrow"]
    assert entity_parts("Foxp3-/- mice")[0] == ["Foxp3-/- mice"]
    assert entity_parts("GPR81/HCAR1")[0] == ["GPR81", "HCAR1"]               # real alternatives still split


def test_trial_wording_counts_as_intervention_and_control():
    from bmira.evidence import verify_methods

    def run(span):
        c = _claim("c", "p", "decreases", perturbation_class="pharmacological", study_type="human_rct",
                   system="human_in_vivo")
        c.span, c.comparator_present = span, True
        return verify_methods(c, span)
    for span in ["Patients were randomly assigned to empagliflozin 10 mg or placebo; it reduced hospitalization.",
                 "Vitamin D3 2000 IU daily reduced autoimmune disease incidence compared with placebo.",
                 "In a pooled analysis of 5 randomized trials, SGLT2 inhibitors lowered hospitalization.",
                 "Dapagliflozin reduced worsening heart failure versus usual care."]:
        c = run(span)
        assert c.perturbation_class == "pharmacological" and c.comparator_present, (span, c.method_checks)


def test_null_findings_contradict_required_and_modulating_steps():
    null = _claim("n", "p", "no_effect")
    assert pf._contradicts("required_for", null) and pf._contradicts("sufficient_for", null)
    assert pf._contradicts("modulates", null)
    assert not pf._contradicts("required_for", _claim("i", "p", "increases"))
    assert not pf._contradicts("binds", null)
    key = pf.link_key("LOCAL:a", "required_for", "LOCAL:b")
    nulls = [_claim("n1", "p1", "no_effect"), _claim("n2", "p2", "no_effect")]     # moderate, with comparator
    assert pf.build_links(nulls, {}, {}, {}, Settings(), extra={key})[key].status == "contradicted"


def test_pair_verdict_has_no_unused_rationale():
    from bmira.schemas import PairAdjudication
    assert "rationale" not in PairAdjudication.model_fields                  # ~100 output tokens per pair


# M1-M5: exposure direction ("NAD+ decline", "TET2 loss") and randomized-trial grading.
def test_loss_words_are_stripped_from_entities_but_not_from_phenotypes():
    from bmira.normalize import entity_change, entity_of
    assert entity_of("age-related NAD+ decline")[0] == "NAD+" and entity_change("age-related NAD+ decline") == "down"
    assert entity_of("Tet2 loss")[0] == "Tet2" and entity_change("Tet2 loss") == "down"
    assert entity_of("vitamin D deficiency")[0] == "vitamin D"
    assert entity_of("bone loss")[0] == "bone loss" and entity_change("bone loss") == ""        # a phenotype
    assert entity_change("Tet2-deficient macrophages") == "" and entity_change("Tet2") == ""     # a cell descriptor


def test_parse_strips_exposure_direction_and_keeps_it():
    from types import SimpleNamespace
    from bmira.graph import parse
    from bmira.schemas import ParsedQuestion

    class LLM:
        def structured(self, task, schema, system, user, role="reasoning", ctx=None, n_items=1):
            return ParsedQuestion(population_model="macrophages", exposure="age-related NAD+ decline",
                                  comparator="young", outcome="inflammaging", mechanism_hypothesis="h",
                                  expected_direction="up")
    rt = SimpleNamespace(llm=LLM(), resolver=EntityResolver(Settings(ontology_provider="off")))
    out = parse({"question": "q"}, rt)
    assert out["parsed"].exposure == "NAD+" and out["parsed"].exposure_change == "down"
    assert out["exposure"] == rt.resolver.resolve("NAD+").id


def test_expected_sign_follows_the_exposure_change():
    assert pf.pathway_sign("up", "down") == "down" and pf.pathway_sign("down", "down") == "up"
    assert pf.pathway_sign("up", "up") == "up" and pf.pathway_sign("none", "down") == "none"
    keys = [pf.link_key("N", "decreases", "I")]                       # NAD+ -| inflammation
    assert "sign_mismatch" not in pf.logic_check(keys, {}, {}, pf.pathway_sign("up", "down"))[1]
    assert "sign_mismatch" in pf.logic_check(keys, {}, {}, "up")[1]


def test_loss_of_function_claims_are_restated_for_the_bare_entity():
    from bmira.graph import normalize
    from bmira.schemas import ParsedQuestion
    rt, _ = offline_runtime()
    c = _claim("c1", "p", "")
    c.subject, c.object, c.relation, c.relation_raw = "Tet2 loss", "IL-6", "increased", "increased"
    parsed = ParsedQuestion(population_model="m", exposure="Tet2", comparator="c", outcome="o",
                            mechanism_hypothesis="h")
    out = normalize({"claims": [c], "parsed": parsed}, rt)["claims"]
    assert out[0].subject_label == "Tet2" and out[0].relation_norm == "decreases"      # loss increases = Tet2 decreases
    again = normalize({"claims": out, "parsed": parsed}, rt)["claims"]
    assert again[0].relation_norm == "decreases"                                       # flipped once, not every round


def test_randomized_evidence_can_grade_strong():
    from bmira.evidence import grade_claim

    def rct(**kw):
        c = _claim("r", "p", "decreases", study_type="human_rct", system="human_in_vivo",
                   perturbation_class="pharmacological")
        c.text_access = "full_text"
        for k, v in kw.items():
            setattr(c, k, v)
        return grade_claim(c, "human")
    assert rct().grade == "strong"                                       # randomization stands in for a rescue arm
    assert rct(comparator_present=False).grade != "strong"
    assert rct(text_access="abstract_only").grade == "moderate"          # the abstract cap still applies
    assert rct(study_type="meta_analysis").grade == "strong"
    animal = _claim("a", "p", "decreases", perturbation_class="pharmacological")
    animal.text_access = "full_text"
    assert grade_claim(animal, "any").grade == "moderate"                # animal pharmacology unchanged


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


# N1-N11: fixes from the pilot4 live run (butyrate -> Treg).
def _onto(table):
    """A stand-in for OLS: lookup key -> (id, label)."""
    import bmira.normalize as nz
    return lambda n: nz.Concept(*table[nz.lookup_key(n)], "chemical", "ols", 0.9) if nz.lookup_key(n) in table else None


def test_salt_acid_and_given_forms_name_the_parent_chemical():
    import bmira.normalize as nz
    assert nz.entity_of("Sodium butyrate treatment")[0] == "Sodium butyrate"
    assert nz.entity_of("Butyrate supplementation")[0] == "Butyrate"
    assert nz.entity_of("provision of butyrate")[0] == "butyrate"
    assert nz.parent_chemical("sodium butyrate") == "butyrate" and nz.parent_chemical("butyric acid") == "butyrate"
    assert nz.parent_chemical("nucleic acid") == "nucleic acid"
    assert nz.parent_chemical("magnesium sulfate") == "magnesium sulfate"     # the metal is the agent
    r = nz.EntityResolver(Settings(ontology_provider="ols"))
    r._ols = _onto({"nab": ("CHEBI:64103", "sodium butyrate"), "butyrate": ("CHEBI:17968", "butyrate")})
    assert {r.resolve(x).id for x in ("NaB", "sodium butyrate", "butyric acid", "butyrate")} == {"CHEBI:17968"}


def test_ols_rejects_measurements_strains_and_bad_ids_and_prefers_labels(monkeypatch):
    """Real OLS answers from pilot4's names."""
    import bmira.normalize as nz

    def ols(*docs):
        class R:
            def json(self):
                return {"response": {"docs": [{"obo_id": i, "label": lab, "synonym": syn, "ontology_name": o,
                                               "iri": ""} for i, lab, syn, o in docs]}}
        monkeypatch.setattr(nz.requests, "get", lambda *a, **k: R())
    r = nz.EntityResolver(Settings(ontology_provider="ols"))
    ols(("CHEBI:17154", "nicotinamide", ["niacin"], "chebi"), ("CHEBI:15940", "nicotinic acid", ["Niacin"], "chebi"),
        ("NCIT:C689", "Niacin", [], "ncit"))
    assert r._ols("niacin").id == "NCIT:C689"                     # was nicotinamide
    ols(("PR:O13754", "Hsp70/Hsp90 co-chaperone cns1 (Schizosaccharomyces pombe 972h-)", ["CNS1"], "pr"))
    assert r._ols("CNS1") is None                                  # the Foxp3 enhancer is not a yeast protein
    ols(("NCIT:C166072", "Forkhead Box Protein P3 Measurement", ["forkhead box p3"], "ncit"))
    assert r._ols("forkhead box p3") is None
    ols(("NCIT:C74814", "Interleukin 18 Measurement", ["interleukin 18"], "ncit"),
        ("NCIT:C20520", "Interleukin-18", [], "ncit"))
    assert r._ols("interleukin 18").id == "NCIT:C20520"
    assert r._ols("interleukin 18 measurement").id == "NCIT:C74814"   # asked for by name
    ols(("1318", "C3", [], "mondo"))
    assert r._ols("C3") is None


def test_modifications_and_inhibition_name_the_bare_entity_in_claims_and_pathways():
    from bmira.graph import _set_concepts
    from bmira.normalize import entity_change, entity_of, with_mark
    from bmira.schemas import ProposedLink, ProposedPathway
    assert entity_of("HDAC inhibition")[0] == "HDAC" and entity_change("HDAC inhibition") == "down"
    assert entity_of("inhibition of HDAC")[0] == "HDAC"
    assert entity_of("histone deacetylase")[0] == "histone deacetylase"
    span = "Butyrate enhanced histone H3 acetylation in the promoter of Foxp3."
    assert with_mark("Histone H3", "modification", span) == "Histone H3 acetylation"
    assert with_mark("Histone H3", "amount", span) == "Histone H3"
    rt, _ = offline_runtime()
    c = _claim("c", "p", "increases", object_attribute="modification")
    c.subject, c.object, c.span = "butyrate", "Histone H3", span
    _set_concepts(c, rt)
    r = rt.resolver
    assert c.object_concept == r.resolve("Histone H3 acetylation").id != r.resolve("histone lactylation").id
    pw = ProposedPathway(name="p", links=[ProposedLink(source="HDAC inhibition", relation="increases",
                                                       target="Histone H3 acetylation")])
    keys, _ = pf.proposal_keys(pw, r)                                  # the pathway names the same node
    assert keys == [pf.link_key(r.resolve("HDAC").id, "decreases", c.object_concept)]


def test_local_cell_subtypes_inherit_the_cell_type_they_name():
    import bmira.normalize as nz
    treg = nz.Concept("CL:0000815", "regulatory T cell", "cell_type", "ols", 0.9, ("CL:0000084",))
    dc = nz.Concept("CL:0000451", "dendritic cell", "cell_type", "ols", 0.9)
    r = nz.EntityResolver(Settings(ontology_provider="hybrid"))
    r._ols = lambda n: {"regulatory t cell": treg, "dendritic cell": dc}.get(nz.lookup_key(n))
    sub = r._labelled("FOXP3-positive regulatory T cell", "cell_type", 0.9)
    assert sub.id.startswith("LOCAL:") and sub.parents == ("CL:0000815",) and "CL:0000084" in sub.ancestors
    assert r._labelled("Slc5a8-null dendritic cell", "cell_type", 0.9).parents == ("CL:0000451",)
    assert r._labelled("regulatory T cell balance", "phenotype", 0.9).ancestors == ()   # not a cell type
    link = pf.link_key("CHEBI:17968", "increases", "CL:0000815")
    claims = [_claim("A", "p1", "increases", subj="CHEBI:17968", obj=sub.id),
              _claim("B", "p2", "increases", subj="CHEBI:17968", obj="CL:0000815")]
    ln = pf.build_links(claims, {}, {}, {}, Settings(), extra={link}, ancestors={sub.id: sub.ancestors})[link]
    assert ln.n_studies == 2                                              # the subtype finding counts (R9)


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


def test_associative_wording_and_emphasis_do_not_trigger_overclaim():
    """The four sentences pilot4's verifier flagged, from the report itself."""
    flagged = [
        "Evidence for specifically *induced* Tregs is weaker, and none of the proposed molecular routes "
        "is established end to end [C31521614_0][NO_EVIDENCE].",
        "[L2] Butyrate is associated with increased induced Tregs in animal studies, but this link has "
        "only weak evidence [C31521614_0][C32010146_0][C34035164_8].",
        "[L3] Butyrate is associated with reduced histone deacetylase in human Tregs [C35148177_4].",
        "[L10] HCAR2 is associated with anti-inflammatory properties in macrophages, while [L12] macrophages "
        "are associated with increased Tregs; both links have weak evidence [C24412617_1][C24412617_2]."]
    assert [sentence_tier(x) <= 1 for x in flagged] == [True] * 4          # at most associative: weak evidence allows it
    # the lead-in excuses only the change word right behind it, never a second clause or verb
    assert sentence_tier("Butyrate is associated with Tregs and induces Treg differentiation.") == 4
    assert sentence_tier("Butyrate is associated with Tregs, which promotes colitis recovery.") == 3
    assert sentence_tier("Butyrate induced regulatory T cells in mice.") == 4          # a verb here, not a name
    assert sentence_tier("Butyrate is associated with more induced regulatory T cells.") == 1


def test_direct_routes_take_no_pathway_slot():
    """Pilot4: H5 and its subtype H6 took two of six slots, so expansion never had room."""
    s = Settings(max_hypotheses=2)
    k = lambda a, b: pf.link_key(a, "increases", b)
    hyps = [Hypothesis(id=i, name=i, origin=o, links=[k(a, b)]) for i, o, a, b in (
        ("H1", "ledger_path", "A", "B"), ("H2", "ledger_path", "A", "C"),
        ("H3", "llm_seed", "A", "M1"), ("H4", "llm_seed", "A", "M2"), ("H5", "llm_seed", "A", "M3"))]
    claims = [_claim("c1", "p1", "increases", subj="A", obj="B"), _claim("c2", "p2", "increases", subj="A", obj="B"),
              _claim("c3", "p3", "increases", grade="weak", subj="A", obj="C")]
    links = pf.build_links(claims, {}, {}, {}, s, extra={h.links[0] for h in hyps})
    kept = [h.id for h in pf.evaluate(hyps, links, {}, "none", s)]
    assert "H1" in kept and "H2" in kept                       # both direct routes survive
    assert sum(h in kept for h in ("H3", "H4", "H5")) == 2     # mechanism routes keep the cap
    assert pf.is_direct(hyps[0]) and not pf.is_direct(hyps[2])
    longer = Hypothesis(id="H6", name="x", origin="ledger_path", links=[k("A", "M1"), k("M1", "B")])
    assert not pf.is_direct(longer)                            # a found multi-step route is a mechanism


def test_seeded_pathways_end_at_the_first_readout():
    """Pilot4's H1 ended 'FOXP3 -> regulatory T cell': FOXP3 is a readout, so that link is a
    definition and no paper states it. The step stayed 'not found' and the pathway stuck at 0."""
    from bmira.schemas import ProposedLink, ProposedPathway
    r = EntityResolver(Settings(ontology_provider="off"))
    chain = [("butyrate", "decreases", "HDAC"), ("HDAC", "decreases", "Histone H3"),
             ("Histone H3", "increases", "FOXP3"), ("FOXP3", "increases", "regulatory T cell")]
    pw = ProposedPathway(name="p", links=[ProposedLink(source=s, relation=rel, target=t) for s, rel, t in chain])
    full, _ = pf.proposal_keys(pw, r)
    cut, _ = pf.proposal_keys(pw, r, {r.resolve("FOXP3").id, r.resolve("regulatory T cell").id})
    assert len(full) == 4 and cut == full[:3]
    assert pf.proposal_keys(pw, r, {r.resolve("butyrate").id})[0] == full           # a start is not an end


def test_binding_has_no_direction_and_signed_effects_show_modulation():
    """Pilot4: 'butyrate binds HCAR2' was split from 'HCAR2 binds butyrate' (the only moderate
    paper), and 'butyrate modulates DCs' was 'not found' beside 'butyrate increases DCs'."""
    binds = pf.link_key("B", "binds", "H")
    claims = [_claim("A", "p1", "binds", subj="H", obj="B"), _claim("C", "p2", "binds", grade="weak", subj="B", obj="H")]
    ln = pf.build_links(claims, {}, {}, {}, Settings(), extra={binds})[binds]
    assert ln.status == "supported" and ln.n_studies == 2 and ln.grade == "moderate"
    mod = pf.link_key("B", "modulates", "D")
    claims = [_claim("E", "p3", "increases", subj="B", obj="D"), _claim("F", "p4", "decreases", subj="B", obj="D"),
              _claim("G", "p5", "no_effect", grade="strong", subj="B", obj="D")]
    ln = pf.build_links(claims[:2], {}, {}, {}, Settings(), extra={mod})[mod]
    assert ln.status == "supported" and ln.n_studies == 2
    assert pf.build_links(claims, {}, {}, {}, Settings(), extra={mod})[mod].n_contra_studies == 1   # a null still refutes
    inc = pf.link_key("B", "increases", "D")
    assert pf.build_links(claims[1:2], {}, {}, {}, Settings(), extra={inc})[inc].n_studies == 0     # no loosening here
