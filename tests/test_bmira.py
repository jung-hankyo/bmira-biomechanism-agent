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
