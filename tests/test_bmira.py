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
