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


def _claim(i, pmid, rel, grade="moderate", subj="LOCAL:a", obj="LOCAL:b", ctx="cd8"):
    return Claim(id=i, pmid=pmid, claim_type="observation", subject="a", relation=rel, object="b",
                 span="x" * 30, study_type="animal", text_access="abstract_only",
                 subject_concept=subj, object_concept=obj, relation_norm=rel, grade=grade,
                 context_cell_type=ctx)


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
    contra = strong[:1] + [_claim("c4", "p4", "decreases"), _claim("c5", "p5", "no_effect")]
    assert pf.build_links(contra, {}, {}, labels, s)[k].status == "contradicted"
    # the same opposing findings, judged context-dependent by conflict triage, do not count
    ctx = {frozenset(("c1", "c4")), frozenset(("c1", "c5"))}
    ln = pf.build_links(contra, {}, {}, labels, s, discounted=ctx)[k]
    assert ln.status == "insufficient" and ln.context_dependent_ids == ["c4", "c5"]


# P5: negation-aware verbs; tags required.
def test_verification():
    assert sentence_tier("Lactylation does not induce IFNG [C1].") == 1
    c = _claim("C1", "p1", "associated_with", "moderate")
    v = verify_text("Lactate drives IFNG loss in T cells [C1].", [c], ["H1"])
    assert v["overclaims"] and v["missing_tags"] == ["H1"]


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
