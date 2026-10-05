"""Blocking tests and the mediation evidence model (EM-1..5, EM-7). Cases from pilot5 (Slc5a8, GPR109A)
and the offline lactate scenario. Offline: no keys, no network."""
from types import SimpleNamespace

import pytest

from bmira import portfolio as pf
from bmira.config import Settings
from bmira.graph import extract, normalize
from bmira.offline import offline_runtime
from bmira.schemas import ClaimList, ExtractedClaim, Paper

from helpers import StubLLM, make_claim, parsed_question

TEXT = ("Butyrate expanded colonic regulatory T cells in wild-type mice compared with untreated controls. "
        "Slc5a8-null DCs did not induce IDO1 in response to butyrate. "
        "Butyrate failed to expand colonic Tregs in Gpr109a-/- mice compared with wild-type littermates. "
        "Ffar2 was dispensable: butyrate expanded Tregs equally in Ffar2-/- and wild-type mice. "
        "Hdac9 deficiency increased Foxp3 expression.")


def _extract(*claims, text=TEXT, judge=None, log=None):
    """Run the extract node on one paper with scripted claims; returns (kept, dropped)."""
    defaults = {"claim_type": "observation", "system": "animal_in_vivo", "comparator_present": True}
    ecs = [ExtractedClaim(**{**defaults, **c}) for c in claims]
    paper = Paper(pmid="p5", title="t", abstract=text, source_text=text, study_type="animal")
    rt = SimpleNamespace(settings=Settings(), llm=StubLLM(ClaimList(claims=ecs)), judge=judge,
                         source=SimpleNamespace(fulltext=lambda p: p))
    out = extract({"paper": paper, "question": "q", "focus": [], "focus_keys": [], "n_existing": 0,
                   "seen": set(), "round": 1}, rt)
    if log is not None:
        log += out["judge_log"]
    return out["claims"], out["dropped_claims"]


SLC5A8 = {"subject": "Slc5a8", "object": "IDO1", "relation": "did not induce",
          "span": "Slc5a8-null DCs did not induce IDO1 in response to butyrate.",
          "perturbation_class": "knockout", "effect_exposure": "butyrate", "effect_result": "abolished"}
GPR109A = {"subject": "Gpr109a", "object": "Tregs", "relation": "failed to expand",
           "span": "Butyrate failed to expand colonic Tregs in Gpr109a-/- mice compared with wild-type littermates.",
           "perturbation_class": "knockout", "effect_exposure": "butyrate", "effect_result": "abolished",
           "subject_lost": True}                       # the extractor's error: a blocking test is not restated


def test_blocking_tests_keep_their_treatment_and_are_typed_from_the_result():
    kept, dropped = _extract(SLC5A8, GPR109A)
    assert not dropped and all(c.is_blocking_test for c in kept)
    gpr = kept[1]
    assert not gpr.subject_lost and "subject_lost ignored on a blocking test" in gpr.method_checks
    rt, _ = offline_runtime()
    out = normalize({"claims": kept, "parsed": parsed_question(exposure="butyrate")}, rt)["claims"]
    assert [(c.relation_norm, c.relation_source) for c in out] == [("required_for", "blocking_test")] * 2
    assert out[0].effect_exposure_concept == rt.resolver.resolve("butyrate").id and out[0].effect_exposure_label
    again = normalize({"claims": out, "parsed": parsed_question(exposure="butyrate")}, rt)["claims"]
    assert [c.relation_norm for c in again] == ["required_for"] * 2              # never restated or retyped


@pytest.mark.parametrize("change, reason", [
    ({"effect_exposure": "propionate"}, "blocking-test treatment not named in quote"),
    ({"perturbation_class": "none"}, "blocking test without a removal or blocking perturbation"),
    ({"perturbation_class": "overexpression"}, "blocking test without a removal or blocking perturbation")])
def test_unverifiable_blocking_tests_are_dropped_with_a_reason(change, reason):
    kept, dropped = _extract({**SLC5A8, **change})
    assert not kept and dropped[0].drop_reason == reason


def test_a_cue_miss_lowers_the_grade_but_keeps_the_experiment():
    """Pilot5: 'Slc5a8-null' is a knockout the cue list does not know. The experiment stays a blocking test;
    its uncredited knockout is recorded and graded as such."""
    (c,), _ = _extract(SLC5A8)
    assert c.is_blocking_test and c.perturbation_class == "none"
    assert "perturbation 'knockout' not evidenced" in c.method_checks


def test_the_treatment_may_be_named_in_the_sentence_before():
    kept, _ = _extract({"subject": "Hdac9", "object": "Foxp3", "relation": "increased",
                        "span": "Hdac9 deficiency increased Foxp3 expression.", "perturbation_class": "knockout",
                        "effect_exposure": "butyrate", "effect_result": "enhanced"},
                       text="Butyrate was given to all mice. Hdac9 deficiency increased Foxp3 expression.")
    assert kept and kept[0].is_blocking_test


def test_half_filled_blocking_fields_make_an_ordinary_claim():
    kept, dropped = _extract({**SLC5A8, "effect_result": "", "relation": "did not induce"})
    assert not dropped and not kept[0].is_blocking_test and kept[0].effect_exposure == ""
    assert "incomplete blocking-test fields ignored" in kept[0].method_checks


def test_unchanged_and_enhanced_results_stay_out_of_binary_links():
    """'Butyrate expanded Tregs equally in Ffar2-/- mice' is not 'Ffar2 no_effect Treg'."""
    kept, _ = _extract({"subject": "Ffar2", "object": "Tregs", "relation": "expanded equally",
                        "span": "Ffar2 was dispensable: butyrate expanded Tregs equally in Ffar2-/- and wild-type mice.",
                        "perturbation_class": "knockout", "effect_exposure": "butyrate", "effect_result": "unchanged"})
    rt, _ = offline_runtime()
    (c,) = normalize({"claims": kept, "parsed": parsed_question(exposure="butyrate")}, rt)["claims"]
    assert c.relation_norm == "no_effect" and c.is_blocking_test
    key = pf.link_key(c.subject_concept, "no_effect", c.object_concept)
    links = pf.build_links([c], {}, {}, {}, Settings(), extra={key})
    assert links[key].support_ids == [] and links[key].n_studies == 0
    plain = make_claim("x", "p", "no_effect", subj=c.subject_concept, obj=c.object_concept)
    assert pf.build_links([plain], {}, {}, {}, Settings(), extra={key})[key].support_ids == ["x"]


def test_the_judge_sees_kept_and_rejected_blocking_tests():
    """JV-15 in shadow: a kept blocking test is asked whether the quote reports the test and its result; one
    the code rejected is asked whether it is a blocking test at all."""
    from bmira.judge import SurrogateJudge
    log = []
    _extract(SLC5A8, {**GPR109A, "effect_exposure": "propionate"}, judge=SurrogateJudge(), log=log)
    by = {(e["task"], e["item_id"]): e for e in log}
    assert by[("blocking_test", "Cp5_0")]["current"] is True and by[("blocking_test", "Cp5_1")]["current"] is False
    assert by[("blocking.effect_result", "Cp5_0")]["jev"] == "abolished"
    assert ("stance", "Cp5_1") not in by                        # a rejected blocking test has no stance to judge


# ── EM-3: the mediation index ───────────────────────────────────────────────
def _block(i, pmid, result, grade="moderate", mediator="LOCAL:m", outcome="LOCAL:y", exposure="LOCAL:x", **kw):
    c = make_claim(i, pmid, "required_for", grade=grade, subj=mediator, obj=outcome, **kw)
    c.effect_exposure, c.effect_result, c.effect_exposure_concept = "x", result, exposure
    return c


def _index(claims, ancestors=None):
    return pf.build_mediation(claims, "LOCAL:x", {"LOCAL:y", "LOCAL:r"}, ancestors or {}, {}, Settings())


KEY = pf.mediation_key("LOCAL:x", "LOCAL:m", "LOCAL:y")


def test_one_moderate_blocking_test_demonstrates_and_two_replicate():
    m = _index([_block("a", "p1", "abolished")])[KEY]
    assert m.status == "demonstrated" and m.reason == "1 study, moderate" and m.support_ids == ["a"]
    m = _index([_block("a", "p1", "abolished"), _block("b", "p2", "attenuated", grade="strong")])[KEY]
    assert m.reason == "replicated in 2 studies, strong" and m.n_support_papers == 2


def test_weak_reviewed_or_other_blocking_tests_do_not_demonstrate():
    assert _index([_block("a", "p1", "abolished", grade="weak")])[KEY].reason == "only weak blocking evidence (1 study)"
    rev = _index([_block("a", "p1", "abolished", study_type="review")])[KEY]
    assert rev.status == "insufficient" and rev.uncounted == {"a": "secondary source (review)"}       # R1
    assert _index([_block("a", "p1", "abolished", exposure="LOCAL:other")]) == {}   # another treatment's effect
    assert _index([_block("a", "p1", "abolished", mediator="LOCAL:x")]) == {}       # the exposure cannot mediate
    assert _index([_block("a", "p1", "enhanced")])[KEY].uncounted == {"a": "effect larger without the mediator"}


def test_an_unchanged_effect_refutes_only_when_it_is_as_strong_as_the_support():
    strong_null = [_block("a", "p1", "abolished"), _block("n1", "p2", "unchanged"), _block("n2", "p3", "unchanged")]
    m = _index(strong_null)[KEY]
    assert m.status == "refuted" and m.against_ids == ["n1", "n2"]
    weak_null = [_block("a", "p1", "abolished"), _block("n", "p2", "unchanged", grade="weak")]
    m = _index(weak_null)[KEY]
    assert m.status == "demonstrated" and m.uncounted == {"n": "null result weaker than the support"}      # R5
    no_control = _block("n", "p2", "unchanged")
    no_control.comparator_present = False
    assert _index([_block("a", "p1", "abolished"), no_control])[KEY].status == "demonstrated"
    tied = _index([_block("a", "p1", "abolished"), _block("n", "p2", "unchanged")])[KEY]
    assert tied.status == "refuted"                         # 1 of 2 papers against: share 0.5 >= threshold


def test_a_subtype_outcome_supports_its_parent_but_never_refutes_it():
    anc = {"LOCAL:sub": ("LOCAL:y",)}
    m = _index([_block("a", "p1", "abolished", outcome="LOCAL:sub")], anc)[KEY]
    assert m.status == "demonstrated"                                                                    # R9
    m = _index([_block("a", "p1", "abolished"), _block("n", "p2", "unchanged", outcome="LOCAL:sub")], anc)[KEY]
    assert m.status == "demonstrated" and "never counts against" in m.uncounted["n"]


# ── EM-4, EM-5: route verdicts ──────────────────────────────────────────────
def _route(*statuses, contexts=None, origin="llm_seed"):
    names = ["LOCAL:x", "LOCAL:m", "LOCAL:y"] if len(statuses) == 2 else ["LOCAL:x", "LOCAL:y"]
    keys = [pf.link_key(a, "increases", b) for a, b in zip(names, names[1:])]
    links = {k: pf.LinkEvidence(key=k, subject=k.split("|")[0], relation="increases", object=k.split("|")[2],
                                subject_label=k.split("|")[0][6:], object_label=k.split("|")[2][6:],
                                status=st, completeness=0.7 if st == "supported" else 0.0,
                                contexts=(contexts or {}).get(i, []), reason=st)
             for i, (k, st) in enumerate(zip(keys, statuses))}
    from bmira.schemas import Hypothesis
    return Hypothesis(id="H1", name="r", origin=origin, links=keys), links


def _verdict(h, links, mediation=None, ancestors=None):
    (out,) = pf.evaluate([h], links, {}, "up", Settings(), mediation, ancestors)
    return out.status, out.reason


def test_route_verdict_precedence():
    shown = _index([_block("a", "p1", "abolished")])
    refuted = _index([_block("a", "p1", "abolished"), _block("n1", "p2", "unchanged"), _block("n2", "p3", "unchanged")])
    assert _verdict(*_route("supported", "supported"))[0] == "assembled"
    assert _verdict(*_route("supported", "insufficient"), shown)[0] == "demonstrated"   # one blocking test suffices
    assert _verdict(*_route("contradicted", "supported"), shown)[0] == "contradicted"   # a contradicted step wins
    status, reason = _verdict(*_route("supported", "supported"), refuted)
    assert status == "assembled" and "effect persisted" in reason                       # handoff order, said aloud
    assert _verdict(*_route("supported", "insufficient"), refuted)[0] == "refuted"
    assert _verdict(*_route("insufficient", "insufficient"))[0] == "insufficient"
    direct = _route("supported", origin="ledger_path")
    assert _verdict(*direct, shown)[0] == "supported"                                    # direct routes keep steps


def test_assembled_needs_compatible_contexts():
    anc = {"CL:treg": ("CL:tcell",)}
    assert _verdict(*_route("supported", "supported", contexts={0: ["CL:treg"], 1: ["CL:tcell"]}), None, anc)[0] \
        == "assembled"                                                                   # a subtype fits its parent
    assert _verdict(*_route("supported", "supported", contexts={0: ["unspecified"], 1: ["CL:dc"]}))[0] == "assembled"
    status, reason = _verdict(*_route("supported", "supported", contexts={0: ["CL:treg"], 1: ["CL:dc"]}), None, anc)
    assert status == "insufficient" and "different cell types" in reason
    assert pf.contexts_compatible([], ["CL:dc"]) and not pf.contexts_compatible(["CL:a"], ["CL:b"])


def test_convergence_needs_a_mechanism_that_no_rival_can_overtake():
    from bmira.schemas import Hypothesis
    s = Settings()
    lead = Hypothesis(id="H1", name="a", origin="llm_seed", links=["X|i|M", "M|i|Y"], status="assembled",
                      score=0.7, open=False)
    rival = Hypothesis(id="H2", name="b", origin="llm_seed", links=["X|i|N", "N|i|Y"], status="insufficient",
                       logic_factor=0.5, open=True)
    assert pf.decide([lead, rival], ["t"], 1, s) == ("search_more", "TARGETED")       # the rival could be shown
    lead.status = "demonstrated"
    assert pf.decide([lead, rival], ["t"], 1, s) == ("done", "CONVERGED")             # same tier at best, lower score
    rival.logic_factor = 0.9
    assert pf.decide([lead, rival], ["t"], 1, s)[1] == "TARGETED"
    direct = Hypothesis(id="H3", name="d", origin="ledger_path", links=["X|i|Y"], status="supported", score=1.0)
    rival.logic_factor = 0.5
    assert pf.decide([direct, rival], ["t"], 1, s)[1] == "TARGETED"                   # a direct route answers 'whether'
    assert pf.decide([direct], ["t"], 1, s)[1] == "CONVERGED"                         # ... unless it is all there is


def test_the_report_shows_the_blocking_test_and_the_direct_effect(offline):
    _, final = offline
    report = final["report"]
    assert "| Blocking test |" in report and "Shown by a blocking experiment" in report
    assert "blocking Glycolytic flux on CD8 T cell effector function: shown (1 study, moderate) [CS020_0]" in report
    assert "blocking NAD+: not tested" in report and report.count("Direct effect:") == 1


def test_follow_up_context_carries_the_blocking_tests(offline):
    from bmira.chat import run_context
    _, final = offline
    ctx = run_context(final)
    assert "blocking Glycolytic flux on CD8 T cell effector function: shown" in ctx and "[CS020_0]" in ctx


def test_a_one_link_route_proposed_by_the_model_keeps_step_verdicts():
    h, links = _route("supported", origin="llm_seed")
    assert _verdict(h, links) == ("supported", "every step has independent support")


def test_states_saved_before_v3_still_replay(tmp_path):
    """Pilot states predate mediation, judge_log, blocking fields and read counters; a replay must work."""
    import json
    from helpers import run_session
    run_session(tmp_path, "only question\n")
    path = tmp_path / "s_q1.state.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    st = data["state"]
    for k in ("mediation", "judge_log"):
        st.pop(k, None)
    for c in st["claims"] + st["dropped_claims"]:
        for f in ("effect_exposure", "effect_result", "effect_exposure_concept", "effect_exposure_label"):
            c.pop(f, None)
    for p in st["papers"]:
        p.pop("n_reads", None), p.pop("chars_read", None)
    for h in st["hypotheses"]:
        h["status"] = "supported" if h["status"] in {"demonstrated", "assembled"} else h["status"]   # v2 vocabulary
    path.write_text(json.dumps(data), encoding="utf-8")
    session = run_session(tmp_path, "only question\n", "--replay", str(path), out="r.json")
    run = session["runs"][0]
    assert run["status"] == "completed" and run["verification"]["passed"]
    assert run["pathways"]["ranked"][0]["verdict"] == "Assembled from separate studies"   # no blocking test left


def test_a_persisting_effect_still_counts_against_required_for():
    """Review finding: an 'unchanged' blocking test left binary links entirely, so it could no longer refute
    'M required_for Y' while 'abolished' ones still supported it. It counts against, never for (R5 applies)."""
    sup = _block("a", "p1", "abolished")
    sup.relation_norm = "required_for"
    null = _block("n", "p2", "unchanged")
    null.relation_norm = "no_effect"
    key = pf.link_key("LOCAL:m", "required_for", "LOCAL:y")
    ln = pf.build_links([sup, null], {}, {}, {}, Settings(), extra={key})[key]
    assert ln.contradicting_ids == ["n"] and ln.status == "contradicted"
    assert pf.link_key("LOCAL:m", "no_effect", "LOCAL:y") not in pf.build_links([null], {}, {}, {}, Settings())
