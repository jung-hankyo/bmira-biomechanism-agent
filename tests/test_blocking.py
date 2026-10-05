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
    ({"perturbation_class": "none"}, "blocking test without a perturbation")])
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
