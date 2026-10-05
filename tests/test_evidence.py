"""Claim grading, method checks and report verification (bmira.evidence). Offline: no keys, no network."""
from bmira.evidence import sentence_tier, verify_text

from helpers import StubLLM, make_claim


# P5: negation-aware verbs; tags required.
def test_verification():
    assert sentence_tier("Lactylation does not induce IFNG [C1].") == 1
    c = make_claim("C1", "p1", "associated_with", "moderate")
    v = verify_text("Lactate drives IFNG loss in T cells [C1].", [c], ["H1"])
    assert v["overclaims"] and v["missing_tags"] == ["H1"]


def test_method_and_design_rules():
    from bmira.evidence import claim_study_type, grade_claim, verify_methods
    from bmira.sources import study_type_from_pubtypes
    c = make_claim("c", "p", "increases", perturbation_class="knockout", rescue_arm=True)
    c.span = "Lactate increased IFNG in mouse T cells."               # no knockout, rescue or control
    c.comparator_present = True
    verify_methods(c, c.span)
    assert c.perturbation_class == "none" and not c.rescue_arm and not c.comparator_present
    assert claim_study_type("animal", None, "human_primary_cells") == "human_primary"   # mixed paper
    assert claim_study_type("human_primary", None, "animal_in_vivo") == "animal"
    assert study_type_from_pubtypes(["Systematic Review"]) == "review"
    # design forced to 3 so only the indirectness rule can cap this mouse in vivo claim
    strong = make_claim("s", "p", "increases", system="animal_in_vivo", study_type="human_primary",
                        perturbation_class="knockout", rescue_arm=True)
    strong.text_access = "full_text"
    assert grade_claim(strong, "human").grade == "moderate" and "indirect_system" in strong.grade_detail["caps"]


def test_method_cues_cover_real_wording():
    from bmira.evidence import verify_methods

    def run(span, **kw):
        c = make_claim("c", "p", "increases")
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


def test_entity_names_and_negated_scope_do_not_trigger_overclaim():
    weak = [make_claim("C1", "p", "increases", "weak")]
    ok = ["Butyrate is associated with more induced regulatory T cells [C1].",
          "These weak findings do not establish that **gut-produced** butyrate induces colonic Treg [NO_EVIDENCE].",
          "Experiments would need to test effects on induced [L15] and other regulatory T cells [L16][NO_EVIDENCE]."]
    for text in ok:
        assert verify_text(text, weak, [])["overclaims"] == [], text
    for text in ["Butyrate induced regulatory T cells in mice [C1].", "Butyrate induces IFNG in T cells [C1]."]:
        assert verify_text(text, weak, [])["overclaims"], text             # real overclaims still caught


def test_trial_wording_counts_as_intervention_and_control():
    from bmira.evidence import verify_methods

    def run(span):
        c = make_claim("c", "p", "decreases", perturbation_class="pharmacological", study_type="human_rct",
                       system="human_in_vivo")
        c.span, c.comparator_present = span, True
        return verify_methods(c, span)
    for span in ["Patients were randomly assigned to empagliflozin 10 mg or placebo; it reduced hospitalization.",
                 "Vitamin D3 2000 IU daily reduced autoimmune disease incidence compared with placebo.",
                 "In a pooled analysis of 5 randomized trials, SGLT2 inhibitors lowered hospitalization.",
                 "Dapagliflozin reduced worsening heart failure versus usual care."]:
        c = run(span)
        assert c.perturbation_class == "pharmacological" and c.comparator_present, (span, c.method_checks)


def test_randomized_evidence_can_grade_strong():
    from bmira.evidence import grade_claim

    def rct(**kw):
        c = make_claim("r", "p", "decreases", study_type="human_rct", system="human_in_vivo",
                       perturbation_class="pharmacological")
        c.text_access = "full_text"
        for k, v in kw.items():
            setattr(c, k, v)
        return grade_claim(c, "human")
    assert rct().grade == "strong"                                       # randomization stands in for a rescue arm
    assert rct(comparator_present=False).grade != "strong"
    assert rct(text_access="abstract_only").grade == "moderate"          # the abstract cap still applies
    assert rct(study_type="meta_analysis").grade == "strong"
    animal = make_claim("a", "p", "decreases", perturbation_class="pharmacological")
    animal.text_access = "full_text"
    assert grade_claim(animal, "any").grade == "moderate"                # animal pharmacology unchanged


def test_entailment_judge_sees_the_study_system_and_design():
    """Pilot5: 9 of 12 entailment issues said the citation 'does not establish' an animal / human /
    cross-sectional study. The judge was shown only grade, relation and cell type."""
    from types import SimpleNamespace
    from bmira.graph import _entailment
    from bmira.schemas import EntailmentBatch
    llm = StubLLM(EntailmentBatch(judgements=[]))
    c = make_claim("C1", "p", "increases", system="human_primary_cells", study_type="human_cohort")
    _entailment("Butyrate increases Treg readouts in a human cohort [C1].", [c], SimpleNamespace(llm=llm))
    assert "system: human_primary_cells" in llm.prompts[0] and "design: human_cohort" in llm.prompts[0]


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


def test_verifier_false_alarms_from_pilot5():
    """Pilot5's overclaim samples. The first four are not causal claims; the fifth is."""
    ok = [
        "A source-traced experiment establishing that **microbiota-produced** butyrate induces **colonic** Tregs "
        "is not identified here [NO_EVIDENCE].",
        "The proposed FOXP3, fatty-acid-oxidation, receptor, and histone-acetylation mechanisms remain less "
        "established than the Treg increase itself [C34006836_0][C34035164_1].",
        "Link [L2] also connects butyrate with Treg readouts; this broader route does not specify a distinct "
        "mediator [C40983096_0][C30446387_3].",
        "Link [L15] associates butyrate with reduced histone-deacetylase readouts [C35148177_4][C38944008_1]."]
    assert all(sentence_tier(x) <= 1 for x in ok), [sentence_tier(x) for x in ok]
    bad = ("The FOXP3 findings differ by T-cell context: butyrate increases FOXP3 in naive CD4+ cells in one "
           "animal study [C34006836_0].")
    assert sentence_tier(bad) == 3                                   # a weak claim still may not say 'increases'
    assert sentence_tier("Butyrate induces Tregs, which is not established in humans.") == 4   # asserted, then qualified
    assert sentence_tier("Butyrate is the mediator of this effect.") == 0
    assert sentence_tier("Butyrate mediates this effect.") == 4
    assert sentence_tier("Butyrate associates with Tregs and increases FOXP3.") == 3
    assert sentence_tier("Source-trace microbial butyrate while measuring colonic Treg induction, and test "
                         "whether blocking FFAR2 changes it [NO_EVIDENCE].") == 0         # a proposed experiment
    assert sentence_tier("Testing shows butyrate induces Tregs.") == 4                   # 'Testing' is no imperative
    assert sentence_tier("Knockout of Ffar2 abolished butyrate-induced Treg expansion [C1].") == 4   # a finding
    assert sentence_tier("Knock out FFAR2 and measure Tregs.") == 0
    assert sentence_tier("Fatty acids that increase FOXP3 are made by Clostridia [C2].") == 3  # 'that' + verb


def test_citations_after_the_period_stay_with_their_sentence():
    """Pilot6's synthesis wrote 'text. [C1_0][C2_0] **[L5] ...' - split naively, each tag block cited the NEXT
    sentence, so 6 overclaims and 8 of 13 entailment issues were judged against the wrong claims."""
    import re
    from bmira.evidence import prose_sentences
    text = ("- **[L3] Butyrate → FAO; [L4] FAO → induced Tregs:** Single-study, weak findings associate butyrate "
            "with higher FAO. [C1_0][C2_0] Test FAO inhibition during butyrate exposure. [C1_0][C2_0]\n"
            "- **[L5] Butyrate → HIF-1α:** Butyrate reduces HIF-1α in primary human T cells. [C3_0]\n"
            "- The reported increases differ from no-effect findings. [C3_0]")
    sents = prose_sentences(text)
    assert [re.findall(r"\[(C\d_0)\]", s) for s in sents if "C" in s] == [["C1_0", "C2_0"], ["C1_0", "C2_0"], ["C3_0"], ["C3_0"]]
    assert sents[0].startswith("**[L3]") and "Test FAO" not in sents[0] and sents[2].startswith("**[L5]")
    claims = [make_claim("C1_0", "p1", "increases", grade="weak"), make_claim("C2_0", "p1", "increases", grade="weak"),
              make_claim("C3_0", "p2", "decreases", grade="moderate")]
    out = verify_text(text, claims, [])["overclaims"]
    assert out == []                       # 'induced' in a link label is no verb; 'reported increases' is a noun
    bad = verify_text("Butyrate increases FAO. [C1_0]", claims, [])["overclaims"]
    assert len(bad) == 1                   # a weak claim may still not say 'increases'
    assert sentence_tier("**[H5] Butyrate drives Treg induction.**") == 4    # bold claim, not a label


def test_verifier_false_alarms_from_pilot7():
    """Pilot7's 11 overclaims: 4 were 'a test is missing' sentences, 1 a 'none of the routes establishes' sentence,
    2 read the noun 'induction' as a verb; both 'unsupported' entailment verdicts judged proposed experiments."""
    from bmira.evidence import is_proposal
    absent = ["An experiment establishing that the proposed acetylation change increases FOXP3 is also missing [NO_EVIDENCE].",
              "A test showing that those macrophages increase colonic Tregs is missing [NO_EVIDENCE]."]
    assert not verify_text(" ".join(absent), [], [])["overclaims"]
    assert verify_text("Butyrate increases colonic Tregs [NO_EVIDENCE].", [], [])["overclaims"]      # still an assertion
    assert sentence_tier("FOXP3 is a supported accompanying response, but none of the proposed multi-step routes "
                         "establishes that its intermediate steps are necessary for Treg induction [NO_EVIDENCE].") == 0
    assert sentence_tier("IL-10 findings differ across settings; measure IL-10 and Tregs together rather than "
                         "treating IL-10 as a proxy for induction.") == 0
    assert sentence_tier("An IL-10 increase is therefore not a uniform explanation for Treg induction.") == 0
    assert sentence_tier("Butyrate induces Tregs.") == 4 and sentence_tier("Butyrate is an inducer of Tregs.") == 4
    assert is_proposal("For mechanism, prioritize butyrate exposure with receptor loss and restoration [C1].")
    assert is_proposal("Compare matched naive-CD4+ cultures with and without TGF-b1 [C1][C2].")
    assert not is_proposal("In mice, butyrate increases Tregs [C1].")

    from types import SimpleNamespace
    from bmira.graph import _entailment
    from bmira.schemas import EntailmentBatch
    llm = StubLLM(EntailmentBatch(judgements=[]))
    c = make_claim("C1", "p", "increases")
    _entailment("Compare matched cultures with and without TGF-b1 [C1].", [c], SimpleNamespace(llm=llm))
    assert not llm.prompts                                           # a proposal is not sent to the judge

