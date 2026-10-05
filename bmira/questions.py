"""Every question asked of the decision model (judge) and every threshold that acts on an answer.

One file, pinned to one model version: a new judge version is a new model (new cache file, shadow
again on the gold sets, recalibrate). The texts are the v3 handoff's starting points (Appendix B);
edit them with the owner, then calibrate. THRESHOLDS are starting points for shadow mode: no
decision reads them until a touchpoint is calibrated and switched to act (handoff 4.5).

Writing rules (TypeSafe guidance applied to B-MiRA): one atomic judgment per question, combined in
code; the exact condition and its boundary cases go in `criteria`; small states (one paragraph, one
claim, one pair); no negated criteria; state fields named in backticks.
"""
MODEL = "jev-1.13.0"

SCREEN = {  # JV-1, JV-2
    "relevance": {"type": "score",
        "instructions": "How directly can `abstract` inform `question`?",
        "criteria": [
            "Unrelated to the exposure and the outcome",
            "Background, review or commentary only; no new data on them",
            "Indirect: a related exposure, another system, or only one link of the question",
            "Original experiment or analysis on the exposure and the outcome, or on a listed step"]},
    "original_data": {"type": "noul",
        "instructions": "Does `abstract` report original experimental or observational data?",
        "criteria": {"true": "New measurements, experiments, trials or cohort analyses by the authors",
                     "false": "A review, commentary, editorial, protocol or summary of other work"}},
}


def step_question(step_text: str, blocking: bool = False) -> dict:  # JV-1 step Nouls, LC-1
    ask = ("Does `abstract` report a test of whether the exposure's effect in `step` persists when the "
           "intermediate is removed or blocked?" if blocking else
           "Does `abstract` report an experiment or analysis that tests `step`?")
    return {"type": "noul", "instructions": {"step": step_text, "question": ask},
            "criteria": {"true": "The abstract reports a result on this, positive, negative or null",
                         "false": "This is not tested in the abstract"}}


def screen_questions(steps: list[str]) -> dict:
    """SCREEN plus one Noul per target step the paper was retrieved for, keyed reports_step_<n>."""
    return {**SCREEN, **{f"reports_step_{n}": step_question(s) for n, s in enumerate(steps, 1)}}


PASSAGE = {  # JV-3, one paragraph per request
    "reports_result": {"type": "noul",
        "instructions": "Does `paragraph` report a measured result about an entity in `focus`?",
        "criteria": {"true": "It states what was observed or measured, including null results",
                     "false": "It gives background, hypotheses, or interpretation of other work"}},
    "method_detail": {"type": "noul",
        "instructions": "Does `paragraph` describe how an entity in `focus` was manipulated or compared?",
        "criteria": {"true": "Knockout, knockdown, deficient animals, inhibitor, antagonist, dose, control group, randomization",
                     "false": "No manipulation or comparison of these entities is described"}},
    "blocking_test": {"type": "noul",
        "instructions": "Does `paragraph` report whether an effect of the exposure in `focus` persisted when another entity was removed or blocked?",
        "criteria": {"true": "A knockout, knockdown, inhibitor or antagonist experiment tests whether the effect still occurs",
                     "false": "No such test is reported"}},
}

STANCE = {"stance": {"type": "choice",  # JV-4
    "instructions": "How does `quote`, read with `previous_sentence`, relate to `claim`?",
    "criteria": {"supports": "The quote states the claim's finding, or directly implies it",
                 "contradicts": "The quote states the opposite finding, or that the effect did not occur",
                 "says_nothing": "The quote does not report this finding either way"}}}

METHODS = {  # JV-5
    "comparator": {"type": "noul",
        "instructions": "Do `quote` or `methods` compare the treated or perturbed group with a control group?",
        "criteria": {"true": "A control, vehicle, untreated, wild-type, baseline, placebo or usual-care group is named or clearly implied",
                     "false": "No comparison group is described"}},
    "perturbation": {"type": "choice",
        "instructions": "What did the experiment in `quote` and `methods` do to produce the finding?",
        "criteria": {"none": "Nothing was manipulated; observation or measurement only",
                     "genetic_association": "A natural genetic variant was associated with the finding",
                     "pharmacological": "A drug, compound, inhibitor, agonist or antagonist was given",
                     "environmental": "A diet, supplement, culture condition or exposure was applied",
                     "knockdown": "Expression was reduced (siRNA, shRNA, antisense)",
                     "knockout": "A gene was deleted (knockout, -/- or deficient animals, CRISPR)",
                     "overexpression": "A gene was overexpressed or transduced",
                     "transfer": "Cells, microbiota or tissue were transferred"}},
    "rescue": {"type": "noul",
        "instructions": "Do `quote` or `methods` report restoring the removed entity and recovering the effect?",
        "criteria": {"true": "Re-expression, add-back or reconstitution restored the effect",
                     "false": "No restoration experiment is reported"}},
    "orthogonal": {"type": "noul",
        "instructions": "Do `quote` or `methods` confirm the finding with a second independent method or model?",
        "criteria": {"true": "A second method, reagent, model or cohort confirms the same finding",
                     "false": "The finding rests on one method"}},
}

SUBJECT_LOST = {"subject_lost": {"type": "noul",  # JV-6
    "instructions": "In the group whose result `quote` reports, was `subject` removed, deleted, knocked out, knocked down, depleted or blocked?",
    "criteria": {"true": "The reported result is what happened when the subject was missing or blocked",
                 "false": "The subject was present or was itself given; it may be the agent that inhibits or reduces something"}}}

RELATION = {"relation": {"type": "choice",  # JV-8; keep 'unresolved' last
    "instructions": "Which relation states what `relation_wording` in `quote` says about `subject` and `object`, without making it stronger?",
    "criteria": {"increases": "The subject raised the amount, activity or number of the object",
                 "decreases": "The subject lowered the amount, activity or number of the object",
                 "modulates": "The subject changed the object, direction not stated",
                 "no_effect": "The subject was tested and did not change the object",
                 "required_for": "The object, or its change, did not occur without the subject",
                 "sufficient_for": "The subject alone produced the object or its change",
                 "associated_with": "The two co-occur or correlate; no causal test",
                 "not_associated": "No association was found",
                 "predicts": "The subject forecasts the object in a prognostic or diagnostic analysis",
                 "binds": "The two bind physically",
                 "modifies": "The subject chemically modifies the object (phosphorylation, acetylation, lactylation)",
                 "unresolved": "None of these states the wording faithfully"}}}


def term_question(candidates: list) -> dict:  # JV-9; candidates: (id, label, synonyms, ontology, definition)
    crit = {cid: {"label": lab, "synonyms": list(syn)[:5], "ontology": onto, "definition": (dfn or "")[:200]}
            for cid, lab, syn, onto, dfn in candidates}
    crit["none"] = "None of these terms names the entity as it is used in the quote"
    return {"term": {"type": "choice", "instructions": "Which term names `surface` as it is used in `quote`?",
                     "criteria": crit}}


STRENGTH = {"strength": {"type": "score",  # JV-14; level i corresponds to verifier tier i
    "instructions": "How strong a claim about cause does `sentence` make?",
    "criteria": ["No claim: absence of evidence, a question, a limitation or a proposed experiment",
                 "Association: things occur together or correlate",
                 "Hedged: may, might, suggests, appears, possibly",
                 "Effect: one thing increased, decreased, promoted or inhibited another",
                 "Cause or necessity: causes, drives, mediates, is required for"]}}

ENTAILMENT = {"entailment": {"type": "choice",  # JV-14
    "instructions": "Do `cited_claims` entail `sentence`?",
    "criteria": {"entailed": "The cited claims state everything the sentence asserts, at the same strength and in the same system",
                 "partial": "They support part of it, or the sentence joins them into a stronger statement",
                 "unsupported": "They do not support what the sentence asserts"}}}

BLOCKING = {  # JV-15, EM-2
    "blocking_test": {"type": "noul",
        "instructions": "Does `quote` report what happened to the effect of `effect_exposure` on `object` when `subject` was removed or blocked?",
        "criteria": {"true": "The quote reports this test and its result",
                     "false": "The quote does not report such a test"}},
    "effect_result": {"type": "choice",
        "instructions": "In that test, what happened to the effect of `effect_exposure` on `object`?",
        "criteria": {"abolished": "The effect no longer occurred", "attenuated": "The effect was smaller",
                     "unchanged": "The effect occurred as before", "enhanced": "The effect was larger"}},
}

THRESHOLDS = {  # starting points for shadow mode; replace with calibrated values per MODEL
    "screen_include": 0.5, "passage_keep": 0.3, "passage_keep_discussion": 0.6,
    "stance_drop": 0.8, "stance_exclude": 0.8, "method_confirm": 0.6, "method_add": None,
    "subject_lost_invert": 0.8, "subject_lost_flag": 0.9, "claim_type_fill": 0.6, "claim_type_override": 0.8,
    "relation_accept": 0.6, "term_accept": 0.5, "alias_merge": 0.85,
    "overclaim": 0.6, "unsupported": 0.7,
}


def reversed_options(question: dict) -> dict:
    """The same Choice with its options in reverse order, for the first-option-lean check (handoff 4.5.3).
    Score and Noul questions are returned unchanged."""
    if question.get("type") != "choice":
        return question
    return {**question, "criteria": dict(reversed(list(question["criteria"].items())))}
