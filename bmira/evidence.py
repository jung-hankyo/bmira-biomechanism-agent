"""Claim grading (pure code) and verification of the written synthesis.

Grading is cheap deterministic arithmetic over fields the extractor already produced, so
every claim is graded. (V11 deferred it "to save cost"; there was no cost to save, and the
deferral left most claims ungraded in the synthesis ledger.)
"""
import re

from bmira.normalize import span_is_anchored
from bmira.schemas import (ASSOCIATIVE_RELATIONS, CAUSAL_RELATIONS, DIRECTION, LABEL, SYSTEM_GROUP,
                           NULL_RELATIONS, PHYSICAL_RELATIONS)

DIRECTNESS = {"observation": 3, "author_interpretation": 2, "mechanistic_speculation": 1}
CAUSAL_BASE = {"none": 0, "genetic_association": 1, "pharmacological": 2, "environmental": 2,
               "knockdown": 2, "knockout": 3, "overexpression": 2, "transfer": 2}
DESIGN = {"meta_analysis": 3, "human_rct": 3, "human_cohort": 3, "human_primary": 3,
          "human_crosssectional": 2, "organoid_ipsc": 2, "animal": 2, "computational_cohort": 2,
          "cell_line": 1, "in_silico": 1, "review": 1}


SYSTEM_STUDY = {"human_primary_cells": "human_primary", "animal_in_vivo": "animal",
                "animal_cells": "animal", "organoid": "organoid_ipsc", "cell_line": "cell_line",
                "in_silico": "in_silico"}
SYSTEM_RANK = {"human": 3, "animal": 2, "cell": 1, "in_silico": 1}
TARGET_RANK = {"human": 3, "animal": 2, "cell": 1, "any": 0}


def claim_study_type(paper_type, pubtype, system) -> str:
    """Design of ONE claim. Mixed papers (mouse in vivo + human cells) were graded with a
    single paper-level label. Pooled or secondary designs (meta-analysis, review) apply to
    every claim; otherwise the claim's own system decides."""
    if pubtype in {"meta_analysis", "review"}:
        return pubtype
    if system == "human_in_vivo":
        if pubtype:
            return pubtype
        human = {"human_rct", "human_cohort", "human_crosssectional", "computational_cohort"}
        return paper_type if paper_type in human else "human_crosssectional"
    return SYSTEM_STUDY.get(system) or paper_type or "cell_line"


# Wording that shows a compound or condition was given: 'provision of X', '0.25 mM', 'in vitro'.
GIVEN = (r"administ|provision|provid|receiv|\bfed\b|feeding|gavage|inject|incubat|stimulat|titrat|exogenous|"
         r"\d\s?(?:mm|[µμu]m|nm|mg|mcg|iu|g)\b|in vitro|in vivo|ex vivo|"
         r"randomi[sz]|randomly|placebo|usual care|standard care|standard therapy|daily|weekly")
METHOD_CUES = {
    "knockout": r"knock-?out|delet|deficien|-/-|null mice|crispr|\bko\b",
    "knockdown": r"knock-?down|sirna|shrna|silenc|antisense",
    "overexpression": r"overexpress|transduc|transfect|forced expression",
    "pharmacological": r"inhibit|agonist|antagonist|treat|block|drug|compound|supplement|precursor|\bdose|" + GIVEN,
    "environmental": r"expos|treat|cultur|condition|supplement|medium|hypoxi|diet|acid|" + GIVEN,
    "transfer": r"transfer|adoptive|transplant",
    "genetic_association": r"variant|polymorphism|snp|allele|gwas|mendelian",
}
RESCUE_CUE = r"rescu|restor|re-?express|reconstitut|add-?back"
ORTHOGONAL_CUE = r"independent|orthogonal|second|alternative|validat|confirm"
COMPARATOR_CUE = (r"compar|versus|\bvs\.?|relative to|than|control|wild-?type|\bwt\b|vehicle|untreated|baseline|"
                  r"scrambled|absence of|\bwhereas\b|\bwhile\b|\black(?:s|ed|ing)?\b|\bunlike\b|in contrast|"
                  r"\bbut not\b|\bwithout\b|\balone\b|randomi[sz]|\btrials?\b|placebo|usual care|"
                  r"standard care|standard therapy|\bsham\b")


def verify_methods(c, source: str):
    """Method fields drive the causal grade, so each needs textual evidence in the claim's
    quote or an anchored methods sentence; unsupported fields are reset and recorded."""
    text = c.span + (" " + c.methods_span if c.methods_span and span_is_anchored(c.methods_span, source) else "")
    text = text.lower().replace("−", "-").replace("–", "-")      # 'Tbx21−/−' uses a minus sign
    checks = []
    if c.perturbation_class != "none" and not re.search(METHOD_CUES[c.perturbation_class], text):
        checks.append(f"perturbation '{c.perturbation_class}' not evidenced")
        c.perturbation_class = "none"
    for field, cue in (("rescue_arm", RESCUE_CUE), ("orthogonal_validation", ORTHOGONAL_CUE),
                       ("comparator_present", COMPARATOR_CUE)):
        if getattr(c, field) and not re.search(cue, text):
            checks.append(f"{field} not evidenced")
            setattr(c, field, False)
    c.method_checks += checks
    return c


def causal_support(c) -> int | None:
    if c.relation_norm not in CAUSAL_RELATIONS:
        return None                                # not applicable, not "zero evidence"
    s = CAUSAL_BASE.get(c.perturbation_class, 0)
    # randomization with a control arm stands in for a rescue arm: it removes confounding
    checked = c.rescue_arm or c.orthogonal_validation or (
        s >= 2 and c.comparator_present and c.study_type in {"human_rct", "meta_analysis"})
    if s >= 2 and checked:
        s = 3
    if c.perturbation_class == "pharmacological" and not checked:
        s = min(s, 2)                              # off-target confounding
    if not c.comparator_present:
        s = min(s, 1)
    return s


def grade_claim(c, target_system: str = "any"):
    d = DIRECTNESS.get(c.claim_type, 1) - (1 if c.readout_is_inferred else 0)
    design = DESIGN.get(c.study_type, 1)
    causal = causal_support(c)
    axes = {"directness": max(1, d), "design": design}
    if c.relation_norm == "no_effect" and not c.comparator_present:
        axes["null_comparison"] = 1
    if causal is not None:
        axes["causal"] = causal
    tier = max(1, min(axes.values()))
    uncapped, caps = tier, []
    if c.relation_norm in ASSOCIATIVE_RELATIONS and c.study_type != "meta_analysis" and tier > 2:
        tier, caps = 2, caps + ["associative"]
    if c.relation_norm == "no_effect" and tier > 2:
        tier, caps = 2, caps + ["null_without_power"]
    if c.study_type == "review" and tier > 1:
        tier, caps = 1, caps + ["secondary_source"]
    if c.text_access == "abstract_only" and tier > 2:
        tier, caps = 2, caps + ["abstract_only"]
    if c.relation_norm in {"unresolved", ""}:
        tier, caps = 1, caps + ["unresolved_relation"]
    if "relation wording not in quote" in c.method_checks and tier > 1:
        tier, caps = 1, caps + ["relation_unverified"]
    rank = SYSTEM_RANK.get(SYSTEM_GROUP.get(c.system, ""), 0)
    if rank and rank < TARGET_RANK.get(target_system, 0) and tier > 2:
        tier, caps = 2, caps + ["indirect_system"]           # GRADE indirectness
    c.grade = LABEL[tier]
    c.grade_detail = {**axes, "causal_applicable": causal is not None, "uncapped_tier": uncapped,
                      "caps": caps, "limiting_axis": min(axes, key=axes.get)}
    return c


def stance(c) -> str:
    """Polarity used for conflicts and corroboration. Unresolved relations have none."""
    if c.relation_norm in {"", "unresolved"}:
        return "unknown"
    if c.relation_norm in NULL_RELATIONS:
        return "null"
    if c.relation_norm in DIRECTION:
        return DIRECTION[c.relation_norm]
    if c.relation_norm in PHYSICAL_RELATIONS:
        return "physical"
    return "effect"


# ── verification ────────────────────────────────────────────────────────────
VERB_TIER = {
    1: [r"\bassociated with\b", r"\bcorrelat\w*", r"\blinked to\b", r"\bconsistent with\b"],
    2: [r"\bsuggest\w*", r"\bmay\b", r"\bappears? to\b", r"\bpossibl\w*"],
    3: [r"\bpromot\w*", r"\binhibit\w*", r"\benhanc\w*", r"\breduc\w*", r"\bimpair\w*",
        r"\bsuppress\w*", r"\battenuat\w*", r"\baugment\w*", r"\bmodulat\w*",
        r"\bincreas\w*", r"\bdecreas\w*", r"\belevat\w*", r"\blower\w*", r"\bdeplet\w*"],
    4: [r"\bcause[sd]?\b", r"\bdrive[sn]?\b", r"\bdrove\b", r"\binduc\w*", r"\btrigger\w*",
        r"\babolish\w*", r"\bmediat(?:e[sd]?|ing|ion)\b", r"\brequired for\b", r"\bnecessary for\b",   # not 'mediator'
        r"\bsufficient\b", r"\blead(?:s|ing)? to\b"],
}
NEGATION = re.compile(r"\b(?:not|no|never|without|fail(?:s|ed)? to|did not|does not|do not)\b")
MAX_TIER = {"ungraded": 1, "weak": 1, "moderate": 3, "strong": 4}
STRUCTURE_TAG = re.compile(r"^(?:[HL]\d+|NO_EVIDENCE)$")


# 'more induced regulatory T cells': 'induced' names a cell type here, it is not a causal verb
NOUN_INDUCED = re.compile(
    r"\b(more|fewer|of|on|the|and|or|for|with|than|in|to|specifically|particularly|especially|only)\s+"
    r"(?:induced|inducible)\s+(?=(?:and other\s+)?"
    r"(?:(?:regulatory|t[- ]regulatory|helper|effector|memory)\s+)*(?:t[- ]?)?(?:cells?|tregs?|itregs?)\b)")
# 'associated with increased X': the change word describes the association, it asserts no cause.
# Only directly after the lead-in (up to two plain words), never across 'and', a comma or a clause.
ASSOCIATIVE_LEAD = re.compile(
    r"\b(?:associat(?:es?|ed|ing)\s+(?:[\w-]+\s+){0,3}?with|association with|linked to|correlated with|"
    r"correlation with|consistent with)\s+"
    r"(?:(?!and\b|or\b|but\b|which\b|that\b|while\b|whereas\b)[\w-]+\s+){0,2}$")
# 'X induces Y is not identified here': the claim is mentioned, then denied. Only when no comma or
# semicolon sits between the verb and the denial ('X induces Y, which is not established in humans' is asserted).
DENIED = re.compile(r"\b(?:is|are|was|were|has been|have been|remains?)\s+not\s+(?:yet\s+)?(?:been\s+)?"
                    r"(?:identified|established|shown|demonstrated|observed|found)\b")
# 'the Treg increase itself', 'a decrease': the change word is a noun here
# ponytail: also masks 'the data increase Tregs' (a plural verb after one word); rare, and a verb check needs a parser
NOUN_CHANGE = re.compile(r"\b(?:the|an?|this|that|its|their|any)\s+(?:[\w-]+\s+)?(?:increase|decrease|reduction|elevation)\b")
# what follows these is mentioned, not asserted: 'do not establish that X induces Y', 'would need to test'
NOT_ASSERTED = re.compile(r"\b(?:do|does|did|can|could)\s*not\s+(?:establish|show|demonstrate|prove|support|"
                          r"confirm|indicate|imply|identify|reveal)\b|\bneeds? to\b|\bwhether\b")


# 'Source-trace microbial butyrate while measuring Treg induction, and test whether ...': a proposed
# experiment, not a finding. Only at the start of the sentence.
IMPERATIVE = re.compile(r"^\W*(?:source-trace|trace|test|measure|compare|knock\w*|block|delete|run|perform|treat|"
                        r"repeat|assess|determine|quantify|isolate|stratify|randomi[sz]e)\b")


def sentence_tier(sentence: str) -> int:
    """Strongest verb tier; a verb negated within three words reads as a null statement."""
    s, best = re.sub(r"\[[A-Za-z0-9_\-]+\]", " ", sentence.lower()), 0
    if IMPERATIVE.match(s):
        return 0
    s = re.sub(r"\*+|(?<!\w)_|_(?!\w)", "", s)          # markdown emphasis: '*induced* Tregs' is a cell name
    s = NOUN_CHANGE.sub(" ", NOUN_INDUCED.sub(r"\1 ", s))
    if m := NOT_ASSERTED.search(s):
        s = s[:m.start()]
    denied = DENIED.search(s)
    for tier, pats in VERB_TIER.items():
        for p in pats:
            for m in re.finditer(p, s):
                before = " ".join(s[:m.start()].split()[-3:])
                associative = NEGATION.search(before) or ASSOCIATIVE_LEAD.search(s[:m.start()]) or (
                    denied and m.end() < denied.start() and not re.search(r"[,;:]", s[m.end():denied.start()]))
                best = max(best, 1 if associative else tier)
    return best


def claim_cap(c) -> int:
    cap = MAX_TIER.get(c.grade, 1)
    return min(cap, 1) if c.relation_norm in ASSOCIATIVE_RELATIONS else cap


def prose_sentences(text: str) -> list[str]:
    lines, fence = [], False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            fence = not fence
            continue
        if fence or not line or line.startswith("#") or line.startswith("|"):
            continue
        lines.append(re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", " ".join(lines)) if s.strip()]


def verify_text(text: str, claims: list, required_tags: list[str]) -> dict:
    by_id = {c.id: c for c in claims}
    uncited, overclaims, unknown = [], [], []
    sentences = prose_sentences(text)
    for sent in sentences:
        tags = re.findall(r"\[([A-Za-z0-9_\-]+)\]", sent)
        ids = [t for t in tags if not STRUCTURE_TAG.match(t)]
        if not ids:
            if "NO_EVIDENCE" in tags:
                if sentence_tier(sent) > 1:
                    overclaims.append({"sentence": sent, "used": sentence_tier(sent), "allowed": 1})
            elif len(sent.split()) >= 5:
                uncited.append(sent)
            continue
        unknown += [i for i in ids if i not in by_id]
        known = [by_id[i] for i in ids if i in by_id]
        if known:
            allowed, used = min(claim_cap(c) for c in known), sentence_tier(sent)
            if used > allowed:
                overclaims.append({"sentence": sent, "used": used, "allowed": allowed})
    missing = [t for t in required_tags if f"[{t}]" not in text]
    return {"n_sentences": len(sentences), "uncited": uncited, "overclaims": overclaims,
            "unknown_ids": sorted(set(unknown)), "missing_tags": missing,
            "passed": not (uncited or overclaims or unknown or missing)}

