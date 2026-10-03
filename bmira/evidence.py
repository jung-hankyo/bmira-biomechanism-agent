"""Claim grading (pure code) and verification of the written synthesis.

Grading is cheap deterministic arithmetic over fields the extractor already produced, so
every claim is graded. (V11 deferred it "to save cost"; there was no cost to save, and the
deferral left most claims ungraded in the synthesis ledger.)
"""
import re

from bmira.schemas import (ASSOCIATIVE_RELATIONS, CAUSAL_RELATIONS, DIRECTION, LABEL,
                           NULL_RELATIONS, PHYSICAL_RELATIONS)

DIRECTNESS = {"observation": 3, "author_interpretation": 2, "mechanistic_speculation": 1}
CAUSAL_BASE = {"none": 0, "genetic_association": 1, "pharmacological": 2, "environmental": 2,
               "knockdown": 2, "knockout": 3, "overexpression": 2, "transfer": 2}
DESIGN = {"meta_analysis": 3, "human_rct": 3, "human_cohort": 3, "human_primary": 3,
          "human_crosssectional": 2, "organoid_ipsc": 2, "animal": 2, "computational_cohort": 2,
          "cell_line": 1, "in_silico": 1, "review": 1}


def causal_support(c) -> int | None:
    if c.relation_norm not in CAUSAL_RELATIONS:
        return None                                # not applicable, not "zero evidence"
    s = CAUSAL_BASE.get(c.perturbation_class, 0)
    if s >= 2 and (c.rescue_arm or c.orthogonal_validation):
        s = 3
    if c.perturbation_class == "pharmacological" and not (c.rescue_arm or c.orthogonal_validation):
        s = min(s, 2)                              # off-target confounding
    if not c.comparator_present:
        s = min(s, 1)
    return s


def grade_claim(c):
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
        r"\babolish\w*", r"\bmediat\w*", r"\brequired for\b", r"\bnecessary for\b",
        r"\bsufficient\b", r"\blead(?:s|ing)? to\b"],
}
NEGATION = re.compile(r"\b(?:not|no|never|without|fail(?:s|ed)? to|did not|does not|do not)\b")
MAX_TIER = {"ungraded": 1, "weak": 1, "moderate": 3, "strong": 4}
STRUCTURE_TAG = re.compile(r"^(?:[HL]\d+|NO_EVIDENCE)$")


def sentence_tier(sentence: str) -> int:
    """Strongest verb tier; a verb negated within three words reads as a null statement."""
    s, best = sentence.lower(), 0
    for tier, pats in VERB_TIER.items():
        for p in pats:
            for m in re.finditer(p, s):
                before = " ".join(s[:m.start()].split()[-3:])
                best = max(best, 1 if NEGATION.search(before) else tier)
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

