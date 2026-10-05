"""Shadow touchpoints: ask the judge what today's executor already decided, log both, change nothing.

Each function returns judge_log entries {task, item_id, current, jev, probability, ...}: `current` is
today's decision, `jev` the judge's answer read at the starting threshold in bmira/questions.py,
`probability` the judge's probability for that answer (P(yes) for yes/no tasks). Entries feed the
telemetry agreement table and, with owner labels, calibration (bmira.eval). No pipeline decision
reads them. With no judge (rt.judge is None) every function returns [] at once.

    python -m bmira.shadow runs/pilot5_q1.state.json            # every touchpoint over a saved run
    python -m bmira.shadow runs/pilot5_q1.state.json --offline  # wiring check with the surrogate

Touchpoints: JV-1/2 screen (include, original data, per-step findability), JV-4 stance,
JV-5 method fields, JV-6 subject_lost, JV-8 relation typing, JV-14 sentence strength, overclaim
and entailment, JV-15 blocking tests.
"""
import argparse
import functools
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from bmira import questions as Q
from bmira.evidence import claim_cap, is_proposal, prose_sentences, sentence_tier
from bmira.judge import expected_level, level_probs, p_yes, top
from bmira.normalize import _previous_sentence

POLARITY_DROPS = {"quote negates the claimed effect", "null claim but the quote reports an effect"}
BLOCKING_DROPS = {"blocking test without a removal or blocking perturbation", "blocking-test treatment not named in quote"}
LOSS_PERTURBATIONS = {"knockout", "knockdown", "pharmacological"}
STEP = re.compile(r"^reports_step_(\d+)$")


def entry(task, item_id, current, jev, probability, **extra) -> dict:
    return {"task": task, "item_id": item_id, "current": current, "jev": jev,
            "probability": None if probability is None else round(float(probability), 4), **extra}


def _safe(fn):
    """Shadow mode must never cost a run: any error inside a touchpoint is logged and swallowed."""
    @functools.wraps(fn)
    def wrapped(rt, *a, **k):
        if getattr(rt, "judge", None) is None:
            return []
        try:
            return fn(rt, *a, **k)
        except Exception as e:                    # noqa: BLE001 - a shadow check is never worth a failed run
            print(f"[shadow][WARN] {fn.__name__} skipped ({type(e).__name__}: {str(e)[:120]})")
            return []
    return wrapped


def _sentence_id(s: str) -> str:
    return "S" + hashlib.sha1(s.encode()).hexdigest()[:10]


def question_summary(p) -> str:
    return (f"Exposure: {p.exposure}" + (" (a decrease of it)" if p.exposure_change == "down" else "")
            + f"; outcome: {p.outcome}; readouts: {', '.join(p.outcome_readouts) or 'none'}; "
            f"population: {p.target_system}")


# ── JV-1, JV-2: screening ───────────────────────────────────────────────────
@_safe
def screen(rt, papers, parsed, step_labels: dict) -> list:
    """`papers`: screened this round (status and study type set). `step_labels`: pmid -> [(key, label)]."""
    papers = [p for p in papers if p.screen_status != "unscreened"]
    items = []
    for p in papers:
        steps = step_labels.get(p.pmid, [])
        qs = Q.screen_questions([label for _, label in steps])
        current = {"relevance": 3 if p.screen_status == "included" else 1,
                   "original_data": p.study_type != "review"}
        state = {"question": question_summary(parsed), "title": p.title, "abstract": p.abstract[:3000],
                 "publication_types": p.publication_types}
        items.append((state, qs, {"current": current}))
    out = []
    for p, (_, qs, _), a in zip(papers, items, rt.judge.ask_many("screen", items)):
        if a is None:
            continue
        probs = level_probs(a.get("relevance"), Q.SCREEN["relevance"]["criteria"])
        if probs:
            p_in = probs[3] + 0.5 * probs[2]
            out.append(entry("screen.include", p.pmid, p.screen_status == "included",
                             p_in >= Q.THRESHOLDS["screen_include"], p_in))
        if (po := p_yes(a.get("original_data"))) is not None:
            out.append(entry("screen.original_data", p.pmid, p.study_type != "review", po >= 0.5, po))
        steps = step_labels.get(p.pmid, [])
        for qid in qs:
            if (m := STEP.match(qid)) and (ps := p_yes(a.get(qid))) is not None:
                key, label = steps[int(m[1]) - 1]
                out.append(entry("screen.step", f"{p.pmid}|{key}", None, ps >= 0.5, ps, step=label))
    return out


# ── JV-4, JV-5, JV-6: claim checks ──────────────────────────────────────────
CLAIM_QUESTIONS = {**Q.STANCE, **Q.METHODS, **Q.SUBJECT_LOST}


@_safe
def claims(rt, paper, records) -> list:
    """`records`: (as extracted, after the code checks) per claim; dropped claims carry drop_reason.
    Kept claims get stance, method fields and subject_lost; claims dropped for polarity get stance."""
    items, meta = [], []
    for before, after in records:
        if after.drop_reason and after.drop_reason not in POLARITY_DROPS | BLOCKING_DROPS:
            continue                                  # quote not found / entity not named: no stance to judge
        prev = _previous_sentence(before.span, paper.source_text or paper.abstract)
        state = {"claim": f"{before.subject} | {before.relation} | {before.object}", "subject": before.subject,
                 "object": before.object, "quote": before.span, "previous_sentence": prev,
                 "methods": before.methods_span, "effect_exposure": before.effect_exposure}
        if after.drop_reason in BLOCKING_DROPS:      # JV-15: the code check rejected it as a blocking test
            qs, cur = {"blocking_test": Q.BLOCKING["blocking_test"]}, {"blocking_test": False}
        elif after.drop_reason:
            qs, cur = Q.STANCE, {"stance": "contradicts"}
        else:
            qs = dict(CLAIM_QUESTIONS)
            if not (before.subject_lost or before.perturbation_class in LOSS_PERTURBATIONS):
                qs.pop("subject_lost")                # loss is only plausible after a perturbation
            cur = {"stance": "supports", "comparator": after.comparator_present,
                   "perturbation": after.perturbation_class, "rescue": after.rescue_arm,
                   "orthogonal": after.orthogonal_validation, "subject_lost": after.subject_lost}
            if after.is_blocking_test:                # JV-15
                qs.update(Q.BLOCKING)
                cur.update(blocking_test=True, effect_result=after.effect_result)
        items.append((state, qs, {"current": cur}))
        meta.append((before, after, cur))
    out = []
    for (before, after, cur), a in zip(meta, rt.judge.ask_many("claims", items)):
        if a is None:
            continue
        cid = after.id
        if (pb := p_yes(a.get("blocking_test"))) is not None:
            out.append(entry("blocking_test", cid, cur.get("blocking_test"), pb >= 0.5, pb, drop_reason=after.drop_reason))
        choice, conf, probs = top(a.get("effect_result"))
        if choice:
            out.append(entry("blocking.effect_result", cid, cur.get("effect_result"), choice, probs.get(choice, conf),
                             probabilities=probs))
        choice, conf, probs = top(a.get("stance"))
        if choice:
            out.append(entry("stance", cid, cur["stance"], choice, probs.get(choice, conf),
                             probabilities=probs, drop_reason=after.drop_reason))
        if after.drop_reason:
            continue
        llm = {"comparator": before.comparator_present, "rescue": before.rescue_arm,
               "orthogonal": before.orthogonal_validation, "subject_lost": before.subject_lost}
        for qid in ("comparator", "rescue", "orthogonal", "subject_lost"):
            if (p := p_yes(a.get(qid))) is not None:
                thr = Q.THRESHOLDS["subject_lost_invert"] if qid == "subject_lost" else Q.THRESHOLDS["method_confirm"]
                out.append(entry(f"methods.{qid}" if qid != "subject_lost" else "subject_lost", cid,
                                 cur[qid], p >= thr, p, llm=llm[qid]))
        choice, conf, probs = top(a.get("perturbation"))
        if choice:
            out.append(entry("methods.perturbation", cid, cur["perturbation"], choice, probs.get(choice, conf),
                             probabilities=probs, llm=before.perturbation_class))
    return out


# ── JV-8: relation typing ───────────────────────────────────────────────────
@_safe
def relations(rt, typed) -> list:
    """`typed`: (claim, relation as typed from its wording, before any loss restatement, source)."""
    typed = [t for t in typed if t[1]]
    items = [({"subject": c.subject, "relation_wording": c.relation_raw or c.relation, "object": c.object,
               "quote": c.span}, Q.RELATION, {"current": {"relation": rel}}) for c, rel, _ in typed]
    out = []
    for (c, rel, source), a in zip(typed, rt.judge.ask_many("relation", items)):
        choice, conf, probs = top(None if a is None else a.get("relation"))
        if choice:
            out.append(entry("relation", c.id, rel, choice, probs.get(choice, conf), confidence=conf,
                             source=source, probabilities=probs))
    return out


# ── JV-14: report sentences ─────────────────────────────────────────────────
@_safe
def report(rt, text, claims, flagged: set, entailment: dict) -> list:
    """`flagged`: sentences the code verifier called overclaims. `entailment`: sentence -> verdict from
    today's entailment check, for the sentences it judged."""
    by_id = {c.id: c for c in claims}
    sents = prose_sentences(text)
    items, meta = [], []
    for s in sents:
        ids = [i for i in re.findall(r"\[([A-Za-z0-9_\-]+)\]", s) if i in by_id]
        cited = [{"id": i, "grade": by_id[i].grade, "relation": by_id[i].relation_norm,
                  "subject": by_id[i].subject_label, "object": by_id[i].object_label,
                  "system": by_id[i].system, "design": by_id[i].study_type} for i in ids]
        qs = dict(Q.STRENGTH)
        if s in entailment:
            qs.update(Q.ENTAILMENT)
        tier = sentence_tier(s)
        cap = min((claim_cap(by_id[i]) for i in ids), default=1 if "[NO_EVIDENCE]" in s else None)
        cur = {"strength": tier, "entailment": entailment.get(s)}
        items.append(({"sentence": s, "cited_claims": cited}, qs, {"current": cur}))
        meta.append((s, tier, cap, cur))
    out = []
    for (s, tier, cap, cur), a in zip(meta, rt.judge.ask_many("report", items)):
        if a is None:
            continue
        sid = _sentence_id(s)
        probs = level_probs(a.get("strength"), Q.STRENGTH["strength"]["criteria"])
        if probs:
            out.append(entry("report.strength", sid, tier, round(expected_level(probs)), max(probs),
                             text=s, probabilities=probs))
            if cap is not None and not is_proposal(s):
                p_over = sum(probs[cap + 1:])
                out.append(entry("report.overclaim", sid, s in flagged, p_over >= Q.THRESHOLDS["overclaim"],
                                 p_over, text=s, cap=cap))
        choice, conf, eprobs = top(a.get("entailment"))
        if choice:
            out.append(entry("report.entailment", sid, cur["entailment"], choice, eprobs.get(choice, conf),
                             text=s, probabilities=eprobs))
    return out


# ── summary ─────────────────────────────────────────────────────────────────
def agreement(log: list) -> dict:
    """Per task: items judged, items with a current decision, agreement rate between the two."""
    by = defaultdict(list)
    for e in log:
        by[e["task"]].append(e)
    out = {}
    for task, es in sorted(by.items()):
        both = [e for e in es if e["current"] is not None and e["jev"] is not None]
        agree = sum(e["current"] == e["jev"] for e in both)
        out[task] = {"judged": len(es), "compared": len(both), "agree": agree,
                     "rate": round(agree / len(both), 3) if both else None,
                     "disagreements": [{"item_id": e["item_id"], "current": e["current"], "jev": e["jev"],
                                        "probability": e["probability"]} for e in both if e["current"] != e["jev"]][:10]}
    return out


# ── replay over a saved run ─────────────────────────────────────────────────
def _reconstructed(c):
    """A saved claim as extracted: fields the cue check reset are restored from method_checks."""
    before = c.model_copy()
    reset = " ".join(c.method_checks)
    for f in ("rescue_arm", "orthogonal_validation", "comparator_present", "subject_lost"):
        if f"{f} not evidenced" in reset:
            setattr(before, f, True)
    if m := re.search(r"perturbation '(\w+)' not evidenced", reset):
        before.perturbation_class = m[1]
    return before


def typed_relation(c) -> str:
    """The relation as typed from the wording, undoing normalize's loss restatement ('Tet2 loss increases
    IL-6' is stored as Tet2 decreases IL-6). The restatement is its own inverse, so the same test undoes it."""
    from bmira.normalize import entity_change
    from bmira.schemas import DIRECTION
    lost = c.subject_lost or entity_change(c.subject) == "down"
    if c.relation_norm in DIRECTION and lost != (entity_change(c.object) == "down"):
        return "decreases" if c.relation_norm == "increases" else "increases"
    return c.relation_norm


def shadow_state(path, rt) -> list:
    """Every touchpoint over a saved run, without searching, extracting or calling the LLM."""
    from bmira.telemetry import load_state
    st = load_state(path, SimpleNamespace(pair_cache={}, conflict_cache={}, alias_verdicts={}))
    papers = {p.pmid: p for p in st.get("papers", [])}
    links = st.get("links", {})
    labels = {k: f"{ln.subject_label} --{ln.relation}--> {ln.object_label}" for k, ln in links.items()}
    steps = {p.pmid: [(k, labels[k]) for k in p.retrieved_for if k in labels] for p in papers.values()}
    log = screen(rt, list(papers.values()), st["parsed"], steps)
    by_paper = defaultdict(list)
    for c in st.get("claims", []) + st.get("dropped_claims", []):
        by_paper[c.pmid].append((_reconstructed(c), c))
    for pmid, records in by_paper.items():
        if pmid in papers:
            log += claims(rt, papers[pmid], records)
    log += relations(rt, [(c, typed_relation(c), c.relation_source) for c in st.get("claims", [])
                          if c.relation_source != "blocking_test"])          # typed from the result, not the wording
    if st.get("synthesis"):
        v = st.get("verification", {})
        flagged = {o["sentence"] for o in v.get("overclaims", [])}
        ent = {e["sentence"]: e["verdict"] for e in v.get("entailment", []) if e.get("sentence")}
        log += report(rt, st["synthesis"], st.get("claims", []), flagged, ent)
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", type=Path, nargs="+")
    ap.add_argument("--offline", action="store_true", help="surrogate judge: wiring check, no key or network")
    ap.add_argument("--out", type=Path, help="write the judge log as JSONL")
    ap.add_argument("--cache-dir", default="runs/cache")
    a = ap.parse_args(argv)
    from bmira.config import Settings
    from bmira.judge import SurrogateJudge, make_judge
    s = Settings(judge_provider="jev", cache_dir=a.cache_dir)
    judge = SurrogateJudge() if a.offline else make_judge(s)
    rt = SimpleNamespace(judge=judge, settings=s)
    log = []
    for path in a.states:
        part = shadow_state(path, rt)
        log += [{**e, "state": str(path)} for e in part]
        print(f"{path}: {len(part)} judgments")
    judge.save()
    if a.out:
        a.out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in log), encoding="utf-8")
    print("\n| Task | Judged | Compared | Agree | Rate |\n|---|---|---|---|---|")
    for task, r in agreement(log).items():
        print(f"| {task} | {r['judged']} | {r['compared']} | {r['agree']} | {r['rate']} |")
    cost = sum(judge.tokens_in.values()) * s.prices.get(s.judge_model, (0, 0))[0] / 1e6
    print(f"\njudge calls {sum(judge.calls.values())}, cache hits {sum(judge.cache_hits.values())}, "
          f"failures {sum(judge.failures.values())}, input tokens {sum(judge.tokens_in.values())}"
          + (f", est. ${cost:.4f}" if not a.offline else "") + (f"; disabled: {judge.disabled}" if judge.disabled else ""))
    return log


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
