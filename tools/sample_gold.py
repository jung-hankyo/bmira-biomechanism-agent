"""EV-2/EV-3: label forms for the gold sets, sampled from saved runs. The owner fills in `labels`.

    python -m tools.sample_gold claims runs/pilot5_q1.state.json runs/pilot6_q1.state.json --n 200
    python -m tools.sample_gold papers runs/pilot*_q1.state.json --n 150
    python -m tools.sample_gold report runs/pilot*_q1.state.json
    python -m tools.sample_gold disagreements runs/pilot5_shadow.jsonl --agree-share 0.2

Each line: {"item_id", "source", "context": {what the labeler reads}, "today": {current decisions},
"labels": {task: null}}. Label keys are the judge-log task names, so `python -m bmira.eval` scores the
filled file directly. Never put gold items into prompts or few-shot examples (gold leakage).
Sampling is seeded: the same inputs give the same form.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from bmira.evidence import prose_sentences, sentence_tier
from bmira.normalize import _previous_sentence
from bmira.shadow import _sentence_id

CLAIM_LABELS = ["stance", "relation", "methods.comparator", "methods.perturbation", "methods.rescue",
                "methods.orthogonal", "subject_lost", "system", "claim_type", "blocking_test"]
PAPER_LABELS = ["screen.include", "screen.original_data"]
REPORT_LABELS = ["report.overclaim", "report.entailment", "report.strength"]


def _state(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["state"]


def strata(c: dict) -> list[str]:
    """Every stratum a claim belongs to (EV-2): relation type, nulls, required_for, subject_lost,
    perturbed, abstract vs full text, claim type."""
    out = [f"relation:{c.get('relation_norm') or 'pending'}", f"access:{c.get('text_access')}",
           f"type:{c.get('claim_type')}"]
    if c.get("relation_norm") in {"no_effect", "not_associated"}:
        out.append("null")
    if c.get("subject_lost"):
        out.append("subject_lost")
    if c.get("perturbation_class", "none") != "none":
        out.append("perturbed")
    if c.get("effect_exposure"):
        out.append("blocking_test")
    return out


def stratified(items: list, keys, n: int, seed: int) -> list:
    """Round-robin over strata (rarest first), each stratum shuffled: rare kinds are never crowded out."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        for k in keys(it):
            groups[k].append(it)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups, key=lambda k: (len(groups[k]), k))
    picked, seen = [], set()
    while len(picked) < n and any(groups[k] for k in order):
        for k in order:
            while groups[k]:
                it = groups[k].pop()
                if id(it) not in seen:
                    seen.add(id(it))
                    picked.append(it)
                    break
            if len(picked) >= n:
                break
    return picked


def claims_form(paths, n=200, seed=7) -> list[dict]:
    items = []
    for path in paths:
        st = _state(path)
        texts = {p["pmid"]: p.get("source_text") or p.get("abstract", "") for p in st.get("papers", [])}
        items += [(str(path), c, texts.get(c["pmid"], "")) for c in st.get("claims", [])]
    rows = []
    for path, c, text in stratified(items, lambda it: strata(it[1]), n, seed):
        rows.append({"item_id": c["id"], "source": path, "pmid": c["pmid"],
                     "context": {"claim": f"{c['subject']} | {c['relation']} | {c['object']}", "quote": c["span"],
                                 "previous_sentence": _previous_sentence(c["span"], text),
                                 "methods": c.get("methods_span", ""),
                                 "effect_exposure": c.get("effect_exposure", "")},
                     "today": {k: c.get(k) for k in ("relation_norm", "comparator_present", "perturbation_class",
                                                     "rescue_arm", "orthogonal_validation", "subject_lost",
                                                     "system", "claim_type", "grade", "method_checks")},
                     "strata": strata(c), "labels": dict.fromkeys(CLAIM_LABELS)})
    return rows


def papers_form(paths, n=150, seed=7) -> list[dict]:
    items = [(str(path), p) for path in paths for p in _state(path).get("papers", [])
             if p.get("screen_status") in {"included", "excluded"}]
    keys = lambda it: [f"status:{it[1]['screen_status']}", f"design:{it[1].get('study_type')}",
                       "targeted" if it[1].get("retrieved_for") else "coverage"]
    return [{"item_id": p["pmid"], "source": path,
             "context": {"title": p.get("title", ""), "abstract": p.get("abstract", ""),
                         "publication_types": p.get("publication_types", [])},
             "today": {"screen_status": p["screen_status"], "relevance_score": p.get("relevance_score"),
                       "study_type": p.get("study_type")},
             "labels": dict.fromkeys(PAPER_LABELS)} for path, p in stratified(items, keys, n, seed)]


def report_form(paths) -> list[dict]:
    """Every sentence of every report (no sampling: reports are short)."""
    rows = []
    for path in paths:
        st = _state(path)
        v = st.get("verification", {})
        flagged = {o["sentence"] for o in v.get("overclaims", [])}
        ent = {e.get("sentence"): e.get("verdict") for e in v.get("entailment", [])}
        for s in prose_sentences(st.get("synthesis", "")):
            rows.append({"item_id": _sentence_id(s), "source": str(path), "context": {"sentence": s},
                         "today": {"overclaim": s in flagged, "tier": sentence_tier(s), "entailment": ent.get(s)},
                         "labels": dict.fromkeys(REPORT_LABELS)})
    return rows


def disagreements_form(logs, agree_share=0.2, seed=7) -> list[dict]:
    """EV-3 active labeling: every item where today and the judge disagree, plus a share of agreements."""
    rng = random.Random(seed)
    rows = []
    for path in logs:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            e = json.loads(line) if line.strip() else None
            if not e or e.get("current") is None or e.get("jev") is None:
                continue
            if e["current"] != e["jev"] or rng.random() < agree_share:
                rows.append({"item_id": e["item_id"], "source": str(path), "task": e["task"],
                             "context": {k: v for k, v in e.items() if k in {"text", "step", "drop_reason"}},
                             "today": e["current"], "judge": e["jev"], "disagree": e["current"] != e["jev"],
                             "labels": {e["task"]: None}})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["claims", "papers", "report", "disagreements"])
    ap.add_argument("inputs", type=Path, nargs="+", help="state files (judge logs for 'disagreements')")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--agree-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.kind == "claims":
        rows = claims_form(a.inputs, a.n or 200, a.seed)
    elif a.kind == "papers":
        rows = papers_form(a.inputs, a.n or 150, a.seed)
    elif a.kind == "report":
        rows = report_form(a.inputs)
    else:
        rows = disagreements_form(a.inputs, a.agree_share, a.seed)
    out = a.out or Path("eval") / "forms" / f"{a.kind}_gold.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} items -> {out}")
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
