"""Score executors against owner-labeled gold sets (handoff section 10).

    python -m bmira.eval --labels eval/claims_gold.jsonl --log runs/pilot5_shadow.jsonl
    python -m bmira.eval --labels eval/claims_gold.jsonl --log runs/pilot5_shadow.jsonl --task stance
    python -m bmira.eval --mechanisms eval/mechanisms_gold.json --state runs/pilot8_q1.state.json

Gold files are JSONL, one item per line: {"item_id": ..., "labels": {task: label, ...}, ...context}.
Labels are true/false for yes/no tasks, an option name for Choice tasks, a level for Score tasks;
null means "not labeled" and is skipped. The judge log (bmira.shadow, or `judge_log` in a state) gives
today's decision (`current`) and the judge's (`jev`, `probability`) per item; both are scored.

Metrics per task: accuracy; for yes/no tasks precision, recall and F1, and for the judge AUROC,
expected calibration error and the thresholds that reach a target precision or recall; for Choice
and Score tasks a confusion table and (Score) mean absolute error. Extraction comparisons need at
least two repeats (claim sets overlap only 0.58-0.60 between repeats); differences inside that
spread are not effects.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise ValueError(f"{path}:{n}: not JSON ({e})") from None
    return rows


def gold_labels(rows: list[dict]) -> dict:
    """(task, item_id) -> label, skipping unlabeled (null) entries."""
    out = {}
    for r in rows:
        for task, label in (r.get("labels") or {}).items():
            if label is not None:
                out[(task, str(r["item_id"]))] = label
    return out


# ── metrics ─────────────────────────────────────────────────────────────────
def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Probability that a random positive outranks a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def ece(probs: list[float], correct: list[bool], bins: int = 10) -> float | None:
    """Expected calibration error: |accuracy - confidence| per probability bin, weighted by bin size."""
    if not probs:
        return None
    groups = defaultdict(list)
    for p, c in zip(probs, correct):
        groups[min(bins - 1, int(p * bins))].append((p, c))
    return round(sum(len(g) / len(probs) * abs(sum(c for _, c in g) / len(g) - sum(p for p, _ in g) / len(g))
                     for g in groups.values()), 4)


def prf(pred: list[bool], labels: list[bool]) -> dict:
    tp = sum(p and y for p, y in zip(pred, labels))
    fp = sum(p and not y for p, y in zip(pred, labels))
    fn = sum(not p and y for p, y in zip(pred, labels))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    r = lambda x: None if x is None else round(x, 4)
    return {"precision": r(precision), "recall": r(recall), "f1": r(f1), "tp": tp, "fp": fp, "fn": fn}


def threshold_for(probs: list[float], labels: list[bool], precision: float | None = None,
                  recall: float | None = None) -> float | None:
    """Give exactly one target. The lowest threshold whose precision reaches `precision` (most recall
    at that precision), or the highest whose recall reaches `recall`. None if no threshold does. Use on
    gold labels only, never on the items a prompt was tuned on."""
    if (precision is None) == (recall is None):
        raise ValueError("give exactly one of precision, recall")
    metric, target = ("precision", precision) if precision is not None else ("recall", recall)
    ok = [t for t in sorted(set(probs))
          if (v := prf([p >= t for p in probs], labels)[metric]) is not None and v >= target]
    return (min(ok) if metric == "precision" else max(ok)) if ok else None


def score_task(task: str, entries: list[dict], gold: dict) -> dict:
    """Both executors on one task: `current` (today) and `jev` (the judge), against the gold labels."""
    rows = [(e, gold[(task, str(e["item_id"]))]) for e in entries if (task, str(e["item_id"])) in gold]
    out = {"task": task, "labeled": len(rows)}
    if not rows:
        return out
    binary = all(isinstance(y, bool) for _, y in rows)
    for who in ("current", "jev"):
        pairs = [(e[who], y) for e, y in rows if e.get(who) is not None]
        if not pairs:
            out[who] = None
            continue
        res = {"n": len(pairs), "accuracy": round(sum(p == y for p, y in pairs) / len(pairs), 4)}
        if binary:
            res.update(prf([bool(p) for p, _ in pairs], [y for _, y in pairs]))
        else:
            res["confusion"] = {f"{y} -> {p}": n for (p, y), n in Counter(pairs).most_common() if p != y}
            if all(isinstance(p, (int, float)) and isinstance(y, (int, float)) for p, y in pairs):
                res["mean_abs_error"] = round(sum(abs(p - y) for p, y in pairs) / len(pairs), 4)
        out[who] = res
    probs = [(e["probability"], e["jev"], y) for e, y in rows if e.get("probability") is not None]
    if probs and binary:                         # yes/no tasks log P(yes) as the probability
        p_yes, ys = [p for p, _, _ in probs], [y for _, _, y in probs]
        out["jev"].update({"auroc": auroc(p_yes, ys), "ece": ece(p_yes, [(p >= 0.5) == y for p, y in zip(p_yes, ys)]),
                           "threshold_precision_0.95": threshold_for(p_yes, ys, precision=0.95),
                           "threshold_recall_0.95": threshold_for(p_yes, ys, recall=0.95)})
    elif probs:
        out["jev"]["ece"] = ece([p for p, _, _ in probs], [j == y for _, j, y in probs])
    return out


def evaluate(log: list[dict], gold_rows: list[dict], tasks=None) -> list[dict]:
    gold = gold_labels(gold_rows)
    by_task = defaultdict(list)
    for e in log:
        by_task[e["task"]].append(e)
    wanted = tasks or sorted({t for t, _ in gold} & set(by_task))
    return [score_task(t, by_task.get(t, []), gold) for t in wanted]


# ── mechanisms ──────────────────────────────────────────────────────────────
def score_mechanisms(gold: dict, state: dict, question: str | None = None) -> dict:
    """Route recall against CONFIRMED expected routes of one question. A route matches when every
    intermediate it names appears, in order, among one pathway's node labels (synonyms allowed).
    Draft entries are never scored."""
    from bmira.normalize import lookup_key
    q = question or state.get("question", "")
    entry = next((g for g in gold.get("questions", []) if lookup_key(g["question"]) == lookup_key(q)), None)
    if not entry:
        return {"question": q, "status": "no gold entry"}
    if entry.get("status") != "confirmed":
        return {"question": q, "status": f"gold entry is {entry.get('status', 'draft')}; not scored"}
    links = state.get("links", {})

    def label(x, side):
        x = x if isinstance(x, dict) else x.model_dump()
        return lookup_key(x[f"{side}_label"])

    paths = []
    for h in state.get("hypotheses", []):
        h = h if isinstance(h, dict) else h.model_dump()
        ks = [k for k in h["links"] if k in links]
        nodes = [label(links[ks[0]], "subject")] + [label(links[k], "object") for k in ks] if ks else []
        paths.append((h["id"], h["status"], nodes))
    found = []
    for route in entry.get("routes", []):
        names = [{lookup_key(n) for n in ([step] if isinstance(step, str) else step)} for step in route["via"]]
        hit = None
        for hid, status, nodes in paths:
            i = 0
            for node in nodes:
                if i < len(names) and node in names[i]:
                    i += 1
            if i == len(names):
                hit = (hid, status)
                break
        found.append({"route": route["name"], "matched": hit is not None, "pathway": hit and hit[0],
                      "verdict": hit and hit[1]})
    n = len(found)
    return {"question": q, "status": "scored", "routes": found,
            "route_recall": round(sum(f["matched"] for f in found) / n, 3) if n else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, help="gold JSONL")
    ap.add_argument("--log", type=Path, nargs="*", default=[], help="judge logs (JSONL from bmira.shadow)")
    ap.add_argument("--state", type=Path, nargs="*", default=[], help="saved run states (their judge_log, or "
                    "the pathways for --mechanisms)")
    ap.add_argument("--task", nargs="*")
    ap.add_argument("--mechanisms", type=Path, help="mechanisms gold (JSON)")
    a = ap.parse_args(argv)
    states = [json.loads(p.read_text(encoding="utf-8"))["state"] for p in a.state]
    if a.mechanisms:
        gold = json.loads(a.mechanisms.read_text(encoding="utf-8"))
        result = [score_mechanisms(gold, st) for st in states]
        print(json.dumps(result, indent=1, ensure_ascii=False))
        return result
    if not a.labels:
        ap.error("--labels is required unless --mechanisms is given")
    log = [e for p in a.log for e in read_jsonl(p)] + [e for st in states for e in st.get("judge_log", [])]
    result = evaluate(log, read_jsonl(a.labels), a.task)
    print("| Task | Labeled | Today: accuracy | Judge: accuracy | Judge AUROC | Judge ECE |\n|---|---|---|---|---|---|")
    for r in result:
        cur, jev = r.get("current") or {}, r.get("jev") or {}
        print(f"| {r['task']} | {r['labeled']} | {cur.get('accuracy')} | {jev.get('accuracy')} | "
              f"{jev.get('auroc')} | {jev.get('ece')} |")
    return result


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
