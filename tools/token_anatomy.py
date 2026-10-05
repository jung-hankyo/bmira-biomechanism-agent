"""HY-3: where a run's tokens and dollars went, from session files (and their saved states).

    python tools/token_anatomy.py runs/session_*.json            # one table per session file
    python tools/token_anatomy.py runs/pilot5.json --json        # machine-readable

Per run: cost and token share of every LLM task, reasoning tokens, extraction tokens per paper read,
reads and re-reads, characters per read by text access, reviews extracted, and claim types. The
state file <session>_q<n>.state.json next to a session file adds the paper-level numbers when the
session itself predates them. Read-only; no network.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _share(a, b):
    return round(a / b, 3) if b else None


def from_state(path: Path) -> dict:
    """Paper and claim numbers from a saved run state (source texts are blanked there)."""
    st = json.loads(path.read_text(encoding="utf-8"))["state"]
    read = set(st.get("extracted_pmids", []))
    papers = [p for p in st.get("papers", []) if p["pmid"] in read]
    reads = sum(p.get("n_reads", 0) for p in papers)
    by_access = {}
    for acc in ("full_text", "abstract_only"):
        ps = [p for p in papers if p.get("text_access") == acc]
        n = sum(p.get("n_reads", 0) for p in ps)
        by_access[acc] = {"papers": len(ps), "chars_per_read": round(sum(p.get("chars_read", 0) for p in ps) / n)
                          if n else None}
    claims = st.get("claims", [])
    return {"papers_read": len(papers), "reads_recorded": reads,
            "re_reads": sum(max(0, p.get("n_reads", 0) - 1) for p in papers),
            "by_access": by_access, "reviews_extracted": sum(p.get("study_type") == "review" for p in papers),
            "claim_types": dict(Counter(c.get("claim_type") for c in claims).most_common()),
            "claims": len(claims)}


def anatomy(run: dict, state: Path | None = None) -> dict:
    llm = run.get("llm", {})
    tasks = llm.get("per_task", {})
    cost = sum(v.get("est_cost_usd") or 0 for v in tasks.values())
    tokens = sum(v.get("tokens_in", 0) + v.get("tokens_out", 0) for v in tasks.values())
    papers = run.get("papers", {})
    extract = tasks.get("extract", {})
    out = {
        "question_number": run.get("question_number"), "status": run.get("status"),
        "est_cost_usd": round(cost, 4) if cost else llm.get("est_cost_usd"),
        "cost_complete": llm.get("cost_complete"),
        "tokens": tokens, "tokens_reasoning": llm.get("total_tokens_reasoning", 0),
        "tasks": {t: {"cost_share": _share(v.get("est_cost_usd") or 0, cost),
                      "token_share": _share(v.get("tokens_in", 0) + v.get("tokens_out", 0), tokens),
                      "calls": v.get("calls", 0), "tokens_in": v.get("tokens_in", 0),
                      "tokens_out": v.get("tokens_out", 0), "tokens_reasoning": v.get("tokens_reasoning", 0),
                      "est_cost_usd": v.get("est_cost_usd")}
                  for t, v in sorted(tasks.items(), key=lambda kv: -(kv[1].get("est_cost_usd") or 0))},
        "extraction": {"papers_read": papers.get("extracted"),
                       "tokens_in_per_paper": _share(extract.get("tokens_in", 0), papers.get("extracted")),
                       "reasoning_per_paper": _share(extract.get("tokens_reasoning", 0), papers.get("extracted")),
                       "reads": papers.get("extraction_reads"), "re_reads": papers.get("re_reads"),
                       "chars_per_read": papers.get("chars_per_read"),
                       "reviews_extracted": papers.get("reviews_extracted"),
                       "full_text_share": papers.get("full_text_share_of_extracted")},
        "claim_types": run.get("extraction", {}).get("claim_types"),
    }
    if state and state.exists():
        out["state"] = from_state(state)
    return out


def analyse(session_path: Path) -> list[dict]:
    data = json.loads(session_path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    return [anatomy(r, session_path.with_name(f"{session_path.stem}_q{r.get('question_number')}.state.json"))
            for r in runs]


def markdown(name: str, rows: list[dict]) -> str:
    lines = [f"## {name}"]
    for r in rows:
        x = r["extraction"]
        lines += [f"\n### Q{r['question_number']} ({r['status']}): ${r['est_cost_usd']} "
                  f"({'complete' if r['cost_complete'] else 'prices missing for some models'}), "
                  f"{r['tokens']} tokens, {r['tokens_reasoning']} reasoning",
                  "", "| Task | Cost share | Token share | Calls | In | Out | Reasoning |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {t} | {v['cost_share']} | {v['token_share']} | {v['calls']} | {v['tokens_in']} | "
                  f"{v['tokens_out']} | {v['tokens_reasoning']} |" for t, v in r["tasks"].items()]
        lines.append(f"\nExtraction: {x['papers_read']} papers, {x['tokens_in_per_paper']} input and "
                     f"{x['reasoning_per_paper']} reasoning tokens per paper; reads {x['reads']} "
                     f"(re-reads {x['re_reads']}); chars per read {x['chars_per_read']}; reviews extracted "
                     f"{x['reviews_extracted']}; full-text share {x['full_text_share']}.")
        lines.append(f"Claim types: {r['claim_types']}")
        if "state" in r:
            lines.append(f"From the saved state: {r['state']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", type=Path, nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    result = {str(p): analyse(p) for p in a.sessions}
    if a.json:
        print(json.dumps(result, indent=1))
    else:
        print("\n\n".join(markdown(k, v) for k, v in result.items()))
    return result


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
