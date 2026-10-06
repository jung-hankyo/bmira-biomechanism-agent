"""A/B the extraction reasoning effort on a fixed set of saved papers (live model, no search).

    python -m bmira.ab_extract                                  # papers read in BOTH pilot5 and pilot6
    python -m bmira.ab_extract --reps 2 --efforts medium low --limit 16 --out runs/ab_extract.json
    python -m bmira.ab_extract --states runs/pilot7_q1.state.json --pmids 24412617 38319728 \
        --efforts medium --out runs/ab_blocking.json            # chosen papers; question from the state

Each paper is extracted `reps` times per effort from the same stored text. Two repeats of ONE effort
measure run-to-run noise; the gap between the efforts is only meaningful beyond that noise. Reports claims,
nulls, required_for, method fields kept, cost, and claim-set overlap (same subject/object pairs). The output
file also lists every claim (`claims`) and the blocking tests among them (`blocking_tests`).
"""
import argparse
import contextlib
import io
import json
import os
import statistics as st
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from bmira import Settings
from bmira.graph import extract
from bmira.llm import LangChainLLM
from bmira.normalize import lookup_key
from bmira.schemas import Paper
from bmira.telemetry import estimate_cost

ROOT = Path(__file__).resolve().parent.parent
NULLS = {"no_effect", "not_associated"}


def load(states, settings, cache, pmids=None):
    """Papers read in ALL the saved runs (or the given `pmids`), with their real text. State files blank
    `source_text` to stay small, so the text is fetched again (PubMed abstract, Europe PMC full text when
    open) and cached, so that both arms and every repeat read identical text."""
    from bmira.sources import PubMedSource
    docs = [json.load(open(p, encoding="utf-8"))["state"] for p in states]
    shared = sorted(pmids) if pmids else sorted(set.intersection(*[{c["pmid"] for c in d["claims"]} for d in docs]))
    if cache.exists():
        papers = [Paper.model_validate(x) for x in json.loads(cache.read_text(encoding="utf-8"))]
        if [p.pmid for p in papers] == shared:
            return docs[-1]["question"], papers
    src = PubMedSource(settings)
    papers = [src.fulltext(p) for p in src.fetch(shared)]
    cache.write_text(json.dumps([p.model_dump() for p in papers], ensure_ascii=False), encoding="utf-8")
    return docs[-1]["question"], papers


def run_once(rt, paper, question):
    payload = {"paper": paper, "question": question, "focus": [], "focus_keys": [], "n_existing": 0,
               "seen": set(), "round": 1}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = extract(payload, rt)
    if "failed" in buf.getvalue():            # extract() swallows model errors and returns {}
        raise RuntimeError(buf.getvalue().strip()[:200])
    return out.get("claims", []), out.get("dropped_claims", [])


def summarize(arm):
    kept = [c for r in arm for c in r["claims"]]
    n = max(len(arm), 1)
    return {"papers_x_reps": len(arm), "claims_per_paper": round(len(kept) / n, 2),
            "dropped_per_paper": round(sum(len(r["dropped"]) for r in arm) / n, 2),
            "null_claims_per_paper": round(sum(c.relation_norm in NULLS or c.relation in NULLS
                                               or any(w in c.relation.lower() for w in ("no ", "not ", "did not", "failed"))
                                               for c in kept) / n, 2),
            "knockout_claims_per_paper": round(sum(c.perturbation_class == "knockout" for c in kept) / n, 2),
            "rescue_orthogonal_comparator_per_paper": round(sum(c.rescue_arm + c.orthogonal_validation + c.comparator_present
                                                                for c in kept) / n, 2),
            "method_checks_reset_per_paper": round(sum(bool(c.method_checks) for c in kept) / n, 2),
            # EM-1/2: blocking tests the extractor recorded, and those the code checks then dropped
            "blocking_tests_per_paper": round(sum(bool(c.effect_exposure) for c in kept) / n, 2),
            "blocking_tests_dropped_per_paper": round(sum(bool(c.effect_exposure) for r in arm for c in r["dropped"]) / n, 2),
            "tokens_out": sum(r["out"] for r in arm), "tokens_reasoning": sum(r["reason"] for r in arm),
            "est_cost_usd": round(sum(r["cost"] for r in arm), 4)}


def listing(arm, only_blocking=False) -> list[dict]:
    """Claims of every read, kept or dropped, for reading by hand against their quotes. A blocking test the model
    did not record shows here as an ordinary claim (or is absent) beside the sentence that carried it."""
    return [{"pmid": c.pmid, "rep": r.get("rep"), "kept": not c.drop_reason, "drop_reason": c.drop_reason,
             "subject": c.subject, "relation": c.relation, "object": c.object, "effect_exposure": c.effect_exposure,
             "effect_result": c.effect_result, "perturbation": c.perturbation_class, "span": c.span[:240]}
            for r in arm for c in r["claims"] + r["dropped"] if c.effect_exposure or not only_blocking]


def blocking_tests(arm) -> list[dict]:
    """Every blocking-test claim, kept or dropped."""
    return listing(arm, only_blocking=True)


def pairs(r):
    return {f"{lookup_key(c.subject)}|{lookup_key(c.object)}" for c in r["claims"]}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", type=Path, nargs="+",
                    default=[ROOT / "runs" / "pilot5_q1.state.json", ROOT / "runs" / "pilot6_q1.state.json"])
    ap.add_argument("--efforts", nargs="+", default=["medium", "low"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--pmids", nargs="+", help="read these papers instead of those shared by --states")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "ab_extract.json")
    a = ap.parse_args(argv)

    s = Settings(provider=a.provider)
    question, papers = load(a.states, s, a.out.with_suffix(".texts.json"), a.pmids)
    papers = [p for p in papers if len(p.source_text) > 500][:a.limit]
    if a.pmids and (missing := sorted(set(a.pmids) - {p.pmid for p in papers})):
        print(f"[WARN] no usable text for {missing}; they are skipped")
    if not papers:
        raise SystemExit("no paper text could be fetched (network? NCBI_EMAIL?); nothing to extract")
    print("text: " + ", ".join(f"{p.pmid}={p.text_access[:4]}/{len(p.source_text)}" for p in papers))
    llm = LangChainLLM(s, os.environ.get(f"{a.provider.upper()}_API_KEY"))
    rt = SimpleNamespace(settings=s, llm=llm, source=SimpleNamespace(fulltext=lambda p: p))
    model = s.models[a.provider]["reasoning"]
    print(f"{len(papers)} papers x {a.reps} reps x {a.efforts} = {len(papers) * a.reps * len(a.efforts)} extractions "
          f"on {model}; expect about ${0.025 * len(papers) * a.reps * len(a.efforts):.1f}")
    arms, failed = {e: [] for e in a.efforts}, 0
    for rep in range(a.reps):
        for e in a.efforts:
            s.reasoning_effort["extract"] = e
            for p in papers:
                i0, o0, r0 = (llm.tokens_in["extract"], llm.tokens_out["extract"], llm.tokens_reasoning["extract"])
                try:
                    kept, dropped = run_once(rt, p, question)
                except Exception as ex:
                    print(f"[{e} rep{rep} {p.pmid}] failed: {ex}")
                    failed += 1
                    if failed >= 4 and not any(arms.values()):
                        raise SystemExit("4 extractions failed and none succeeded; stopping before more spend")
                    continue
                d_in, d_out = llm.tokens_in["extract"] - i0, llm.tokens_out["extract"] - o0
                arms[e].append({"pmid": p.pmid, "rep": rep, "claims": kept, "dropped": dropped, "out": d_out,
                                "reason": llm.tokens_reasoning["extract"] - r0,
                                "cost": estimate_cost(model, d_in, d_out, s.prices) or 0.0})
            print(f"rep {rep} {e}: {sum(len(r['claims']) for r in arms[e] if r['rep'] == rep)} claims", flush=True)

    result = {"model": model, "papers": [p.pmid for p in papers], "arms": {e: summarize(v) for e, v in arms.items()},
              "blocking_tests": {e: blocking_tests(v) for e, v in arms.items()},
              "claims": {e: listing(v) for e, v in arms.items()}}
    # overlap of the claimed subject/object pairs, paper by paper: within one effort = noise, across = effort
    within, across = {e: [] for e in a.efforts}, []
    for p in papers:
        by = {e: [pairs(r) for r in arms[e] if r["pmid"] == p.pmid] for e in a.efforts}
        for e in a.efforts:
            within[e] += [jaccard(x, y) for x, y in combinations(by[e], 2)]
        for e1, e2 in combinations(a.efforts, 2):
            across += [jaccard(x, y) for x in by[e1] for y in by[e2]]
    mean = lambda v: round(st.mean(v), 3) if v else None
    result["pair_overlap"] = {"within_" + e: mean(v) for e, v in within.items()} | {"across_efforts": mean(across)}
    a.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "claims"}, indent=1, ensure_ascii=False))
    print(f"\nwritten: {a.out} (every claim of every read is under 'claims')\nRead it: if across_efforts is about as high as the within_* values, the effort "
          f"changes little beyond noise; compare claims_per_paper, null_claims and knockout_claims between arms.")


if __name__ == "__main__":
    main()
