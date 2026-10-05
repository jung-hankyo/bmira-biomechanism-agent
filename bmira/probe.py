"""Generalization probe: parse -> plan -> PubMed search for every question, nothing read (~$0.03 each).

    python -m bmira.probe                       # all of experiments/questions.txt
    python -m bmira.probe "Does exercise ..."   # your own questions
    python -m bmira.probe --offline             # wiring check

Shows what the agent made of each question: scope, exposure/outcome (resolved), members, readouts,
expected direction, and the round-1 queries with their hit counts. Zero hits, ignored terms, an
unresolved exposure or an odd outcome here would cost a whole run later.
"""
import argparse
import os

from bmira import Runtime, Settings
from bmira.experiments import ROOT, load_questions
from bmira.graph import parse, plan


def probe(question: str, rt) -> None:
    print(f"\n=== {question}", flush=True)
    state = {"question": question, **parse({"question": question}, rt)}
    p = state["parsed"]
    if not p.in_scope:
        print(f"  OUT OF SCOPE: {p.scope_note}")
        return
    r = rt.resolver
    print(f"  exposure  {p.exposure!r} -> {r.concepts[state['exposure']].label} [{state['exposure']}] "
          f"change={p.exposure_change} members={p.exposure_members}")
    print(f"  outcome   {p.outcome!r} -> {r.concepts[state['outcome']].label} [{state['outcome']}] "
          f"readouts={p.outcome_readouts} expected={p.expected_direction} population={p.target_system}")
    for q in plan(state, rt)["queries"]:
        try:
            res = rt.source.search(q.query, 20)
            print(f"  {q.intent:18} {len(res['pmids']):3} hits  ignored={res.get('ignored_terms') or '-'}  {q.query[:110]}")
        except Exception as e:
            print(f"  {q.intent:18} FAILED {type(e).__name__}  {q.query[:110]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    questions = a.questions or load_questions(ROOT / "experiments" / "questions.txt")
    if a.offline:
        from bmira.offline import offline_runtime
        rt, scenario = offline_runtime()
        questions = [scenario["question"]]
    else:
        rt = Runtime.live(Settings(provider=a.provider, ncbi_email=os.environ.get("NCBI_EMAIL", ""),
                                   ncbi_api_key=os.environ.get("NCBI_API_KEY", ""),
                                   cache_dir=str(ROOT / "runs" / "cache")))
    for q in questions:
        probe(q, rt)
    cost = sum(getattr(rt.llm, "tokens_in", {}).values()), sum(getattr(rt.llm, "tokens_out", {}).values())
    print(f"\ntokens in/out: {cost[0]}/{cost[1]}")


if __name__ == "__main__":
    main()
