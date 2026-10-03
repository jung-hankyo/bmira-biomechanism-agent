"""Run a list of questions and write ONE session summary file.

    python -m bmira.experiments                                  # all of experiments/questions.txt, live
    python -m bmira.experiments --only 1 3 --max-rounds 3        # a subset, cheaper
    python -m bmira.experiments --offline                        # wiring check, no keys or network

The file (default runs/session_<timestamp>.json) is rewritten after every question, so an
interrupted session keeps its finished runs. Upload it to a new Claude session together
with the experiment protocol doc to plan revisions.
"""
import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from bmira import Runtime, Settings, __version__
from bmira.telemetry import execute, git_state, preflight, summarize

ROOT = Path(__file__).resolve().parent.parent


def load_questions(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False,
                      default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o))


def _runtime(a, overrides):
    if a.offline:
        from bmira.offline import offline_runtime
        return offline_runtime(**overrides)
    rt = Runtime.live(Settings(provider=a.provider, ncbi_email=os.environ.get("NCBI_EMAIL", ""),
                               ncbi_api_key=os.environ.get("NCBI_API_KEY", ""), **overrides))
    return rt, None


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=ROOT / "experiments" / "questions.txt")
    ap.add_argument("--only", type=int, nargs="*", help="1-based question numbers to run")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--budget-tokens", type=int, default=None,
                    help="soft token cap per run (input + output); searching stops when reached")
    ap.add_argument("--no-preflight", action="store_true", help="skip the checks before the session")
    ap.add_argument("--offline", action="store_true", help="scripted model and synthetic papers")
    ap.add_argument("--quiet", action="store_true", help="do not echo the pipeline log")
    a = ap.parse_args(argv)

    if not a.offline and not os.environ.get("NCBI_EMAIL"):
        print("[WARN] NCBI_EMAIL is not set; NCBI asks every client for a contact address.")
    questions = load_questions(a.questions)
    picked = [(i, q) for i, q in enumerate(questions, 1) if not a.only or i in a.only]
    out = a.out or ROOT / "runs" / f"session_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    overrides = {k: v for k, v in (("max_rounds", a.max_rounds), ("budget_tokens", a.budget_tokens)) if v}
    probe, _ = _runtime(a, overrides)
    s = probe.settings
    session = {"session": {
        "started": datetime.now().isoformat(timespec="seconds"), "bmira_version": __version__,
        "git": git_state(), "python": platform.python_version(), "mode": "offline" if a.offline else "live",
        "provider": a.provider, "models": s.models.get(s.provider), "temperature": s.temperature,
        "reasoning_effort": s.reasoning_effort, "max_rounds": s.max_rounds, "budget_tokens": s.budget_tokens,
        "questions_file": str(a.questions), "argv": sys.argv[1:] if argv is None else argv}, "runs": []}

    def save():
        session["session"]["finished"] = datetime.now().isoformat(timespec="seconds")
        out.write_text(_dump(session), encoding="utf-8")

    if session["session"]["git"]["uncommitted_changes"]:
        print(f"[WARN] running with uncommitted changes: {session['session']['git']['changed_files']}")
    if not a.no_preflight:
        checks = preflight(probe)
        session["session"]["preflight"] = checks
        for c in checks:
            print(f"[preflight] {'ok  ' if c['ok'] else 'FAIL'} {c['check']} ({c['seconds']}s) "
                  f"{c.get('detail') or c.get('error', '')}")
        failed = [c for c in checks if c["required"] and not c["ok"]]
        if failed:
            session["session"]["aborted"] = "preflight failed: " + "; ".join(c["check"] for c in failed)
            save()
            print(f"\nAborted before spending: {session['session']['aborted']}\nSession summary: {out}")
            return out

    for i, q in picked:
        t0 = time.time()
        try:
            rt, scenario = _runtime(a, overrides)
            if scenario:
                q = scenario["question"]                   # the scripted model only knows this one
            print(f"\n=== [{i}/{len(questions)}] {q}", flush=True)
            final, info = execute(q, rt, echo=not a.quiet)
            run = summarize(final, rt, info)
        except Exception as e:                             # e.g. the runtime could not be built
            info = {"fatal": False}
            run = {"status": "failed", "run": {"question": q, "wall_seconds": round(time.time() - t0, 1)},
                   "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
        run["question_number"] = i
        session["runs"].append(run)
        save()
        lead = (run.get("pathways", {}).get("ranked") or [{}])[0]
        cost = run.get("llm", {}).get("est_cost_usd")
        print(f"=== {run['status']} in {run['run']['wall_seconds']}s"
              + (f" | est. cost ${cost}" if cost is not None else "")
              + (f" | error: {run['error'][:200]}" if run.get("error") else
                 f" | leader: {lead.get('name')} ({lead.get('verdict')})")
              + f" | signals: {[x['signal'] for x in run.get('signals', [])] or 'none'}", flush=True)
        if info.get("fatal"):
            session["session"]["aborted"] = f"fatal error in question {i}: {run.get('error', '')[:300]}"
            save()
            print(f"\nSession aborted: {session['session']['aborted']}")
            break
    print(f"\nSession summary: {out}")
    return out


if __name__ == "__main__":
    main()
