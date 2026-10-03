"""Run telemetry: execute one investigation with timing and log capture, then condense the
final state into one JSON-ready summary with revision signals.

The summary is built for revising the code afterwards: every section maps to a pipeline
stage, and `signals` lists measured values that crossed a heuristic threshold, each with
the code area to inspect. Thresholds are starting points, not validated cut-offs.
"""
import contextlib
import io
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from bmira import __version__
from pydantic import BaseModel

from bmira.graph import build_agent
from bmira.llm import FatalLLMError
from bmira.normalize import lookup_key
from bmira.portfolio import STATUS_LABEL, STOP_LABEL


class _Tee(io.TextIOBase):
    """Keeps every printed line (from any graph thread) and optionally echoes it."""

    def __init__(self, echo: bool):
        self.out, self.echo = sys.stdout, echo
        self.lines, self._lock, self._buf = [], threading.Lock(), ""

    def write(self, s):
        with self._lock:                       # whole lines only: parallel prints stay apart
            self._buf += s
            *done, self._buf = self._buf.split("\n")
            for d in done:
                if self.echo:
                    self.out.write(d + "\n")
                if d.strip():
                    self.lines.append(d)
        return len(s)


NODES = {"parse", "plan", "search", "screen", "extract", "normalize", "semantic", "portfolio",
         "review", "synthesize", "verify"}


def _failing_node(e: BaseException) -> str | None:
    """Deepest pipeline node in the traceback; LangGraph's own note is missing for some errors."""
    node = None
    for frame, _ in traceback.walk_tb(e.__traceback__):
        if frame.f_code.co_filename.endswith("graph.py") and frame.f_code.co_name in NODES:
            node = frame.f_code.co_name
    if node is None:
        hit = re.search(r"task with name '([^']+)'", " ".join(getattr(e, "__notes__", [])))
        node = hit.group(1) if hit else None
    return node


def execute(question: str, rt, on_progress=None, echo: bool = True):
    """Run one investigation. Returns (final_state, run_info). `on_progress(nodes, new_lines)`
    is called on the caller's thread after every graph step (used by the Streamlit app)."""
    agent = build_agent(rt)
    cfg = {"configurable": {"thread_id": f"bmira-{uuid.uuid4()}"}, "recursion_limit": 250}
    log, shown = _Tee(echo), 0
    node_seconds, t0 = Counter(), time.perf_counter()
    last = t0
    info = {"question": question, "status": "completed", "error": None, "failed_node": None, "fatal": False}
    with contextlib.redirect_stdout(log):
        try:
            for update in agent.stream({"question": question}, cfg, stream_mode="updates"):
                now = time.perf_counter()
                nodes = [n for n in update if not n.startswith("__")]
                for n in nodes:                # parallel extractions share the elapsed time
                    node_seconds[n] += (now - last) / max(1, len(nodes))
                last = now
                if on_progress:
                    on_progress(nodes, log.lines[shown:])
                    shown = len(log.lines)
        except (Exception, FatalLLMError, KeyboardInterrupt) as e:
            # Keep what was paid for: the last saved graph state still holds every
            # completed step. Fatal errors and Ctrl-C abort the session; others fail one run.
            fatal = isinstance(e, (FatalLLMError, KeyboardInterrupt))
            info.update(status="aborted" if fatal else "failed", fatal=fatal,
                        error=f"{type(e).__name__}: {e}"[:2000], failed_node=_failing_node(e),
                        traceback=traceback.format_exc()[-4000:])
            print(f"[run] {info['status'].upper()} at step {info['failed_node']}: {info['error'][:300]}")
    info.update(wall_seconds=round(time.perf_counter() - t0, 1), log=log.lines,
                node_seconds={k: round(v, 1) for k, v in node_seconds.most_common()})
    return agent.get_state(cfg).values, info


# ── summary ─────────────────────────────────────────────────────────────────
def _count(values) -> dict:
    return dict(Counter(values).most_common())


def _share(a, b) -> float | None:
    return round(a / b, 3) if b else None


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                              cwd=Path(__file__).parent, timeout=5).stdout
    except Exception:
        return ""


def git_commit() -> str:
    return _git("rev-parse", "--short", "HEAD").strip() or "unknown"


def git_state() -> dict:
    """Commit plus local edits: a run from hand-edited code must not look like a release."""
    changed = [ln[3:] for ln in _git("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
    return {"commit": git_commit(), "uncommitted_changes": bool(changed), "changed_files": changed}


def estimate_cost(model, tokens_in, tokens_out, prices) -> float | None:
    p = prices.get(model)
    return round((tokens_in * p[0] + tokens_out * p[1]) / 1e6, 4) if p else None


def summarize(final: dict, rt, run: dict) -> dict:
    s, llm, res = rt.settings, rt.llm, rt.resolver
    papers, claims = final.get("papers", []), final.get("claims", [])
    dropped, links = final.get("dropped_claims", []), final.get("links", {})
    hyps, history = final.get("hypotheses", []), final.get("portfolio_history", [])
    qlog, verif = final.get("search_log", []), final.get("verification", {})
    extracted = set(final.get("extracted_pmids", []))
    included = [p for p in papers if p.screen_status == "included"]
    read = [p for p in papers if p.pmid in extracted]
    tasks = sorted(set(llm.calls) | set(getattr(llm, "failures", {})))
    concept_ids = {i for c in claims for i in (c.subject_concept, c.object_concept)}
    concepts = [res.concepts[i] for i in concept_ids if i in res.concepts]
    leaders = [h["hypotheses"][0][0] if h["hypotheses"] else None for h in history]
    pairs = list(rt.pair_cache.values())
    n_extracted = len(claims) + len(dropped)
    by_label = {}                      # one label naming several concepts = a split node
    for c in concepts:
        by_label.setdefault(lookup_key(c.label), set()).add(res.canonical(c).id)

    zero = Counter()
    per_task = {}
    for t in tasks:
        model = getattr(llm, "model_of", {}).get(t)
        t_in, t_out = getattr(llm, "tokens_in", zero)[t], getattr(llm, "tokens_out", zero)[t]
        per_task[t] = {"model": model, "calls": llm.calls[t], "items": llm.items[t],
                       "failures": getattr(llm, "failures", zero)[t], "tokens_in": t_in, "tokens_out": t_out,
                       "tokens_reasoning": getattr(llm, "tokens_reasoning", zero)[t],
                       "seconds": round(getattr(llm, "seconds", zero)[t], 1),
                       "est_cost_usd": estimate_cost(model, t_in, t_out, s.prices)}
    costs = [v["est_cost_usd"] for v in per_task.values()]
    summary = {
        "status": run.get("status", "completed"), "error": run.get("error"), "failed_node": run.get("failed_node"),
        "run": {
            "question": final.get("question") or run.get("question"), "bmira_version": __version__,
            "git_commit": git_commit(),
            "wall_seconds": run["wall_seconds"], "node_seconds": run["node_seconds"],
            "rounds": final.get("round_idx"), "stop_reason": STOP_LABEL.get(final.get("gate"), final.get("gate")),
            "provider": s.provider, "models": s.models.get(s.provider),
            "llm_client": type(llm).__name__, "embedder": getattr(rt.embedder, "name", "?"),
            "literature_source": getattr(rt.source, "name", "?"),
            # session files get shared: never write the NCBI key or the contact email
            "settings": {**asdict(s), "ncbi_api_key": "set" if s.ncbi_api_key else "",
                         "ncbi_email": "set" if s.ncbi_email else ""}},
        "parsed_question": final["parsed"].model_dump() if final.get("parsed") else None,
        "llm": {
            "per_task": per_task,
            "total_tokens_in": sum(v["tokens_in"] for v in per_task.values()),
            "total_tokens_out": sum(v["tokens_out"] for v in per_task.values()),
            "total_tokens_reasoning": sum(v["tokens_reasoning"] for v in per_task.values()),
            "total_failures": sum(v["failures"] for v in per_task.values()),
            "est_cost_usd": round(sum(c for c in costs if c is not None), 4) if any(c is not None for c in costs) else None,
            "cost_complete": all(c is not None for c in costs),
            "budget_tokens": s.budget_tokens},
        "retrieval": {
            "queries": len(qlog), "queries_failed": sum(not q["ok"] for q in qlog),
            "queries_zero_hits": sum(q["ok"] and q.get("hits", 0) == 0 for q in qlog),
            "queries_with_ignored_terms": sum(bool(q.get("ignored_terms")) for q in qlog),
            "mean_hits_per_query": round(mean([q["hits"] for q in qlog if q["ok"]]), 1)
            if any(q["ok"] for q in qlog) else None,
            "ignored_term_samples": [{"query": q["query"], "ignored": q["ignored_terms"]}
                                     for q in qlog if q.get("ignored_terms")][:10],
            "zero_hit_samples": [q["query"] for q in qlog if q["ok"] and q.get("hits", 0) == 0][:10]},
        "papers": {
            "retrieved": len(papers), "screened": sum(p.screen_status != "unscreened" for p in papers),
            "included": len(included),
            "excluded": sum(p.screen_status == "excluded" for p in papers),
            "retracted_excluded": sum(p.retracted for p in papers),
            "inclusion_rate": _share(len(included), sum(p.screen_status != "unscreened" for p in papers)),
            "extracted": len(read), "included_never_extracted": len([p for p in included if p.pmid not in extracted]),
            "full_text_share_of_extracted": _share(sum(p.text_access == "full_text" for p in read), len(read)),
            "study_types": _count(p.study_type for p in included),
            "retrieved_for_a_step": sum(bool(p.retrieved_for) for p in papers),
            "unread_for_their_step": sum(bool(set(p.retrieved_for) - set(p.read_for)) for p in included),
            "years": _count(p.year[:3] + "0s" for p in included if p.year)},
        "extraction": {
            "claims_extracted": n_extracted, "claims_kept": len(claims), "claims_dropped": len(dropped),
            "drop_rate": _share(len(dropped), n_extracted), "drop_reasons": _count(c.drop_reason for c in dropped),
            "dropped_samples": [{"pmid": c.pmid, "reason": c.drop_reason, "claim": f"{c.subject} | {c.relation} | {c.object}",
                                 "span": c.span[:200]} for c in dropped][:15],
            "claims_per_extracted_paper": _share(len(claims), len(read)),
            "claim_types": _count(c.claim_type for c in claims), "systems": _count(c.system for c in claims),
            "attributes": _count([c.subject_attribute for c in claims] + [c.object_attribute for c in claims]),
            "method_fields_uncredited": _count(m for c in claims for m in c.method_checks),
            "uncredited_samples": [{"id": c.id, "checks": c.method_checks, "span": c.span[:160]}
                                   for c in claims if c.method_checks][:10],
            "relation_sources": _count(c.relation_source for c in claims),
            "relations": _count(c.relation_norm or "pending" for c in claims)},
        "normalization": {
            "concepts_used": len(concepts), "concept_sources": _count(c.source for c in concepts),
            "unresolved_local_share": _share(sum(c.source == "local" for c in concepts), len(concepts)),
            "ontology_share": _share(sum(c.source == "ols" for c in concepts), len(concepts)),
            "unresolved_samples": sorted(c.label for c in concepts if c.source == "local")[:20],
            "categories": _count(c.category for c in concepts), "alias_merges": len(res.alias),
            # distinct entities / entity slots in kept claims: near 1.0 = every claim names
            # new nodes (fragmented graph); lower = claims share nodes and can connect
            "fragmentation": _share(len(concept_ids), 2 * len(claims)),
            "duplicate_labels": sorted(k for k, ids in by_label.items() if len(ids) > 1),
            "entity_cache_reused": getattr(res, "disk_hits", 0),
            "tissues": _count(t for c in claims for t in c.context_tissue.split(", ") if t),
            "distinct_contexts": len({c.context_concept for c in claims if c.context_concept})},
        "grading": {
            "grades": _count(c.grade for c in claims),
            "caps": _count(cap for c in claims for cap in c.grade_detail.get("caps", [])),
            "limiting_axis": _count(c.grade_detail.get("limiting_axis") for c in claims)},
        "comparison": {
            "semantic_status": final.get("semantic_status"), "pairs_judged": len(pairs),
            "pairs_asked": getattr(llm, "items", zero)["pair"],
            "same_finding": sum(v.same_finding for v in pairs), "same_context": sum(v.same_context for v in pairs),
            "conflict_status": final.get("conflict_status"),
            "conflicts": _count(c.verdict for c in final.get("conflicts", []))},
        "steps": {
            "count": len(links), "verdicts": _count(STATUS_LABEL[ln.status] for ln in links.values()),
            "reasons": _count(re.sub(r"\d+", "N", ln.reason) for ln in links.values()),
            "not_counted": _count(r for ln in links.values() for r in ln.uncounted.values()),
            "papers_per_step": _count(str(ln.n_studies) if ln.n_studies < 3 else "3+" for ln in links.values()),
            "searched_out": sum(ln.exhausted for ln in links.values())},
        "pathways": {
            "count": len(hyps), "verdicts": _count(STATUS_LABEL[h.status] for h in hyps),
            "origins": _count(h.origin for h in hyps), "logic_flags": _count(f for h in hyps for f in h.logic_flags),
            "leader_per_round": leaders,
            "leader_changes": sum(a != b for a, b in zip(leaders, leaders[1:])),
            "targets_per_round": [len(h["targets"]) for h in history],
            "ranked": [{"id": h.id, "name": h.name, "origin": h.origin, "verdict": STATUS_LABEL[h.status],
                        "reason": h.reason, "score": h.score, "logic_flags": h.logic_flags,
                        "steps": [f"{links[k].subject_label} -{links[k].relation}-> {links[k].object_label}: "
                                  f"{STATUS_LABEL[links[k].status]} ({links[k].reason})" for k in h.links if k in links]}
                       for h in hyps]},
        "verification": {
            "passed": verif.get("passed"), "uncited": len(verif.get("uncited", [])),
            "overclaims": len(verif.get("overclaims", [])), "unknown_ids": len(verif.get("unknown_ids", [])),
            "missing_tags": len(verif.get("missing_tags", [])), "entailment_issues": len(verif.get("entailment", [])),
            "overclaim_samples": [o["sentence"][:200] for o in verif.get("overclaims", [])][:5]},
        # per-claim and per-step tables: the hand spot-check (protocol step 5) needs them
        "claims": [{"id": c.id, "pmid": c.pmid, "subject": f"{c.subject_label} [{c.subject_concept}]",
                    "relation": c.relation_norm, "object": f"{c.object_label} [{c.object_concept}]",
                    "grade": c.grade, "limiting_axis": c.grade_detail.get("limiting_axis"),
                    "system": c.system, "span": c.span[:120]} for c in claims],
        "links": [{"key": ln.key, "step": f"{ln.subject_label} -{ln.relation}-> {ln.object_label}",
                   "status": STATUS_LABEL[ln.status], "reason": ln.reason, "n_studies": ln.n_studies,
                   "grade": ln.grade, "support": ln.support_ids, "corroborating": ln.corroborating_ids}
                  for ln in links.values()],
        "warnings": final.get("warnings", []),
        "log": {"problems": [ln for ln in run["log"] if re.search(r"fail|WARN|error", ln, re.I)][:30],
                "tail": run["log"][-40:], "traceback": run.get("traceback")},
        "report": final.get("report", ""),
    }
    summary["signals"] = signals(summary)
    return summary


# ── preflight ───────────────────────────────────────────────────────────────
class Ping(BaseModel):
    ok: bool


def preflight(rt) -> list[dict]:
    """Seconds-long checks before any spending: each configured model answers one tiny
    structured call (same path, parameters and effort as real calls), PubMed answers one
    search, and the ontology service answers one lookup. A required failure aborts."""
    checks = []

    def check(name, fn, required=True):
        t0 = time.perf_counter()
        try:
            checks.append({"check": name, "ok": True, "required": required, "detail": str(fn())[:200]})
        except (Exception, FatalLLMError) as e:
            checks.append({"check": name, "ok": False, "required": required,
                           "error": f"{type(e).__name__}: {e}"[:500]})
        checks[-1]["seconds"] = round(time.perf_counter() - t0, 1)

    for role in ("reasoning", "cheap"):
        model = rt.settings.models.get(rt.settings.provider, {}).get(role, "?")
        check(f"LLM {role} model ({model})", lambda r=role: rt.llm.structured(
            "preflight", Ping, "Answer with ok = true.", "ping", role=r, ctx={"schema": Ping}).ok)
    check("PubMed search", lambda: f"{len(rt.source.search('butyrate regulatory T cells', 1)['pmids'])} hit(s)")
    if rt.settings.ontology_provider in {"ols", "hybrid"}:
        def ols():
            hit = rt.resolver._ols("regulatory T cell")     # returns None on network errors too
            if hit is None:
                raise RuntimeError("no answer from OLS (unreachable, or exact match failed)")
            return f"{hit.id} {hit.label}"
        check("Ontology lookup (OLS)", ols, required=False)
    return checks


# ── revision signals ────────────────────────────────────────────────────────
RULES = [
    # (name, value(summary), fires(value), threshold text, where to look)
    ("claim drop rate", lambda m: m["extraction"]["drop_rate"], lambda v: v is not None and v > 0.30,
     "> 0.30", "normalize.check_claim / extract prompt: quote checks may be too strict, or quotes are paraphrased"),
    ("unverified relation wording", lambda m: _share(m["grading"]["caps"].get("relation_unverified", 0),
                                                     m["extraction"]["claims_kept"]),
     lambda v: v is not None and v > 0.30, "> 0.30 of kept claims",
     "normalize.check_claim verb matching; extract prompt rule 4 (verbatim relation)"),
    ("uncredited method fields", lambda m: _share(sum(m["extraction"]["method_fields_uncredited"].values()),
                                                  m["extraction"]["claims_kept"]),
     lambda v: v is not None and v > 0.40, "> 0.40 per kept claim",
     "evidence.METHOD_CUES / COMPARATOR_CUE too narrow, or methods_span not being quoted"),
    ("unresolved concepts", lambda m: m["normalization"]["unresolved_local_share"],
     lambda v: v is not None and v > 0.50, "> 0.50", "normalize.EntityResolver._ols: ontology list, exact matching"),
    ("fragmented graph", lambda m: (m["normalization"]["fragmentation"], m["extraction"]["claims_kept"]),
     lambda v: v[0] is not None and v[0] > 0.7 and v[1] >= 20, "> 0.7 with >= 20 claims",
     "normalize.entity_of (attribute, tissue, modifiers), alias merging, extract prompt rule 3"),
    ("no alias merges", lambda m: (m["normalization"]["alias_merges"], m["extraction"]["claims_kept"]),
     lambda v: v[0] == 0 and v[1] >= 30, "0 merges with >= 30 claims", "normalize._maybe_alias prefilter"),
    ("reading backlog", lambda m: _share(m["papers"]["included_never_extracted"], m["papers"]["included"]),
     lambda v: v is not None and v > 0.30, "> 0.30 of included papers", "config.max_extract_per_round"),
    ("mostly abstracts", lambda m: m["papers"]["full_text_share_of_extracted"],
     lambda v: v is not None and v < 0.20, "< 0.20 full text", "grades are capped at moderate; expected for paywalled fields"),
    ("query syntax problems", lambda m: _share(m["retrieval"]["queries_with_ignored_terms"], m["retrieval"]["queries"]),
     lambda v: v is not None and v > 0.25, "> 0.25 of queries", "llm PROMPTS['plan']; graph._synonym_query"),
    ("zero-hit queries", lambda m: _share(m["retrieval"]["queries_zero_hits"], m["retrieval"]["queries"]),
     lambda v: v is not None and v > 0.40, "> 0.40 of queries", "query construction; labels too specific"),
    ("low inclusion", lambda m: m["papers"]["inclusion_rate"], lambda v: v is not None and v < 0.15,
     "< 0.15", "PROMPTS['screen'] or round-1 queries off target"),
    ("pending relations", lambda m: m["extraction"]["relations"].get("pending", 0), lambda v: v > 0,
     "> 0", "relation typing failed; see llm failures"),
    ("unresolved relations", lambda m: _share(m["extraction"]["relations"].get("unresolved", 0),
                                              m["extraction"]["claims_kept"]),
     lambda v: v is not None and v > 0.20, "> 0.20", "normalize.RELATION_LEXICON / PROMPTS['relation']"),
    ("no pathway supported", lambda m: m["pathways"]["verdicts"].get("Supported", 0),
     lambda v: v == 0, "0", "evidence too sparse, or min_studies_per_link / R8 too strict for this field"),
    ("many unfound steps", lambda m: _share(m["steps"]["reasons"].get("no study found in N targeted searches", 0),
                                            m["steps"]["count"]),
     lambda v: v is not None and v > 0.50, "> 0.50 of steps", "search recall: synonym query, plan prompt"),
    ("leader not settled", lambda m: m["pathways"]["leader_per_round"][-2:],
     lambda v: len(v) == 2 and v[0] != v[1], "leader changed in the last round", "more rounds, or check scoring"),
    ("verification failed", lambda m: m["verification"]["passed"], lambda v: v is False, "failed",
     "PROMPTS['synthesize']; evidence.VERB_TIER"),
    ("LLM failures", lambda m: m["llm"]["total_failures"], lambda v: v > 0, "> 0",
     "schemas vs model structured output; see llm.per_task"),
    ("run did not finish", lambda m: m["status"], lambda v: v != "completed", "failed / aborted",
     "see status, error and failed_node; metrics cover completed steps only"),
    ("token budget reached", lambda m: m["run"]["stop_reason"], lambda v: v == STOP_LABEL["BUDGET"],
     "budget hit", "Settings.budget_tokens, or cost drivers in llm.per_task"),
    ("semantic layer degraded", lambda m: m["comparison"]["semantic_status"],
     lambda v: v in {"PARTIAL", "UNAVAILABLE"}, "PARTIAL / UNAVAILABLE", "semantic.adjudicate batches"),
    ("pair verdicts lost", lambda m: _share(m["comparison"]["pairs_judged"], m["comparison"]["pairs_asked"]),
     lambda v: v is not None and v < 0.5, "< 0.5 of asked pairs judged",
     "semantic.adjudicate pair numbering; PROMPTS['pair']; the model returned fewer or mis-numbered verdicts"),
    ("one label, several concepts", lambda m: m["normalization"]["duplicate_labels"],
     lambda v: len(v) > 0, "> 0 labels", "normalize.EntityResolver._labelled / consolidate_aliases"),
]


def signals(summary: dict) -> list[dict]:
    out = []
    for name, value, fires, threshold, look in RULES:
        try:
            v = value(summary)
            if fires(v):
                out.append({"signal": name, "value": v, "threshold": threshold, "look_at": look})
        except (KeyError, TypeError, IndexError):
            continue
    return out
