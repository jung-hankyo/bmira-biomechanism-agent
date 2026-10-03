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
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from bmira import __version__
from bmira.graph import build_agent
from bmira.portfolio import STATUS_LABEL, STOP_LABEL


class _Tee(io.TextIOBase):
    """Keeps every printed line (from any graph thread) and optionally echoes it."""

    def __init__(self, echo: bool):
        self.out, self.echo = sys.stdout, echo
        self.lines, self._lock, self._buf = [], threading.Lock(), ""

    def write(self, s):
        with self._lock:
            if self.echo:
                self.out.write(s)
            self._buf += s
            *done, self._buf = self._buf.split("\n")
            self.lines += [d for d in done if d.strip()]
        return len(s)


def execute(question: str, rt, on_progress=None, echo: bool = True):
    """Run one investigation. Returns (final_state, run_info). `on_progress(nodes, new_lines)`
    is called on the caller's thread after every graph step (used by the Streamlit app)."""
    agent = build_agent(rt)
    cfg = {"configurable": {"thread_id": f"bmira-{uuid.uuid4()}"}, "recursion_limit": 250}
    log, shown = _Tee(echo), 0
    node_seconds, t0 = Counter(), time.perf_counter()
    last = t0
    with contextlib.redirect_stdout(log):
        for update in agent.stream({"question": question}, cfg, stream_mode="updates"):
            now = time.perf_counter()
            nodes = [n for n in update if not n.startswith("__")]
            for n in nodes:                    # parallel extractions share the elapsed time
                node_seconds[n] += (now - last) / max(1, len(nodes))
            last = now
            if on_progress:
                on_progress(nodes, log.lines[shown:])
                shown = len(log.lines)
    return agent.get_state(cfg).values, {
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "node_seconds": {k: round(v, 1) for k, v in node_seconds.most_common()},
        "log": log.lines}


# ── summary ─────────────────────────────────────────────────────────────────
def _count(values) -> dict:
    return dict(Counter(values).most_common())


def _share(a, b) -> float | None:
    return round(a / b, 3) if b else None


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, encoding="utf-8",
                              cwd=Path(__file__).parent, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


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

    summary = {
        "run": {
            "question": final.get("question"), "bmira_version": __version__, "git_commit": git_commit(),
            "wall_seconds": run["wall_seconds"], "node_seconds": run["node_seconds"],
            "rounds": final.get("round_idx"), "stop_reason": STOP_LABEL.get(final.get("gate"), final.get("gate")),
            "provider": s.provider, "models": s.models.get(s.provider),
            "llm_client": type(llm).__name__, "embedder": getattr(rt.embedder, "name", "?"),
            "literature_source": getattr(rt.source, "name", "?"), "settings": asdict(s)},
        "parsed_question": final["parsed"].model_dump() if final.get("parsed") else None,
        "llm": {
            "per_task": {t: {"calls": llm.calls[t], "items": llm.items[t],
                             "failures": getattr(llm, "failures", Counter())[t],
                             "tokens_in": getattr(llm, "tokens_in", Counter())[t],
                             "tokens_out": getattr(llm, "tokens_out", Counter())[t],
                             "seconds": round(getattr(llm, "seconds", Counter())[t], 1)} for t in tasks},
            "total_tokens_in": sum(getattr(llm, "tokens_in", Counter()).values()),
            "total_tokens_out": sum(getattr(llm, "tokens_out", Counter()).values()),
            "total_failures": sum(getattr(llm, "failures", Counter()).values())},
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
            "distinct_contexts": len({c.context_concept for c in claims if c.context_concept})},
        "grading": {
            "grades": _count(c.grade for c in claims),
            "caps": _count(cap for c in claims for cap in c.grade_detail.get("caps", [])),
            "limiting_axis": _count(c.grade_detail.get("limiting_axis") for c in claims)},
        "comparison": {
            "semantic_status": final.get("semantic_status"), "pairs_judged": len(pairs),
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
        "warnings": final.get("warnings", []),
        "log": {"problems": [ln for ln in run["log"] if re.search(r"fail|WARN|error", ln, re.I)][:30],
                "tail": run["log"][-40:]},
        "report": final.get("report", ""),
    }
    summary["signals"] = signals(summary)
    return summary


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
    ("semantic layer degraded", lambda m: m["comparison"]["semantic_status"],
     lambda v: v in {"PARTIAL", "UNAVAILABLE"}, "PARTIAL / UNAVAILABLE", "semantic.adjudicate batches"),
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
