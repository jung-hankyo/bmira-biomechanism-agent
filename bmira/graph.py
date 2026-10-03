"""LangGraph wiring. Nodes are plain functions of (state, runtime).

parse -> plan -> search -> screen -> extract* -> normalize -> semantic -> portfolio -> review
           ^                                                                  |
           +---------------------- search_more -------------------------------+-> synthesize -> verify
"""
import operator
import re
import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from bmira import portfolio as pf
from bmira.config import Settings
from bmira.evidence import grade_claim, prose_sentences, verify_text
from bmira.llm import PROMPTS
from bmira.normalize import EntityResolver, consolidate_aliases, lexical_relation, span_is_anchored
from bmira.schemas import (Claim, ClaimList, Conflict, EntailmentBatch, Hypothesis, LinkEvidence,
                           Paper, ParsedQuestion, PathwayProposal, QueryPlan,
                           RelationResolutionBatch, Screen, SearchQuery)
from bmira.semantic import adjudicate, candidate_pairs, clusters, conflict_candidates, triage


@dataclass
class Runtime:
    """Swappable dependencies plus per-run caches (each item is judged by the LLM once)."""
    settings: Settings
    llm: object
    source: object
    embedder: object
    resolver: EntityResolver | None = None
    pair_cache: dict = field(default_factory=dict)
    alias_verdicts: dict = field(default_factory=dict)
    conflict_cache: dict = field(default_factory=dict)

    def __post_init__(self):
        self.resolver = self.resolver or EntityResolver(self.settings, self.llm)

    @classmethod
    def live(cls, settings: Settings, api_key: str | None = None):
        from bmira.llm import LangChainLLM
        from bmira.semantic import SentenceEmbedder
        from bmira.sources import PubMedSource
        return cls(settings, LangChainLLM(settings, api_key), PubMedSource(settings), SentenceEmbedder())


# ── state ───────────────────────────────────────────────────────────────────
def _as(model, x):
    """Checkpoints may hand back dicts instead of pydantic objects."""
    return model.model_validate(x) if isinstance(x, dict) else x


def _merge(model, key):
    def reducer(old, new):
        merged = {getattr(_as(model, x), key): _as(model, x) for x in (old or [])}
        merged.update({getattr(_as(model, x), key): _as(model, x) for x in (new or [])})
        return list(merged.values())
    return reducer


class State(TypedDict, total=False):
    question: str
    parsed: ParsedQuestion
    exposure: str
    outcome: str
    queries: list
    papers: Annotated[list, _merge(Paper, "pmid")]
    claims: Annotated[list, _merge(Claim, "id")]
    dropped_claims: Annotated[list, operator.add]
    extracted_pmids: Annotated[list, operator.add]
    search_status: str
    target_yield: dict
    search_log: Annotated[list, operator.add]
    semantic_status: str
    semantic_edges: list
    conflicts: list
    conflict_status: str
    links: dict
    hypotheses: list
    targets: list
    gate: str
    decision: str
    seed_status: str
    round_idx: int
    portfolio_history: Annotated[list, operator.add]
    synthesis: str
    link_tags: dict
    warnings: list
    verification: dict
    report: str


# ── nodes ───────────────────────────────────────────────────────────────────
def parse(state, rt):
    p = rt.llm.structured("parse", ParsedQuestion, PROMPTS["parse"], state["question"],
                          ctx={"question": state["question"]})
    exp, out = rt.resolver.resolve(p.exposure), rt.resolver.resolve(p.outcome)
    print(f"[parse] exposure={exp.label} ({exp.id}) outcome={out.label} ({out.id}) "
          f"expected={p.expected_direction}")
    return {"parsed": p, "exposure": exp.id, "outcome": out.id, "round_idx": 0, "targets": []}


def plan(state, rt):
    links = {k: _as(LinkEvidence, v) for k, v in state.get("links", {}).items()}
    targets = [links[k] for k in state.get("targets", []) if k in links]
    p = _as(ParsedQuestion, state["parsed"])
    user = f"Question: {state['question']}\nStructured: {p.model_dump_json()}\n"
    if targets:
        user += "Targets:\n" + "\n".join(
            f"[{t.key}] {t.subject_label} --{t.relation}--> {t.object_label} "
            f"(papers={t.n_studies}, status={t.status})" for t in targets)
    out = rt.llm.structured("plan", QueryPlan, PROMPTS["plan"], user,
                            ctx={"parsed": p, "targets": targets})
    if not targets:                             # LLM output is a trust boundary: degrade, don't crash
        queries = out.queries or [SearchQuery(query=f"{p.exposure} {p.outcome}", intent="broad")]
        missing = {"broad", "mechanism", "contradiction", "negative_result"} - {q.intent for q in queries}
        if missing:
            print(f"[plan][WARN] coverage round lacks intents {sorted(missing)}")
    else:
        keys = {t.key for t in targets}
        queries = [q for q in out.queries if q.target in keys]
        for t in targets:                       # never let a target go unsearched silently
            if not any(q.target == t.key for q in queries):
                queries.append(SearchQuery(query=f'"{t.subject_label}" AND "{t.object_label}"',
                                           intent="gap_positive", target=t.key))
    print(f"[plan] round {state.get('round_idx', 0) + 1}: {len(queries)} queries"
          + (f" for {len(targets)} target links" if targets else " (coverage round)"))
    return {"queries": queries}


def search(state, rt):
    known = {_as(Paper, p).pmid for p in state.get("papers", [])}
    log, new_ids, per_target = [], [], {}
    for q in state["queries"]:
        q = _as(SearchQuery, q)
        try:
            res = rt.source.search(q.query, rt.settings.max_papers_per_query)
        except Exception as e:
            log.append({"query": q.query, "target": q.target, "ok": False, "error": type(e).__name__})
            continue
        fresh = [i for i in res["pmids"] if i not in known]
        new_ids += [i for i in fresh if i not in new_ids]
        if q.target:
            per_target.setdefault(q.target, set()).update(fresh)
        log.append({"query": q.query, "target": q.target, "ok": True, "hits": len(res["pmids"]),
                    "new": len(fresh), "translation": res.get("translation", ""),
                    "ignored_terms": res.get("ignored_terms", [])})
    executed = [x for x in log if x["ok"]]
    papers = rt.source.fetch(new_ids) if new_ids else []
    status = ("SEARCH_FAILED" if not executed else
              "NEW_RESULTS" if papers else "ZERO_NEW_RESULTS")
    print(f"[search] {status}: {len(executed)}/{len(log)} queries ran, {len(papers)} new papers")
    return {"papers": papers, "search_status": status, "search_log": log,
            "target_yield": {k: len(v) for k, v in per_target.items()}}


def screen(state, rt):
    todo = [_as(Paper, p) for p in state["papers"] if _as(Paper, p).screen_status == "unscreened"]
    for p in todo:
        if p.retracted:
            p.screen_status, p.relevance_reason = "excluded", "retracted / expression of concern"
    live = [p for p in todo if not p.retracted]
    if live:
        users = [f"Question: {state['question']}\n\nTitle: {p.title}\nAbstract: {p.abstract[:3000]}"
                 for p in live]
        try:
            verdicts = rt.llm.structured_many("screen", Screen, PROMPTS["screen"], users, ctxs=live)
        except Exception as e:                  # papers stay unscreened and are retried next round
            print(f"[screen] batch failed ({type(e).__name__}); retrying next round")
            verdicts = []
        for p, r in zip(live, verdicts):
            p.screen_status = "included" if r.relevant else "excluded"
            p.relevance_score, p.relevance_reason = r.relevance_score, r.reason
            p.study_type = p.pubtype_study_type or r.study_type   # metadata outranks the model
    todo = [p for p in todo if p.screen_status != "unscreened"]
    print(f"[screen] {len(todo)} screened, {sum(p.screen_status == 'included' for p in todo)} "
          f"included, {sum(p.retracted for p in todo)} retracted excluded")
    return {"papers": todo}


def fan_out(state, rt):
    done = set(state.get("extracted_pmids", []))
    pool = sorted((_as(Paper, p) for p in state["papers"]),
                  key=lambda p: (-p.relevance_score, p.pmid))
    todo = [p for p in pool if p.screen_status == "included" and p.pmid not in done]
    todo = todo[:rt.settings.max_extract_per_round]   # the rest wait for the next round
    return [Send("extract", {"paper": p, "question": state["question"]}) for p in todo] or ["normalize"]


def extract(payload, rt):
    paper = rt.source.fulltext(payload["paper"])
    s = rt.settings
    try:
        out = rt.llm.structured(
            "extract", ClaimList, PROMPTS["extract"].format(max_claims=s.max_claims_per_paper),
            f"Question: {payload['question']}\n\nPaper (PMID {paper.pmid}, {paper.text_access}):\n"
            f"{paper.source_text[:s.fulltext_char_limit]}", ctx={"paper": paper})
    except Exception as e:
        print(f"[extract] PMID {paper.pmid} failed ({type(e).__name__}); retried next round")
        return {}
    claims = [Claim(**ec.model_dump(), id=f"C{paper.pmid}_{i}", pmid=paper.pmid,
                    study_type=paper.study_type or "cell_line", text_access=paper.text_access,
                    relation_raw=ec.relation)
              for i, ec in enumerate(out.claims[:s.max_claims_per_paper])]
    kept = [c for c in claims if span_is_anchored(c.span, paper.source_text)]
    for c in kept:
        c.anchored = True
    dropped = [c for c in claims if not c.anchored]
    print(f"[extract] {paper.pmid}: {len(kept)} anchored, {len(dropped)} dropped ({paper.text_access})")
    return {"claims": kept, "dropped_claims": dropped, "extracted_pmids": [paper.pmid],
            "papers": [paper]}


def _set_concepts(c, rt):
    s, o = rt.resolver.resolve(c.subject), rt.resolver.resolve(c.object)
    c.subject_concept, c.subject_label, c.subject_category = s.id, s.label, s.category
    c.object_concept, c.object_label, c.object_category = o.id, o.label, o.category
    c.object_ancestors = list(o.ancestors)


def normalize(state, rt):
    claims = [_as(Claim, c) for c in state.get("claims", [])]
    for c in claims:
        if not c.relation_norm:
            c.relation_norm, c.relation_source = lexical_relation(c.relation_raw), "lexical"
    pending = [c for c in claims if not c.relation_norm]      # only never-resolved claims
    for i in range(0, len(pending), 20):
        chunk = pending[i:i + 20]
        text = "\n\n".join(f"[CLAIM {c.id}] {c.subject} | {c.relation_raw} | {c.object}\n"
                           f"Span: {c.span[:600]}" for c in chunk)
        try:
            out = rt.llm.structured("relation", RelationResolutionBatch, PROMPTS["relation"], text,
                                    ctx={"claims": chunk}, n_items=len(chunk))
            by_id = {c.id: c for c in chunk}
            for r in out.resolutions:
                if r.claim_id in by_id:
                    c = by_id[r.claim_id]
                    c.relation_norm, c.relation_source = r.canonical_relation, "llm"
                    c.relation_confidence = r.confidence
        except Exception as e:
            print(f"[relation] batch failed ({type(e).__name__}); claims stay pending for retry")
    for c in claims:
        _set_concepts(c, rt)
    merged = consolidate_aliases(claims, rt.resolver, rt.llm, rt.alias_verdicts)
    for c in claims:
        _set_concepts(c, rt)                               # cheap: cached + alias registry
        grade_claim(c)
    print(f"[normalize] {len(claims)} claims, {len(pending)} relations sent to LLM, "
          f"{merged} alias merges")
    return {"claims": claims}


def semantic(state, rt):
    claims = [_as(Claim, c) for c in state.get("claims", [])]
    thr = rt.settings.similarity_threshold.get(rt.embedder.name, 0.6)
    pairs = candidate_pairs(claims, rt.embedder, thr, rt.settings.max_candidates_per_claim)
    new = [p for p in pairs if frozenset(p[:2]) not in rt.pair_cache]
    failed = adjudicate(claims, pairs, rt.llm, rt.pair_cache)
    status = ("NO_CANDIDATES" if not pairs else "COMPLETE" if not failed else
              "UNAVAILABLE" if failed == len(new) else "PARTIAL")
    fams = clusters(claims, rt.pair_cache)
    cands = conflict_candidates(claims, fams, rt.pair_cache)
    fresh = [f for f in cands if frozenset(f.claim_ids) not in rt.conflict_cache]
    verdicts, cstatus = triage(claims, fresh, rt.llm) if fresh else ([], "CACHED")
    for f in fresh:
        hit = next((v for v in verdicts if v.cluster_key == f.key), None)
        if hit:
            rt.conflict_cache[frozenset(f.claim_ids)] = hit.model_copy(update={"claim_ids": f.claim_ids})
    conflicts = [rt.conflict_cache[frozenset(f.claim_ids)] for f in cands
                 if frozenset(f.claim_ids) in rt.conflict_cache]
    print(f"[semantic] {len(pairs)} candidate pairs ({len(new)} new) status={status}; "
          f"{len(cands)} conflict candidates, {len(conflicts)} triaged")
    return {"semantic_status": status, "semantic_edges": list(rt.pair_cache.values()),
            "conflicts": conflicts, "conflict_status": cstatus if fresh else "COMPLETE"}


def _canon(cid, rt):
    while cid in rt.resolver.alias:
        cid = rt.resolver.alias[cid]
    return cid


def _canon_key(key, rt):
    s, r, o = pf.split_key(key)
    return pf.link_key(_canon(s, rt), r, _canon(o, rt))


def _claims_summary(claims, n=120):
    return "\n".join(f"[{c.id}] {c.subject_label} --{c.relation_norm}--> {c.object_label} "
                     f"({c.grade}, {c.study_type}, {c.context_cell_type})" for c in claims[:n])


def _add(hyps, keys, origin, name, rationale, diverse):
    if keys and not pf.is_duplicate(keys, hyps, diverse):
        n = max((int(h.id[1:]) for h in hyps), default=0) + 1     # ids never reused after drops
        hyps.append(Hypothesis(id=f"H{n}", name=name, origin=origin, links=keys,
                               rationale=rationale))
        return True
    return False


def portfolio(state, rt):
    s, r = rt.settings, rt.resolver
    claims = [_as(Claim, c) for c in state.get("claims", [])]
    prior = {}
    for k, v in state.get("links", {}).items():          # re-key: alias merges may have happened
        v = _as(LinkEvidence, v)
        prior.setdefault(_canon_key(k, rt), v)
    for k in state.get("targets", []):                    # zero-yield accounting for last round
        ck = _canon_key(k, rt)
        searched_ok = state.get("search_status") != "SEARCH_FAILED"
        if ck in prior and searched_ok and state.get("target_yield", {}).get(k, 0) == 0:
            prior[ck].zero_yield_count += 1
            prior[ck].exhausted = prior[ck].zero_yield_count >= s.zero_yield_rounds_to_exhaust
    hyps = [_as(Hypothesis, h) for h in state.get("hypotheses", [])]
    for h in hyps:
        h.links = [_canon_key(k, rt) for k in h.links]
    labels = {cid: c.label for cid, c in r.concepts.items()}
    p = _as(ParsedQuestion, state["parsed"])
    exposure, outcome = _canon(state["exposure"], rt), _canon(state["outcome"], rt)
    outcomes = {outcome} | {cid for cid, c in r.concepts.items() if outcome in c.ancestors}
    seed_status = state.get("seed_status", "")

    if not seed_status:                                   # seed exactly once, even if it fails
        try:
            out = rt.llm.structured(
                "seed", PathwayProposal, PROMPTS["seed"].format(k=s.n_seed_hypotheses),
                f"Question: {state['question']}\nHypothesis: {p.mechanism_hypothesis}\n"
                f"Exposure: {p.exposure}\nOutcome: {p.outcome}\nClaims:\n{_claims_summary(claims)}",
                ctx={"parsed": p, "claims": claims})
            for pw in out.pathways[:s.n_seed_hypotheses]:
                keys, lab = pf.proposal_keys(pw, r)
                labels.update(lab)
                _add(hyps, keys, "llm_seed", pw.name, pw.rationale, diverse=True)
            seed_status = "OK"
        except Exception as e:
            print(f"[portfolio] seeding failed ({type(e).__name__}); ledger paths only")
            seed_status = "FAILED"

    # opposing claims judged context-dependent / not comparable do not count against a link
    conflicts = [_as(Conflict, c) for c in state.get("conflicts", [])]
    discounted = {frozenset((a, b)) for c in conflicts if c.verdict != "true_conflict"
                  for a in c.claim_ids for b in c.claim_ids if a < b}
    extra = {k for h in hyps for k in h.links}
    links = pf.build_links(claims, prior, rt.pair_cache, labels, s, extra, discounted)
    for path in pf.ledger_paths(links, exposure, outcomes, s.max_path_len):
        if len(hyps) >= s.max_hypotheses:      # no slot: adding then trimming would churn ids
            break
        via = [links[k].object_label for k in path[:-1]]
        _add(hyps, path, "ledger_path", "Literature-graph route" + (f" via {', '.join(via)}" if via
             else ": direct"), "found by graph search over supported steps", False)
    novel = pf.novel_intermediates(links, hyps, exposure, outcomes)
    if novel and len(hyps) < s.max_hypotheses:            # gated expansion, at most one per round
        try:
            out = rt.llm.structured(
                "expand", PathwayProposal, PROMPTS["expand"],
                f"Question: {state['question']}\nUnused intermediates: "
                f"{[labels.get(n, n) for n in novel]}\nCurrent pathways:\n" + "\n".join(
                    pf.label_pathway(h, links) for h in hyps) + f"\nClaims:\n{_claims_summary(claims)}",
                ctx={"novel": [labels.get(n, n) for n in novel], "parsed": p})
            for pw in out.pathways:
                keys, lab = pf.proposal_keys(pw, r)
                labels.update(lab)
                if _add(hyps, keys, "llm_expansion", pw.name, pw.rationale, diverse=True):
                    break
        except Exception as e:
            print(f"[portfolio] expansion failed ({type(e).__name__})")
    links = pf.build_links(claims, links, rt.pair_cache, labels, s,
                           {k for h in hyps for k in h.links}, discounted)

    cats = {cid: c.category for cid, c in r.concepts.items()}
    hyps = pf.evaluate(hyps, links, cats, p.expected_direction, s)
    rnd = state.get("round_idx", 0) + 1
    targets = pf.allocate(hyps, links, s, rnd)
    decision, gate = pf.decide(hyps, targets, rnd, s)
    if decision == "done":
        for k in targets:
            links[k].times_targeted -= 1
        targets = []
    print(f"[portfolio] round {rnd}: " + ", ".join(f"{h.id}={pf.STATUS_LABEL[h.status]}:{h.score}"
                                                    for h in hyps)
          + f" | {pf.STOP_LABEL[gate]} | targets={[f'{links[k].subject_label}->{links[k].object_label}' for k in targets]}")
    return {"links": links, "hypotheses": hyps, "targets": targets, "decision": decision,
            "gate": gate, "round_idx": rnd, "seed_status": seed_status,
            "portfolio_history": [{"round": rnd, "gate": gate,
                                   "hypotheses": [(h.id, pf.STATUS_LABEL[h.status], h.score) for h in hyps],
                                   "targets": [f"{links[k].subject_label} -> {links[k].object_label}"
                                               for k in targets]}]}


def review(state, rt):
    """Optional human checkpoint right after seeding: the cheapest point to correct the search.
    Kept LLM-free because LangGraph re-runs an interrupted node from its start."""
    if not (rt.settings.interactive and state["round_idx"] == 1):
        return {}
    hyps = [_as(Hypothesis, h) for h in state["hypotheses"]]
    answer = interrupt({"message": "Drop pathways by id, e.g. {'drop': ['H2']}",
                        "pathways": [(h.id, h.name, pf.STATUS_LABEL[h.status], h.score) for h in hyps]}) or {}
    return {"hypotheses": [h for h in hyps if h.id not in set(answer.get("drop", []))]}


def run_warnings(state, rt) -> list[str]:
    w = []
    claims = [_as(Claim, c) for c in state.get("claims", [])]
    ids = {i for c in claims for i in (c.subject_concept, c.object_concept)}
    local = [i for i in ids if rt.resolver.concepts.get(i) and rt.resolver.concepts[i].source == "local"]
    if ids and len(local) / len(ids) > 0.5:
        w.append(f"Entity resolution DEGRADED: {len(local)}/{len(ids)} concepts unresolved (LOCAL).")
    if rt.embedder.name != "sentence":
        w.append(f"Candidate pairs came from the '{rt.embedder.name}' encoder, not sentence embeddings.")
    if state.get("semantic_status") in {"PARTIAL", "UNAVAILABLE"}:
        w.append(f"Semantic adjudication {state['semantic_status']}: absent conflicts do not mean concordance.")
    if state.get("conflict_status") == "LLM_FAILED":
        w.append("Conflict triage failed: opposing findings were all counted as contradictions.")
    if state.get("gate") == "MAX_ROUNDS":
        w.append("Search stopped at the round limit while some pathways were still open.")
    if state.get("seed_status") == "FAILED":
        w.append("Pathway seeding failed; only ledger-derived pathways were considered.")
    pending = sum(1 for c in claims if not c.relation_norm)
    if pending:
        w.append(f"{pending} claim relations could not be resolved and were excluded from support.")
    done = set(state.get("extracted_pmids", []))
    waiting = [p for p in state.get("papers", []) if _as(Paper, p).screen_status == "included"
               and _as(Paper, p).pmid not in done]
    if waiting:
        w.append(f"{len(waiting)} included papers were never extracted (per-round budget).")
    return w


def synthesize(state, rt):
    links = {k: _as(LinkEvidence, v) for k, v in state["links"].items()}
    hyps = [_as(Hypothesis, h) for h in state["hypotheses"]]
    tags = {}
    for h in hyps:
        for k in h.links:
            tags.setdefault(k, f"L{len(tags) + 1}")
    used = {i for k in tags for i in links[k].support_ids + links[k].corroborating_ids
            + links[k].contradicting_ids + links[k].context_dependent_ids}
    claims = [c for c in (_as(Claim, x) for x in state["claims"]) if c.id in used]
    conflicts = [_as(Conflict, c) for c in state.get("conflicts", [])]
    warnings = run_warnings(state, rt)
    user = (f"Question: {state['question']}\nGate: {state.get('gate')} | semantic: "
            f"{state.get('semantic_status')} | warnings: {warnings or 'none'}\n\nPathways:\n" +
            "\n".join(f"[{h.id}] {h.name} | {pf.STATUS_LABEL[h.status]} ({h.reason}) | score={h.score}"
                      f" | flags={[pf.FLAG_LABEL[f] for f in h.logic_flags]}\n" +
                      "\n".join(f"  [{tags[k]}] {links[k].subject_label} --{links[k].relation}--> "
                                f"{links[k].object_label} | {pf.STATUS_LABEL[links[k].status]} "
                                f"({links[k].reason}) | grade={links[k].grade} | support="
                                f"{links[k].support_ids} contra={links[k].contradicting_ids} "
                                f"context-dependent={links[k].context_dependent_ids}"
                                for k in h.links) for h in hyps) +
            "\n\nConflicts:\n" + ("\n".join(f"- {c.verdict}: {c.explanation}" for c in conflicts) or "none") +
            "\n\nClaim ledger:\n" + "\n".join(
                f"[{c.id}] ({c.grade}) {c.subject_label} {c.relation_norm} {c.object_label} | "
                f"{c.study_type} | {c.context_cell_type}" for c in claims))
    text = rt.llm.text("synthesize", PROMPTS["synthesize"], user,
                       ctx={"hypotheses": hyps, "links": links, "tags": tags, "claims": claims})
    return {"synthesis": text, "link_tags": tags, "warnings": warnings}


def _entailment(text, claims, rt):
    by_id = {c.id: c for c in claims}
    sents = prose_sentences(text)
    items = [(i, s, [x for x in re.findall(r"\[([A-Za-z0-9_\-]+)\]", s) if x in by_id])
             for i, s in enumerate(sents)]
    items = [x for x in items if x[2]]
    if not items:
        return []
    user = "\n\n".join(f"[SENTENCE {i}] {s}\nCited:\n" + "\n".join(
        f"  [{c}] ({by_id[c].grade}) {by_id[c].subject_label} {by_id[c].relation_norm} "
        f"{by_id[c].object_label} | {by_id[c].context_cell_type}" for c in ids) for i, s, ids in items)
    try:
        out = rt.llm.structured("entailment", EntailmentBatch, PROMPTS["entailment"], user,
                                ctx={"items": items}, n_items=len(items))
    except Exception as e:
        return [{"verdict": "check_unavailable", "sentence": type(e).__name__}]
    return [{"sentence": sents[j.sentence_index], "verdict": j.verdict, "why": j.rationale}
            for j in out.judgements if j.verdict != "entailed" and j.sentence_index < len(sents)]


def _step_label(ln) -> str:
    label = pf.STATUS_LABEL[ln.status]
    return label if ln.status == "contradicted" or not ln.support_ids else f"{label}, {ln.grade}"


def verify(state, rt):
    links = {k: _as(LinkEvidence, v) for k, v in state["links"].items()}
    hyps = [_as(Hypothesis, h) for h in state["hypotheses"]]
    claims = [_as(Claim, c) for c in state["claims"]]
    tags = state["link_tags"]
    v = verify_text(state["synthesis"], claims, [h.id for h in hyps] + list(tags.values()))
    v["entailment"] = _entailment(state["synthesis"], claims, rt)
    v["passed"] = v["passed"] and not any(e["verdict"] == "unsupported" for e in v["entailment"])
    table = ["| Pathway | Verdict | Why | Score | Source | Logic warnings | Steps |",
             "|---|---|---|---|---|---|---|"]
    for h in hyps:
        table.append(f"| {h.id} {h.name} | {pf.STATUS_LABEL[h.status]} | {h.reason} | {h.score} | "
                     f"{pf.ORIGIN_LABEL[h.origin]} | {'; '.join(pf.FLAG_LABEL[f] for f in h.logic_flags) or '-'} | "
                     + "; ".join(f"{tags[k]} {links[k].subject_label}→{links[k].object_label} "
                                 f"({_step_label(links[k])})" for k in h.links) + " |")
    stop = (f"Search stopped after {state['round_idx']} rounds: {pf.STOP_LABEL[state['gate']]}. "
            "Scores rank pathways; they are not probabilities.")
    report = (state["synthesis"] + "\n\n---\n## Pathway portfolio (computed)\n" + stop + "\n\n"
              + "\n".join(table)
              + "\n\n## Run-quality warnings\n" + ("\n".join(f"- {x}" for x in state["warnings"]) or "- none")
              + "\n\n## Verification\n" + ("- PASS" if v["passed"] else "\n".join(
                  [f"- uncited: {s[:120]}" for s in v["uncited"][:5]] +
                  [f"- overclaim (tier {o['used']} > {o['allowed']}): {o['sentence'][:120]}" for o in v["overclaims"][:5]] +
                  [f"- unknown claim id: {i}" for i in v["unknown_ids"][:5]] +
                  [f"- missing tag: {t}" for t in v["missing_tags"][:10]] +
                  [f"- entailment {e['verdict']}: {e['sentence'][:120]}" for e in v["entailment"][:5]])))
    print(f"[verify] {'PASS' if v['passed'] else 'VIOLATIONS'}: uncited={len(v['uncited'])} "
          f"overclaims={len(v['overclaims'])} missing_tags={len(v['missing_tags'])} "
          f"entailment_issues={len(v['entailment'])}")
    return {"verification": v, "report": report}


# ── assembly ────────────────────────────────────────────────────────────────
def build_agent(rt: Runtime, checkpointer=None):
    g = StateGraph(State)
    for name, fn in [("parse", parse), ("plan", plan), ("search", search), ("screen", screen),
                     ("extract", extract), ("normalize", normalize), ("semantic", semantic),
                     ("portfolio", portfolio), ("review", review), ("synthesize", synthesize),
                     ("verify", verify)]:
        g.add_node(name, partial(fn, rt=rt))
    g.add_edge(START, "parse")
    g.add_edge("parse", "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "screen")
    g.add_conditional_edges("screen", partial(fan_out, rt=rt), ["extract", "normalize"])
    g.add_edge("extract", "normalize")
    g.add_edge("normalize", "semantic")
    g.add_edge("semantic", "portfolio")
    g.add_edge("portfolio", "review")
    g.add_conditional_edges("review", lambda s: "plan" if s["decision"] == "search_more" else "synthesize",
                            ["plan", "synthesize"])
    g.add_edge("synthesize", "verify")
    g.add_edge("verify", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def run(question: str, rt: Runtime, thread_id: str | None = None):
    agent = build_agent(rt)
    cfg = {"configurable": {"thread_id": thread_id or f"bmira-{uuid.uuid4()}"}, "recursion_limit": 250}
    return agent.invoke({"question": question}, cfg)
