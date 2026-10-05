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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from bmira import portfolio as pf
from bmira import shadow
from bmira.config import Settings
from bmira.evidence import grade_claim, is_proposal, prose_sentences, verify_text
from bmira.llm import PROMPTS, spent_usd
from bmira.evidence import claim_study_type, verify_methods
from bmira.normalize import (EntityResolver, _mentioned, _previous_sentence, abbreviations, check_claim,
                             consolidate_aliases, expand_abbreviations, entity_change, entity_of, entity_parts,
                             lexical_relation, lookup_key, split_change, with_mark)
from bmira.schemas import (BLOCKING_RELATION, Claim, ClaimList, Conflict, DIRECTION, EntailmentBatch, Hypothesis,
                           LinkEvidence, Paper, ParsedQuestion, PathwayProposal, QueryPlan,
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
    judge: object | None = None                   # decision model in shadow mode; None = off

    def __post_init__(self):
        self.resolver = self.resolver or EntityResolver(self.settings, self.llm)

    @classmethod
    def live(cls, settings: Settings, api_key: str | None = None):
        from bmira.judge import make_judge
        from bmira.llm import LangChainLLM
        from bmira.semantic import SentenceEmbedder
        from bmira.sources import PubMedSource
        llm = LangChainLLM(settings, api_key)
        judge = make_judge(settings)
        return cls(settings, llm, PubMedSource(settings), SentenceEmbedder(), judge=judge)


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
    outcome_ids: list
    queries: list
    papers: Annotated[list, _merge(Paper, "pmid")]
    claims: Annotated[list, _merge(Claim, "id")]
    dropped_claims: Annotated[list, operator.add]
    extracted_pmids: Annotated[list, operator.add]
    search_status: str
    target_searches: dict
    search_log: Annotated[list, operator.add]
    semantic_status: str
    semantic_edges: list
    conflicts: list
    conflict_candidates: int
    conflict_status: str
    links: dict
    mediation: dict                                # blocking tests by exposure|mediator|outcome (EM-3)
    hypotheses: list
    targets: list
    gate: str
    decision: str
    seed_status: str
    round_idx: int
    portfolio_history: Annotated[list, operator.add]
    judge_log: Annotated[list, operator.add]       # shadow judgments beside today's decisions
    synthesis: str
    link_tags: dict
    warnings: list
    verification: dict
    report: str


# ── nodes ───────────────────────────────────────────────────────────────────
def parse(state, rt):
    p = rt.llm.structured("parse", ParsedQuestion, PROMPTS["parse"], state["question"],
                          ctx={"question": state["question"]})
    if not p.in_scope:                                     # no search, no cost: say why and stop
        print(f"[parse][WARN] question out of scope: {p.scope_note}")
        return {"parsed": p, "round_idx": 0, "report": (
            f"This question was not investigated: {p.scope_note or 'it is not an exposure-outcome question'} "
            "B-MiRA answers 'does X affect Y, and through which mechanisms?' from PubMed literature.")}
    bare, change = split_change(p.exposure)                # 'NAD+ decline' -> 'NAD+', exposure_change down
    p = p.model_copy(update={"exposure": bare, "exposure_change": change or p.exposure_change})
    exp, out = rt.resolver.resolve(entity_of(p.exposure)[0]), rt.resolver.resolve(entity_of(p.outcome)[0])
    readouts = [rt.resolver.resolve(entity_of(x)[0]) for x in p.outcome_readouts]
    for name in p.exposure_members:     # papers name 'empagliflozin', the question says 'SGLT2 inhibitors': one node
        m = rt.resolver.resolve(entity_of(name)[0])
        if m.id != exp.id:
            rt.resolver.alias[m.id] = exp.id
    print(f"[parse] exposure={exp.label} outcome={out.label} readouts={[r.label for r in readouts]} "
          f"members={p.exposure_members} expected={p.expected_direction} exposure_change={p.exposure_change} "
          f"population={p.target_system}")
    return {"parsed": p, "exposure": exp.id, "outcome": out.id, "round_idx": 0, "targets": [],
            "outcome_ids": [out.id] + [r.id for r in readouts]}


def plan(state, rt):
    links = {k: _as(LinkEvidence, v) for k, v in state.get("links", {}).items()}
    targets = [links[k] for k in state.get("targets", []) if k in links]
    p = _as(ParsedQuestion, state["parsed"])
    user = f"Question: {state['question']}\nStructured: {p.model_dump_json()}\n"
    if targets:
        user += "Targets:\n" + "\n".join(      # numbered: models can't reliably echo ontology-id keys (K5, P4)
            f"[T{n}] {t.subject_label} --{t.relation}--> {t.object_label} "
            f"(papers={t.n_studies}, status={t.status})" for n, t in enumerate(targets, 1))
    out = rt.llm.structured("plan", QueryPlan, PROMPTS["plan"], user,
                            ctx={"parsed": p, "targets": targets})
    if not targets:                             # LLM output is a trust boundary: degrade, don't crash
        queries = out.queries or [SearchQuery(query=f"{p.exposure} {p.outcome}", intent="broad")]
        missing = {"broad", "mechanism", "contradiction", "negative_result"} - {q.intent for q in queries}
        if missing:
            print(f"[plan][WARN] coverage round lacks intents {sorted(missing)}")
    else:
        keys = {t.key for t in targets}
        norm = lambda k: k.strip().strip("[]`'\" ")             # the model copies ids with brackets or quotes
        by_key = {norm(k): k for k in keys}

        def target_of(q):                                        # 'T2', '2', '[T2]'; a scripted model echoes the key
            m = re.fullmatch(r"\D*(\d{1,2})\D*", q.target.strip())
            return by_key.get(norm(q.target)) or (targets[int(m[1]) - 1].key if m and 1 <= int(m[1]) <= len(targets)
                                                  else None)
        queries = [q.model_copy(update={"target": t}) for q in out.queries if (t := target_of(q))]
        if len(queries) < len(out.queries) or not queries:       # pilot5: 3 queries for 3 targets, no sign why
            print(f"[plan][WARN] {len(out.queries) - len(queries)} of {len(out.queries)} model queries named "
                  f"no known target and were dropped; targets {sorted(keys)}, model targets "
                  f"{sorted({q.target for q in out.queries})}")
        queries += [_synonym_query(t, rt) for t in targets]   # every step: one query built by code
    print(f"[plan] round {state.get('round_idx', 0) + 1}: {len(queries)} queries"
          + (f" for {len(targets)} target links" if targets else " (coverage round)"))
    return {"queries": queries}


def _synonym_query(t, rt) -> SearchQuery:
    """(names of the subject) AND (names of the object), from every surface form that
    resolved to each concept. Exact quoted labels often match nothing and then looked like
    an absence of evidence."""
    def names(cid, label):
        forms = {label} | {k for k, c in rt.resolver.cache.items() if rt.resolver.canonical(c).id == cid}
        return "(" + " OR ".join(f"({n})" for n in sorted(forms, key=len)[:4]) + ")"
    return SearchQuery(query=f"{names(t.subject, t.subject_label)} AND {names(t.object, t.object_label)}",
                       intent="gap_alternative_terms", target=t.key)


def search(state, rt):
    known = {p.pmid: p for p in (_as(Paper, x) for x in state.get("papers", []))}
    log, new_ids, hits_for, stats = [], [], {}, {}
    for q in state["queries"]:
        q = _as(SearchQuery, q)
        st = stats.setdefault(q.target, {"ok": 0, "clean": 0}) if q.target else {}
        try:
            res = rt.source.search(q.query, rt.settings.max_papers_per_target_query if q.target
                                   else rt.settings.max_papers_per_query)
        except Exception as e:
            log.append({"query": q.query, "target": q.target, "ok": False, "error": type(e).__name__})
            continue
        new_ids += [i for i in res["pmids"] if i not in known and i not in new_ids]
        if q.target:
            st["ok"] += 1
            st["clean"] += not res.get("ignored_terms")
            for pid in res["pmids"]:
                hits_for.setdefault(pid, set()).add(q.target)
        log.append({"query": q.query, "target": q.target, "ok": True, "hits": len(res["pmids"]),
                    "translation": res.get("translation", ""), "ignored_terms": res.get("ignored_terms", [])})
    try:
        papers = rt.source.fetch(new_ids) if new_ids else []
        fetched = True
    except Exception as e:
        print(f"[search] fetch failed ({type(e).__name__}); this round does not count as a search")
        papers, fetched = [], False
    for p in papers:
        p.retrieved_for = sorted(hits_for.get(p.pmid, ()))
    updated = []                                  # known papers found again for a new step
    for pid, steps in hits_for.items():
        if pid in known and not steps <= set(known[pid].retrieved_for):
            updated.append(known[pid].model_copy(update={
                "retrieved_for": sorted(set(known[pid].retrieved_for) | steps)}))
    ran = sum(x["ok"] for x in log)
    status = "SEARCH_FAILED" if not ran or not fetched else "NEW_RESULTS" if papers else "ZERO_NEW_RESULTS"
    print(f"[search] {status}: {ran}/{len(log)} queries ran, {len(papers)} new papers, "
          f"{len(updated)} known papers matched new steps")
    return {"papers": papers + updated, "search_status": status, "search_log": log,
            "target_searches": stats}


def screen(state, rt):
    todo = [_as(Paper, p) for p in state["papers"] if _as(Paper, p).screen_status == "unscreened"]
    for p in todo:
        if p.retracted:
            p.screen_status, p.relevance_reason = "excluded", "retracted / expression of concern"
    live = [p for p in todo if not p.retracted]
    links = {k: _as(LinkEvidence, v) for k, v in state.get("links", {}).items()}

    def steps(p):                               # judge a targeted hit against its step, not only the question
        named = [links[k] for k in p.retrieved_for if k in links]
        return ("\nRetrieved for these mechanism steps:\n" + "\n".join(
            f"- {t.subject_label} --{t.relation}--> {t.object_label}" for t in named)) if named else ""
    if live:
        users = [f"Question: {state['question']}{steps(p)}\n\nTitle: {p.title}\nAbstract: {p.abstract[:3000]}"
                 for p in live]
        try:
            verdicts = rt.llm.structured_many("screen", Screen, PROMPTS["screen"], users, ctxs=live)
        except Exception as e:                  # papers stay unscreened and are retried next round
            print(f"[screen] batch failed ({type(e).__name__}); retrying next round")
            verdicts = []
        for p, r in zip(live, verdicts):
            if r is None:                       # this paper's reply failed: stays unscreened, retried next round
                continue
            keep = r.relevant and r.relevance_score >= rt.settings.min_relevance
            p.screen_status = "included" if keep else "excluded"
            p.relevance_score = r.relevance_score
            p.relevance_reason = r.reason if keep or not r.relevant else f"below relevance cut-off: {r.reason}"
            p.study_type = p.pubtype_study_type or r.study_type   # metadata outranks the model
    todo = [p for p in todo if p.screen_status != "unscreened"]
    print(f"[screen] {len(todo)} screened, {sum(p.screen_status == 'included' for p in todo)} "
          f"included, {sum(p.retracted for p in todo)} retracted excluded")
    log = shadow.screen(rt, [p for p in todo if not p.retracted], _as(ParsedQuestion, state["parsed"]), {
        p.pmid: [(k, f"{links[k].subject_label} --{links[k].relation}--> {links[k].object_label}")
                 for k in p.retrieved_for if k in links] for p in todo})
    return {"papers": todo, "judge_log": log}


def _readable(p) -> bool:
    """Included and primary. TE-1: a review's claims never count (R1), so it is not extracted; its abstract
    goes to pathway seeding instead (pilot4 extracted 8 reviews for 28 uncounted claims)."""
    return p.screen_status == "included" and p.study_type != "review"


def fan_out(state, rt):
    """Papers retrieved for a step are read FOR that step first (even if read before);
    then unread papers by relevance. Previously a targeted round could 'search' a step
    without anyone reading the hits for it."""
    done = set(state.get("extracted_pmids", []))
    links = {k: _as(LinkEvidence, v) for k, v in state.get("links", {}).items()}
    claims = [_as(Claim, c) for c in state.get("claims", [])] + \
             [_as(Claim, c) for c in state.get("dropped_claims", [])]
    pool = sorted((p for p in (_as(Paper, x) for x in state["papers"]) if _readable(p)),
                  key=lambda p: (-p.relevance_score, p.pmid))
    pending = lambda p: [k for k in p.retrieved_for if k not in p.read_for]
    todo = [p for p in pool if pending(p)] + [p for p in pool if p.pmid not in done and not pending(p)]
    sends = []
    for p in todo[:rt.settings.max_extract_per_round]:          # the rest wait, not dropped
        mine = [c for c in claims if c.pmid == p.pmid]
        sends.append(Send("extract", {
            "paper": p, "question": state["question"], "round": state.get("round_idx", 0) + 1,
            "focus": [links[k] for k in pending(p) if k in links], "focus_keys": pending(p),
            "n_existing": len(mine),
            "seen": {(lookup_key(c.span), lookup_key(c.subject), lookup_key(c.relation), lookup_key(c.object))
                     for c in mine}}))
    return sends or ["normalize"]


def _check(c, paper, abbrevs, kept, dropped):
    reason, warnings = check_claim(c, paper.source_text, abbrevs)
    if reason:
        c.drop_reason = reason
        dropped.append(c)
        return
    c.anchored, c.method_checks = True, warnings
    perturbed = c.perturbation_class != "none"            # as extracted, before the cue check below
    verify_methods(c, paper.source_text)
    if reason := _blocking_check(c, paper, abbrevs, perturbed):
        c.drop_reason = reason
        dropped.append(c)
        return
    kept.append(c)


def _blocking_check(c, paper, abbrevs, perturbed: bool) -> str:
    """EM-2: a blocking test needs a perturbation and its treatment named in the quote or the sentence
    before it; otherwise its meaning ('Gpr81-/- cells: lactate no longer reduced IFNG') cannot be stated as
    an ordinary edge either, so it is dropped with its reason. Half-filled fields are ignored.
    `perturbed` is the extractor's perturbation: the cue check misses notations such as 'Slc5a8-null'
    (pilot5), and that miss already lowers the grade; it must not also discard the experiment."""
    if not (c.effect_exposure.strip() or c.effect_result):
        return ""
    if not (c.effect_exposure.strip() and c.effect_result):
        c.effect_exposure, c.effect_result = "", ""
        c.method_checks.append("incomplete blocking-test fields ignored")
        return ""
    if not perturbed:
        return "blocking test without a perturbation"
    context = _previous_sentence(c.span, paper.source_text) + " " + c.span
    if not _mentioned(c.effect_exposure, context, abbrevs):
        return "blocking-test treatment not named in quote"
    if c.subject_lost:                          # the result already says what the loss did: never flip it
        c.subject_lost = False
        c.method_checks.append("subject_lost ignored on a blocking test")
    return ""


def extract(payload, rt):
    paper = rt.source.fulltext(payload["paper"])
    s = rt.settings
    focus = "".join(f"\n- {t.subject_label} --{t.relation}--> {t.object_label}" for t in payload["focus"])
    text = paper.source_text[:s.fulltext_char_limit + 2000]
    try:
        out = rt.llm.structured(
            "extract", ClaimList, PROMPTS["extract"].format(max_claims=s.max_claims_per_paper),
            f"Question: {payload['question']}\n" + (f"Focus steps (report any finding on them, "
                                                    f"including null or opposite):{focus}\n" if focus else "")
            + f"\nPaper (PMID {paper.pmid}, {paper.text_access}):\n{text}",
            ctx={"paper": paper, "focus": payload["focus"]})
    except Exception as e:
        print(f"[extract] PMID {paper.pmid} failed ({type(e).__name__}); retried next round")
        return {}
    kept, dropped, n, records = [], [], payload["n_existing"], []
    abbrevs = abbreviations(paper.source_text)
    for ec in out.claims[:s.max_claims_per_paper]:
        subjects, s_tissue = entity_parts(ec.subject)        # 'NFAT1 and SMAD3' -> one claim each
        objects, o_tissue = entity_parts(ec.object)
        tissue = ", ".join(t for t in (ec.context_tissue, s_tissue, o_tissue) if t)
        for subj in subjects:
            for obj in objects:
                sig = (lookup_key(ec.span), lookup_key(subj), lookup_key(ec.relation), lookup_key(obj))
                if sig in payload["seen"]:                  # re-read for a step: no duplicates
                    continue
                payload["seen"].add(sig)
                c = Claim(**{**ec.model_dump(), "subject": subj, "object": obj, "context_tissue": tissue},
                          id=f"C{paper.pmid}_{n}", pmid=paper.pmid, round=payload["round"],
                          study_type=claim_study_type(paper.study_type, paper.pubtype_study_type, ec.system),
                          text_access=paper.text_access, relation_raw=ec.relation)
                n += 1
                before = c.model_copy()
                _check(c, paper, abbrevs, kept, dropped)
                records.append((before, c))
    reads = sorted(set(paper.read_for) | set(payload["focus_keys"]))
    print(f"[extract] {paper.pmid}: {len(kept)} kept, {len(dropped)} dropped"
          + (f" ({', '.join(c.drop_reason for c in dropped)})" if dropped else "")
          + (f"; read for {len(payload['focus_keys'])} step(s)" if payload["focus_keys"] else ""))
    return {"claims": kept, "dropped_claims": dropped, "extracted_pmids": [paper.pmid],
            "papers": [paper.model_copy(update={"read_for": reads, "n_reads": paper.n_reads + 1,
                                                "chars_read": paper.chars_read + len(text)})],
            "judge_log": shadow.claims(rt, paper, records)}


def _surfaces(c) -> tuple[str, str]:
    return (with_mark(c.subject, c.subject_attribute, c.span), with_mark(c.object, c.object_attribute, c.span))


def _long_forms(state) -> dict:
    """pmid -> {abbreviation: long form} from each paper's own definitions ('sodium butyrate (SB)')."""
    return {p.pmid: expand_abbreviations(abbreviations(p.source_text or f"{p.title}. {p.abstract}"))
            for p in (_as(Paper, x) for x in state.get("papers", []))}


def _long(x, forms) -> str | None:
    lf = forms.get(x.strip().lower())
    return lf if lf and x.strip().lower() not in lf.split() else None


def _set_concepts(c, rt, forms=None):
    """`forms`: this paper's abbreviations. A long form replaces an abbreviation only when the
    abbreviation resolved to nothing but a LOCAL id and the long form did better: pilot5 left 'SB' (sodium
    butyrate) as LOCAL:sb, while 'GPR109A' and 'iTreg' already resolve and must not move."""
    local = lambda x: rt.resolver.resolve(entity_of(x)[0]).id.startswith("LOCAL:")
    pick = lambda x: lf if (lf := _long(x, forms or {})) and local(x) and not local(lf) else x
    (se, sa, st), (oe, oa, ot) = (entity_of(pick(x)) for x in _surfaces(c))
    if c.subject_attribute == "none":
        c.subject_attribute = sa
    if c.object_attribute == "none":
        c.object_attribute = oa
    tissues = [t for t in (c.context_tissue, st, ot) if t]
    c.context_tissue = ", ".join(dict.fromkeys(", ".join(tissues).split(", "))) if tissues else ""
    s, o = rt.resolver.resolve(se), rt.resolver.resolve(oe)
    c.subject_concept, c.subject_label, c.subject_category = s.id, s.label, s.category
    c.object_concept, c.object_label, c.object_category = o.id, o.label, o.category
    c.subject_parents, c.object_parents = list(s.parents), list(o.parents)
    c.context_concept = rt.resolver.resolve(c.context_cell_type).id if c.context_cell_type else ""
    if c.is_blocking_test:
        x = rt.resolver.resolve(entity_of(pick(c.effect_exposure))[0])
        c.effect_exposure_concept, c.effect_exposure_label = x.id, x.label


def normalize(state, rt):
    claims = [_as(Claim, c) for c in state.get("claims", [])]
    fresh = [c for c in claims if not c.relation_norm]        # typed this round: restate loss claims once
    for c in claims:
        if not c.relation_norm and c.is_blocking_test:        # EM-1: the result, not the wording, types it
            c.relation_norm, c.relation_source = BLOCKING_RELATION[c.effect_result], "blocking_test"
        elif not c.relation_norm:
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
    log = shadow.relations(rt, [(c, c.relation_norm, c.relation_source) for c in fresh])   # as worded, before restatement
    for c in fresh:      # 'Tet2 loss increases IL-6' and 'Gpr109a-/- mice show fewer DCs' are 'Tet2 decreases IL-6'
        lost = c.subject_lost or entity_change(c.subject) == "down"      # for the bare entity; one flip, not two
        if c.is_blocking_test:                                           # typed from its result: never restated
            continue
        if c.relation_norm in DIRECTION and lost != (entity_change(c.object) == "down"):
            c.relation_norm = "decreases" if c.relation_norm == "increases" else "increases"
    forms = _long_forms(state)
    rt.resolver.resolve_many([entity_of(e)[0] for c in claims for e in _surfaces(c)]
                             + [entity_of(lf)[0] for c in claims for e in _surfaces(c)
                                if (lf := _long(e, forms.get(c.pmid, {})))]
                             + [c.context_cell_type for c in claims if c.context_cell_type]
                             + [entity_of(c.effect_exposure)[0] for c in claims if c.is_blocking_test])
    for c in claims:
        _set_concepts(c, rt, forms.get(c.pmid))
    nodes = [n for h in state.get("hypotheses", []) for k in _as(Hypothesis, h).links
             for n in (pf.split_key(k)[0], pf.split_key(k)[2])]          # pathway nodes join the merge
    merged = consolidate_aliases(claims, rt.resolver, rt.llm, rt.alias_verdicts, extra_ids=nodes)
    target = _as(ParsedQuestion, state["parsed"]).target_system
    for c in claims:
        _set_concepts(c, rt, forms.get(c.pmid))            # cheap: cached + alias registry
        grade_claim(c, target)
    rt.resolver.save()
    print(f"[normalize] {len(claims)} claims, {len(pending)} relations sent to LLM, "
          f"{merged} alias merges")
    return {"claims": claims, "judge_log": log}


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
            "conflicts": conflicts, "conflict_status": cstatus if fresh else "COMPLETE",
            "conflict_candidates": len(cands)}


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


def _review_digest(papers, n=20, chars=600) -> str:
    """TE-1: included reviews, read by title and abstract only, as background for proposing pathways."""
    reviews = sorted((p for p in papers if p.screen_status == "included" and p.study_type == "review"),
                     key=lambda p: (-p.relevance_score, p.pmid))[:n]
    return ("\nReviews (background for proposing pathways; they count as no evidence):\n" + "\n".join(
        f"- {p.title}: {p.abstract[:chars]}" for p in reviews)) if reviews else ""


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
    # R7: a step's search counts toward "no study found" only if it ran cleanly (>= 2 queries,
    # >= 1 with no terms ignored by PubMed), every hit was read FOR the step, and no new claim
    # on the step's entity pair appeared. Otherwise the round simply does not count.
    this_round = state.get("round_idx", 0) + 1
    papers = [_as(Paper, x) for x in state.get("papers", [])]
    for k in state.get("targets", []):
        ck = _canon_key(k, rt)
        if ck not in prior:
            continue
        st = state.get("target_searches", {}).get(k, {})
        clean = (state.get("search_status") != "SEARCH_FAILED" and st.get("ok", 0) >= 2
                 and st.get("clean", 0) >= 1)
        unread = any(k in p.retrieved_for and k not in p.read_for and _readable(p) for p in papers)
        s_id, _, o_id = pf.split_key(ck)
        found = any(c.round == this_round and (c.subject_concept, c.object_concept) == (s_id, o_id)
                    for c in claims)
        if clean and not unread and not found:
            prior[ck].zero_yield_count += 1
            prior[ck].exhausted = prior[ck].zero_yield_count >= s.zero_yield_rounds_to_exhaust
    hyps = [_as(Hypothesis, h) for h in state.get("hypotheses", [])]
    for h in hyps:
        h.links = [_canon_key(k, rt) for k in h.links]
    labels = {cid: c.label for cid, c in r.concepts.items()}
    ancestors = {cid: tuple(_canon(a, rt) for a in c.ancestors) for cid, c in r.concepts.items() if c.ancestors}
    p = _as(ParsedQuestion, state["parsed"])
    exposure = _canon(state["exposure"], rt)
    named = {_canon(i, rt) for i in state.get("outcome_ids", [state["outcome"]])}
    outcomes = named | {cid for cid, c in r.concepts.items() if named & set(c.ancestors)}
    seed_status = state.get("seed_status", "")

    if not seed_status:                                   # seed exactly once, even if it fails
        try:
            out = rt.llm.structured(
                "seed", PathwayProposal, PROMPTS["seed"].format(k=s.n_seed_hypotheses),
                f"Question: {state['question']}\nHypothesis: {p.mechanism_hypothesis}\n"
                f"Exposure: {p.exposure}" + (" (the question concerns a DECREASE of it: write every link as the "
                                           "effect of an INCREASE of its source)" if p.exposure_change == "down" else "")
                + f"\nOutcome: {p.outcome}\nClaims:\n{_claims_summary(claims)}" + _review_digest(papers),
                ctx={"parsed": p, "claims": claims})
            for pw in out.pathways[:s.n_seed_hypotheses]:
                keys, lab = pf.proposal_keys(pw, r, outcomes)
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
    links = pf.build_links(claims, prior, rt.pair_cache, labels, s, extra, discounted, ancestors)
    slots = lambda: sum(not pf.is_direct(h) for h in hyps)       # direct routes take none (N8)
    direct_to = {pf.split_key(h.links[0])[2] for h in hyps if pf.is_direct(h)}
    for path in pf.ledger_paths(links, exposure, outcomes, s.max_path_len):    # strongest first
        if len(path) == 1:
            end = links[path[0]].object
            # one direct route per named outcome or readout, the strongest: pilot5 had seven, four to Treg
            # differing only in relation, and a subtype's route repeats the broader one
            if end not in named or end in direct_to:
                continue
            direct_to.add(end)
        elif slots() >= s.max_hypotheses - 1:     # the last slot stays free for expansion; no slot, no churn
            continue
        via = [links[k].object_label for k in path[:-1]]
        # multi-step routes that share half their intermediates with a seed repeat it: pilot5's 'via FFAR2'
        _add(hyps, path, "ledger_path", "Literature-graph route" + (f" via {', '.join(via)}" if via
             else f": direct to {links[path[-1]].object_label}"), "found by graph search over supported steps",
             len(path) > 1)
    novel = pf.novel_intermediates(links, hyps, exposure, outcomes)
    if novel and slots() < s.max_hypotheses:              # gated expansion, at most one per round
        try:
            out = rt.llm.structured(
                "expand", PathwayProposal, PROMPTS["expand"],
                f"Question: {state['question']}\nUnused intermediates: "
                f"{[labels.get(n, n) for n in novel]}\nCurrent pathways:\n" + "\n".join(
                    pf.label_pathway(h, links) for h in hyps) + f"\nClaims:\n{_claims_summary(claims)}",
                ctx={"novel": [labels.get(n, n) for n in novel], "parsed": p})
            for pw in out.pathways:
                keys, lab = pf.proposal_keys(pw, r, outcomes)
                labels.update(lab)
                if _add(hyps, keys, "llm_expansion", pw.name, pw.rationale, diverse=True):
                    break
        except Exception as e:
            print(f"[portfolio] expansion failed ({type(e).__name__})")
    links = pf.build_links(claims, links, rt.pair_cache, labels, s,
                           {k for h in hyps for k in h.links}, discounted, ancestors)

    cats = {cid: c.category for cid, c in r.concepts.items()}
    mediation = pf.build_mediation(claims, exposure, named, ancestors, labels, s, canon=lambda x: _canon(x, rt))
    hyps = pf.evaluate(hyps, links, cats, pf.pathway_sign(p.expected_direction, p.exposure_change), s,
                       mediation, ancestors)
    rnd = this_round
    targets = pf.allocate(hyps, links, s, rnd)
    decision, gate = pf.decide(hyps, targets, rnd, s)
    used = sum(getattr(rt.llm, "tokens_in", {}).values()) + sum(getattr(rt.llm, "tokens_out", {}).values())
    if decision == "search_more" and s.budget_tokens and used >= s.budget_tokens:
        decision, gate = "done", "BUDGET"            # soft cap: stop searching, still report
    if decision == "search_more" and s.budget_usd is not None:
        spent, _ = spent_usd(rt.llm, rt.judge, s.prices)
        if spent >= max(0.0, s.budget_usd):
            decision, gate = "done", "COST_BUDGET"
    if decision == "done":
        for k in targets:
            links[k].times_targeted -= 1
        targets = []
    print(f"[portfolio] round {rnd}: " + ", ".join(f"{h.id}={pf.ROUTE_LABEL[h.status]}:{h.score}"
                                                    for h in hyps)
          + f" | {pf.STOP_LABEL[gate]} | targets={[f'{links[k].subject_label}->{links[k].object_label}' for k in targets]}")
    return {"links": links, "mediation": mediation, "hypotheses": hyps, "targets": targets, "decision": decision,
            "gate": gate, "round_idx": rnd, "seed_status": seed_status,
            "portfolio_history": [{"round": rnd, "gate": gate,
                                   "hypotheses": [(h.id, pf.ROUTE_LABEL[h.status], h.score) for h in hyps],
                                   "targets": [f"{links[k].subject_label} -> {links[k].object_label}"
                                               for k in targets]}]}


def review(state, rt):
    """Optional human checkpoint right after seeding: the cheapest point to correct the search.
    Kept LLM-free because LangGraph re-runs an interrupted node from its start."""
    if not (rt.settings.interactive and state["round_idx"] == 1):
        return {}
    hyps = [_as(Hypothesis, h) for h in state["hypotheses"]]
    answer = interrupt({"message": "Drop pathways by id, e.g. {'drop': ['H2']}",
                        "pathways": [(h.id, h.name, pf.ROUTE_LABEL[h.status], h.score) for h in hyps]}) or {}
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
    if state.get("gate") == "BUDGET":
        w.append("Search stopped early because the token budget was reached; open pathways were not searched further.")
    if state.get("gate") == "COST_BUDGET":
        w.append("Search stopped early because the cost budget was reached; open pathways were not searched further.")
    if rt.settings.budget_usd is not None and (missing := spent_usd(rt.llm, rt.judge, rt.settings.prices)[1]):
        w.append(f"The cost budget cannot see models without a price ({', '.join(missing)}); their tokens were "
                 "not counted against it.")
    if state.get("gate") == "MAX_ROUNDS":
        w.append("Search stopped at the round limit while some pathways were still open.")
    if state.get("seed_status") == "FAILED":
        w.append("Pathway seeding failed; only ledger-derived pathways were considered.")
    dropped = [_as(Claim, c) for c in state.get("dropped_claims", [])]
    if dropped:
        why = {}
        for c in dropped:
            why[c.drop_reason] = why.get(c.drop_reason, 0) + 1
        w.append(f"{len(dropped)} extracted claims failed their quote checks and were dropped: "
                 + ", ".join(f"{n} {r}" for r, n in sorted(why.items())) + ".")
    reset = sum(bool(c.method_checks) for c in claims)
    if reset:
        w.append(f"{reset} claims had method details (perturbation, rescue, controls) without "
                 "supporting text; those details were not credited.")
    pending = sum(1 for c in claims if not c.relation_norm)
    if pending:
        w.append(f"{pending} claim relations could not be resolved and were excluded from support.")
    done = set(state.get("extracted_pmids", []))
    papers = [_as(Paper, p) for p in state.get("papers", [])]
    waiting = [p for p in papers if _readable(p) and p.pmid not in done]
    if waiting:
        w.append(f"{len(waiting)} included papers were never extracted (per-round budget).")
    reviews = [p for p in papers if p.screen_status == "included" and p.study_type == "review" and p.pmid not in done]
    if reviews:
        w.append(f"{len(reviews)} included reviews were not extracted (they count as no evidence); their "
                 "abstracts informed pathway proposals only.")
    return w


def synthesize(state, rt):
    links = {k: _as(LinkEvidence, v) for k, v in state["links"].items()}
    hyps = [_as(Hypothesis, h) for h in state["hypotheses"]]
    tags = {}
    for h in hyps:
        for k in h.links:
            tags.setdefault(k, f"L{len(tags) + 1}")
    mediation = {k: _as(pf.MediationEvidence, v) for k, v in state.get("mediation", {}).items()}
    used = {i for k in tags for i in links[k].support_ids + links[k].corroborating_ids
            + links[k].contradicting_ids + list(links[k].uncounted)}
    used |= {i for m in mediation.values() for i in m.support_ids + m.against_ids + list(m.uncounted)}
    claims = [c for c in (_as(Claim, x) for x in state["claims"]) if c.id in used]
    conflicts = [_as(Conflict, c) for c in state.get("conflicts", [])]
    warnings = run_warnings(state, rt)
    user = (f"Question: {state['question']}\nGate: {state.get('gate')} | semantic: "
            f"{state.get('semantic_status')} | warnings: {warnings or 'none'}\n\nPathways:\n" +
            "\n".join(f"[{h.id}] {h.name} | {pf.ROUTE_LABEL[h.status]} ({h.reason}) | score={h.score}"
                      f" | flags={[pf.FLAG_LABEL[f] for f in h.logic_flags]}\n"
                      + "".join(f"  {line}\n" for line in _blocking_lines(h, links, mediation)) +
                      "\n".join(f"  [{tags[k]}] {links[k].subject_label} --{links[k].relation}--> "
                                f"{links[k].object_label} | {pf.STATUS_LABEL[links[k].status]} "
                                f"({links[k].reason}) | grade={links[k].grade} | support="
                                f"{links[k].support_ids} contra={links[k].contradicting_ids} "
                                f"not counted={links[k].uncounted}"
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
    items = [x for x in items if x[2] and not is_proposal(x[1])]      # a proposed experiment is not entailed by claims
    if not items:
        return []
    user = "\n\n".join(f"[SENTENCE {i}] {s}\nCited:\n" + "\n".join(
        # the system and design are what 'in animal studies' / 'a human cohort' are judged against; without
        # them 9 of pilot5's 12 issues read 'does not establish that the study was animal / human'
        f"  [{c}] ({by_id[c].grade}) {by_id[c].subject_label} {by_id[c].relation_norm} "
        f"{by_id[c].object_label} | system: {by_id[c].system}; design: {by_id[c].study_type}; "
        f"cell type: {by_id[c].context_cell_type or 'unspecified'}" for c in ids) for i, s, ids in items)
    try:
        out = rt.llm.structured("entailment", EntailmentBatch, PROMPTS["entailment"], user,
                                ctx={"items": items}, n_items=len(items))
    except Exception as e:
        return [{"verdict": "check_unavailable", "sentence": type(e).__name__}]
    return [{"sentence": sents[j.sentence_index], "verdict": j.verdict, "why": j.rationale}
            for j in out.judgements if j.verdict != "entailed" and j.sentence_index < len(sents)]


def _entailment_verdicts(text, claims, issues) -> dict:
    """sentence -> today's entailment verdict, for every sentence the check judged ('entailed' unless flagged)."""
    if any(e["verdict"] == "check_unavailable" for e in issues):
        return {}
    ids = {c.id for c in claims}
    judged = {s: "entailed" for s in prose_sentences(text)
              if any(x in ids for x in re.findall(r"\[([A-Za-z0-9_\-]+)\]", s)) and not is_proposal(s)}
    return {**judged, **{e["sentence"]: e["verdict"] for e in issues if e.get("sentence") in judged}}


def _blocking_lines(h, links, mediation, short=False) -> list[str]:
    """EM-7: per intermediate of a mechanism route, what blocking it did to the exposure's effect."""
    if pf.is_direct(h):
        return []
    by_m = pf.mediation_of(h, mediation)
    out = []
    for node in pf.nodes(h.links)[1:-1]:
        label = pf.labels_of(node, links)
        meds = by_m.get(node, [])
        if not meds:
            out.append(f"blocking {label}: not tested" if short else
                       f"Blocking {label}: no study tested whether removing or blocking it removes the effect")
            continue
        for m in meds:
            ids = " ".join(f"[{i}]" for i in m.support_ids + m.against_ids)
            out.append(f"blocking {label} on {m.outcome_label}: {pf.MEDIATION_LABEL[m.status]} ({m.reason})"
                       + (f" {ids}" if ids else ""))
    return out


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
    log = shadow.report(rt, state["synthesis"], claims, {o["sentence"] for o in v["overclaims"]},
                        _entailment_verdicts(state["synthesis"], claims, v["entailment"]))
    v["passed"] = v["passed"] and not any(e["verdict"] == "unsupported" for e in v["entailment"])
    mediation = {k: _as(pf.MediationEvidence, v) for k, v in state.get("mediation", {}).items()}
    table = ["| Pathway | Verdict | Why | Score | Source | Logic warnings | Blocking test | Steps |",
             "|---|---|---|---|---|---|---|---|"]
    for h in hyps:
        table.append(f"| {h.id} {h.name} | {pf.ROUTE_LABEL[h.status]} | {h.reason} | {h.score} | "
                     f"{pf.ORIGIN_LABEL[h.origin]} | {'; '.join(pf.FLAG_LABEL[f] for f in h.logic_flags) or '-'} | "
                     f"{'; '.join(_blocking_lines(h, links, mediation, short=True)) or '-'} | "
                     + "; ".join(f"{tags[k]} {links[k].subject_label}→{links[k].object_label} "
                                 f"({_step_label(links[k])})" for k in h.links) + " |")
    p = _as(ParsedQuestion, state["parsed"])
    direct = [h for h in hyps if pf.is_direct(h)]
    headline = ("Direct effect: " + "; ".join(f"{links[h.links[0]].subject_label} → {links[h.links[0]].object_label}: "
                                             f"{pf.ROUTE_LABEL[h.status]} ({links[h.links[0]].reason})" for h in direct)
                if direct else "Direct effect: no study in the ledger tested the exposure against the outcome itself")
    stop = (f"{headline}. Search stopped after {state['round_idx']} rounds: {pf.STOP_LABEL[state['gate']]}. "
            f"Outcome measured as: {', '.join([p.outcome] + p.outcome_readouts)}. "
            + (f"Question analysed as a decrease of {p.exposure}. " if p.exposure_change == "down" else "") +
            f"Population: {p.target_system}. Scores rank pathways; they are not probabilities.")
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
    return {"verification": v, "report": report, "judge_log": log}


# ── assembly ────────────────────────────────────────────────────────────────
def build_agent(rt: Runtime, checkpointer=None):
    g = StateGraph(State)
    for name, fn in [("parse", parse), ("plan", plan), ("search", search), ("screen", screen),
                     ("extract", extract), ("normalize", normalize), ("semantic", semantic),
                     ("portfolio", portfolio), ("review", review), ("synthesize", synthesize),
                     ("verify", verify)]:
        g.add_node(name, partial(fn, rt=rt))
    g.add_edge(START, "parse")
    g.add_conditional_edges("parse", lambda s: "plan" if _as(ParsedQuestion, s["parsed"]).in_scope else END,
                            ["plan", END])
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
    # Checkpoints hold B-MiRA's pydantic models; newer LangGraph releases refuse to restore
    # unregistered types, so they are allowed explicitly.
    allowed = [("bmira.schemas", n) for n in ("ParsedQuestion", "SearchQuery", "Paper", "Claim",
                                               "PairAdjudication", "Conflict", "LinkEvidence", "Hypothesis",
                                               "MediationEvidence")]
    saver = checkpointer or InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))
    return g.compile(checkpointer=saver)


def run(question: str, rt: Runtime, thread_id: str | None = None):
    agent = build_agent(rt)
    cfg = {"configurable": {"thread_id": thread_id or f"bmira-{uuid.uuid4()}"}, "recursion_limit": 250}
    return agent.invoke({"question": question}, cfg)
