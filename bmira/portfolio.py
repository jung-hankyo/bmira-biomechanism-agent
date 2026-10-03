"""Pathway portfolio over one shared mechanism-evidence graph.

Links are keyed by concept ids, so evidence accumulates across rounds and is shared by
every pathway that uses a link. Everything here is deterministic code; only seeding and
gated expansion (in graph.py) ask the LLM.
"""
import math
from collections import defaultdict

from bmira.evidence import stance
from bmira.schemas import (ASSOCIATIVE_RELATIONS, DIRECTION, NULL_RELATIONS, RELATION_SET,
                           TIER, Hypothesis, LinkEvidence)

QUALITY = {"ungraded": 0.0, "weak": 0.4, "moderate": 0.7, "strong": 1.0}
SIGNED = {"increases": 1, "decreases": -1, "required_for": 1, "sufficient_for": 1}
MOLECULAR = {"gene_or_protein", "chemical"}
DOWNSTREAM_ONLY = {"phenotype", "disease"}
LOGIC_PENALTY = {"disconnected": 0.5, "sign_mismatch": 0.5, "cycle": 0.7,
                 "reverse_order": 0.8, "cross_context": 0.85, "sign_unknown": 0.9}

# Display labels: the only names a reader of the report sees.
STATUS_LABEL = {"supported": "Supported", "contradicted": "Contradicted",
                "insufficient": "Insufficient evidence"}
ORIGIN_LABEL = {"llm_seed": "LLM proposal", "llm_expansion": "LLM expansion",
                "ledger_path": "Found in literature graph", "user": "Added by user"}
FLAG_LABEL = {"disconnected": "steps do not connect",
              "sign_mismatch": "net direction conflicts with the question",
              "cycle": "pathway loops back on itself",
              "reverse_order": "a phenotype acts on a molecule",
              "cross_context": "steps shown in different cell types",
              "sign_unknown": "net direction undefined"}
STOP_LABEL = {"MAX_ROUNDS": "round limit reached", "NO_TARGETS": "nothing left to search",
              "CONVERGED": "leading pathway cannot be overtaken", "TARGETED": "searching"}


def link_key(subject: str, relation: str, obj: str) -> str:
    return f"{subject}|{relation}|{obj}"


def split_key(key: str):
    return tuple(key.split("|"))


# ── link evidence ───────────────────────────────────────────────────────────
def _contradicts(link_rel: str, claim) -> bool:
    st = stance(claim)
    if link_rel in DIRECTION:
        return st == "null" or (st in {"up", "down"} and st != DIRECTION[link_rel])
    if link_rel in ASSOCIATIVE_RELATIONS - NULL_RELATIONS:
        return claim.relation_norm in NULL_RELATIONS
    if link_rel in NULL_RELATIONS:
        return st in {"up", "down"}
    return False


def _corroborates(link_rel: str, claim) -> bool:
    """Same direction only; associations corroborate associative links only."""
    if link_rel in DIRECTION:
        return stance(claim) == DIRECTION[link_rel]
    if link_rel in ASSOCIATIVE_RELATIONS:
        return claim.relation_norm == link_rel
    return False


def build_links(claims, prior: dict, pair_cache: dict, labels: dict, settings, extra=(),
                discounted=frozenset()) -> dict:
    """`discounted` holds claim-id pairs whose conflict was judged context-dependent or not
    comparable: such a claim is listed on the link but does not count against it."""
    by_pair = defaultdict(list)
    by_id = {c.id: c for c in claims}
    for c in claims:
        if c.relation_norm not in {"", "unresolved"}:
            by_pair[(c.subject_concept, c.object_concept)].append(c)
    partners = defaultdict(set)
    for p, v in pair_cache.items():
        if v.same_finding:
            a, b = tuple(p)
            partners[a].add(b)
            partners[b].add(a)

    keys = {link_key(c.subject_concept, c.relation_norm, c.object_concept)
            for cs in by_pair.values() for c in cs} | set(prior) | set(extra)
    m = settings.min_studies_per_link
    out = {}
    for key in keys:
        s, r, o = split_key(key)
        same_pair = by_pair.get((s, o), [])
        support = [c for c in same_pair if c.relation_norm == r]
        sup_ids = {c.id for c in support}
        opposing = [c for c in same_pair if _contradicts(r, c)]
        ctx_dep = [c for c in opposing if any(frozenset((c.id, i)) in discounted for i in sup_ids)]
        contra = [c for c in opposing if c not in ctx_dep]
        corro = {c.id: c for c in same_pair if c.id not in sup_ids and _corroborates(r, c)}
        for c in support:                       # semantic partners judged same question+endpoint
            for pid in partners[c.id]:
                other = by_id.get(pid)
                if other and pid not in sup_ids and _corroborates(r, other):
                    corro[pid] = other
        papers = {c.pmid for c in support} | {c.pmid for c in corro.values()}
        contra_papers = {c.pmid for c in contra} - {c.pmid for c in support}
        grade = max((c.grade for c in support), key=TIER.get, default="ungraded")
        if r in ASSOCIATIVE_RELATIONS and TIER[grade] > 1:
            grade = "weak"                      # an association caps the link, however powered
        contra_grade = max((c.grade for c in contra), key=TIER.get, default="ungraded")
        total = len(papers) + len(contra_papers)
        share = len(contra_papers) / total if total else 0.0
        old = prior.get(key)
        exhausted = old.exhausted if old else False
        if contra_papers and share >= settings.contradiction_threshold and TIER[contra_grade] >= 2:
            status, reason = "contradicted", f"{len(contra_papers)} contradicting vs {len(papers)} supporting papers"
        elif support and len(papers) >= m and share < settings.contradiction_threshold:
            status, reason = "supported", f"{len(papers)} papers, best grade {grade}"
        elif exhausted:
            status, reason = "insufficient", "no study found after repeated searches"
        elif not support:
            status, reason = "insufficient", "not found yet"
        elif share >= settings.contradiction_threshold:
            status, reason = "insufficient", "weak evidence on both sides"
        else:
            status, reason = "insufficient", f"{len(papers)} of {m} required papers"
        out[key] = LinkEvidence(
            key=key, subject=s, relation=r, object=o,
            subject_label=labels.get(s, old.subject_label if old else s),
            object_label=labels.get(o, old.object_label if old else o),
            support_ids=sorted(sup_ids), corroborating_ids=sorted(corro),
            contradicting_ids=sorted(c.id for c in contra),
            context_dependent_ids=sorted(c.id for c in ctx_dep), n_studies=len(papers),
            n_contra_studies=len(contra_papers), grade=grade, contra_grade=contra_grade,
            contradiction_share=round(share, 3),
            completeness=round(min(1.0, len(papers) / m) * QUALITY[grade] * (1 - share), 3)
            if support else 0.0,
            status=status, reason=reason, contexts=sorted({c.context for c in support}),
            times_targeted=old.times_targeted if old else 0,
            zero_yield_count=old.zero_yield_count if old else 0, exhausted=exhausted)
    return out


# ── pathways ────────────────────────────────────────────────────────────────
def proposal_keys(pathway, resolver) -> tuple[list[str], dict]:
    keys, labels = [], {}
    for ln in pathway.links:
        if ln.relation not in RELATION_SET - {"unresolved"}:
            continue
        s, o = resolver.resolve(ln.source), resolver.resolve(ln.target)
        labels[s.id], labels[o.id] = s.label, o.label
        keys.append(link_key(s.id, ln.relation, o.id))
    return keys, labels


def nodes(keys) -> list[str]:
    out = []
    for k in keys:
        s, _, o = split_key(k)
        out += [n for n in (s, o) if not out or out[-1] != n]
    return out


def logic_check(keys, links, categories, expected: str):
    """Deterministic biological-logic screen; returns (factor, flags)."""
    flags = []
    pairs = [split_key(k) for k in keys]
    if any(pairs[i][2] != pairs[i + 1][0] for i in range(len(pairs) - 1)):
        flags.append("disconnected")
    ns = nodes(keys)
    if len(ns) != len(set(ns)):
        flags.append("cycle")
    signs = [SIGNED.get(r) for _, r, _ in pairs]
    if None in signs:
        flags.append("sign_unknown")
    elif expected in {"up", "down"} and math.prod(signs) != (1 if expected == "up" else -1):
        flags.append("sign_mismatch")
    if any(categories.get(s) in DOWNSTREAM_ONLY and categories.get(o) in MOLECULAR
           for s, _, o in pairs):
        flags.append("reverse_order")
    ctx = [set(links[k].contexts) - {"unspecified"} for k in keys if k in links]
    if any(a and b and not (a & b) for a, b in zip(ctx, ctx[1:])):
        flags.append("cross_context")
    factor = math.prod(LOGIC_PENALTY.get(f, 1.0) for f in flags)
    return round(factor, 3), flags


def is_duplicate(keys, hyps, diverse: bool) -> bool:
    """Exact duplicate always; for LLM proposals also >= 50% shared intermediates."""
    mid = set(nodes(keys)[1:-1])
    for h in hyps:
        if h.links == keys:
            return True
        other = set(nodes(h.links)[1:-1])
        if diverse and mid and other and len(mid & other) / len(mid | other) >= 0.5:
            return True
    return False


def ledger_paths(links, exposure: str, outcomes: set, max_len: int) -> list[list[str]]:
    """Simple exposure -> outcome paths through supported steps, strongest weakest-step first.

    # ponytail: exhaustive DFS bounded by max_len; switch to k-shortest paths if the
    # graph grows to thousands of links.
    """
    adj = defaultdict(list)
    for k, ln in links.items():
        if ln.support_ids and ln.relation not in NULL_RELATIONS:
            adj[ln.subject].append(k)
    found = []

    def walk(node, path, seen):
        if node in outcomes and path:
            found.append(list(path))
            return
        if len(path) == max_len:
            return
        for k in adj[node]:
            nxt = links[k].object
            if nxt not in seen:
                walk(nxt, path + [k], seen | {nxt})

    walk(exposure, [], {exposure})
    return sorted(found, key=lambda p: -min(links[k].completeness for k in p))


def novel_intermediates(links, hyps, exposure, outcomes) -> list[str]:
    """Nodes on a supported link that touches the portfolio but that no pathway uses."""
    used = {n for h in hyps for n in nodes(h.links)} | {exposure} | set(outcomes)
    return sorted({a for ln in links.values() if ln.status == "supported"
                   for a, b in ((ln.subject, ln.object), (ln.object, ln.subject))
                   if a not in used and b in used})


def evaluate(hyps, links, categories, expected, settings):
    """Score and label from evidence only. A pathway stays open for search while it is
    insufficient and none of its steps has been searched out."""
    for h in hyps:
        h.logic_factor, h.logic_flags = logic_check(h.links, links, categories, expected)
        ls = [links[k] for k in h.links]
        h.score = round(h.logic_factor * min((ln.completeness for ln in ls), default=0.0), 3)
        step = lambda ln: f"{ln.subject_label} → {ln.object_label}"
        bad = next((ln for ln in ls if ln.status == "contradicted"), None)
        gap = next((ln for ln in ls if ln.status == "insufficient" and ln.exhausted), None)
        if bad:
            h.status, h.reason = "contradicted", f"{step(bad)}: {bad.reason}"
        elif ls and all(ln.status == "supported" for ln in ls):
            h.status, h.reason = "supported", "every step has independent support"
        else:
            weakest = gap or min(ls, key=lambda ln: ln.completeness)
            h.status, h.reason = "insufficient", f"{step(weakest)}: {weakest.reason}"
        h.open = h.status == "insufficient" and gap is None
    hyps.sort(key=lambda h: (-h.score, h.id))
    keep = [h for h in hyps if h.status == "supported"]
    rest = [h for h in hyps if h.status != "supported"]
    return keep + rest[:max(0, settings.max_hypotheses - len(keep))]


def allocate(hyps, links, settings, round_idx: int) -> list[str]:
    """Spend the round's search budget: exploit the most decisive links, explore at least one."""
    live = [h for h in hyps if h.open]
    if not live:
        return []
    raw = [math.exp(h.score / settings.softmax_temperature) for h in live]
    w = {h.id: x / sum(raw) for h, x in zip(live, raw)}
    prio, owner = defaultdict(float), defaultdict(set)
    for h in live:
        for k in set(h.links):
            ln = links[k]
            if ln.exhausted or ln.status == "supported":
                continue
            prio[k] += w[h.id] * (1 - ln.completeness)
            owner[k].add(h.id)
    bonus = {k: settings.ucb_beta * math.sqrt(math.log(round_idx + 2) / (1 + links[k].times_targeted))
             for k in prio}
    ranked = sorted(prio, key=lambda k: (-(prio[k] + bonus[k]), k))
    n_explore = min(settings.exploration_slots, settings.targets_per_round)
    chosen = ranked[:settings.targets_per_round - n_explore]
    leader = max(live, key=lambda h: (h.score, h.id)).id
    explore = [k for k in sorted(prio, key=lambda k: (-bonus[k], -prio[k], k))
               if k not in chosen and owner[k] - {leader}]
    chosen += explore[:n_explore]
    chosen += [k for k in ranked if k not in chosen][:settings.targets_per_round - len(chosen)]
    for k in chosen:
        links[k].times_targeted += 1
    return chosen


def decide(hyps, targets, completed_rounds: int, settings) -> tuple[str, str]:
    if not targets:                                # checked first: an honest stop reason
        return "done", "NO_TARGETS"
    if completed_rounds >= settings.max_rounds:
        return "done", "MAX_ROUNDS"
    leader = hyps[0] if hyps else None
    rivals = [h.logic_factor for h in hyps[1:] if h.open]
    if leader and leader.status == "supported" and leader.score >= max(rivals, default=0):
        return "done", "CONVERGED"                 # no rival can overtake even if fully supported
    return "search_more", "TARGETED"


def label_pathway(h: Hypothesis, links) -> str:
    out = []
    for k in h.links:
        ln = links[k]
        out.append(f"{ln.subject_label} -{ln.relation}-> {ln.object_label}")
    return " ; ".join(out)
