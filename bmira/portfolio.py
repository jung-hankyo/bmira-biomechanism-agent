"""Pathway portfolio over one shared mechanism-evidence graph.

Links are keyed by concept ids, so evidence accumulates across rounds and is shared by
every pathway that uses a link. Everything here is deterministic code; only seeding and
gated expansion (in graph.py) ask the LLM.
"""
import math
from collections import defaultdict

from bmira.evidence import SYSTEM_RANK, stance
from bmira.normalize import entity_change, entity_of
from bmira.schemas import (ASSOCIATIVE_RELATIONS, BLOCKING_ONLY, DIRECTION, NULL_RELATIONS, RELATION_SET,
                           SYSTEM_GROUP, TIER, Hypothesis, LinkEvidence, MediationEvidence)

QUALITY = {"ungraded": 0.0, "weak": 0.4, "moderate": 0.7, "strong": 1.0}
SIGNED = {"increases": 1, "decreases": -1, "required_for": 1, "sufficient_for": 1}
MODULATING = set(SIGNED)
MOLECULAR = {"gene_or_protein", "chemical"}
DOWNSTREAM_ONLY = {"phenotype", "disease"}
LOGIC_PENALTY = {"disconnected": 0.5, "sign_mismatch": 0.5, "cycle": 0.7,
                 "reverse_order": 0.8, "cross_context": 0.85, "sign_unknown": 0.9}

# Display labels: the only names a reader of the report sees. Steps (links) use STATUS_LABEL; pathways use
# ROUTE_LABEL (EM-4, owner decision D3: proposed labels). "Assembled" never displays as "Supported" (D6).
STATUS_LABEL = {"supported": "Supported", "contradicted": "Contradicted",
                "insufficient": "Insufficient evidence"}
ROUTE_LABEL = {"demonstrated": "Shown by a blocking experiment", "assembled": "Assembled from separate studies",
               "refuted": "Mediator not required", "supported": "Supported", "contradicted": "Contradicted",
               "insufficient": "Insufficient evidence"}
MEDIATION_LABEL = {"demonstrated": "shown", "refuted": "effect persisted", "insufficient": "inconclusive"}
# Sort and convergence order. A direct route's 'supported' ranks with 'assembled'.
ROUTE_TIER = {"demonstrated": 4, "assembled": 3, "supported": 3, "insufficient": 2, "refuted": 1, "contradicted": 0}
ORIGIN_LABEL = {"llm_seed": "LLM proposal", "llm_expansion": "LLM expansion",
                "ledger_path": "Found in literature graph", "user": "Added by user"}
FLAG_LABEL = {"disconnected": "steps do not connect",
              "sign_mismatch": "net direction conflicts with the question",
              "cycle": "pathway loops back on itself",
              "reverse_order": "a phenotype acts on a molecule",
              "cross_context": "steps shown in different cell types",
              "sign_unknown": "net direction undefined"}
STOP_LABEL = {"MAX_ROUNDS": "round limit reached", "NO_TARGETS": "nothing left to search",
              "BUDGET": "token budget reached", "COST_BUDGET": "cost budget reached",
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
    if link_rel in {"required_for", "sufficient_for", "modulates"}:
        return st == "null"                  # 'GPR81-/- did not change the suppression' refutes 'required for'
    return False


def _corroborates(link_rel: str, claim) -> bool:
    """Same direction only; associations corroborate associative links only."""
    if link_rel in DIRECTION:
        return stance(claim) == DIRECTION[link_rel]
    if link_rel in ASSOCIATIVE_RELATIONS:
        return claim.relation_norm == link_rel
    return False


def _relevance(c) -> int:
    return SYSTEM_RANK.get(SYSTEM_GROUP.get(c.system, ""), 0)


def context_of(c) -> tuple:
    """Recorded context of a claim: (system group, normalized cell type, tissue)."""
    return (SYSTEM_GROUP.get(c.system, "unclear"), c.context_concept or c.context, c.context_tissue.lower())


def build_links(claims, prior: dict, pair_cache: dict, labels: dict, settings, extra=(),
                discounted=frozenset(), ancestors=None) -> dict:
    """One evidence record per step. Rules that decide what counts:
    R1 reviews never count as independent papers;
    R5 a null result counts against a step only with a comparator and a grade at least as
       high as the best support;
    R6 an opposing finding triaged as context-dependent is set aside only if the recorded
       contexts really differ AND it comes from a system no closer to humans than the support
       (a contradiction in a more relevant system is never 'just context');
    R8 Supported needs at least one moderate-or-better claim, not just two papers;
    R9 a directional finding about a subtype ('butyrate increases iTreg') also supports the link
       to its ontology ancestor ('... increases Treg'), as support only: a null or opposite
       finding in a subtype never counts against the broader link, and the subject never rolls up
       ('butyrate' evidence is not evidence about 'short-chain fatty acids')."""
    by_pair = defaultdict(list)
    by_id = {c.id: c for c in claims}
    for c in claims:
        if c.relation_norm not in {"", "unresolved"} and c.effect_result not in BLOCKING_ONLY:
            by_pair[(c.subject_concept, c.object_concept)].append(c)
    partners = defaultdict(set)
    for p, v in pair_cache.items():
        if v.same_finding:
            a, b = tuple(p)
            partners[a].add(b)
            partners[b].add(a)

    keys = {link_key(c.subject_concept, c.relation_norm, c.object_concept)
            for cs in by_pair.values() for c in cs} | set(prior) | set(extra)
    wanted = {n for k in keys for n in (split_key(k)[0], split_key(k)[2])}
    rolled = defaultdict(list)                                                           # R9
    for c in claims:
        if c.relation_norm in DIRECTION:
            for up in set((ancestors or {}).get(c.object_concept, ())) & wanted - {c.object_concept}:
                rolled[(c.subject_concept, up)].append(c)
    m, thr = settings.min_studies_per_link, settings.contradiction_threshold
    out = {}
    for key in sorted(keys):                    # a set: its order changed with the string-hash seed, and with it pathway ids
        s, r, o = split_key(key)
        exact = by_pair.get((s, o), [])
        if r == "binds":                        # 'HCAR2 binds butyrate' is 'butyrate binds HCAR2'
            exact = exact + [c for c in by_pair.get((o, s), []) if c.relation_norm == "binds"]
        same_pair = exact + rolled.get((s, o), [])
        # a 'modulates' step is shown by any signed effect: 'butyrate increases DCs' modulates DCs
        support = [c for c in same_pair if c.relation_norm == r or (r == "modulates" and c.relation_norm in MODULATING)]
        sup_ids = {c.id for c in support}
        corro = {c.id: c for c in same_pair if c.id not in sup_ids and _corroborates(r, c)}
        for c in support:                       # semantic partners judged to report the same finding
            for pid in partners[c.id]:
                other = by_id.get(pid)
                if other and pid not in sup_ids and _corroborates(r, other):
                    corro[pid] = other
        uncounted = {}
        primary = [c for c in support if c.study_type != "review"]
        for c in support + list(corro.values()):
            if c.study_type == "review":
                uncounted[c.id] = "secondary source (review)"                       # R1
        grade = max((c.grade for c in primary), key=TIER.get, default="ungraded")
        if r in ASSOCIATIVE_RELATIONS and TIER[grade] > 1:
            grade = "weak"                      # an association caps the step, however powered
        contra = []
        for c in (c for c in exact if _contradicts(r, c)):
            if c.study_type == "review":
                uncounted[c.id] = "secondary source (review)"                       # R1
            elif any(frozenset((c.id, x.id)) in discounted and context_of(c) != context_of(x)
                     and _relevance(c) <= _relevance(x) for x in support):
                uncounted[c.id] = "different context (triaged as context-dependent)"  # R6
            elif stance(c) == "null" and not (c.comparator_present and TIER[c.grade] >= TIER[grade]):
                uncounted[c.id] = "null result weaker than the support"            # R5
            else:
                contra.append(c)
        papers = {c.pmid for c in primary} | {c.pmid for c in corro.values() if c.study_type != "review"}
        contra_papers = {c.pmid for c in contra} - {c.pmid for c in support}
        contra_grade = max((c.grade for c in contra), key=TIER.get, default="ungraded")
        total = len(papers) + len(contra_papers)
        share = len(contra_papers) / total if total else 0.0
        old = prior.get(key)
        exhausted = old.exhausted if old else False
        searches = old.zero_yield_count if old else 0
        if contra_papers and share >= thr and TIER[contra_grade] >= 2:
            status, reason = "contradicted", f"{len(contra_papers)} opposing vs {len(papers)} supporting papers"
        elif papers and len(papers) >= m and share < thr and TIER[grade] >= 2:
            status, reason = "supported", f"{len(papers)} papers, best grade {grade}"
        elif papers and len(papers) >= m and share < thr:
            status, reason = "insufficient", f"only weak evidence ({len(papers)} papers)"           # R8
        elif not papers and support:
            status, reason = "insufficient", "only secondary sources (reviews)"
        elif not papers:
            status, reason = ("insufficient", f"no study found in {searches} targeted searches"
                              if exhausted else f"not found in {searches} targeted search"
                              + ("es" if searches != 1 else "") if searches else "not found yet")
        elif share >= thr:
            status, reason = "insufficient", "weak evidence on both sides"
        else:
            status, reason = "insufficient", f"{len(papers)} of {m} required papers" + (
                "; targeted searches found no more" if exhausted else "")
        out[key] = LinkEvidence(
            key=key, subject=s, relation=r, object=o,
            subject_label=labels.get(s, old.subject_label if old else s),
            object_label=labels.get(o, old.object_label if old else o),
            support_ids=sorted(sup_ids), corroborating_ids=sorted(corro),
            contradicting_ids=sorted(c.id for c in contra), uncounted=uncounted,
            n_studies=len(papers), n_contra_studies=len(contra_papers), grade=grade,
            contra_grade=contra_grade, contradiction_share=round(share, 3),
            completeness=round(min(1.0, len(papers) / m) * QUALITY[grade] * (1 - share), 3)
            if papers else 0.0,
            status=status, reason=reason, contexts=sorted({context_of(c)[1] for c in primary}),
            times_targeted=old.times_targeted if old else 0, zero_yield_count=searches,
            exhausted=exhausted)
    return out


# ── mediation (blocking tests) ──────────────────────────────────────────────
def mediation_key(exposure: str, mediator: str, outcome: str) -> str:
    return f"{exposure}|{mediator}|{outcome}"


def build_mediation(claims, exposure: str, named: set, ancestors: dict, labels: dict, settings,
                    canon=lambda x: x) -> dict:
    """EM-3: blocking tests indexed by (exposure, mediator, outcome or readout). Rules mirror the step rules:
    R1 reviews never count; R5 an 'unchanged' result counts against only with a comparator and a grade at
    least the best support; R9 a finding on a descendant of a named outcome supports it but never counts
    against it. Demonstrated: one moderate-or-better supporting paper and an opposing share below the
    contradiction threshold ('1 study' or 'replicated'). Refuted: moderate-or-better opposing papers at or
    above the threshold."""
    groups = defaultdict(lambda: {"support": [], "against": [], "uncounted": {}})
    for c in claims:
        if not c.is_blocking_test or not c.effect_exposure_concept:
            continue
        if canon(c.effect_exposure_concept) != exposure:
            continue                                  # a blocking test of another treatment's effect
        m, y = canon(c.subject_concept), canon(c.object_concept)
        if m == exposure or m in named:
            continue                                  # the exposure or the outcome cannot mediate itself
        targets = [(y, True)] if y in named else [
            (a, False) for a in sorted(set(ancestors.get(y, ())) & named)]
        for target, exact in targets:
            g = groups[mediation_key(exposure, m, target)]
            if c.study_type == "review":
                g["uncounted"][c.id] = "secondary source (review)"                     # R1
            elif c.effect_result in {"abolished", "attenuated"}:
                g["support"].append(c)
            elif c.effect_result == "unchanged" and exact:
                g["against"].append(c)
            elif c.effect_result == "unchanged":
                g["uncounted"][c.id] = "a subtype finding never counts against the parent"   # R9
            else:
                g["uncounted"][c.id] = "effect larger without the mediator"
    out, thr = {}, settings.contradiction_threshold
    for key in sorted(groups):
        g = groups[key]
        x, m, y = split_key(key)
        best = max((c.grade for c in g["support"]), key=TIER.get, default="ungraded")
        against = []
        for c in g["against"]:
            if c.comparator_present and TIER[c.grade] >= TIER[best]:
                against.append(c)
            else:
                g["uncounted"][c.id] = "null result weaker than the support"           # R5
        sup_papers = {c.pmid for c in g["support"]}
        con_papers = {c.pmid for c in against} - sup_papers
        con_grade = max((c.grade for c in against), key=TIER.get, default="ungraded")
        total = len(sup_papers) + len(con_papers)
        share = len(con_papers) / total if total else 0.0
        n = len(sup_papers)
        if sup_papers and TIER[best] >= 2 and share < thr:
            status, reason = "demonstrated", (f"replicated in {n} studies" if n > 1 else "1 study") + f", {best}"
        elif con_papers and TIER[con_grade] >= 2 and share >= thr:
            status, reason = "refuted", f"effect persisted without it in {len(con_papers)} of {total} studies"
        elif sup_papers and share >= thr:
            status, reason = "insufficient", "blocking tests disagree"
        elif sup_papers:
            status, reason = "insufficient", f"only weak blocking evidence ({n} {'study' if n == 1 else 'studies'})"
        elif g["uncounted"] and not against:
            status, reason = "insufficient", "no blocking test that counts"
        else:
            status, reason = "insufficient", "only weak opposing evidence"
        out[key] = MediationEvidence(
            key=key, exposure=x, mediator=m, outcome=y, mediator_label=labels.get(m, m),
            outcome_label=labels.get(y, y), support_ids=sorted(c.id for c in g["support"]),
            against_ids=sorted(c.id for c in against), uncounted=g["uncounted"], n_support_papers=n,
            n_against_papers=len(con_papers), grade=best, status=status, reason=reason)
    return out


def mediation_of(h, mediation: dict) -> dict:
    """Intermediate node -> its mediation records, for the intermediates of one route."""
    mids = set(nodes(h.links)[1:-1])
    out = defaultdict(list)
    for med in mediation.values():
        if med.mediator in mids:
            out[med.mediator].append(med)
    return out


def contexts_compatible(a, b, ancestors=None) -> bool:
    """EM-5: adjacent steps fit together when either context is unspecified, they share one, or one is
    an ancestor of the other (a step in regulatory T cells and one in T cells)."""
    a, b = set(a) - {"unspecified", ""}, set(b) - {"unspecified", ""}
    if not a or not b or a & b:
        return True
    anc = ancestors or {}
    return any(x in anc.get(y, ()) or y in anc.get(x, ()) for x in a for y in b)


# ── pathways ────────────────────────────────────────────────────────────────
def proposal_keys(pathway, resolver, stop_at=frozenset()) -> tuple[list[str], dict]:
    """Concept keys of a proposed pathway. It ends at the first link that reaches the outcome or
    one of its readouts: 'FOXP3 -> regulatory T cell' is a definition, no paper tests it (pilot4's H1)."""
    keys, labels = [], {}
    for ln in pathway.links:
        if ln.relation not in RELATION_SET - {"unresolved"}:
            continue
        # same entity/attribute split as claims, or 'IFNG expression' would miss node 'IFNG'
        s, o = resolver.resolve(entity_of(ln.source)[0]), resolver.resolve(entity_of(ln.target)[0])
        labels[s.id], labels[o.id] = s.label, o.label
        rel = ln.relation         # and the same restatement: 'HDAC inhibition increases X' = HDAC decreases X
        if rel in DIRECTION and (entity_change(ln.source) == "down") != (entity_change(ln.target) == "down"):
            rel = "decreases" if rel == "increases" else "increases"
        keys.append(link_key(s.id, rel, o.id))
        if resolver.canonical(o).id in stop_at:
            break
    return as_chain(keys), labels


def as_chain(keys: list[str]) -> list[str]:
    """A proposal whose links share a source ('HIF -> Th17' and 'HIF -> Treg' after 'butyrate -> HIF') is a
    fan, not a pathway: keep the connected route from the first source to the last target through the most
    proposed steps, so a shortcut never replaces the mechanism (pilot6's H7 was flagged 'steps do not
    connect' and failed on a side branch). Unchanged if no route exists."""
    if len(keys) < 2:
        return keys
    start, goal = split_key(keys[0])[0], split_key(keys[-1])[2]

    def routes(node, path, seen):                 # proposals have a handful of links: exhaustive is fine
        if node == goal and path:
            yield path
        for k in keys:
            s, _, o = split_key(k)
            if s == node and o not in seen:
                yield from routes(o, path + [k], seen | {o})
    return max(routes(start, [], {start}), key=len, default=keys)


def nodes(keys) -> list[str]:
    out = []
    for k in keys:
        s, _, o = split_key(k)
        out += [n for n in (s, o) if not out or out[-1] != n]
    return out


def pathway_sign(expected: str, exposure_change: str) -> str:
    """Net direction a correct pathway from the bare exposure must have. 'NAD+ decline -> more
    inflammaging' is expected 'up' for the decline, so NAD+ itself must act 'down' on it."""
    if exposure_change == "down" and expected in {"up", "down"}:
        return "down" if expected == "up" else "up"
    return expected


def logic_check(keys, links, categories, expected: str, ancestors=None):
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
    ctx = [links[k].contexts for k in keys if k in links]
    if not all(contexts_compatible(a, b, ancestors) for a, b in zip(ctx, ctx[1:])):
        flags.append("cross_context")
    factor = math.prod(LOGIC_PENALTY.get(f, 1.0) for f in flags)
    return round(factor, 3), flags


def is_direct(h) -> bool:
    """A one-step exposure -> outcome route found in the literature: the answer to 'does X affect Y',
    not a mechanism. It takes no pathway slot (pilot4: two of six slots went to H5 and its subtype H6,
    which left no room for expansion)."""
    return h.origin == "ledger_path" and len(h.links) == 1


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
    # ties: signed relations before 'modulates'/'associated_with' (more informative), then by key
    return sorted(found, key=lambda p: (-min(links[k].completeness for k in p),
                                        sum(links[k].relation not in SIGNED for k in p), p))


def novel_intermediates(links, hyps, exposure, outcomes) -> list[str]:
    """Nodes on a supported link that touches the portfolio but that no pathway uses."""
    used = {n for h in hyps for n in nodes(h.links)} | {exposure} | set(outcomes)
    return sorted({a for ln in links.values() if ln.status == "supported"
                   for a, b in ((ln.subject, ln.object), (ln.object, ln.subject))
                   if a not in used and b in used})


def evaluate(hyps, links, categories, expected, settings, mediation=None, ancestors=None):
    """Score and label from evidence only (EM-4). Precedence for a mechanism route: contradicted (a step is)
    -> demonstrated (blocking an intermediate removed the exposure's effect, no step contradicted) ->
    assembled (every step supported, adjacent contexts compatible) -> refuted (removing an intermediate left
    the effect, none shown) -> insufficient. A direct route keeps the step verdicts. A pathway stays open
    for search while it is insufficient and none of its steps has been searched out."""
    mediation = mediation or {}
    for h in hyps:
        h.logic_factor, h.logic_flags = logic_check(h.links, links, categories, expected, ancestors)
        ls = [links[k] for k in h.links]
        h.score = round(h.logic_factor * min((ln.completeness for ln in ls), default=0.0), 3)
        step = lambda ln: f"{ln.subject_label} → {ln.object_label}"
        bad = next((ln for ln in ls if ln.status == "contradicted"), None)
        gap = next((ln for ln in ls if ln.status == "insufficient" and ln.exhausted), None)
        meds = [m for ms in mediation_of(h, mediation).values() for m in ms]
        shown = next((m for m in meds if m.status == "demonstrated"), None)
        refuted = next((m for m in meds if m.status == "refuted"), None)
        all_supported = bool(ls) and all(ln.status == "supported" for ln in ls)
        coherent = all(contexts_compatible(a.contexts, b.contexts, ancestors) for a, b in zip(ls, ls[1:]))
        if bad:
            h.status, h.reason = "contradicted", f"{step(bad)}: {bad.reason}"
        elif is_direct(h) or len(nodes(h.links)) <= 2:       # no intermediate: nothing to block or assemble
            h.status, h.reason = ("supported", "every step has independent support") if all_supported else \
                ("insufficient", f"{step(ls[0])}: {ls[0].reason}")
        elif shown:
            h.status, h.reason = "demonstrated", (f"blocking {shown.mediator_label} removed the effect on "
                                                  f"{shown.outcome_label} ({shown.reason})")
        elif all_supported and coherent:
            untested = [labels_of(n, links) for n in nodes(h.links)[1:-1]]
            h.status, h.reason = "assembled", ("every step supported by separate studies; " + (
                f"but blocking {refuted.mediator_label}: {refuted.reason}" if refuted     # precedence: handoff EM-4
                else "no blocking test of " + ", ".join(untested)))
        elif refuted:
            h.status, h.reason = "refuted", (f"blocking {refuted.mediator_label}: {refuted.reason}")
        elif all_supported:
            h.status, h.reason = "insufficient", "every step supported, but the steps were shown in different cell types"
        else:
            weakest = gap or min(ls, key=lambda ln: ln.completeness)
            done = sum(ln.status == "supported" for ln in ls)
            h.status, h.reason = "insufficient", f"{step(weakest)}: {weakest.reason}" + (
                f" ({done} of {len(ls)} steps supported)" if done and len(ls) > 1 else "")
        h.open = h.status == "insufficient" and gap is None
    # the score is gated by the weakest step, so on sparse literature every route scores 0: order those by progress
    progress = {h.id: sum(links[k].completeness for k in h.links) / max(1, len(h.links)) for h in hyps}
    hyps.sort(key=lambda h: (-ROUTE_TIER[h.status], -h.score, -progress[h.id], h.id))
    keep = [h for h in hyps if ROUTE_TIER[h.status] >= ROUTE_TIER["assembled"]]
    rest = [h for h in hyps if ROUTE_TIER[h.status] < ROUTE_TIER["assembled"]]
    room = max(0, settings.max_hypotheses - sum(not is_direct(h) for h in keep))
    cut = {h.id for h in [h for h in rest if not is_direct(h)][room:]}          # direct routes are never cut
    return keep + [h for h in rest if h.id not in cut]


def allocate(hyps, links, settings, round_idx: int) -> list[str]:
    """Spend the round's search budget: exploit the most decisive links, explore at least one."""
    live = [h for h in hyps if h.open]
    if not live:
        return []
    # Weight by progress along the route, not by the weakest step (the verdict score): every
    # unstarted mechanism route scores 0 there and got the same weight, and a direct route's
    # 0.4 outweighed all of them (pilot4: the HDAC route was never targeted).
    progress = {h.id: sum(links[k].completeness for k in h.links) / max(1, len(h.links)) for h in live}
    raw = [math.exp(progress[h.id] / settings.softmax_temperature) for h in live]
    w = {h.id: x / sum(raw) for h, x in zip(live, raw)}
    prio, owner, near = defaultdict(float), defaultdict(set), {}
    for h in live:
        for i, k in enumerate(h.links):
            ln = links[k]
            if ln.exhausted or ln.status == "supported":
                continue
            prio[k] += w[h.id] * (1 - ln.completeness) / (1 + i)       # nearest the exposure first
            owner[k].add(h.id)
            near[k] = min(near.get(k, i), i)
    bonus = {k: settings.ucb_beta * math.sqrt(math.log(round_idx + 2) / (1 + links[k].times_targeted))
             for k in prio}
    ranked = sorted(prio, key=lambda k: (-(prio[k] + bonus[k]), near[k], k))
    n_explore = min(settings.exploration_slots, settings.targets_per_round)
    chosen = ranked[:settings.targets_per_round - n_explore]
    leader = max(live, key=lambda h: (h.score, h.id)).id
    explore = [k for k in sorted(prio, key=lambda k: (-bonus[k], -prio[k], near[k], k))
               if k not in chosen and owner[k] - {leader}]
    chosen += explore[:n_explore]
    chosen += [k for k in ranked if k not in chosen][:settings.targets_per_round - len(chosen)]
    for k in chosen:
        links[k].times_targeted += 1
    return chosen


def reachable_tier(h) -> int:
    """The highest tier an open route could still reach: a route with an intermediate can be shown by a
    blocking test; a one-step route can at best be supported."""
    return ROUTE_TIER["demonstrated"] if len(nodes(h.links)) > 2 else ROUTE_TIER["assembled"]


def decide(hyps, targets, completed_rounds: int, settings) -> tuple[str, str]:
    """CONVERGED only when the leading mechanism route is at least assembled and no open rival could reach a
    higher tier, or the same tier with a better score. Direct routes answer 'does X affect Y', not 'how',
    so they lead only when there is no mechanism route."""
    if not targets:                                # checked first: an honest stop reason
        return "done", "NO_TARGETS"
    if completed_rounds >= settings.max_rounds:
        return "done", "MAX_ROUNDS"
    mech = [h for h in hyps if not is_direct(h)]
    leader = (mech or hyps or [None])[0]
    if leader and ROUTE_TIER[leader.status] >= ROUTE_TIER["assembled"]:
        lt = ROUTE_TIER[leader.status]
        rivals = [h for h in hyps if h is not leader and h.open and (h in mech or not mech)]
        if not any(reachable_tier(h) > lt or (reachable_tier(h) == lt and h.logic_factor > leader.score)
                   for h in rivals):
            return "done", "CONVERGED"
    return "search_more", "TARGETED"


def labels_of(node: str, links) -> str:
    """Display label of a node, from any link that touches it."""
    for ln in links.values():
        if ln.subject == node:
            return ln.subject_label
        if ln.object == node:
            return ln.object_label
    return node


def label_pathway(h: Hypothesis, links) -> str:
    out = []
    for k in h.links:
        ln = links[k]
        out.append(f"{ln.subject_label} -{ln.relation}-> {ln.object_label}")
    return " ; ".join(out)
