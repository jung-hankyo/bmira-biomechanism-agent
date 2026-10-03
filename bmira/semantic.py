"""Claim-level semantic comparison: embeddings propose pairs, the LLM judges them,
code builds complete-linkage families and conflict candidates. Verdicts are cached by
claim pair, so each pair is judged once per run."""
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np

from bmira.evidence import stance
from bmira.llm import PROMPTS
from bmira.normalize import complete_linkage
from bmira.schemas import Cluster, ConflictBatch, PairBatch


@lru_cache(maxsize=2)
def _sentence_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


class SentenceEmbedder:
    name = "sentence"

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = model

    def encode(self, texts):
        return np.asarray(_sentence_model(self.model).encode(texts, normalize_embeddings=True))


def claim_text(c) -> str:
    return (f"{c.subject_label} {c.relation_norm or c.relation_raw} {c.object_label}; "
            f"context {c.context_cell_type or 'unspecified'} {c.context_model or ''}")


STEM_STOP = {"cell", "cells", "human", "mouse", "protein", "gene", "level", "levels", "expression",
             "activity", "function", "signaling", "tumor", "tumour"}


def _block_keys(concept: str, parents, label: str) -> set:
    """Concept, its direct ontology parents, and 5-letter word stems: un-merged synonyms
    ('sodium lactate' / 'lactate') still meet; embeddings and the top-K cap prune the rest."""
    stems = {"stem:" + w[:5] for w in re.findall(r"[a-z0-9+]+", label.lower())
             if len(w) >= 4 and w not in STEM_STOP}
    return {concept, *parents} | stems


def candidate_pairs(claims, embedder, threshold: float, k: int):
    blocks = defaultdict(set)
    for i, c in enumerate(claims):
        for sk in _block_keys(c.subject_concept, c.subject_parents, c.subject_label):
            for ok in _block_keys(c.object_concept, c.object_parents, c.object_label):
                blocks[(sk, ok)].add(i)
    pairs = {tuple(sorted((i, j))) for idx in blocks.values() for i in idx for j in idx if i < j}
    if not pairs:
        return []
    emb = embedder.encode([claim_text(c) for c in claims])
    scored = sorted(((claims[i].id, claims[j].id, float(emb[i] @ emb[j])) for i, j in pairs),
                    key=lambda x: (-x[2], x[0], x[1]))
    out, count = [], defaultdict(int)
    for a, b, sim in scored:
        if sim >= threshold and count[a] < k and count[b] < k:   # V11 used `or`: no real cap
            out.append((a, b, sim))
            count[a] += 1
            count[b] += 1
    return out


def adjudicate(claims, pairs, llm, cache: dict, batch=30):
    """Return the number of pairs left unjudged. `cache` maps frozenset(ids) -> PairAdjudication.

    Pairs are numbered 1..n within a batch and verdicts come back by that number: the first live
    run judged 0 of 227 pairs because verdicts that had to echo two long claim ids never matched."""
    by_id = {c.id: c for c in claims}
    todo = [(a, b) for a, b, _ in pairs if frozenset((a, b)) not in cache]
    failed = 0
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        text = "\n\n".join(
            f"[PAIR {n}]\nA: {by_id[a].subject_label} --{by_id[a].relation_norm}--> "
            f"{by_id[a].object_label} | context={by_id[a].context_cell_type}; {by_id[a].context_model}\n"
            f"B: {by_id[b].subject_label} --{by_id[b].relation_norm}--> {by_id[b].object_label} "
            f"| context={by_id[b].context_cell_type}; {by_id[b].context_model}"
            for n, (a, b) in enumerate(chunk, 1))
        try:
            out = llm.structured("pair", PairBatch, PROMPTS["pair"], text,
                                 ctx={"pairs": [(by_id[a], by_id[b]) for a, b in chunk]},
                                 n_items=len(chunk))
        except Exception as e:
            print(f"[semantic] batch failed: {type(e).__name__}")
            failed += len(chunk)
            continue
        stray = 0
        for v in out.pairs:
            if 1 <= v.pair <= len(chunk):
                cache[frozenset(chunk[v.pair - 1])] = v
            else:
                stray += 1
        if stray:
            print(f"[semantic] {stray} verdict(s) numbered outside 1..{len(chunk)} were ignored")
        failed += sum(1 for p in chunk if frozenset(p) not in cache)
    return failed


POLAR = {"up", "down", "null"}


def clusters(claims, cache: dict) -> list[Cluster]:
    """Same finding in the same context, for EVERY pair in the family.

    Relation family is deliberately not required: an 'increases' and a 'decreases' claim
    about one endpoint must share a family, or no contradiction could ever be seen.
    """
    ok = {p for p, v in cache.items() if v.same_finding and v.same_context}
    by_id = {c.id: c for c in claims}
    out = []
    for g in complete_linkage([c.id for c in claims], ok):
        polar = [st for st in (stance(by_id[i]) for i in g) if st in POLAR]
        discordant = len(polar) - max(map(polar.count, set(polar)), default=0)
        out.append(Cluster(key=f"SEM::{g[0]}", claim_ids=g, discordant=discordant))
    return out


def conflict_candidates(claims, families, cache):
    """Discordant families, plus same-finding pairs with opposite findings that families
    split apart (typically because their contexts differ)."""
    by_id = {c.id: c for c in claims}
    cands, covered = [], set()
    for f in families:
        if f.discordant > 0:
            cands.append(f)
            covered |= {frozenset((a, b)) for a in f.claim_ids for b in f.claim_ids if a < b}
    for p, v in cache.items():
        a, b = sorted(p)
        sa, sb = stance(by_id[a]), stance(by_id[b])
        if v.same_finding and p not in covered and sa != sb and {sa, sb} <= POLAR:
            cands.append(Cluster(key=f"PAIR::{a}::{b}", claim_ids=[a, b], discordant=1))
    return cands


def triage(claims, cands, llm):
    if not cands:
        return [], "NO_CANDIDATES"
    by_id = {c.id: c for c in claims}
    asked = cands[:20]
    text = "\n\n".join(
        f"[CANDIDATE {n}]\n" + "\n".join(
            f"[{i}] {by_id[i].subject_label} --{by_id[i].relation_norm}--> {by_id[i].object_label}"
            f" | stance={stance(by_id[i])} | context={by_id[i].context_cell_type}; "
            f"{by_id[i].context_model} | PMID {by_id[i].pmid}" for i in f.claim_ids[:30])
        for n, f in enumerate(asked, 1))
    try:
        out = llm.structured("conflict", ConflictBatch, PROMPTS["conflict"], text,
                             ctx={"candidates": [(f, [by_id[i] for i in f.claim_ids]) for f in asked]},
                             n_items=len(asked))
    except Exception as e:
        print(f"[conflict] triage failed: {type(e).__name__}")
        return [], "LLM_FAILED"
    by_key = {f.key: f for f in asked}
    kept = []
    for c in out.conflicts:
        # Verdicts name their candidate by number; echoing 'SEM::C22724664_0' lost 10 of 11 in pilot5
        # (the same fault K5 fixed for pair verdicts). A scripted model may still return the key.
        m = re.fullmatch(r"\D*(\d{1,2})\D*", c.cluster_key.strip())
        f = by_key.get(c.cluster_key.strip()) or (asked[int(m.group(1)) - 1] if m and 1 <= int(m.group(1)) <= len(asked) else None)
        if f:
            kept.append(c.model_copy(update={"cluster_key": f.key}))
    return kept, "COMPLETE"
