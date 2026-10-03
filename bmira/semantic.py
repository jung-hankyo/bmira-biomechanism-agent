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
    """Return (new verdicts, n_failed_pairs). `cache` maps frozenset(ids) -> PairAdjudication."""
    by_id = {c.id: c for c in claims}
    todo = [(a, b) for a, b, _ in pairs if frozenset((a, b)) not in cache]
    failed = 0
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        text = "\n\n".join(
            f"[PAIR {a} vs {b}]\nA: {by_id[a].subject_label} --{by_id[a].relation_norm}--> "
            f"{by_id[a].object_label} | context={by_id[a].context_cell_type}; {by_id[a].context_model}\n"
            f"B: {by_id[b].subject_label} --{by_id[b].relation_norm}--> {by_id[b].object_label} "
            f"| context={by_id[b].context_cell_type}; {by_id[b].context_model}" for a, b in chunk)
        try:
            out = llm.structured("pair", PairBatch, PROMPTS["pair"], text,
                                 ctx={"pairs": [(by_id[a], by_id[b]) for a, b in chunk]},
                                 n_items=len(chunk))
        except Exception as e:
            print(f"[semantic] batch failed: {type(e).__name__}")
            failed += len(chunk)
            continue
        asked = {frozenset(p) for p in chunk}
        for v in out.pairs:
            if frozenset((v.claim_a, v.claim_b)) in asked:
                cache[frozenset((v.claim_a, v.claim_b))] = v
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
    text = "\n\n".join(
        f"CANDIDATE {f.key}\n" + "\n".join(
            f"[{i}] {by_id[i].subject_label} --{by_id[i].relation_norm}--> {by_id[i].object_label}"
            f" | stance={stance(by_id[i])} | context={by_id[i].context_cell_type}; "
            f"{by_id[i].context_model} | PMID {by_id[i].pmid}" for i in f.claim_ids[:30])
        for f in cands[:20])
    try:
        out = llm.structured("conflict", ConflictBatch, PROMPTS["conflict"], text,
                             ctx={"candidates": [(f, [by_id[i] for i in f.claim_ids]) for f in cands[:20]]},
                             n_items=len(cands[:20]))
    except Exception as e:
        print(f"[conflict] triage failed: {type(e).__name__}")
        return [], "LLM_FAILED"
    keys = {f.key for f in cands}
    return [c for c in out.conflicts if c.cluster_key in keys], "COMPLETE"
