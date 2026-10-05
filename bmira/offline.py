"""Offline kit: run the real graph with a scripted LLM, a fixture corpus and a hashing
encoder. This tests WIRING and INVARIANTS, not scientific accuracy. Fixture papers are
synthetic and must never be cited as literature."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from bmira.config import Settings
from bmira.normalize import lookup_key
from bmira.portfolio import ROUTE_LABEL, STATUS_LABEL
from bmira.schemas import (AliasBatch, AliasVerdict, ClaimList, Conflict, ConflictBatch,
                           EntailmentBatch, EntityResolution, PairAdjudication, PairBatch, Paper,
                           ParsedQuestion, PathwayProposal, QueryPlan, RelationResolution,
                           RelationResolutionBatch, Screen, SearchQuery)
from bmira.sources import is_retracted, study_type_from_pubtypes

FIXTURES = Path(__file__).parent / "fixtures"


class HashingEmbedder:
    """Deterministic bag of words + character trigrams. Lexical, so thresholds differ."""
    name = "hashing"

    def encode(self, texts, dim=512):
        out = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            t = t.lower()
            for f in re.findall(r"[a-z0-9+]+", t) + [t[j:j + 3] for j in range(len(t) - 2)]:
                out[i, int(hashlib.md5(f.encode()).hexdigest(), 16) % dim] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FixtureCorpus:
    """search / fetch / fulltext over a JSON corpus; a paper matches when one of its
    keyword phrases occurs in the query."""
    name = "fixture"

    def __init__(self, scenario: dict):
        self.papers = {p["pmid"]: p for p in scenario["papers"]}

    def search(self, query, retmax):
        q = query.lower()
        hits = [pid for pid, p in self.papers.items() if any(k in q for k in p["keywords"])]
        return {"pmids": hits[:retmax], "translation": query, "ignored_terms": []}

    def fetch(self, pmids):
        out = []
        for pid in pmids:
            p = self.papers[pid]
            pt = p.get("publication_types", ["Journal Article"])
            out.append(Paper(pmid=pid, title=p["title"], abstract=p["abstract"], year=p.get("year", ""),
                             journal="SYNTHETIC FIXTURE", source_text=p["abstract"],
                             publication_types=pt, pubtype_study_type=study_type_from_pubtypes(pt),
                             retracted=is_retracted(pt)))
        return out

    def fulltext(self, paper):
        ft = self.papers[paper.pmid].get("full_text")
        return paper.model_copy(update={"source_text": ft, "text_access": "full_text"}) if ft else paper


class SurrogateLLM:
    """Scripted stand-in for the model. Reads `ctx`, never the prompt text."""

    def __init__(self, scenario: dict):
        self.s = scenario
        self.calls, self.items, self.failures = Counter(), Counter(), Counter()
        self.tokens_in, self.tokens_out, self.seconds = Counter(), Counter(), Counter()
        self.tokens_reasoning, self.model_of = Counter(), {}
        self.alias_groups = [{lookup_key(x) for x in g} for g in scenario["alias_groups"]]

    def _count(self, task, prompt_chars, n_out):
        """Approximate tokens (4 characters each) so budgets and costs work offline."""
        self.model_of[task] = "surrogate"
        self.tokens_in[task] += prompt_chars // 4
        self.tokens_out[task] += 60 * n_out

    def structured_many(self, task, schema, system, users, role="cheap", ctxs=None):
        self.calls[task] += 1
        self.items[task] += len(users)
        self._count(task, sum(len(system) + len(u) for u in users), len(users))
        return [self._screen(p) for p in ctxs]

    def structured(self, task, schema, system, user, role="reasoning", ctx=None, n_items=1):
        self.calls[task] += 1
        self.items[task] += n_items
        self._count(task, len(system) + len(user), n_items)
        return getattr(self, f"_{task}")(ctx or {})

    def text(self, task, system, user, role="reasoning", ctx=None):
        self.calls[task] += 1
        self.items[task] += 1
        self._count(task, len(system) + len(user), 1)
        return getattr(self, f"_{task}")(ctx)

    # ── scripted behaviours ──
    def _parse(self, ctx):
        return ParsedQuestion(**self.s["parsed"])

    def _plan(self, ctx):
        if not ctx["targets"]:
            return QueryPlan(queries=[SearchQuery(**q) for q in self.s["round1_queries"]])
        qs = []
        for t in ctx["targets"]:
            base = f"{t.subject_label} {t.object_label}"
            qs += [SearchQuery(query=base, intent="gap_positive", target=t.key),
                   SearchQuery(query=f"{base} mechanism", intent="gap_alternative_terms", target=t.key),
                   SearchQuery(query=f"{base} no effect", intent="gap_null", target=t.key)]
        return QueryPlan(queries=qs)

    def _screen(self, paper):
        p = next(x for x in self.s["papers"] if x["pmid"] == paper.pmid)
        return Screen(relevant=p.get("relevant", True), reason="fixture",
                      relevance_score=p.get("relevance", 70), study_type=p["study_type"])

    def _extract(self, ctx):
        p = next(x for x in self.s["papers"] if x["pmid"] == ctx["paper"].pmid)
        return ClaimList(claims=p["claims"])

    def _relation(self, ctx):
        m = self.s["relation_map"]
        return RelationResolutionBatch(resolutions=[
            RelationResolution(claim_id=c.id, canonical_relation=m.get(c.relation_raw.lower(), "unresolved"),
                               confidence=0.9) for c in ctx["claims"]])

    def _entities(self, ctx):
        from bmira.schemas import EntityBatch, EntityItem
        items = []
        for s in ctx["surfaces"]:
            r = self._entity({"surface": s})
            items.append(EntityItem(surface=s, normalized_label=r.normalized_label, category=r.category,
                                    confidence=r.confidence))
        return EntityBatch(items=items)

    def _entity(self, ctx):
        e = self.s["entities"].get(lookup_key(ctx["surface"]))
        if not e:
            return EntityResolution(normalized_label=ctx["surface"], category="other", confidence=0.3)
        return EntityResolution(normalized_label=e[0], category=e[1], confidence=0.8)

    def _alias(self, ctx):
        same = lambda a, b: any(lookup_key(a) in g and lookup_key(b) in g for g in self.alias_groups)
        return AliasBatch(verdicts=[AliasVerdict(label_a=a, label_b=b, same_entity=same(a, b))
                                    for a, b in ctx["pairs"]])

    def _pair(self, ctx):
        return PairBatch(pairs=[PairAdjudication(
            pair=n, same_context=(a.system, a.context_concept) == (b.system, b.context_concept),
            same_finding=(a.subject_concept, a.object_concept) == (b.subject_concept, b.object_concept))
            for n, (a, b) in enumerate(ctx["pairs"], 1)])

    def _conflict(self, ctx):
        out = []
        for fam, claims in ctx["candidates"]:
            ctxs = {(c.system, c.context_concept) for c in claims}
            out.append(Conflict(cluster_key=fam.key,
                                verdict="true_conflict" if len(ctxs) == 1 else "context_dependent",
                                explanation=f"{len(claims)} claims, contexts {sorted(ctxs)}",
                                discriminating_experiment="Repeat both designs in one system with matched dose."))
        return ConflictBatch(conflicts=out)

    def _seed(self, ctx):
        return PathwayProposal(pathways=self.s["seed_pathways"])

    def _expand(self, ctx):
        novel = {lookup_key(n) for n in ctx["novel"]}
        for pw in self.s["expansion_pathways"]:
            if any(lookup_key(l["target"]) in novel or lookup_key(l["source"]) in novel for l in pw["links"]):
                return PathwayProposal(pathways=[pw])
        return PathwayProposal(pathways=[])

    def _preflight(self, ctx):
        return ctx["schema"](ok=True)

    def _entailment(self, ctx):
        return EntailmentBatch(judgements=[])

    def _synthesize(self, ctx):
        hyps, links, tags = ctx["hypotheses"], ctx["links"], ctx["tags"]
        grade_of = {c.id: c.grade for c in ctx["claims"]}
        cite = lambda ids: " ".join(f"[{i}]" for i in ids)
        lines = ["## Summary"]
        lead = next((h for h in hyps if h.status in {"demonstrated", "assembled", "supported"}), None)
        if lead:
            ids = [i for k in lead.links for i in links[k].support_ids][:4]
            lines.append(f"The best-supported route is {lead.name} [{lead.id}] {cite(ids)}.")
        else:
            lines.append("No pathway reached adequate support for every step [NO_EVIDENCE].")
        for h in hyps:
            lines.append(f"\n### [{h.id}] {h.name}: {ROUTE_LABEL[h.status]} (score {h.score})")
            for k in h.links:
                ln, t = links[k], tags[k]
                if not ln.support_ids:
                    lines.append(f"[{t}] No study in the ledger examined the step from "
                                 f"{ln.subject_label} to {ln.object_label} [NO_EVIDENCE].")
                    continue
                weakest = min((grade_of.get(i, "weak") for i in ln.support_ids),
                              key=["weak", "moderate", "strong"].index)
                verb = ("is associated with changes in" if weakest == "weak" else
                        {"increases": "increases", "decreases": "reduces"}.get(ln.relation, "is associated with"))
                lines.append(f"[{t}] {ln.subject_label} {verb} {ln.object_label} {cite(ln.support_ids)}.")
                if ln.contradicting_ids:
                    lines.append(f"[{t}] Other studies report null or opposite findings for this step "
                                 f"{cite(ln.contradicting_ids)}.")
        return "\n".join(lines)


    def _repair(self, ctx):
        """Every flagged sentence becomes an association, its bracketed tags kept in order."""
        from bmira.schemas import Rewrite, RewriteBatch
        return RewriteBatch(rewrites=[
            Rewrite(n=n, sentence="These findings are associated with the reported outcome "
                                  + " ".join(f"[{t}]" for t in re.findall(r"\[([A-Za-z0-9_\-]+)\]", s)) + ".")
            for n, s, _ in ctx["items"]])

    def _chat(self, ctx):
        """Keyword lookup over the run: enough to exercise the chat path offline."""
        st, words = ctx["state"], {w for w in re.findall(r"[a-z0-9+]+", ctx["question"].lower()) if len(w) > 3}
        links = st["links"]
        hit = lambda h: words & set(re.findall(r"[a-z0-9+]+", (h.name + " " + " ".join(
            f"{links[k].subject_label} {links[k].object_label}" for k in h.links)).lower()))
        hyps = [h for h in st["hypotheses"] if hit(h)] or st["hypotheses"]
        out = []
        for h in hyps:
            out.append(f"[{h.id}] {h.name}: {ROUTE_LABEL[h.status]} ({h.reason}).")
            for k in h.links:
                ln = links[k]
                ids = " ".join(f"[{i}]" for i in ln.support_ids + ln.contradicting_ids) or "[NO_EVIDENCE]"
                out.append(f"- Step {ln.subject_label} to {ln.object_label}: {STATUS_LABEL[ln.status]} {ids}.")
        return "\n".join(out)


def load_scenario(name="lactate_cd8"):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def offline_runtime(name="lactate_cd8", **overrides):
    """Any judge_provider other than "off" gets the offline SurrogateJudge (no network)."""
    from bmira.graph import Runtime
    from bmira.judge import SurrogateJudge
    scenario = load_scenario(name)
    settings = Settings(ontology_provider="llm", **overrides)   # OLS needs network
    judge = SurrogateJudge() if settings.judge_provider != "off" else None
    return Runtime(settings, SurrogateLLM(scenario), FixtureCorpus(scenario), HashingEmbedder(),
                   judge=judge), scenario
