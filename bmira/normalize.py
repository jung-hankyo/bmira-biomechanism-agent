"""Entity resolution, alias registry, lexical relation typing and span anchoring."""
import re
from dataclasses import dataclass
from urllib.parse import quote

import requests

from bmira.llm import PROMPTS
from bmira.schemas import AliasBatch, EntityResolution, RELATION_SET

OLS = "https://www.ebi.ac.uk/ols4/api"
PREFIX_CATEGORY = {
    "CL": "cell_type", "GO": "process", "CHEBI": "chemical", "HP": "phenotype",
    "MP": "phenotype", "MONDO": "disease", "DOID": "disease", "EFO": "phenotype",
    "PR": "gene_or_protein", "HGNC": "gene_or_protein", "NCBIGENE": "gene_or_protein",
    "UBERON": "anatomy",
}


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\u2013\u2014]", "-", s or "")).strip(" .,;:()[]{}")


def lookup_key(s: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", clean(s).lower()).strip()


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    category: str = "unknown"
    source: str = "local"
    confidence: float = 0.0
    ancestors: tuple = ()


def local_concept(label: str) -> Concept:
    slug = re.sub(r"[^a-z0-9+]+", "_", clean(label).lower()).strip("_")[:120] or "unknown"
    return Concept(f"LOCAL:{slug}", clean(label) or "unknown")


class EntityResolver:
    """surface -> Concept, with one alias registry applied on EVERY lookup.

    V11 cached alias merges under the normalized label while re-lookups used the raw
    surface, so merges were silently undone. Here the cache holds the raw resolution and
    `canonical()` is applied on the way out, so all paths end at the same id.
    """

    def __init__(self, settings, llm=None):
        self.settings, self.llm = settings, llm
        self.cache: dict[str, Concept] = {}
        self.concepts: dict[str, Concept] = {}
        self.alias: dict[str, str] = {}

    def resolve(self, surface: str) -> Concept:
        key = lookup_key(surface)
        if key not in self.cache:
            c = self._resolve(clean(surface))
            self.cache[key] = c
            self.concepts.setdefault(c.id, c)
        return self.canonical(self.cache[key])

    def canonical(self, c: Concept) -> Concept:
        cid = c.id
        while cid in self.alias:
            cid = self.alias[cid]
        return self.concepts.get(cid, c)

    def merge(self, ids: list[str]):
        """Point every id at the best-provenance member of the group."""
        group = [self.concepts[i] for i in ids if i in self.concepts]
        if len(group) < 2:
            return
        best = max(group, key=lambda c: (not c.id.startswith("LOCAL:"), c.confidence, -len(c.label)))
        for c in group:
            if c.id != best.id:
                self.alias[c.id] = best.id

    def _resolve(self, name: str) -> Concept:
        if not name:
            return local_concept("unknown")
        if re.match(r"^[A-Za-z][A-Za-z0-9_.-]*:\S+$", name):
            return Concept(name, name, "identifier", "identifier", 1.0)
        provider = self.settings.ontology_provider
        if provider in {"ols", "hybrid"}:
            c = self._ols(name)
            if c:
                return c
        if provider in {"llm", "hybrid"} and self.llm is not None:
            try:
                r = self.llm.structured("entity", EntityResolution, PROMPTS["entity"],
                                        f"Entity: {name}", role="cheap", ctx={"surface": name})
                base = local_concept(r.normalized_label)
                return Concept(base.id, r.normalized_label, r.category or "unknown", "llm",
                               float(r.confidence))
            except Exception as e:
                print(f"[entity] LLM resolution failed for {name!r}: {type(e).__name__}")
        return local_concept(name)

    def _ols(self, name: str) -> Concept | None:
        """Accept an OLS hit only on an exact label or synonym match; a search rank is not a fact."""
        t = self.settings.ontology_timeout_s
        try:
            docs = requests.get(f"{OLS}/search", timeout=t, params={
                "q": name, "rows": 5, "fieldList": "obo_id,label,synonym,ontology_name,iri"
            }).json().get("response", {}).get("docs", [])
        except Exception:
            return None
        want = lookup_key(name)
        for d in docs:
            names = [d.get("label", "")] + list(d.get("synonym") or [])
            if d.get("obo_id") and want in {lookup_key(n) for n in names}:
                prefix = d["obo_id"].split(":")[0].upper()
                return Concept(d["obo_id"], d.get("label", name),
                               PREFIX_CATEGORY.get(prefix, "unknown"), "ols", 0.9,
                               self._ancestors(d.get("ontology_name", ""), d.get("iri", "")))
        return None

    def _ancestors(self, onto: str, iri: str) -> tuple:
        if not onto or not iri:
            return ()
        try:
            url = f"{OLS}/ontologies/{onto}/terms/{quote(quote(iri, safe=''), safe='')}/hierarchicalAncestors"
            terms = requests.get(url, timeout=self.settings.ontology_timeout_s,
                                 params={"size": 20}).json().get("_embedded", {}).get("terms", [])
            return tuple(t["obo_id"] for t in terms if t.get("obo_id"))
        except Exception:
            return ()


# ── alias consolidation ─────────────────────────────────────────────────────
def _maybe_alias(a: str, b: str) -> bool:
    # ponytail: lexical prefilter; misses opaque synonyms (trade names). Add embedding
    # neighbours here if alias recall turns out to matter.
    ka, kb = lookup_key(a), lookup_key(b)
    ta = {t for t in ka.split() if len(t) > 2}
    tb = {t for t in kb.split() if len(t) > 2}
    if ta & tb or ka[:4] == kb[:4]:
        return True
    tri = lambda s: {s[i:i + 3] for i in range(len(s) - 2)}
    A, B = tri(ka.replace(" ", "")), tri(kb.replace(" ", ""))
    return bool(A and B) and len(A & B) / len(A | B) >= 0.2


def complete_linkage(ids: list[str], ok: set[frozenset]) -> list[list[str]]:
    """Every pair inside a group must be judged compatible: similarity is not transitive."""
    groups: list[list[str]] = []
    for i in sorted(ids):
        for g in groups:
            if all(frozenset((i, m)) in ok for m in g):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def consolidate_aliases(claims, resolver: EntityResolver, llm, verdicts: dict, batch=50, cap=200):
    """Ask the LLM only about new, lexically plausible pairs; merge with complete linkage."""
    concepts = {}
    for c in claims:
        for surface in (c.subject, c.object):
            k = resolver.resolve(surface)
            concepts[k.id] = k
    ids = sorted(concepts)
    todo = []
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            a, b = concepts[ids[x]], concepts[ids[y]]
            pair = frozenset((a.id, b.id))
            compatible = a.category == b.category or "unknown" in (a.category, b.category)
            if pair not in verdicts and compatible and _maybe_alias(a.label, b.label):
                todo.append((a, b))
    for s in range(0, min(len(todo), cap), batch):
        chunk = todo[s:s + batch]
        text = "\n".join(f"[PAIR] A: {a.label} ({a.category}) | B: {b.label} ({b.category})"
                         for a, b in chunk)
        try:
            out = llm.structured("alias", AliasBatch, PROMPTS["alias"], text,
                                 ctx={"pairs": [(a.label, b.label) for a, b in chunk]},
                                 n_items=len(chunk))
        except Exception as e:
            print(f"[alias] batch failed: {type(e).__name__}")
            continue
        by_labels = {(a.label, b.label): (a, b) for a, b in chunk}
        for v in out.verdicts:
            pair = by_labels.get((v.label_a, v.label_b)) or by_labels.get((v.label_b, v.label_a))
            if pair:
                verdicts[frozenset((pair[0].id, pair[1].id))] = v.same_entity
    ok = {p for p, same in verdicts.items() if same}
    merged = 0
    for g in complete_linkage(ids, ok):
        if len(g) > 1:
            resolver.merge(g)
            merged += len(g) - 1
    return merged


# ── relations ───────────────────────────────────────────────────────────────
RELATION_LEXICON = {
    "upregulates": "increases", "up-regulates": "increases", "induces": "increases",
    "increases": "increases", "increased": "increases", "elevates": "increases",
    "elevated": "increases", "enhances": "increases", "enhanced": "increases",
    "promotes": "increases", "promoted": "increases", "stimulates": "increases",
    "downregulates": "decreases", "down-regulates": "decreases", "suppresses": "decreases",
    "suppressed": "decreases", "inhibits": "decreases", "inhibited": "decreases",
    "reduces": "decreases", "reduced": "decreases", "attenuates": "decreases",
    "impairs": "decreases", "impaired": "decreases", "decreases": "decreases",
    "decreased": "decreases",
    "no effect": "no_effect", "no effect on": "no_effect", "does not affect": "no_effect",
    "unchanged": "no_effect",
    "not associated": "not_associated", "not associated with": "not_associated",
    "modulates": "modulates", "affects": "modulates", "regulates": "modulates",
    "required for": "required_for", "is required for": "required_for",
    "necessary for": "required_for", "sufficient for": "sufficient_for",
    "associated with": "associated_with", "correlated with": "associated_with",
    "correlates with": "associated_with", "predicts": "predicts", "predictive of": "predicts",
    "binds": "binds", "interacts with": "binds", "modifies": "modifies",
    "phosphorylates": "modifies", "lactylates": "modifies",
}


def lexical_relation(surface: str) -> str:
    """Exact lexicon hit or canonical name; '' means ambiguous -> LLM."""
    s = re.sub(r"[\s_]+", " ", (surface or "").lower().replace("-", " ")).strip()
    if s in RELATION_LEXICON:
        return RELATION_LEXICON[s]
    if s.replace(" ", "_") in RELATION_SET - {"unresolved"}:
        return s.replace(" ", "_")
    return ""


# ── anchoring ───────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    s = s.lower().replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s).strip()


def _ordered(span_toks, src_toks) -> float:
    i = 0
    for t in src_toks:
        if i < len(span_toks) and t == span_toks[i]:
            i += 1
    return i / len(span_toks) if span_toks else 0.0


def span_is_anchored(span: str, source: str, min_overlap: float = 0.85) -> bool:
    """A span must be ONE local passage of the source, in order (stitched quotes fail)."""
    if not span or not source:
        return False
    s, src = _norm(span), _norm(source)
    if len(s) < 20:
        return False
    if s in src:
        return True
    toks = [t for t in re.findall(r"[a-z0-9]+", s) if len(t) > 2]
    if not toks:
        return False
    for sent in re.split(r"(?<=[.!?])\s+", src):
        if _ordered(toks, [t for t in re.findall(r"[a-z0-9]+", sent) if len(t) > 2]) >= min_overlap:
            return True
    src_toks = [t for t in re.findall(r"[a-z0-9]+", src) if len(t) > 2]
    w = max(len(toks) * 2, len(toks) + 10)
    # ponytail: O(n*w) window scan; fine for one paper, index sentences if full texts get huge
    return any(_ordered(toks, src_toks[i:i + w]) >= min_overlap
               for i in range(0, max(1, len(src_toks) - w + 1)))
