"""Entity resolution, alias registry, lexical relation typing and span anchoring."""
import re
from dataclasses import dataclass
from urllib.parse import quote

import requests

from bmira.llm import PROMPTS
from bmira.schemas import AliasBatch, EntityResolution, RELATION_SET

OLS = "https://www.ebi.ac.uk/ols4/api"
# Ontologies searched, best first. Unrestricted search returns exact matches from obscure
# ontologies (e.g. an unmapped NCIT term) ahead of the right one.
ONTOLOGIES = ["chebi", "pr", "go", "cl", "hp", "mp", "efo", "mondo", "uberon", "ncit"]
PREFIX_CATEGORY = {
    "CL": "cell_type", "GO": "process", "CHEBI": "chemical", "HP": "phenotype",
    "MP": "phenotype", "MONDO": "disease", "DOID": "disease", "EFO": "phenotype",
    "PR": "gene_or_protein", "HGNC": "gene_or_protein", "NCBIGENE": "gene_or_protein",
    "UBERON": "anatomy", "NCIT": "other",
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
    ancestors: tuple = ()      # transitive; used to recognise outcome descendants
    parents: tuple = ()        # direct; used to block claim comparisons


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
        """Accept an OLS hit only on an exact label or synonym match; a search rank is not a fact.
        Among exact matches, the preferred ontology wins."""
        t = self.settings.ontology_timeout_s
        try:
            docs = requests.get(f"{OLS}/search", timeout=t, params={
                "q": name, "rows": 20, "ontology": ",".join(ONTOLOGIES),
                "fieldList": "obo_id,label,synonym,ontology_name,iri"}).json().get("response", {}).get("docs", [])
        except Exception:
            return None
        want = lookup_key(name)
        exact = [d for d in docs if d.get("obo_id") and want in
                 {lookup_key(n) for n in [d.get("label", "")] + list(d.get("synonym") or [])}]
        if not exact:
            return None
        rank = lambda d: ONTOLOGIES.index(d["ontology_name"]) if d.get("ontology_name") in ONTOLOGIES else 99
        d = min(exact, key=rank)
        parents, ancestors = self._hierarchy(d.get("ontology_name", ""), d.get("iri", ""))
        return Concept(d["obo_id"], d.get("label", name),
                       PREFIX_CATEGORY.get(d["obo_id"].split(":")[0].upper(), "unknown"), "ols", 0.9,
                       ancestors, parents)

    def _hierarchy(self, onto: str, iri: str) -> tuple[tuple, tuple]:
        """(direct parents, all ancestors). OLS does not order ancestors by depth, so slicing
        the ancestor list for 'nearest' families was arbitrary."""
        if not onto or not iri:
            return (), ()
        base = f"{OLS}/ontologies/{onto}/terms/{quote(quote(iri, safe=''), safe='')}"
        out = []
        for rel in ("parents", "hierarchicalAncestors"):
            try:
                terms = requests.get(f"{base}/{rel}", timeout=self.settings.ontology_timeout_s,
                                     params={"size": 50}).json().get("_embedded", {}).get("terms", [])
                out.append(tuple(t["obo_id"] for t in terms if t.get("obo_id")))
            except Exception:
                out.append(())
        return out[0], out[1]


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
    for c in claims:                     # the claims' resolved entities, never raw surfaces
        for cid in (c.subject_concept, c.object_concept):
            if cid in resolver.concepts:
                k = resolver.canonical(resolver.concepts[cid])
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


def _tokens(s: str) -> list[str]:
    """Content tokens; short negations are kept so a fabricated 'no' cannot slip through."""
    return [t for t in re.findall(r"[a-z0-9]+", s) if len(t) > 2 or t in {"no", "nor"}]


NEGATION_TOKENS = {"no", "not", "never", "neither", "nor", "without", "unchanged", "unaffected"}


def _ordered(span_toks, src_toks) -> float:
    """Share of span tokens found in order (missing tokens are skipped, not fatal)."""
    j = hit = 0
    for t in span_toks:
        try:
            j = src_toks.index(t, j) + 1
            hit += 1
        except ValueError:
            continue
    return hit / len(span_toks) if span_toks else 0.0


def _match(span_toks, cand_toks, min_overlap) -> bool:
    negs = NEGATION_TOKENS & set(span_toks)          # a negation is never tolerated as noise
    return negs <= set(cand_toks) and _ordered(span_toks, cand_toks) >= min_overlap


def span_is_anchored(span: str, source: str, min_overlap: float = 0.85) -> bool:
    """A span must be ONE local passage of the source, in order (stitched quotes fail)."""
    if not span or not source:
        return False
    s, src = _norm(span), _norm(source)
    if len(s) < 20:
        return False
    if s in src:
        return True
    toks = _tokens(s)
    if not toks:
        return False
    for sent in re.split(r"(?<=[.!?])\s+", src):
        if _match(toks, _tokens(sent), min_overlap):
            return True
    src_toks = _tokens(src)
    w = max(len(toks) * 2, len(toks) + 10)
    # ponytail: O(n*w) window scan; fine for one paper, index sentences if full texts get huge
    return any(_match(toks, src_toks[i:i + w], min_overlap)
               for i in range(0, max(1, len(src_toks) - w + 1)))


# ── claim-span consistency ──────────────────────────────────────────────────
ATTRIBUTE_PATTERNS = [
    (r"\b(?:m?rna |gene |protein )?expression\b|\btranscription\b", "expression"),
    (r"\b(?:levels?|concentrations?|abundance|amounts?|content|secretion|production|release)\b", "amount"),
    (r"\bsignal+ing\b", "activity"),
    (r"\bphosphorylation\b", "modification"),
]


def split_attribute(surface: str) -> tuple[str, str]:
    """'IFNG expression' -> ('IFNG', 'expression'). A graph node is the entity; how it was
    measured is a qualifier on the claim. Cell- and tissue-level phrases are left intact
    ('T cell activation' is a phenotype, not an attribute of T cells)."""
    for pat, attr in ATTRIBUTE_PATTERNS:
        if re.search(pat, surface, flags=re.I):
            rest = re.sub(pat, " ", surface, flags=re.I)
            rest = clean(re.sub(r"^\s*of\s+|\s+of\s*$", " ", rest))
            if len(rest) >= 2 and not re.search(r"\b(cells?|tissues?)\b", rest, flags=re.I):
                return rest, attr
    return clean(surface), "none"


STOP = {"the", "and", "with", "cells", "cell", "human", "mouse", "mice", "levels", "level", "expression"}
NULL_CUE = re.compile(r"\b(?:no|not|never|neither|nor|unchanged|unaffected|without|fail(?:ed|s)?|"
                      r"similar|comparable|independent of)\b")
NEGATORS = {"no", "not", "never", "neither", "nor", "without", "failed", "fail", "fails"}


def _mentioned(surface: str, span: str) -> bool:
    words = [w for w in re.findall(r"[a-z0-9+]+", surface.lower()) if w not in STOP]
    span_words = re.findall(r"[a-z0-9+]+", span.lower())
    long_ = [w for w in words if len(w) >= 3]
    if not long_:                                    # 'pH', 'IL'
        return any(w in span_words for w in words)
    return any(any(sw.startswith(w[:4]) for sw in span_words) for w in long_)


def check_claim(c, source: str) -> tuple[str, list[str]]:
    """(drop reason or '', warnings). The quote must exist, name both entities, and agree
    in polarity with the claimed relation; the structured claim was previously never checked
    against its own quote."""
    if not span_is_anchored(c.span, source):
        return "quote not found in source", []
    span = c.span.lower().replace("n't", " not")
    for role, surface in (("subject", c.subject), ("object", c.object)):
        if not _mentioned(surface, span):
            return f"{role} not named in quote", []
    rel = c.relation.lower().replace("n't", " not")
    if NULL_CUE.search(rel):
        return ("", []) if NULL_CUE.search(span) else ("null claim but the quote reports an effect", [])
    words = re.findall(r"[a-z0-9]+", span)
    verbs = [w for w in re.findall(r"[a-z]+", rel) if len(w) >= 4]
    hits = [i for i, w in enumerate(words) if any(w.startswith(v[:4]) for v in verbs)]
    if not hits:
        return "", ["relation wording not in quote"]
    if any(NEGATORS & set(words[max(0, i - 3):i]) for i in hits):
        return "quote negates the claimed effect", []
    return "", []
