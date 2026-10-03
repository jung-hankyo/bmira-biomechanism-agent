"""Entity resolution, alias registry, lexical relation typing and span anchoring."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import requests

from bmira.llm import PROMPTS
from bmira.schemas import AliasBatch, EntityBatch, EntityResolution, RELATION_SET

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


ABBR_TAIL = re.compile(r"\s*\((?=[^()]*[A-Z])[^()\s]{2,12}\)?$")      # '(HDAC)', or a truncated '(CD25'


def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", re.sub(r"[\u2013\u2014]", "-", s or "")).strip(" .,;:[]{}")
    m = ABBR_TAIL.search(s)
    if m and len(s[:m.start()]) >= 2 and s[:m.start()].lower() not in GENERIC:
        s = s[:m.start()]                          # the abbreviation is an alias, not part of the name
    if s.count("(") != s.count(")") or re.fullmatch(r"\([^()]*\)", s):
        s = s.strip(" .,;:()[]{}")                 # a stray or wrapping parenthesis only
    return s


GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "Δ": "delta",
         "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta", "κ": "kappa",
         "λ": "lambda", "μ": "mu", "µ": "mu", "σ": "sigma", "τ": "tau",
         "ω": "omega"}


def ascii_name(s: str) -> str:
    """Spell out Greek letters and unify charges: 'IFN-γ' and 'IFN-α' used to collapse to
    'ifn', 'IL-1β' and 'IL-1α' to 'il 1'; 'NAD(+)' and 'NAD⁺' did not match 'NAD+'."""
    s = re.sub(r"\(\s*\+\s*\)", "+", s.replace("⁺", "+"))
    return "".join(GREEK.get(ch, ch) for ch in s)


def lookup_key(s: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", ascii_name(clean(s)).lower()).strip()


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
    slug = re.sub(r"[^a-z0-9+]+", "_", ascii_name(clean(label)).lower()).strip("_")[:120] or "unknown"
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
        self.disk_hits = 0
        self._disk = self._load()

    # ── disk cache: ontology and LLM resolutions are reused across runs ──
    def _path(self):
        d = self.settings.cache_dir
        # v2: v1 files hold split LOCAL ids for LLM-normalized labels ('Treg cell' vs 'Treg')
        # v3: v2 files hold salts as their own chemical ('NaB' -> sodium butyrate) and NCIT
        #     measurement terms ('forkhead box p3' -> 'Forkhead Box Protein P3 Measurement')
        return Path(d) / f"entities_v3_{self.settings.ontology_provider}.json" if d else None

    def _load(self) -> dict:
        p = self._path()
        if p and p.exists():
            try:
                return {k: Concept(**{**v, "ancestors": tuple(v["ancestors"]), "parents": tuple(v["parents"])})
                        for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
            except Exception:
                return {}
        return {}

    def save(self):
        p = self._path()
        if p:
            keep = {k: asdict(c) for k, c in self.cache.items() if c.source in {"ols", "llm", "identifier"}}
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({**{k: asdict(c) for k, c in self._disk.items()}, **keep}),
                         encoding="utf-8")

    def _remember(self, key, c: Concept):
        self.cache[key] = c
        self.concepts.setdefault(c.id, c)

    def resolve(self, surface: str) -> Concept:
        name = singular(surface)
        key = lookup_key(name)
        if key not in self.cache:
            if key in self._disk:
                self.disk_hits += 1
                self._remember(key, self._disk[key])
            else:
                self._remember(key, self._resolve(name))
        return self.canonical(self.cache[key])

    def resolve_many(self, surfaces):
        """Resolve all new surfaces at once: identifiers and ontology lookups in parallel,
        then ONE batched LLM call per 30 leftovers instead of one call each."""
        todo = {}
        for surface in surfaces:
            name = singular(surface)
            key = lookup_key(name)
            if not key or key in self.cache or key in todo:
                continue
            if key in self._disk:
                self.disk_hits += 1
                self._remember(key, self._disk[key])
            else:
                todo[key] = name
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=8) as ex:
            found = dict(zip(todo, ex.map(self._known, todo.values())))
        left = [k for k, c in found.items() if c is None]
        if left and self.settings.ontology_provider in {"llm", "hybrid"} and self.llm is not None:
            for i in range(0, len(left), 30):
                chunk = [todo[k] for k in left[i:i + 30]]
                try:
                    out = self.llm.structured("entities", EntityBatch, PROMPTS["entities"],
                                              "\n".join(f"- {n}" for n in chunk), role="cheap",
                                              ctx={"surfaces": chunk}, n_items=len(chunk))
                except Exception as e:
                    print(f"[entity] batch failed ({type(e).__name__}); {len(chunk)} kept unresolved")
                    continue
                by_key = {lookup_key(singular(r.surface)): r for r in out.items}
                names = {lookup_key(singular(r.normalized_label)): singular(r.normalized_label)
                         for r in by_key.values()}
                names = {k: v for k, v in names.items() if k and k not in self.cache and k not in self._disk}
                with ThreadPoolExecutor(max_workers=8) as ex:
                    hits = dict(zip(names, ex.map(self._known, names.values())))
                for n in chunk:
                    r = by_key.get(lookup_key(n))
                    if r:
                        found[lookup_key(n)] = self._labelled(r.normalized_label, r.category, r.confidence, hits)
        for key, name in todo.items():
            self._remember(key, found.get(key) or local_concept(name))

    def _known(self, name: str) -> Concept | None:
        if re.match(r"^[A-Za-z][A-Za-z0-9_.-]*:\S+$", name):
            return Concept(name, name, "identifier", "identifier", 1.0)
        if self.settings.ontology_provider in {"ols", "hybrid"}:
            c = self._ols(parent_chemical(name))
            if c and parent_chemical(c.label) != c.label:      # 'NaB' -> sodium butyrate -> butyrate
                return self._ols(parent_chemical(c.label)) or c
            return c
        return None

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

    def _labelled(self, label: str, category: str, confidence: float, hits: dict | None = None) -> Concept:
        """An LLM-normalized label is looked up like any surface (cache, disk, ontology) before it
        becomes a LOCAL id; otherwise 'Treg cell' -> LOCAL:regulatory_t_cell and 'Treg' ->
        CL:0000815 name one cell type twice and no pathway can connect them."""
        name = singular(label)
        key = lookup_key(name)
        known = self.cache.get(key) or self._disk.get(key) or (hits or {}).get(key) \
            or (None if hits is not None else self._known(name))
        if known:
            self._remember(key, known)
            return known
        base = local_concept(name)
        return Concept(base.id, base.label, category or "unknown", "llm", float(confidence))

    def _resolve(self, name: str) -> Concept:
        if not name:
            return local_concept("unknown")
        c = self._known(name)
        if c:
            return c
        if self.settings.ontology_provider in {"llm", "hybrid"} and self.llm is not None:
            try:
                r = self.llm.structured("entity", EntityResolution, PROMPTS["entity"],
                                        f"Entity: {name}", role="cheap", ctx={"surface": name})
                return self._labelled(r.normalized_label, r.category, r.confidence)
            except Exception as e:
                print(f"[entity] LLM resolution failed for {name!r}: {type(e).__name__}")
        return local_concept(name)

    def _ols(self, name: str) -> Concept | None:
        """Accept an OLS hit only on an exact label or synonym match; a search rank is not a fact.
        Among exact matches a label beats a synonym ('niacin' is a synonym of both nicotinamide
        and nicotinic acid, but the label of NCIT Niacin), then the preferred ontology wins."""
        t = self.settings.ontology_timeout_s
        try:
            docs = requests.get(f"{OLS}/search", timeout=t, params={
                "q": name, "rows": 20, "ontology": ",".join(ONTOLOGIES),
                "fieldList": "obo_id,label,synonym,ontology_name,iri"}).json().get("response", {}).get("docs", [])
        except Exception:
            return None
        want = lookup_key(name)
        exact = [d for d in docs if ":" in (d.get("obo_id") or "") and want in       # MONDO also returns ids like '1318'
                 {lookup_key(n) for n in [d.get("label", "")] + list(d.get("synonym") or [])}]
        exact = [d for d in exact if _species(d.get("label", "")) != "other" and not any(
            w not in name.lower() and re.search(rf"\b{w}\b", d.get("label", ""), re.I) for w in QUALIFIED)]
        if not exact:                                  # e.g. only 'interleukin-10 (chicken)'
            return None
        onto = lambda d: ONTOLOGIES.index(d["ontology_name"]) if d.get("ontology_name") in ONTOLOGIES else 99
        d = min(exact, key=lambda d: (lookup_key(d.get("label", "")) != want,
                                      SPECIES_RANK[_species(d.get("label", ""))], onto(d)))
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


SPECIES_RANK = {"agnostic": 0, "human": 1, "mouse": 2, "rat": 3, "other": 9}
# NCIT terms ABOUT an entity, not the entity: 'HDAC9 wt Allele' for 'HDAC', 'Forkhead Box
# Protein P3 Measurement' for 'forkhead box p3'. Kept only when the name asks for one.
QUALIFIED = ("allele", "measurement")


def _species(label: str) -> str:
    """Protein Ontology labels carry a species suffix, e.g. 'interleukin-10 (chicken)'.
    Species-agnostic terms come first, then human, mouse, rat; other species are rejected."""
    m = re.search(r"\(([^()]+)\)\s*$", label)
    if not m:
        return "agnostic"
    tag = m.group(1).lower()
    for name, words in (("human", ("human", "homo sapiens")), ("mouse", ("mouse", "mus musculus")),
                        ("rat", ("rat", "rattus norvegicus"))):
        if tag in words:
            return name
    # 'chicken', 'yeast', and binomials with a strain: 'Schizosaccharomyces pombe 972h-'
    return "other" if re.fullmatch(r"[a-z .]+", tag) or re.match(r"[a-z]+\.? [a-z]+\b", tag) else "agnostic"


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


def consolidate_aliases(claims, resolver: EntityResolver, llm, verdicts: dict, batch=50, cap=200,
                        extra_ids=()):
    """Ask the LLM only about new, lexically plausible pairs; merge with complete linkage.
    `extra_ids`: concepts used by pathway proposals that no claim names (yet)."""
    concepts = {}
    # the resolved entities, never raw surfaces
    for cid in [i for c in claims for i in (c.subject_concept, c.object_concept)] + list(extra_ids):
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


# ── entity parsing ──────────────────────────────────────────────────────────
# A graph node is an ENTITY. How it was measured (attribute) and where (tissue) are
# qualifiers on the claim; otherwise "colonic Treg induction", "Treg differentiation" and
# "bone marrow regulatory T cells" become separate nodes that no pathway can connect.
PROCESS = r"differentiation|induction|generation|development|conversion|polarization"
ATTRIBUTE_PATTERNS = [
    (r"\b(?:m?rna |gene |protein )?expression\b|\btranscription\b", "expression"),
    (rf"\b(?:{PROCESS})\b", "differentiation"),
    (r"\b(?:levels?|concentrations?|abundance|amounts?|content|secretion|production|release|"
     r"frequency|frequencies|numbers?|proportions?|percentages?|expansion|accumulation)\b", "amount"),
    (r"\bsignal+ing\b", "activity"),
    (r"\bphosphorylation\b", "modification"),
]
GENERIC = {"cell", "cells", "tissue", "tissues", "gene", "protein", "level", "levels"}
TISSUES = [
    (r"colonic|colon", "colon"), (r"small[- ]intestinal|intestinal|gut|ileal|jejunal", "intestine"),
    (r"peyer['’]?s[- ]patch(?:es)?", "Peyer's patch"), (r"lamina propria", "lamina propria"),
    (r"splenic|spleen", "spleen"), (r"bone[- ]marrow|bm", "bone marrow"),
    (r"(?:mesenteric |pancreatic |draining )?lymph[- ]nodes?|mln", "lymph node"),
    (r"thymic|thymus", "thymus"), (r"peripheral[- ]blood|circulating", "blood"),
    (r"tumou?r[- ]infiltrating|intratumou?ral", "tumor"), (r"mucosal", "mucosa"),
    (r"hepatic|liver", "liver"), (r"pulmonary|lung", "lung"), (r"cutaneous|skin", "skin"),
    (r"synovial", "synovium"), (r"(?:visceral )?adipose", "adipose tissue"),
]
_TISSUE_RE = "|".join(f"(?:{p})" for p, _ in TISSUES)
MODIFIERS = r"de novo|extrathymic|in vitro|in vivo|[\w\-+]+-(?:induced|derived|treated|exposed)"
PLACEHOLDERS = {"", "not specified", "unspecified", "unknown", "none", "n/a", "na", "not reported",
                "not applicable", "it", "they", "this", "these", "that", "those"}
NO_SINGULAR = {"diabetes", "herpes", "series", "species", "lupus", "status", "mumps", "measles", "sepsis"}


def split_attribute(surface: str) -> tuple[str, str]:
    """'IFNG expression' -> ('IFNG', 'expression'); 'Treg differentiation' -> ('Treg',
    'differentiation'). A phrase whose remainder would be generic ('cell numbers') is kept."""
    s = clean(surface)
    m = re.match(rf"^(?:[\w\-]+\s+)?({PROCESS})\s+of\s+(?:the\s+)?(.+)$", s, flags=re.I)
    if m:
        return clean(m.group(2)), "differentiation"
    for pat, attr in ATTRIBUTE_PATTERNS:
        if re.search(pat, s, flags=re.I):
            rest = clean(re.sub(r"^\s*of\s+|\s+of\s*$", " ", re.sub(pat, " ", s, flags=re.I)))
            if len(rest) >= 2 and rest.lower() not in GENERIC:
                return rest, attr
    return s, "none"


def split_location(surface: str) -> tuple[str, str]:
    """'Bone marrow, splenic and Peyer's patch regulatory T cells' -> ('regulatory T cells',
    'bone marrow, spleen, Peyer's patch'); also '... in (the) mesenteric lymph nodes'."""
    s, found = clean(surface), []
    while True:
        m = re.match(rf"^({_TISSUE_RE})(?:\s*,\s*|\s+and\s+|\s+)", s, flags=re.I)
        if not m or len(s) - m.end() < 2:
            break
        found.append(m.group(1))
        s = s[m.end():]
    m = re.search(rf"\s+in\s+(?:the\s+)?({_TISSUE_RE})$", s, flags=re.I)
    if m:
        found.append(m.group(1))
        s = s[:m.start()]
    if not found or clean(s).lower() in GENERIC:
        return clean(surface), ""
    names = [next(n for p, n in TISSUES if re.fullmatch(p, f, flags=re.I)) for f in found]
    return clean(s), ", ".join(dict.fromkeys(names))


# 'NAD+ decline', 'Tet2 loss', 'vitamin D deficiency': a decrease of the entity, not another entity.
# Without this the exposure never meets the claims that name the bare 'NAD+' / 'Tet2'.
CHANGE_DOWN = r"loss|deficiency|depletion|decline|knockout|deletion|knockdown"
NOT_AN_ENTITY = {"bone", "weight", "hearing", "muscle", "hair", "fat", "vision", "memory", "tissue", "cell",
                 "cells", "body", "blood", "appetite", "neuron", "neuronal", "synapse", "skin", "lung"}


def split_change(surface: str) -> tuple[str, str]:
    """'age-related NAD+ decline' -> ('NAD+', 'down'); 'bone loss' stays (a phenotype)."""
    s = clean(surface)
    for pat in (rf"^(?:(?:age|aging|ageing)[- ](?:related|associated)\s+)?(.+?)\s+(?:{CHANGE_DOWN})$",
                rf"^(?:{CHANGE_DOWN})\s+of\s+(?:the\s+)?(.+)$"):
        m = re.match(pat, s, flags=re.I)
        if m and len(m.group(1)) >= 2 and m.group(1).lower() not in GENERIC \
                and m.group(1).split()[-1].lower() not in NOT_AN_ENTITY:
            return m.group(1), "down"
    return s, ""


def entity_change(surface: str) -> str:
    return split_change(surface)[1]


# 'sodium butyrate', 'butyric acid' and 'butyrate' are one exposure: an inert counter-ion or the
# protonation state is not the mechanism. Pilot4 split 25 butyrate claims over four nodes.
# ponytail: sodium/potassium salts only; for 'zinc sulfate' or 'magnesium sulfate' the metal is the agent.
SALT = re.compile(r"^(?:sodium|potassium)\s+(\S+ate)$", re.I)
ACID = re.compile(r"^(\S+?)(?<!nucle)ic acid$", re.I)
GIVEN = r"treatment|supplementation|administration|provision|exposure"


def parent_chemical(name: str) -> str:
    """'sodium butyrate' -> 'butyrate', 'butyric acid' -> 'butyrate'; anything else unchanged."""
    if m := SALT.match(name.strip()):
        return m.group(1)
    if m := ACID.match(name.strip()):
        return m.group(1) + "ate"
    return name


def strip_given(surface: str) -> str:
    """'Butyrate supplementation', 'provision of butyrate' -> the agent, not the act of giving it."""
    s = clean(surface)
    m = re.match(rf"^(?:{GIVEN})\s+(?:of|with)\s+(?:the\s+)?(.+)$", s, flags=re.I) \
        or re.match(rf"^(.+?)\s+(?:{GIVEN})$", s, flags=re.I)
    return m.group(1) if m and len(m.group(1)) >= 2 and m.group(1).lower() not in GENERIC else s


def entity_of(surface: str) -> tuple[str, str, str]:
    """surface -> (entity, attribute, tissue)."""
    rest, attr = split_attribute(strip_given(split_change(surface)[0]))
    rest = clean(re.sub(rf"^(?:{MODIFIERS})\s+", "", rest, flags=re.I)) or rest
    rest, tissue = split_location(rest)
    return rest, attr, tissue


def singular(name: str) -> str:
    """Singularize the last word for lookups: 'regulatory T cells' -> 'regulatory T cell'."""
    words = clean(name).split(" ")
    w = words[-1]
    if w.lower() not in NO_SINGULAR and len(w) >= 5 and w[0].isalpha() and w[1:].islower():
        if w.endswith("ies"):
            words[-1] = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
            words[-1] = w[:-1]
    return " ".join(words)


def entity_parts(surface: str) -> tuple[list[str], str]:
    """'NFAT1 and SMAD3' -> (['NFAT1', 'SMAD3'], ''). Only short parts are split, so names
    such as 'signal transducer and activator of transcription 3' stay whole."""
    if re.search(r"[-−–]/[-−–]|\+/[-−–+]", surface):    # Tet2-/-, Foxp3+/+: one genotype
        return [clean(surface)], ""
    rest, tissue = split_location(surface)
    parts = [p.strip() for p in re.split(r"\s*,\s*(?:and\s+|or\s+)?|\s+and\s+|\s*/\s*", rest) if p.strip()]
    if len(parts) < 2 or any(len(p.split()) > 3 or len(p) < 2 for p in parts):
        return [clean(surface)], ""
    return parts, tissue


def abbreviations(text: str) -> list[tuple[str, str]]:
    """(abbreviation, preceding words) for every 'long form (ABBR)' in a paper."""
    out = []
    for m in re.finditer(r"\(\s*([A-Za-z][A-Za-z0-9\-+]{1,12})\s*\)", text):
        abbr = m.group(1)
        if re.search(r"[A-Z]", abbr):
            words = re.findall(r"[\w\-+]+", text[max(0, m.start() - 160):m.start()])[-(len(abbr) + 3):]
            out.append((abbr.lower(), " ".join(words).lower()))
    return out


# ── claim-span consistency ──────────────────────────────────────────────────
STOP = {"the", "and", "with", "cells", "cell", "human", "mouse", "mice", "levels", "level", "expression"}
NULL_CUE = re.compile(r"\b(?:no|not|never|neither|nor|unchanged|unaffected|without|fail(?:ed|s)?|"
                      r"lack(?:s|ed|ing)?|similar|comparable|independent of)\b")
BUT_NOT = re.compile(r"\bbut not\s+[\w\-+]+")      # 'butyrate but not pentanoate exerts': negates the other agent
NEGATORS = {"no", "not", "never", "neither", "nor", "without", "failed", "fail", "fails"}


def _named(surface: str, text: str) -> bool:
    words = [w for w in re.findall(r"[a-z0-9+]+", surface.lower()) if w not in STOP]
    text_words = re.findall(r"[a-z0-9+]+", text.lower())
    long_ = [w for w in words if len(w) >= 3]
    if not long_:                                    # 'pH', 'IL'
        return any(w in text_words for w in words)
    return any(any(tw.startswith(w[:4]) for tw in text_words) for w in long_)


def _mentioned(surface: str, text: str, abbrevs=()) -> bool:
    """Named directly, or through an abbreviation the paper defines ('sodium butyrate (NaB)')."""
    if _named(surface, text):
        return True
    low, text_low = surface.lower(), text.lower()
    for abbr, long_form in abbrevs:
        if _named(surface, long_form) and re.search(rf"\b{re.escape(abbr)}\b", text_low):
            return True
        if low == abbr and _named(long_form.split()[-1] if long_form else "", text):
            return True
    return False


def _previous_sentence(span: str, source: str) -> str:
    i = _norm(source).find(_norm(span))
    if i <= 0:
        return ""
    before = re.split(r"(?<=[.!?])\s+", _norm(source)[:i].strip())
    return before[-1] if before else ""


def check_claim(c, source: str, abbrevs=()) -> tuple[str, list[str]]:
    """(drop reason or '', warnings). The quote must exist, name both entities (directly, by
    a defined abbreviation, or in the sentence just before it), and agree in polarity with
    the claimed relation."""
    for role, surface in (("subject", c.subject), ("object", c.object)):
        if clean(surface).lower() in PLACEHOLDERS:
            return f"{role} not specified", []
    if not span_is_anchored(c.span, source):
        return "quote not found in source", []
    span = c.span.lower().replace("n't", " not")
    context = _previous_sentence(c.span, source) + " " + span
    for role, surface in (("subject", c.subject), ("object", c.object)):
        if not _mentioned(surface, context, abbrevs):
            return f"{role} not named in quote", []
    rel = c.relation.lower().replace("n't", " not")
    if NULL_CUE.search(rel):
        return ("", []) if NULL_CUE.search(span) else ("null claim but the quote reports an effect", [])
    words = re.findall(r"[a-z0-9]+", BUT_NOT.sub(" ", span))
    verbs = [w for w in re.findall(r"[a-z]+", rel) if len(w) >= 4]
    hits = [i for i, w in enumerate(words) if any(w.startswith(v[:4]) for v in verbs)]
    if not hits:
        return "", ["relation wording not in quote"]
    if any(NEGATORS & set(words[max(0, i - 3):i]) for i in hits):
        return "quote negates the claimed effect", []
    return "", []
