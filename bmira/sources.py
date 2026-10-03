"""PubMed / Europe PMC access. Retrieval never judges evidence; it only supplies papers.

FixtureCorpus in bmira.offline offers the same three methods for offline runs.
"""
import re
import time
import xml.etree.ElementTree as ET

import requests

from bmira.schemas import Paper

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# PublicationType -> study design. Structured metadata outranks the screening model's guess.
# A systematic review without a meta-analysis is a secondary source, not pooled evidence.
PUBTYPE_TO_STUDY = {
    "meta-analysis": "meta_analysis", "systematic review": "review",
    "randomized controlled trial": "human_rct", "controlled clinical trial": "human_rct",
    "clinical trial, phase iii": "human_rct", "clinical trial": "human_cohort",
    "observational study": "human_cohort", "multicenter study": "human_cohort",
    "review": "review", "scoping review": "review",
}
PUBTYPE_PRECEDENCE = ["meta_analysis", "human_rct", "human_cohort", "review"]
RETRACTION_TYPES = {"retracted publication", "retraction of publication", "expression of concern"}
SECTION_ORDER = [("RESULTS", r"result|finding"), ("FIGURE LEGENDS", None),
                 ("METHODS", r"method|material|experimental|procedure"),
                 ("DISCUSSION", r"discussion|conclusion")]


def study_type_from_pubtypes(pubtypes):
    mapped = {PUBTYPE_TO_STUDY.get(p.lower()) for p in pubtypes}
    return next((s for s in PUBTYPE_PRECEDENCE if s in mapped), None)


def is_retracted(pubtypes) -> bool:
    return any(p.lower() in RETRACTION_TYPES for p in pubtypes)


def _text(node) -> str:
    return "".join(node.itertext()).strip() if node is not None else ""


def _flat(node, skip=("fig", "table-wrap", "ref-list")) -> str:
    """Text of a node without figures and tables (their captions are collected separately)."""
    parts = [node.text or ""]
    for child in node:
        if child.tag not in skip:
            parts.append(_flat(child, skip))
        parts.append(child.tail or "")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def sectioned_text(abstract: str, body, limit: int) -> str:
    """Abstract, then Results, figure legends, Methods, Discussion, in that priority, within
    `limit` characters. A blind cut of the first N characters loses Results in journals that
    print Methods first."""
    buckets = {name: [] for name, _ in SECTION_ORDER}
    buckets["OTHER"] = []
    for sec in body.findall("./sec"):
        title = _text(sec.find("title")).lower()
        name = next((n for n, pat in SECTION_ORDER if pat and re.search(pat, title)), "OTHER")
        buckets[name].append(_flat(sec))
    buckets["FIGURE LEGENDS"] = [_text(c) for c in body.iter("caption")]
    out, used = [f"ABSTRACT: {abstract}"], len(abstract)
    for name in [n for n, _ in SECTION_ORDER] + ["OTHER"]:
        chunk = " ".join(x for x in buckets[name] if x)
        if chunk and used < limit:
            chunk = chunk[:limit - used]
            out.append(f"{name}: {chunk}")
            used += len(chunk)
    return "\n\n".join(out)


def _request(method, url, tries=3, **kw):
    """Retry transient failures (network, 429, 5xx) with exponential backoff."""
    for i in range(tries):
        try:
            r = requests.request(method, url, timeout=60, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


class PubMedSource:
    name = "pubmed"

    def __init__(self, settings):
        self.settings = settings

    def _call(self, method, path, **params):
        params = {"tool": "bmira", "email": self.settings.ncbi_email, **params}
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        kw = {"data": params} if method == "POST" else {"params": params}
        r = _request(method, f"{NCBI}/{path}", **kw)
        time.sleep(0.12 if self.settings.ncbi_api_key else 0.35)   # NCBI rate limit
        return r

    def search(self, query: str, retmax: int) -> dict:
        res = self._call("GET", "esearch.fcgi", db="pubmed", term=query, retmax=retmax,
                         sort="relevance", retmode="json").json().get("esearchresult", {})
        warn = res.get("warninglist") or {}
        ignored = list(warn.get("phrasesignored", [])) + list(warn.get("quotedphrasesnotfound", []))
        # PubMed silently re-translates unmatched terms; such a query is not a clean search.
        return {"pmids": res.get("idlist", []), "translation": res.get("querytranslation", ""),
                "ignored_terms": ignored}

    def fetch(self, pmids: list[str]) -> list[Paper]:
        papers = []
        for i in range(0, len(pmids), 200):                  # POST: long id lists break GET URLs
            root = ET.fromstring(self._call("POST", "efetch.fcgi", db="pubmed",
                                            id=",".join(pmids[i:i + 200]), retmode="xml").content)
            for art in root.findall(".//PubmedArticle"):
                abstract = " ".join(
                    (f"{a.get('Label').upper()}: " if a.get("Label") else "") + _text(a)
                    for a in art.findall(".//Abstract/AbstractText"))
                pubtypes = [p for p in (_text(x) for x in
                            art.findall(".//PublicationTypeList/PublicationType")) if p]
                year = _text(art.find(".//PubDate/Year")) or \
                    (re.findall(r"\d{4}", _text(art.find(".//PubDate/MedlineDate"))) or [""])[0]
                papers.append(Paper(
                    pmid=_text(art.find(".//PMID")), title=_text(art.find(".//ArticleTitle")),
                    abstract=abstract, journal=_text(art.find(".//Journal/Title")), year=year,
                    source_text=abstract, publication_types=pubtypes,
                    pubtype_study_type=study_type_from_pubtypes(pubtypes),
                    retracted=is_retracted(pubtypes)))
        return papers

    def fulltext(self, paper: Paper) -> Paper:
        """Open-access full text from Europe PMC; abstract stays the anchor source otherwise."""
        if paper.text_access == "full_text":
            return paper
        try:
            hits = _request("GET", f"{EPMC}/search", params={
                "query": f"EXT_ID:{paper.pmid} AND SRC:MED", "resultType": "core",
                "format": "json"}).json().get("resultList", {}).get("result", [])
            if not hits or hits[0].get("isOpenAccess") != "Y" or not hits[0].get("pmcid"):
                return paper
            body = ET.fromstring(_request("GET", f"{EPMC}/{hits[0]['pmcid']}/fullTextXML").content
                                 ).find(".//body")
        except Exception:
            return paper
        if body is None:
            return paper
        text = sectioned_text(paper.abstract, body, self.settings.fulltext_char_limit)
        if len(text) > len(paper.abstract) + 500:
            return paper.model_copy(update={"source_text": text, "text_access": "full_text"})
        return paper
