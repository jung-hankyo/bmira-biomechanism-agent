"""PubMed / Europe PMC access. Retrieval never judges evidence; it only supplies papers.

FixtureCorpus in bmira.offline offers the same three methods for offline runs.
"""
import time
import xml.etree.ElementTree as ET

import requests

from bmira.schemas import Paper

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# PublicationType -> study design. Structured metadata outranks the screening model's guess.
PUBTYPE_TO_STUDY = {
    "meta-analysis": "meta_analysis", "systematic review": "meta_analysis",
    "randomized controlled trial": "human_rct", "controlled clinical trial": "human_rct",
    "clinical trial, phase iii": "human_rct", "clinical trial": "human_cohort",
    "observational study": "human_cohort", "multicenter study": "human_cohort",
    "review": "review", "scoping review": "review",
}
PUBTYPE_PRECEDENCE = ["meta_analysis", "human_rct", "human_cohort", "review"]
RETRACTION_TYPES = {"retracted publication", "retraction of publication", "expression of concern"}


def study_type_from_pubtypes(pubtypes):
    mapped = {PUBTYPE_TO_STUDY.get(p.lower()) for p in pubtypes}
    return next((s for s in PUBTYPE_PRECEDENCE if s in mapped), None)


def is_retracted(pubtypes) -> bool:
    return any(p.lower() in RETRACTION_TYPES for p in pubtypes)


def _text(node) -> str:
    return "".join(node.itertext()).strip() if node is not None else ""


class PubMedSource:
    name = "pubmed"

    def __init__(self, settings):
        self.settings = settings

    def _get(self, path, **params):
        params = {"tool": "bmira", "email": self.settings.ncbi_email, **params}
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        r = requests.get(f"{NCBI}/{path}", params=params, timeout=60)
        r.raise_for_status()
        time.sleep(0.12 if self.settings.ncbi_api_key else 0.35)   # NCBI rate limit
        return r

    def search(self, query: str, retmax: int) -> dict:
        res = self._get("esearch.fcgi", db="pubmed", term=query, retmax=retmax,
                        retmode="json").json().get("esearchresult", {})
        warn = res.get("warninglist") or {}
        ignored = list(warn.get("phrasesignored", [])) + list(warn.get("quotedphrasesnotfound", []))
        # PubMed silently re-translates unmatched terms; surface that instead of trusting hits.
        return {"pmids": res.get("idlist", []), "translation": res.get("querytranslation", ""),
                "ignored_terms": ignored}

    def fetch(self, pmids: list[str]) -> list[Paper]:
        if not pmids:
            return []
        root = ET.fromstring(self._get("efetch.fcgi", db="pubmed", id=",".join(pmids),
                                       retmode="xml").content)
        papers = []
        for art in root.findall(".//PubmedArticle"):
            abstract = " ".join(_text(a) for a in art.findall(".//Abstract/AbstractText"))
            pubtypes = [p for p in (_text(x) for x in
                        art.findall(".//PublicationTypeList/PublicationType")) if p]
            papers.append(Paper(
                pmid=_text(art.find(".//PMID")), title=_text(art.find(".//ArticleTitle")),
                abstract=abstract, journal=_text(art.find(".//Journal/Title")),
                year=_text(art.find(".//PubDate/Year")), source_text=abstract,
                publication_types=pubtypes, pubtype_study_type=study_type_from_pubtypes(pubtypes),
                retracted=is_retracted(pubtypes)))
        return papers

    def fulltext(self, paper: Paper) -> Paper:
        """Open-access full text from Europe PMC; abstract stays the anchor source otherwise."""
        try:
            hits = requests.get(f"{EPMC}/search", timeout=30, params={
                "query": f"EXT_ID:{paper.pmid} AND SRC:MED", "resultType": "core",
                "format": "json"}).json().get("resultList", {}).get("result", [])
            if not hits or hits[0].get("isOpenAccess") != "Y" or not hits[0].get("pmcid"):
                return paper
            xml = requests.get(f"{EPMC}/{hits[0]['pmcid']}/fullTextXML", timeout=60)
            body = ET.fromstring(xml.content).find(".//body") if xml.status_code == 200 else None
            text = " ".join(body.itertext()) if body is not None else ""
        except Exception:
            return paper
        if len(text) > 500:
            return paper.model_copy(update={"source_text": text, "text_access": "full_text"})
        return paper
