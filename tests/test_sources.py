"""PubMed and Europe PMC access (bmira.sources), with the HTTP layer replaced. Offline: no keys, no network."""
import xml.etree.ElementTree as ET

import pytest
import requests

import bmira.sources as src
from bmira.config import Settings
from bmira.schemas import Paper


class Response:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self.payload, self.content = status, payload, content

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def sleeps(monkeypatch):
    """No real waiting; the waits asked for are recorded."""
    waited = []
    monkeypatch.setattr(src.time, "sleep", waited.append)
    return waited


def test_request_retries_transient_errors_with_backoff(monkeypatch, sleeps):
    replies = [Response(503), Response(429), Response(200, {"ok": 1})]
    monkeypatch.setattr(src.requests, "request", lambda *a, **k: replies.pop(0))
    assert src._request("GET", "u").json() == {"ok": 1}
    assert sleeps == [1, 2]                                  # exponential backoff between the tries


def test_request_gives_up_after_the_last_try(monkeypatch, sleeps):
    monkeypatch.setattr(src.requests, "request", lambda *a, **k: Response(502))
    with pytest.raises(requests.HTTPError):
        src._request("GET", "u", tries=3)
    assert sleeps == [1, 2]                                  # no wait after the final failure


def test_search_reports_ignored_terms_and_sends_the_key(monkeypatch, sleeps):
    seen = []
    payload = {"esearchresult": {"idlist": ["1", "2"], "querytranslation": "t",
                                 "warninglist": {"phrasesignored": ["failed to induce"],
                                                 "quotedphrasesnotfound": ["\"x y\""]}}}
    monkeypatch.setattr(src, "_request", lambda method, url, **kw: seen.append((method, url, kw)) or Response(200, payload))
    out = src.PubMedSource(Settings(ncbi_api_key="K", ncbi_email="me@example.org")).search("q", 20)
    assert out == {"pmids": ["1", "2"], "translation": "t", "ignored_terms": ["failed to induce", "\"x y\""]}
    method, url, kw = seen[0]
    assert method == "GET" and url.endswith("esearch.fcgi")
    assert kw["params"]["api_key"] == "K" and kw["params"]["email"] == "me@example.org" and kw["params"]["retmax"] == 20
    assert sleeps == [0.12]                                  # NCBI rate limit with a key (0.35 without)


EFETCH = b"""<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
  <Journal><Title>J One</Title><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
  <ArticleTitle>Trial title</ArticleTitle>
  <Abstract><AbstractText Label="Background">B text.</AbstractText><AbstractText Label="Results">R text.</AbstractText></Abstract>
  <PublicationTypeList><PublicationType>Journal Article</PublicationType>
    <PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>222</PMID><Article>
  <Journal><Title>J Two</Title><JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
  <ArticleTitle>Retracted title</ArticleTitle>
  <Abstract><AbstractText>Plain abstract.</AbstractText></Abstract>
  <PublicationTypeList><PublicationType>Retracted Publication</PublicationType></PublicationTypeList>
</Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_fetch_parses_abstract_sections_dates_and_publication_types(monkeypatch, sleeps):
    monkeypatch.setattr(src, "_request", lambda *a, **k: Response(200, content=EFETCH))
    trial, retracted = src.PubMedSource(Settings()).fetch(["111", "222"])
    assert (trial.pmid, trial.title, trial.journal, trial.year) == ("111", "Trial title", "J One", "2021")
    assert trial.abstract == "BACKGROUND: B text. RESULTS: R text." and trial.source_text == trial.abstract
    assert trial.pubtype_study_type == "human_rct" and not trial.retracted
    assert retracted.year == "2019" and retracted.abstract == "Plain abstract." and retracted.retracted


def test_fetch_posts_ids_in_batches_of_200(monkeypatch, sleeps):
    batches = []

    def post(method, url, **kw):
        batches.append((method, len(kw["data"]["id"].split(","))))
        return Response(200, content=b"<PubmedArticleSet/>")
    monkeypatch.setattr(src, "_request", post)
    assert src.PubMedSource(Settings()).fetch([str(i) for i in range(450)]) == []
    assert batches == [("POST", 200), ("POST", 200), ("POST", 50)]


def _paper(**kw):
    return Paper(pmid="111", title="t", abstract="An abstract.", source_text="An abstract.", **kw)


def _europe_pmc(monkeypatch, open_access="Y", body=None, fail=False):
    results = "" if body is None else body

    def get(method, url, **kw):
        if fail:
            raise requests.ConnectionError("down")
        if url.endswith("/search"):
            return Response(200, {"resultList": {"result": [{"isOpenAccess": open_access, "pmcid": "PMC1"}]}})
        assert url.endswith("PMC1/fullTextXML")
        return Response(200, content=f"<article>{results}</article>".encode())
    monkeypatch.setattr(src, "_request", get)


def test_fulltext_reads_open_access_text_by_section(monkeypatch):
    long_results = "Lactate reduced IFNG in CD8 T cells. " * 30
    _europe_pmc(monkeypatch, body=f"<body><sec><title>Methods</title><p>M.</p></sec>"
                                   f"<sec><title>Results</title><p>{long_results}</p></sec></body>")
    out = src.PubMedSource(Settings()).fulltext(_paper())
    assert out.text_access == "full_text" and out.source_text.startswith("ABSTRACT: An abstract.")
    assert out.source_text.index("RESULTS:") < out.source_text.index("METHODS:")


@pytest.mark.parametrize("case", ["closed access", "no body", "short body", "service down"])
def test_fulltext_keeps_the_abstract_when_no_usable_text(monkeypatch, case):
    if case == "closed access":
        _europe_pmc(monkeypatch, open_access="N", body="<body><sec><p>x</p></sec></body>")
    elif case == "no body":
        _europe_pmc(monkeypatch, body="<front/>")
    elif case == "short body":
        _europe_pmc(monkeypatch, body="<body><sec><title>Results</title><p>Too short to add much.</p></sec></body>")
    else:
        _europe_pmc(monkeypatch, fail=True)
    paper = _paper()
    out = src.PubMedSource(Settings()).fulltext(paper)
    assert out.text_access == paper.text_access and out.source_text == "An abstract."


def test_fulltext_is_not_fetched_twice(monkeypatch):
    monkeypatch.setattr(src, "_request", lambda *a, **k: pytest.fail("no request expected"))
    paper = _paper(text_access="full_text")
    assert src.PubMedSource(Settings()).fulltext(paper) is paper


def test_sectioned_text_respects_the_limit_in_priority_order():
    body = ET.fromstring("<body><sec><title>Discussion</title><p>DDDD</p></sec>"
                         "<sec><title>Results</title><p>RRRRRRRRRR</p></sec></body>")
    text = src.sectioned_text("abs", body, 9)                # 3 characters of abstract, 6 left for sections
    abstract, results = text.split("\n\n")
    assert abstract == "ABSTRACT: abs" and results.startswith("RESULTS: ") and len(results) == len("RESULTS: ") + 6
    assert "DISCUSSION" in src.sectioned_text("abs", body, 1000)   # lower-priority sections come when room is left


def test_publication_types_map_to_the_strongest_design():
    assert src.study_type_from_pubtypes(["Journal Article", "Meta-Analysis", "Randomized Controlled Trial"]) == "meta_analysis"
    assert src.study_type_from_pubtypes(["Journal Article"]) is None
    assert src.is_retracted(["Expression of Concern"]) and not src.is_retracted(["Review"])
