"""Token-efficiency and report levers of v3 Phase 2: reviews are not extracted (TE-1), a cost budget
(TE-10), allowed wording before writing (RP-1) and one repair pass after verification (RP-2).
Offline: no keys, no network."""
from bmira.offline import offline_runtime
from bmira.telemetry import execute, summarize


def _recording(rt, task):
    """Wraps the scripted model so the user prompt of `task` is kept."""
    seen, real = [], rt.llm.structured

    def structured(t, schema, system, user, *a, **k):
        if t == task:
            seen.append(user)
        return real(t, schema, system, user, *a, **k)
    rt.llm.structured = structured
    return seen


def test_reviews_are_not_extracted_but_inform_the_seed():
    rt, sc = offline_runtime()
    seed = _recording(rt, "seed")
    final, info = execute(sc["question"], rt, echo=False)
    review = next(p for p in final["papers"] if p.pmid == "S019")
    assert review.screen_status == "included" and review.study_type == "review" and review.n_reads == 0
    assert "S019" not in final["extracted_pmids"]
    assert "Reviews (background" in seed[0] and review.title in seed[0]
    m = summarize(final, rt, info)["papers"]
    assert m["reviews_extracted"] == 0 and m["reviews_not_extracted"] == 1 and m["included_never_extracted"] == 0
    assert any("reviews were not extracted" in w for w in final["warnings"])


def _searched(study_type):
    """One targeted round in which the step's only hit was this paper, left unread; its zero-yield count."""
    from bmira import portfolio as pf
    from bmira.graph import portfolio
    from bmira.schemas import Hypothesis, Paper
    from helpers import parsed_question
    rt, _ = offline_runtime()
    a, b, y = (rt.resolver.resolve(x).id for x in ("lactate", "GPR81", "CD8 T cell effector function"))
    k = pf.link_key(a, "increases", b)
    prior = {k: pf.LinkEvidence(key=k, subject=a, relation="increases", object=b, subject_label="Lactate",
                                object_label="GPR81")}
    hit = Paper(pmid="r", screen_status="included", study_type=study_type, retrieved_for=[k])
    state = {"claims": [], "links": prior, "targets": [k], "target_searches": {k: {"ok": 3, "clean": 2}},
             "search_status": "NEW_RESULTS", "papers": [hit], "exposure": a, "outcome": y, "outcome_ids": [y],
             "parsed": parsed_question(exposure="lactate", outcome="CD8 T cell effector function"),
             "seed_status": "OK", "round_idx": 1,
             "hypotheses": [Hypothesis(id="H1", name="r", origin="llm_seed", links=[k, pf.link_key(b, "decreases", y)])]}
    return portfolio(state, rt)["links"][k].zero_yield_count


def test_a_step_found_only_in_a_review_can_still_be_searched_out():
    """R7 counts a step's search only when every hit was read for it. A review is never read (TE-1), so it must
    not hold the step open forever; an unread primary paper still does."""
    assert _searched("review") == 1
    assert _searched("animal") == 0
