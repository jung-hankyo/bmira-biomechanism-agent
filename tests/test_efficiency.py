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
    """TE-1. Pilot4 extracted 8 reviews (8 of 36 extraction calls) for 28 claims that R1 never counted."""
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


# ── TE-10: a cost budget ────────────────────────────────────────────────────
PRICED = {"surrogate": (1000.0, 1000.0)}       # the scripted model, priced so a round costs dollars


def test_spend_counts_llm_and_judge_and_names_unpriced_models():
    from collections import Counter
    from types import SimpleNamespace
    from bmira.llm import spent_usd
    llm = SimpleNamespace(tokens_in=Counter(a=1_000_000, b=10), tokens_out=Counter(a=100_000),
                          model_of={"a": "gpt-6-sol", "b": "mystery"})
    judge = SimpleNamespace(tokens_in=Counter(screen=1_000_000), model="jev-1.13.0", model_of={})
    prices = {"gpt-6-sol": (2.0, 10.0), "jev-1.13.0": (0.042, 0.0)}
    assert spent_usd(llm, judge, prices) == (round(3.0 + 0.042, 4), ["mystery"])
    assert spent_usd(llm, None, prices)[0] == 3.0


def test_a_cost_budget_stops_the_search_and_still_reports():
    rt, sc = offline_runtime(budget_usd=0.01, prices=PRICED)
    final, info = execute(sc["question"], rt, echo=False)
    m = summarize(final, rt, info)
    assert final["gate"] == "COST_BUDGET" and final["round_idx"] == 1 and final["report"]
    assert m["run"]["stop_reason"] == "cost budget reached" and "cost budget reached" in {s["signal"] for s in m["signals"]}
    assert any("cost budget was reached" in w for w in final["warnings"]) and m["llm"]["budget_usd"] == 0.01


def test_a_budget_that_cannot_see_the_model_says_so():
    rt, sc = offline_runtime(budget_usd=0.01)                     # the scripted model has no price
    final, _ = execute(sc["question"], rt, echo=False)
    assert final["gate"] != "COST_BUDGET"
    assert any("models without a price (surrogate)" in w for w in final["warnings"])


def test_a_zero_budget_from_the_command_line_stops_after_one_round(tmp_path):
    from helpers import run_session
    session = run_session(tmp_path, "one\n", "--budget-usd", "0")
    assert session["session"]["budget_usd"] == 0.0
    run = session["runs"][0]
    assert run["run"]["rounds"] == 1 and any("unpriced" in w or "without a price" in w for w in run["warnings"])


# ── RP-1: allowed wording, computed before writing ──────────────────────────
def test_the_writer_is_told_the_wording_each_citation_allows():
    from bmira.evidence import WORDING
    rt, sc = offline_runtime()
    seen, real = [], rt.llm.text
    rt.llm.text = lambda task, system, user, **k: seen.append((task, system, user)) or real(task, system, user, **k)
    final, _ = execute(sc["question"], rt, echo=False)
    _, system, user = next(x for x in seen if x[0] == "synthesize")
    assert "may say" in system and "Shown by a blocking experiment" in system
    weak = next(c for c in final["claims"] if c.grade == "weak" and f"[{c.id}]" in user)
    assert f"[{weak.id}]" in user and WORDING[1] in user.split(f"[{weak.id}]")[1].split("\n")[0]
    assert "required for the exposure's effect, in the tested system" in user        # the shown redox route
    assert "do not state this pathway as established" in user                         # the insufficient ones
    assert "strongest wording: " + WORDING[3] in user


# ── RP-2: one repair pass ───────────────────────────────────────────────────
def _with_overclaim(extra_llm=None, **settings):
    """A finished offline run whose report gains one overclaim on a weak claim; verify runs again."""
    from bmira.graph import verify
    rt, sc = offline_runtime(**settings)
    final, _ = execute(sc["question"], rt, echo=False)
    weak = next(c for c in final["claims"] if c.grade == "weak" and c.id in final["synthesis"])
    bad = f"Lactate drives the loss of CD8 T cell effector function [{weak.id}]."
    state = {**final, "synthesis": final["synthesis"] + "\n\n" + bad}
    if extra_llm:
        rt.llm.structured = extra_llm(rt.llm.structured)
    return verify(state, rt), bad, rt, weak


def test_a_flagged_sentence_is_rewritten_to_its_allowed_wording():
    out, bad, rt, weak = _with_overclaim()
    v = out["verification"]
    assert v["passed"] and v["repair"]["rewritten"] == 1 and v["repair"]["before"]["overclaims"] == 1
    assert bad not in out["synthesis"] and bad not in out["report"]
    assert f"associated with the reported outcome [{weak.id}]." in out["synthesis"]
    assert "1 of 1 flagged sentences were rewritten" in out["report"] and rt.llm.calls["repair"] == 1


def _scripted(rewrites):
    from bmira.schemas import Rewrite, RewriteBatch

    def wrap(real):
        def structured(task, schema, system, user, *a, **k):
            if task == "repair":
                if isinstance(rewrites, Exception):
                    raise rewrites
                pairs = rewrites(user) if callable(rewrites) else rewrites
                return RewriteBatch(rewrites=[Rewrite(n=n, sentence=s) for n, s in pairs])
            return real(task, schema, system, user, *a, **k)
        return structured
    return wrap


def test_a_rewrite_that_drops_or_adds_citations_is_rejected():
    out, bad, _, _ = _with_overclaim(_scripted([(1, "Lactate is associated with lower effector function.")]))
    v = out["verification"]
    assert not v["passed"] and v["repair"]["rewritten"] == 0
    assert v["repair"]["rejected"] == [{"sentence": bad, "why": "tags changed"}] and bad in out["report"]


def test_stray_numbers_duplicates_and_failures_change_nothing():
    import re

    def answers(user):
        tag = re.search(r"\[(C\w+)\]", user)[1]
        return [(7, "x"), (0, "y"), (1, f"Lactate is associated with it [{tag}]."), (1, f"Lactate drives it [{tag}].")]
    out, bad, _, _ = _with_overclaim(_scripted(answers))
    assert out["verification"]["repair"]["rewritten"] == 1 and "Lactate is associated with it" in out["synthesis"]
    out, bad, _, _ = _with_overclaim(_scripted(RuntimeError("model down")))
    assert out["verification"]["repair"]["error"] == "RuntimeError" and bad in out["synthesis"]
    assert not out["verification"]["passed"]


def test_no_repair_call_without_a_flag_or_when_switched_off():
    rt, sc = offline_runtime()
    final, _ = execute(sc["question"], rt, echo=False)
    assert final["verification"]["passed"] and "repair" not in final["verification"] and rt.llm.calls["repair"] == 0
    out, bad, rt, _ = _with_overclaim(repair_pass=False)
    assert rt.llm.calls["repair"] == 0 and bad in out["synthesis"] and not out["verification"]["passed"]


def test_tags_after_the_period_are_found_in_the_text():
    from bmira.graph import _replace_sentence
    text = "- First finding is shown. [C12_0][C13_1] Then more text.\n"
    old = "First finding is shown [C12_0][C13_1]."               # as prose_sentences returns it
    assert _replace_sentence(text, old, "First finding is associated [C12_0][C13_1].") == \
        "- First finding is associated [C12_0][C13_1]. Then more text.\n"
    assert _replace_sentence(text, "Not in the text.", "x") is None
