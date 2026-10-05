"""Scoring against gold labels (bmira.eval) and the label forms (tools/sample_gold). Offline."""
import json

import pytest

from bmira import eval as ev
from bmira.judge import SurrogateJudge
from bmira.offline import offline_runtime
from bmira.shadow import _sentence_id, agreement
from bmira.telemetry import execute, save_state
from tools import sample_gold as sg


def test_auroc_ece_and_precision_recall():
    assert ev.auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert ev.auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert ev.auroc([0.5, 0.5], [True, False]) == 0.5 and ev.auroc([0.3], [True]) is None
    assert ev.ece([0.8] * 5, [True, True, True, True, False]) == pytest.approx(0.0)
    assert ev.ece([1.0, 1.0], [False, False]) == 1.0 and ev.ece([], []) is None
    m = ev.prf([True, True, False, False], [True, False, True, False])
    assert (m["precision"], m["recall"], m["tp"], m["fp"], m["fn"]) == (0.5, 0.5, 1, 1, 1)
    assert ev.prf([False], [False])["precision"] is None


def test_thresholds_reach_a_stated_target():
    probs, ys = [0.95, 0.9, 0.7, 0.6, 0.3], [True, True, False, True, False]
    assert ev.threshold_for(probs, ys, precision=0.95) == 0.9      # lowest cut with no false positive
    assert ev.threshold_for(probs, ys, recall=0.95) == 0.6         # highest cut that keeps every positive
    assert ev.threshold_for([0.5, 0.5], [False, False], precision=0.9) is None
    with pytest.raises(ValueError):
        ev.threshold_for(probs, ys, precision=0.9, recall=0.9)


def test_both_executors_are_scored_and_unlabeled_items_skipped():
    log = [{"task": "subject_lost", "item_id": f"c{i}", "current": cur, "jev": j, "probability": p}
           for i, (cur, j, p) in enumerate([(True, True, 0.9), (True, False, 0.2), (False, False, 0.1),
                                            (False, True, 0.7), (True, True, 0.95)])]
    gold = [{"item_id": "c0", "labels": {"subject_lost": True}}, {"item_id": "c1", "labels": {"subject_lost": False}},
            {"item_id": "c2", "labels": {"subject_lost": False}}, {"item_id": "c3", "labels": {"subject_lost": True}},
            {"item_id": "c4", "labels": {"subject_lost": None}}]                      # not labeled
    (r,) = ev.evaluate(log, gold)
    assert r["labeled"] == 4 and r["current"]["accuracy"] == 0.5 and r["jev"]["accuracy"] == 1.0
    assert r["jev"]["auroc"] == 1.0 and r["current"]["precision"] == 0.5
    choice = [{"task": "stance", "item_id": "a", "current": "supports", "jev": "contradicts", "probability": 0.8}]
    (r,) = ev.evaluate(choice, [{"item_id": "a", "labels": {"stance": "contradicts"}}])
    assert r["current"]["confusion"] == {"contradicts -> supports": 1} and r["jev"]["accuracy"] == 1.0
    score = [{"task": "report.strength", "item_id": "s", "current": 4, "jev": 3, "probability": 0.6}]
    (r,) = ev.evaluate(score, [{"item_id": "s", "labels": {"report.strength": 3}}])
    assert r["current"]["mean_abs_error"] == 1 and r["jev"]["mean_abs_error"] == 0


def test_a_malformed_gold_line_names_its_line(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"item_id": "a", "labels": {}}\n\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match="g.jsonl:3"):
        ev.read_jsonl(p)


@pytest.fixture(scope="module")
def judged_state(tmp_path_factory):
    """One offline run with a judge that disagrees on the comparator, saved like a live run."""
    tmp = tmp_path_factory.mktemp("judged")
    rt, sc = offline_runtime(judge_provider="jev")
    rt.judge = SurrogateJudge(disagree={"comparator"})
    final, _ = execute(sc["question"], rt, echo=False)
    path = tmp / "run_q1.state.json"
    save_state(final, rt, path)
    log = tmp / "log.jsonl"
    log.write_text("".join(json.dumps(e) + "\n" for e in final["judge_log"]), encoding="utf-8")
    return path, log, final


def test_forms_from_a_saved_run_are_scored_after_labeling(judged_state, tmp_path):
    path, log, final = judged_state
    rows = sg.claims_form([path], n=50)
    assert len(rows) == len(final["claims"]) and all(set(sg.CLAIM_LABELS) == set(r["labels"]) for r in rows)
    for r in rows:                          # the owner agrees with today's comparator field everywhere
        r["labels"]["methods.comparator"] = r["today"]["comparator_present"]
    (res,) = ev.evaluate(ev.read_jsonl(log), rows, ["methods.comparator"])
    assert res["labeled"] == len(rows) and res["current"]["accuracy"] == 1.0 and res["jev"]["accuracy"] == 0.0
    report = sg.report_form([path])
    logged = {e["item_id"] for e in final["judge_log"] if e["task"] == "report.strength"}
    assert {r["item_id"] for r in report} == logged                    # same sentence ids as the judge log
    assert sg.papers_form([path], n=5) and all(r["labels"] == dict.fromkeys(sg.PAPER_LABELS)
                                               for r in sg.papers_form([path], n=5))


def test_stratified_sampling_keeps_rare_kinds_and_is_seeded(judged_state):
    path, _, final = judged_state
    rows = sg.claims_form([path], n=3)
    assert len(rows) == 3 and rows == sg.claims_form([path], n=3)     # same seed, same form
    rare = min({s for c in final["claims"] for s in sg.strata(c.model_dump())},
               key=lambda s: sum(s in sg.strata(c.model_dump()) for c in final["claims"]))
    assert any(rare in r["strata"] for r in rows)


def test_disagreement_form_takes_every_disagreement(judged_state):
    _, log, final = judged_state
    rows = sg.disagreements_form([log], agree_share=0.0)
    expected = sum(r["compared"] - r["agree"] for r in agreement(final["judge_log"]).values())
    assert len(rows) == expected > 0 and all(r["disagree"] for r in rows)
    assert len(sg.disagreements_form([log], agree_share=1.0)) == sum(
        r["compared"] for r in agreement(final["judge_log"]).values())


def test_mechanism_recall_scores_confirmed_entries_only(judged_state):
    _, _, final = judged_state
    state = json.loads(json.dumps({"question": final["question"], "hypotheses": [h.model_dump() for h in final["hypotheses"]],
                                   "links": {k: v.model_dump() for k, v in final["links"].items()}}))
    gold = {"questions": [{"question": final["question"], "status": "confirmed", "routes": [
        {"name": "redox", "via": [["NAD+", "nicotinamide adenine dinucleotide"], "glycolytic flux"]},
        {"name": "receptor", "via": ["GPR81"]},
        {"name": "unknown", "via": ["MCT1"]}]}]}
    r = ev.score_mechanisms(gold, state)
    assert r["status"] == "scored" and r["route_recall"] == round(2 / 3, 3)
    assert [x["matched"] for x in r["routes"]] == [True, True, False]
    gold["questions"][0]["status"] = "draft"
    assert "not scored" in ev.score_mechanisms(gold, state)["status"]
    assert ev.score_mechanisms({"questions": []}, state)["status"] == "no gold entry"


def test_cli_scores_a_state_with_its_own_judge_log(judged_state, tmp_path, capsys):
    path, _, _ = judged_state
    gold = tmp_path / "g.jsonl"
    gold.write_text(json.dumps({"item_id": _sentence_id("x"), "labels": {"report.overclaim": False}}) + "\n",
                    encoding="utf-8")
    ev.main(["--labels", str(gold), "--state", str(path)])            # nothing matches: still a table
    assert "| Task |" in capsys.readouterr().out
