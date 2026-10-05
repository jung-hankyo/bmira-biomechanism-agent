"""Phase-0 measurement tools over saved runs: token anatomy (HY-3) and the mediation census (EM-0).
Offline: no keys, no network."""
import json

from helpers import run_session
from tools import mediation_census, token_anatomy


def test_token_anatomy_reads_a_session_and_its_state(tmp_path):
    run_session(tmp_path, "only question\n")
    (row,) = token_anatomy.analyse(tmp_path / "s.json")
    assert row["status"] == "completed" and row["tokens"] > 0
    assert abs(sum(v["token_share"] for v in row["tasks"].values()) - 1) < 0.01
    assert row["extraction"]["papers_read"] == row["state"]["papers_read"] > 0
    assert row["state"]["reads_recorded"] == row["extraction"]["reads"]
    assert row["state"]["by_access"]["full_text"]["papers"] >= 1          # the fixture has one open-access paper
    assert "### Q1 (completed)" in token_anatomy.markdown("s.json", [row])


def test_token_anatomy_survives_runs_that_failed_early(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"runs": [{"question_number": 1, "status": "failed", "run": {}}]}), encoding="utf-8")
    (row,) = token_anatomy.analyse(path)                                   # no llm, papers or state sections
    assert row["tokens"] == 0 and row["tasks"] == {} and "state" not in row
    token_anatomy.markdown("old.json", [row])


def _state_with(tmp_path, claims, abstract):
    run_session(tmp_path, "only question\n")
    path = tmp_path / "s_q1.state.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["state"]["papers"].append({"pmid": "X1", "title": "t", "abstract": abstract, "source_text": ""})
    base = {"pmid": "X1", "relation_norm": "required_for", "subject_lost": False, "perturbation_class": "knockout",
            "subject_concept": "LOCAL:m", "object_concept": "LOCAL:y", "grade": "moderate"}
    data["state"]["claims"] += [{**base, **c} for c in claims]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_census_counts_blocking_tests_that_name_the_exposure(tmp_path):
    abstract = ("Separately, Hdac3 was required for IFNG expression in knockout cells. "    # first: nothing before it
                "Gpr81-deficient mice were studied. Lactate failed to reduce IFNG in Gpr81-/- CD8 T cells. "
                "Cells were exposed to lactic acid. Inhibition of MCT1 blocked the loss of cytotoxicity.")
    path = _state_with(tmp_path, [
        {"id": "B1", "subject": "Gpr81", "object": "IFNG", "span": "Lactate failed to reduce IFNG in Gpr81-/- CD8 T cells."},
        {"id": "B2", "subject": "Hdac3", "object": "IFNG",       # no exposure named: not a blocking test
         "span": "Separately, Hdac3 was required for IFNG expression in knockout cells."},
        {"id": "B3", "subject": "MCT1", "object": "cytotoxicity", "relation_norm": "decreases", "subject_lost": True,
         "perturbation_class": "pharmacological",      # exposure named in the previous sentence only
         "span": "Inhibition of MCT1 blocked the loss of cytotoxicity."},
        {"id": "B4", "subject": "Gpr81", "object": "IFNG", "perturbation_class": "none",     # nothing perturbed
         "span": "Lactate failed to reduce IFNG in Gpr81-/- CD8 T cells."}], abstract)
    r = mediation_census.census(path)
    # B1, B3 and the fixture's own blocking test (S020, recorded as such by EM-1)
    assert {x["id"] for x in r["list"]} == {"B1", "B3", "CS020_0"} and r["blocking_tests"] == 3 and r["papers"] == 2
    assert r["perturbed_candidates"] == 4 and r["explicit_blocking_tests"] == 1
    assert "Lactate" in r["exposure_surfaces"] or "lactate" in r["exposure_surfaces"]


def test_census_cli_prints_a_table(tmp_path, capsys):
    run_session(tmp_path, "only question\n")
    rows = mediation_census.main([str(tmp_path / "s_q1.state.json"), "--list"])
    assert rows[0]["blocking_tests"] == rows[0]["explicit_blocking_tests"] == 1     # S020
    assert "| s_q1.state.json |" in capsys.readouterr().out
