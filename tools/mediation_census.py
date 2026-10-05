"""EM-0: how many claims are blocking tests in disguise, before the evidence model is built.

    python tools/mediation_census.py runs/*_q*.state.json
    python tools/mediation_census.py runs/pilot5_q1.state.json --list

A blocking test removes or blocks an entity M and reports whether the exposure's effect on an
endpoint persisted. Today such a finding is stored as a two-entity edge ("Slc5a8 required_for
IDO1") and the exposure is dropped. Counted here: claims typed required_for or flagged
subject_lost, with a knockout, knockdown or pharmacological perturbation, whose quote or the
sentence before it names the exposure (any surface that resolved to the exposure node, the
question's exposure, or a listed class member). Read-only; no network.
"""
import argparse
import json
import sys
from pathlib import Path

from bmira.normalize import _mentioned, _previous_sentence

PERTURBATIONS = {"knockout", "knockdown", "pharmacological"}


def exposure_surfaces(st: dict) -> list[str]:
    exp = st.get("exposure", "")
    parsed = st.get("parsed") or {}
    names = {parsed.get("exposure", "")} | set(parsed.get("exposure_members", []))
    for c in st.get("claims", []):
        if c.get("subject_concept") == exp:
            names.add(c.get("subject", ""))
        if c.get("object_concept") == exp:
            names.add(c.get("object", ""))
    return sorted(n for n in names if n)


def census(path: Path) -> dict:
    st = json.loads(path.read_text(encoding="utf-8"))["state"]
    texts = {p["pmid"]: p.get("source_text") or f"{p.get('title', '')}. {p.get('abstract', '')}"
             for p in st.get("papers", [])}
    surfaces = exposure_surfaces(st)
    found, candidates = [], 0
    for c in st.get("claims", []):
        if not (c.get("relation_norm") == "required_for" or c.get("subject_lost")):
            continue
        if c.get("perturbation_class") not in PERTURBATIONS:
            continue
        candidates += 1
        context = _previous_sentence(c["span"], texts.get(c["pmid"], "")) + " " + c["span"]
        named = [s for s in surfaces if _mentioned(s, context)]
        if named:
            found.append({"id": c["id"], "pmid": c["pmid"], "mediator": c.get("subject_label") or c["subject"],
                          "endpoint": c.get("object_label") or c["object"], "relation": c.get("relation_norm"),
                          "subject_lost": bool(c.get("subject_lost")), "perturbation": c["perturbation_class"],
                          "exposure_named": named, "grade": c.get("grade"), "span": c["span"][:240]})
    explicit = [c for c in st.get("claims", []) if c.get("effect_exposure") and c.get("effect_result")]
    return {"state": str(path), "claims": len(st.get("claims", [])),
            "explicit_blocking_tests": len(explicit),           # recorded as such by EM-1 extraction
            "required_for": sum(c.get("relation_norm") == "required_for" for c in st.get("claims", [])),
            "subject_lost": sum(bool(c.get("subject_lost")) for c in st.get("claims", [])),
            "perturbed_candidates": candidates, "blocking_tests": len(found),
            "papers": len({x["pmid"] for x in found}), "exposure_surfaces": surfaces, "list": found}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", type=Path, nargs="+")
    ap.add_argument("--list", action="store_true", help="print every blocking-test claim")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rows = [census(p) for p in a.states]
    if a.json:
        print(json.dumps(rows, indent=1, ensure_ascii=False))
        return rows
    print("| State | Claims | required_for | subject_lost | Perturbed candidates | Blocking tests | Papers | Explicit (EM-1) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {Path(r['state']).name} | {r['claims']} | {r['required_for']} | {r['subject_lost']} | "
              f"{r['perturbed_candidates']} | {r['blocking_tests']} | {r['papers']} | {r['explicit_blocking_tests']} |")
    if a.list:
        for r in rows:
            for x in r["list"]:
                print(f"- {x['id']} ({x['perturbation']}, {x['grade']}): {x['mediator']} -> {x['endpoint']} "
                      f"[exposure: {', '.join(x['exposure_named'])}] {x['span']}")
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
