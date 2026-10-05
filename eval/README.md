# Gold sets

Owner-labeled items that decide whether a change to a judged decision is an improvement
(handoff section 10). Files here are small JSONL, committed, with no secrets and no full paper texts.

| File | Items | Labels (task names in the judge log) |
|---|---|---|
| `claims_gold.jsonl` | about 200 claims, stratified from pilot states | `stance`, `relation`, `methods.comparator`, `methods.perturbation`, `methods.rescue`, `methods.orthogonal`, `subject_lost`, `system`, `claim_type`, `blocking_test` |
| `papers_gold.jsonl` | about 150 screened papers | `screen.include`, `screen.original_data` |
| `report_gold.jsonl` | every sentence of the pilot reports | `report.overclaim`, `report.entailment`, `report.strength` |
| `mechanisms_gold.json` | expected routes per question Q1-Q8 | route recall (only `status: confirmed` entries are scored) |

## Making a form

```bash
python -m tools.sample_gold claims runs/pilot5_q1.state.json runs/pilot6_q1.state.json runs/pilot7_q1.state.json
python -m tools.sample_gold papers runs/pilot*_q1.state.json
python -m tools.sample_gold report runs/pilot*_q1.state.json
python -m bmira.shadow runs/pilot*_q1.state.json --out runs/shadow.jsonl     # needs TYPESAFE_API_KEY
python -m tools.sample_gold disagreements runs/shadow.jsonl                   # every disagreement + 20% of agreements
```

Forms land in `eval/forms/`. Fill `labels`: `true`/`false` for yes/no tasks, an option name for choice
tasks (`supports`, `contradicts`, `says_nothing`; a relation; a perturbation class), a level 0-4 for
`report.strength`. Leave `null` where you cannot tell; nulls are skipped. Then move the file to `eval/`.

## Scoring

```bash
python -m bmira.eval --labels eval/claims_gold.jsonl --log runs/shadow.jsonl
python -m bmira.eval --mechanisms eval/mechanisms_gold.json --state runs/pilot8_q1.state.json
```

Both executors are scored: today's (`current`) and the judge's. For yes/no tasks the judge also gets
AUROC, calibration error and the thresholds reaching precision or recall 0.95. A threshold chosen here
goes into `bmira/questions.py` with the model version it was calibrated on.

## Rules

- Never put gold items into prompts or few-shot examples.
- Extraction comparisons use at least two repeats; differences inside the repeat spread are noise.
- `mechanisms_gold.json` entries drafted by Claude carry `"status": "draft"`; only the owner sets `"confirmed"`.
