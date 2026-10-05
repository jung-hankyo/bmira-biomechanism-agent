# B-MiRA — Biomedical Mechanism Inference Research Agent

**v2.6 + v3 development** · LangGraph · Python 3.10+

Ask *"Does X affect Y, and through which mechanisms?"*. B-MiRA searches PubMed, extracts and grades claims, weighs **competing pathways** against each other, and writes a report in which every sentence cites a claim and is checked against it.

> Research tool for exploring literature. Not medical advice, and not a substitute for reading the papers.

## How it works

<p align="center"><img src="docs/figure1.svg" alt="B-MiRA architecture: retrieve papers; build the evidence graph; weigh competing pathways, looping back to targeted search until the stop rule fires" width="100%"></p>

**Figure 1.** **a**, The question is parsed (exposure, outcome, readouts, direction, class members) and searched; papers are screened. **b**, Claims are extracted, checked against their quotes, normalized to ontology concepts, graded, and joined into one evidence graph. **c**, Pathways from LLM proposals, the literature graph and gated expansion compete; the next searches go to the steps that would most change the ranking. Hexagons are LLM calls, rectangles code.

- **Evidence graph.** A claim is an edge between two entities; how and where it was measured are qualifiers, so "Treg differentiation" and "colonic regulatory T cells" meet at *regulatory T cell*. Evidence on a step is shared by every pathway that uses it.
- **Blocking tests (v3).** An experiment that removes or blocks an intermediate M (knockout, knockdown, inhibitor) and reports whether the exposure's effect on the outcome persisted keeps the exposure. These tests form a mediation index, separate from the step edges.
- **Decisions in code.** Counting, grades, verdicts, scores, search allocation, budgets and stop rules are deterministic. LLMs parse, write queries, extract, propose pathways and write prose.

### Verdicts

| Level | Verdict | Meaning |
|---|---|---|
| Step | **Supported** | ≥ 2 independent primary papers, one at moderate grade or better, opposing papers under half |
| Step | **Contradicted** | Counted opposing papers at least half, at moderate or better |
| Step | **Insufficient evidence** | Anything else, with the reason (*"1 of 2 required papers"*, *"no study found in 2 targeted searches"*) |
| Pathway | **Shown by a blocking experiment** | Blocking an intermediate removed the effect (moderate or better), and every other intermediate is blocked or has both its steps supported |
| Pathway | **Assembled from separate studies** | Every step Supported, cell-type contexts compatible; no blocking test |
| Pathway | **Mediator not required** | Removing the intermediate left the effect, and nothing better holds |
| Pathway | **Contradicted / Insufficient evidence** | A step is contradicted / anything else |

A direct exposure → outcome route keeps the step verdicts and heads the report. Scores **rank** pathways; they are not probabilities.

### Evidence rules (code, not the model)

- A quote must exist in the paper, name both entities and match the claim's direction; method details count only if the text shows them.
- Each claim is graded by its own system; evidence further from the question's population is capped at moderate.
- Reviews never count and are not extracted; their abstracts only inform pathway proposals.
- A null result counts against a step only with a control and at least the support's grade; the same rule applies to blocking tests.
- A finding on a subtype supports its parent, never refutes it.
- Report wording is limited to what the weakest cited claim allows. The writer is told this limit before writing, flagged sentences get one repair pass, and the verifier checks the result.

## Quick start

```bash
git clone https://github.com/jung-hankyo/bmira-biomechanism-agent.git && cd bmira-biomechanism-agent
pip install -r requirements.txt             # or, for development: pip install -e ".[dev,app]"
python -m pytest -q                          # offline: no keys, no network
streamlit run app.py                         # "Offline demo" in the sidebar, or Live with keys
export OPENAI_API_KEY=... NCBI_EMAIL=you@example.org   # or ANTHROPIC_API_KEY; NCBI_API_KEY optional
```

The chat app runs an investigation on the first message. It answers follow-ups only from that run's evidence, citing claim ids, and `/new <question>` starts over.

## Experiments and measurement

```bash
python -m bmira.experiments [--only 1 2] [--max-rounds 3] [--budget-usd 2] [--budget-tokens N]
python -m bmira.experiments --offline                      # wiring check
python -m bmira.experiments --replay runs/S_q1.state.json  # re-judge a saved run with the current code
python -m bmira.experiments --judge jev                    # decision model in shadow mode (TYPESAFE_API_KEY)
python -m bmira.probe ["question"]                         # parse + round-1 queries + hit counts (~$0.03)
python -m bmira.shadow runs/S_q1.state.json --out runs/shadow.jsonl   # every judge question over a saved run
python -m tools.token_anatomy runs/session_X.json          # where tokens and dollars went
python -m tools.mediation_census runs/*_q*.state.json      # blocking tests in saved runs
python -m tools.sample_gold claims runs/*_q1.state.json    # label forms for the gold sets (eval/README.md)
python -m bmira.eval --labels eval/claims_gold.jsonl --log runs/shadow.jsonl   # score executors on gold labels
```

- **Before spending.** A preflight checks each model, PubMed, the ontology service and the judge (the judge is not required). Fatal errors, such as an empty balance or a bad or missing key, stop the session and are written to the session file.
- **The session file.** Each session writes one file, `runs/session_<timestamp>.json` (git-ignored). It holds per-task tokens, cost and failures; retrieval, screening, extraction, normalization, grading, comparison, step, pathway and mediation tables; verification and repair; the judge's agreement with today's executor; and `signals` naming the code to inspect.
- **Saved state.** Each run also saves `<session>_q<n>.state.json` for replay.

**Decision model ("judge").** TypeSafe Jev, pinned to `jev-1.13.0`, runs in **shadow** mode only. It answers typed questions (screening, quote stance, method fields, loss of function, relation, blocking tests, sentence strength, entailment) next to today's executor and changes nothing. A touchpoint will act only after its thresholds are calibrated on owner-labeled gold sets (`eval/`). Questions and thresholds live in `bmira/questions.py`.

## Repository layout

```
app.py                  Streamlit chat          bmira/graph.py        LangGraph pipeline and report
bmira/config.py         All settings            bmira/portfolio.py    Links, mediation index, verdicts, search, stop
bmira/schemas.py        Data models             bmira/evidence.py     Grading, allowed wording, report verifier
bmira/llm.py            Prompts, LLM client     bmira/normalize.py    Entities, relations, quote checks
bmira/sources.py        PubMed, Europe PMC      bmira/semantic.py     Same finding; conflict triage
bmira/judge.py          Decision-model client   bmira/questions.py    Judge questions and thresholds
bmira/shadow.py         Judge touchpoints       bmira/eval.py         Gold-set scoring
bmira/telemetry.py      Session metrics         bmira/experiments.py  Batch runner, replay
bmira/offline.py        Scripted model, corpus  bmira/fixtures/       Synthetic scenario (invented papers)
tools/                  Token anatomy, mediation census, gold-set forms
eval/                   Gold sets (owner-labeled) and how to make them
tests/                  Offline tests, one file per stage    CLAUDE.md   Conventions for coding agents
```

## Main settings (`bmira/config.py`)

| Setting | Default | Effect |
|---|---|---|
| `max_rounds` / `targets_per_round` | 5 / 3 | Search rounds; steps searched per round |
| `min_studies_per_link` | 2 | Papers a step needs to be Supported |
| `max_extract_per_round` / `max_claims_per_paper` | 20 / 8 | Papers read per round (step hits first); claims per paper |
| `budget_usd` / `budget_tokens` | `None` | Soft caps checked between rounds; the report is still written |
| `repair_pass` | `True` | One rewrite of flagged report sentences to their allowed wording |
| `judge_provider` / `judge_mode` | `"off"` / `"shadow"` | Decision model; only shadow mode exists until calibration |
| `prices` | list prices | USD per 1M tokens; unpriced models are named in the warnings |
| `reasoning_effort`, `cheap_tasks`, `temperature` | per task | Effort per task (OpenAI), tasks on the cheap model, sampling |

## Limitations

- **Live runs cover one question.** Pilots 3-7 all asked Q1 (butyrate and colonic Tregs). No mechanism route reached Supported there; most steps rest on one paper.
- **v3 is not yet validated live.** The blocking-test model, route tiers, repair pass and judge are tested offline on invented papers only. Pilot8 will be the first live run with them, and also the first test of the pilot7 fixes.
- **Not yet built:** EM-6 (searching for blocking tests on purpose) and the gold labels (`eval/`). Without labels, no judge touchpoint can act.
- Two reads of one paper change about 40% of the extracted claims, so single runs show only large effects. Grade weights and thresholds are reasoned defaults, not calibrated values.
- Only open-access full texts are read; abstract-only evidence is capped at moderate.

## Versioning

- **v2.0.0** first public release; **v2.0.1** license, citation, figure; **v2.1.0** evidence-pipeline hardening; **v2.2.0** telemetry and batch runner; **v2.2.1** checkpoint type registration; **v2.2.2** no fixed temperature, explicit encodings; **v2.3.0** safe live runs (preflight, fatal abort, budgets, costs); **v2.4.0** and **v2.5.0** fixes from the first live runs (entity granularity, ontology, pair matching, loss restatement).
- **v2.6.0** fixes from pilots 4-7: replay of saved runs, chemical, ontology and abbreviation rules, knockouts as *required for*, one direct route per outcome, numbered targets and verdicts, verifier false alarms, scope and class members in parsing, `bmira.probe`.
- **Unreleased (v3, handoff Phases 0-2):**
  - Blocking tests and the mediation index, route tiers, context coherence, a direct-effect headline.
  - Reviews are not extracted; a USD budget; allowed wording before writing and one repair pass.
  - The judge in shadow mode; the gold-set harness; the token anatomy and mediation census tools.
  - A missing key is recorded in the session file.

## License and citing

MIT ([LICENSE](LICENSE)). Literature retrieved from PubMed, Europe PMC and EBI OLS is subject to those services' terms and is not stored here; `bmira/fixtures/` papers are invented. Cite via [CITATION.cff](CITATION.cff).
