# B-MiRA: notes for Claude Code

A LangGraph agent that searches PubMed, extracts and grades claims, and weighs competing mechanism
pathways. Read README.md for the design; this file covers working on the code.

## Setup and checks

```bash
pip install -e ".[dev,app]"     # cloud sessions do this in .claude/hooks/session-start.sh
pytest                          # offline: no keys, no network (about 5 s)
ruff check .                    # errors only (pyflakes, likely bugs), not style
python -m bmira.experiments --offline --quiet   # end-to-end wiring check
```

CI (`.github/workflows/tests.yml`) runs `ruff check .` and `pytest` on Python 3.10 and 3.12.

## Where things are

- `bmira/graph.py` is the pipeline (parse, plan, search, screen, extract, normalize, semantic,
  portfolio, review, synthesize, verify). Prompts are in `bmira/llm.py`; tunable values in
  `bmira/config.py`.
- `bmira/offline.py` runs the real graph with a scripted model (`SurrogateLLM`), the synthetic
  corpus `bmira/fixtures/lactate_cd8.json` and a hashing encoder. A new LLM task needs a
  `_<task>` method there, or offline runs fail.
- v3 (handoff Phases 0-2): blocking tests and the mediation index (`portfolio.build_mediation`, route
  tiers in `evaluate`); the decision model ("judge") in `judge.py`, `questions.py`, `shadow.py`, shadow
  mode only, never acting until calibrated on gold labels; gold-set scoring in `eval.py`; measurement
  tools in `tools/`. Fix ids: JV, EM, GR, TE, LC, RP, EV, HY.
- `tests/` has one file per stage (`test_normalize.py` for `bmira/normalize.py`, and so on).
  `test_pipeline.py` runs whole graphs; `test_tools.py` covers `probe`, `ab_extract` and metadata.
  Shared stand-ins are in `tests/helpers.py` (`make_claim`, `parsed_question`, `StubLLM`,
  `stub_ols`, `run_session`, `fail_on_call`); the `offline` and `fake_chat_openai` fixtures are in
  `tests/conftest.py`.

## Conventions

- Tests never touch the network. Replace HTTP with `monkeypatch` (see `test_sources.py`,
  `stub_ols`) and the model with `StubLLM` or the offline runtime.
- A fix found in a live run gets a test named after the behaviour it guarantees, with the pilot
  and the observed numbers in its docstring. Commit messages carry the fix code (`N9:`, `P3:`).
- Fixture papers are invented; never cite them as literature.
- Do not add regex cue lists or lexicon entries to fix a semantic misreading: add the case to the gold
  set and route it to the judge. Do not loosen an evidence rule without gold-set evidence.
- `# ponytail:` marks a known, deliberate limitation and what would lift it.
- A version bump touches `bmira/__init__.py`, `CITATION.cff` (version and date) and a README
  versioning entry; `tests/test_tools.py::test_version_is_the_same_everywhere` checks them.
- Live runs (`python -m bmira.experiments` without `--offline`) spend money and need
  `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` plus `NCBI_EMAIL`. Do not start one unless asked.
  Session files go to `runs/` (git-ignored).
