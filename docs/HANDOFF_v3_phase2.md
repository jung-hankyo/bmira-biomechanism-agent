# B-MiRA handoff: v3 Phases 0-2 done, ready for the first live run

| Item | Value |
|---|---|
| Written | 2026-10-05, by a Claude Code cloud session, for the project owner and the next Claude Code session |
| Branch | `claude/fervent-meitner-9ba4g6`, based on `pilot7-fixes` @ `a8f2a90` (v2.6.0); 23 commits ahead, not merged |
| Design source | `BMIRA_V3_HANDOFF.md` (the v3 plan: sections 4-11, fix ids JV, EM, GR, TE, LC, RP, EV, HY). Not in the repo; ask the owner for it if a task needs the full spec |
| Checks | 187 offline tests pass (`pytest`), `ruff check .` clean, CI workflow in `.github/workflows/tests.yml` |
| Not run | **No live run.** Everything below is tested offline on invented papers only |

**Read order for a new session:** this file, then `CLAUDE.md` (how to work on the code), then `README.md` (what the agent does). The commit messages on the branch explain each fix in detail: `git log a8f2a90..HEAD`.

---

## 1. The agentic workflow in one picture

```
question
  -> parse (LLM)            exposure, outcome, readouts, expected direction, class members, scope
  -> plan (LLM + code)      PubMed queries; later rounds: queries per target step
  -> search (code)          PubMed / Europe PMC
  -> screen (cheap LLM)     include? study type?               [judge asks too, shadow only]
  -> extract (LLM, per paper, in parallel)                     reviews are no longer extracted (TE-1)
       -> code checks: quote exists, names both entities, polarity, method cues, blocking-test check
                                                               [judge asks stance, methods, loss, blocking]
  -> normalize (code + cheap LLM)  entities, relations, aliases, grades   [judge asks relation]
  -> semantic (cheap LLM)   same finding? conflicts?
  -> portfolio (code; LLM seeds once, may expand)
       steps (links) -> mediation index (blocking tests) -> route verdicts -> targets -> stop?
       └── loop back to plan while searching is worth it
  -> synthesize (LLM)       told the allowed wording for every claim, step and route (RP-1)
  -> verify (code + cheap LLM)  overclaims, entailment, one repair pass (RP-2)   [judge asks strength]
  -> report + session file + saved state
```

Rule of thumb: **code decides, models read and write.** Counting, grades, verdicts, scores, targets, budgets and stop rules are deterministic code. The judge (a decision model) only watches and logs.

---

## 2. What changed, by area

### 2.1 Housekeeping and harness (no behaviour change)
- Tests split into one file per pipeline stage (`tests/test_<stage>.py`), with shared helpers in `tests/helpers.py` and `tests/conftest.py`.
- `pyproject.toml` (install with `pip install -e ".[dev,app]"`), ruff (errors only), GitHub Actions CI, a cloud-session hook (`.claude/hooks/session-start.sh`), and `CLAUDE.md`.
- New offline tests for `sources.py`, `probe.py` and `ab_extract.py`, plus a check that the version matches across `bmira/__init__.py`, `CITATION.cff` and the README.

### 2.2 Phase 0: measurement (no behaviour change)
| Id | What | Where |
|---|---|---|
| HY-2 | A live session with a missing API key now writes `aborted: runtime could not be built ...` to the session file instead of crashing | `bmira/experiments.py` |
| HY-3 | Token anatomy: cost and token share per task, reads and re-reads, characters per read. Papers now record `n_reads` and `chars_read` | `tools/token_anatomy.py` |
| EM-0 | Mediation census: how many saved claims are blocking tests in disguise | `tools/mediation_census.py` |
| EV-7 | A test caps package lines naming butyrate or Treg at 37 (now 36). Do not add Q1-specific literals | `tests/test_tools.py` |

### 2.3 Phase 1: the judge, in shadow mode (no behaviour change)
- **What it is:** TypeSafe's decision model "Jev", pinned to `jev-1.13.0`. It answers typed yes/no, choice and score questions with probabilities.
- **What it does now:** at each touchpoint it answers the same question today's code or LLM already decided, and the pair is logged in `state["judge_log"]`. **It never changes a decision.** `judge_mode="act"` is refused on purpose until a touchpoint is calibrated on gold labels.
- **Files:**
  - `bmira/judge.py`: client, cache per model version, retries, circuit breaker, offline `SurrogateJudge`.
  - `bmira/questions.py`: every question and threshold.
  - `bmira/shadow.py`: the touchpoints. `python -m bmira.shadow <state>` also runs them over a saved run.
- **Touchpoints:**
  - JV-1/2: screening.
  - JV-4: quote stance.
  - JV-5: method fields.
  - JV-6: subject_lost.
  - JV-8: relation typing.
  - JV-14: sentence strength, overclaim and entailment.
  - JV-15: blocking tests.
- **Off by default.** `--judge jev` needs `TYPESAFE_API_KEY`.
- **Gold-set harness (EV-1..3):**
  - `bmira/eval.py` scores today's executor and the judge against owner labels: accuracy, precision, recall, AUROC, calibration error, and the thresholds that reach a target precision or recall.
  - `tools/sample_gold.py` writes the label forms.
  - `eval/README.md` explains the procedure.

### 2.4 Phase 2: the evidence model (BEHAVIOUR CHANGES)
**Why:** in pilots 3-7 no mechanism route ever reached Supported. A route needed two papers per step, and a within-paper blocking experiment, the strongest mechanistic evidence, had no place.

| Id | Change |
|---|---|
| EM-1/2 | A **blocking test** ("lactate no longer reduced IFNG in Gpr81-/- cells") keeps its treatment: new claim fields `effect_exposure` and `effect_result` (abolished, attenuated, unchanged, enhanced); the extract prompt was rewritten for this. Code drops a blocking test whose treatment is not named in the quote or the sentence before, or whose perturbation is not a knockout, knockdown or inhibitor |
| EM-3 | **Mediation index** `portfolio.build_mediation`: blocking tests per (exposure, intermediate, outcome). Demonstrated = one moderate-or-better paper with no counted opposition (R1, R5, R9 analogues apply) |
| EM-4 | **New pathway verdicts**, in precedence order: Contradicted -> **Shown by a blocking experiment** (every intermediate covered by a blocking test or by both its steps supported) -> **Assembled from separate studies** (every step Supported, contexts compatible) -> **Mediator not required** -> Insufficient evidence. Direct routes and one-link routes keep step verdicts ("Supported"). Steps (links) keep Supported / Contradicted / Insufficient evidence |
| EM-4 | **Stop rule:** CONVERGED when the leading mechanism route is at least assembled and no open rival could beat it. Runs can now stop earlier than before |
| EM-5 | Adjacent steps in incompatible cell types block "assembled" (ancestor cell types are compatible) |
| EM-7 | Report: a **"Direct effect:"** headline, a **Blocking test** column in the pathway table, and blocking-test lines in the synthesis and chat context |
| TE-1 | **Reviews are not extracted** (pilot4: 8 of 36 extraction calls bought 28 claims that never counted). Their abstracts go into the pathway-seeding prompt |
| TE-10 | `--budget-usd`: a soft cost cap from `Settings.prices`, covering LLM and judge. Unpriced models are named in the warnings |
| RP-1 | The writer is told the allowed wording for every claim, step and route before writing |
| RP-2 | One **repair pass**: flagged report sentences are rewritten once by the cheap model, accepted only if every citation tag is kept, and verified again (`repair_pass`, default on) |

Old saved states (pilots 3-7) still load and replay.

### 2.5 Deliberately not done
| Item | Why |
|---|---|
| EM-6: searching for blocking tests on purpose | Needs `bmira.probe` on live PubMed first to check hit counts. Marked with a `# ponytail:` note in `portfolio.reachable_tier`. Until it exists, a route reaches "Shown" only through blocking tests found incidentally |
| TE-3/4/6: extraction prompt and model changes | Need an `ab_extract` A/B on a live model (claim sets overlap only about 0.6 between repeats) |
| Judge act mode (Phase 3), PubTator grounding (Phase 4), reading control (Phase 5) | Need gold labels, a TypeSafe key and live runs |
| `eval/mechanisms_gold.json` | Template only; the expected routes and PMIDs are the owner's to confirm |

---

## 3. What the owner has to do next

In order:

1. **Review the branch:** `git log a8f2a90..HEAD` and the README verdict table. Decide whether to merge `claude/fervent-meitner-9ba4g6` into `pilot7-fixes`, or open a PR to `main`.
2. **Decisions from the v3 plan:**
   - D3: keep or rename the route labels ("Shown by a blocking experiment", "Assembled from separate studies", "Mediator not required").
   - D6 is implemented as recommended: "Assembled" never displays as "Supported".
   - D1: whether to get a TypeSafe key for the judge.
   - D5: target cost per run.
3. **Hygiene on your machine:**
   - Scrub or rotate the NCBI key stored in `runs/pilot3.json` (HY-2; `runs/` is not in git).
   - Check that GitHub Actions is enabled for the repo; no CI run appeared after the push.
4. **Run the live baseline** (section 4).
5. **Start the gold labels:**
   - Make forms with `tools/sample_gold.py` from the pilot states and label them (about 3-4 hours in total).
   - Confirm `eval/mechanisms_gold.json` for Q1 at least.
   - Phase 3 (judge acting) cannot start without these labels.
6. **Hand the results to a new session:** the session files, the saved states, and the token-anatomy and census outputs from section 4.

---

## 4. Live baseline test (your terminal)

These commands spend money. Run them on your own computer, where PubMed and the model APIs are reachable. The cloud sessions used so far could not reach NCBI or EBI.

### 4.1 Setup (once)
```bash
git clone https://github.com/jung-hankyo/bmira-biomechanism-agent.git
cd bmira-biomechanism-agent
git checkout claude/fervent-meitner-9ba4g6
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,app,openai,embeddings]"             # add ",anthropic" to run on Claude
pytest -q                                                 # expect 187 passed (1 skipped if streamlit missing)

export OPENAI_API_KEY=...            # or ANTHROPIC_API_KEY=... with --provider anthropic
export NCBI_EMAIL=you@example.org
export NCBI_API_KEY=...              # optional, faster PubMed
# export TYPESAFE_API_KEY=...        # only if you add --judge jev
```

### 4.2 Free checks first (no or near-zero cost)
```bash
python -m bmira.experiments --offline --quiet --out runs/offline_check.json   # wiring, no keys used
python -m bmira.experiments --replay runs/pilot7_q1.state.json --out runs/replay_p7_v3.json   # old run, new evidence model
python -m tools.mediation_census runs/pilot5_q1.state.json runs/pilot6_q1.state.json runs/pilot7_q1.state.json --list
python -m tools.token_anatomy runs/pilot7.json                   # where pilot7's money went (use your session file names)
python -m bmira.probe "Does butyrate produced by gut microbiota induce colonic regulatory T cells, and through which mechanisms?"
```
- **The replay** costs only the synthesis and verification calls. It shows how the new verdicts read on real claims. Old claims have no blocking-test fields, so expect "Assembled" rather than "Shown".
- **The census** tells you how many blocking tests pilots 5-7 already contained in disguise.
- Use the real file names in your `runs/` folder.

### 4.3 The baseline run (pilot8, Q1)
```bash
python -m bmira.experiments --only 1 --max-rounds 3 --budget-usd 3 --out runs/pilot8.json
```
- About $1-2 at past pilot costs. `--budget-usd` stops searching between rounds once the estimate passes $3 and still writes the report.
- Outputs:
  - `runs/pilot8.json`: the session summary.
  - `runs/pilot8_q1.state.json`: the saved state, for replays.
- Optional extras:
  - `--provider anthropic`: run on Claude.
  - `--judge jev`: shadow judge on (adds the `judge` section; costs cents).
  - Add `2 6` to `--only`: also run the clinical questions (Q2, Q6) that have never run live.

**Attributing changes.** Pilot8 tests the pilot7 fixes and v3 together. To separate them, run the same command once on `pilot7-fixes` (v2.6) with a different `--out`, then compare the two session files. Live runs read different papers, so compare verdicts on replays or on the same state where possible.

### 4.4 What to look at afterwards
```bash
python -m tools.token_anatomy runs/pilot8.json
python -m tools.mediation_census runs/pilot8_q1.state.json --list
python - <<'EOF'
import json; r = json.load(open("runs/pilot8.json"))["runs"][0]
print(r["status"], r["run"]["rounds"], r["run"]["stop_reason"], r["llm"]["est_cost_usd"])
print(r["pathways"]["verdicts"], r["pathways"]["mediation"])
print(r["extraction"]["blocking_tests"], r["extraction"]["drop_reasons"])
print(r["papers"]["reviews_not_extracted"], r["verification"])
print([s["signal"] for s in r["signals"]])
EOF
```
Questions the run should answer:
1. **Blocking tests:** did extraction record any (`extraction.blocking_tests`)? How many were dropped, and why ("blocking test without a removal or blocking perturbation", "blocking-test treatment not named in quote")?
2. **Mechanism routes:** did any reach "Shown" or "Assembled"? Read `pathways.mediation.shown`.
3. **Reviews:** how many reviews were skipped, and did the cost per paper read fall against pilots 5-7?
4. **Report:** how many overclaims before and after the repair pass (`verification.repair`), and are any left?
5. **Stopping:** did the run stop early with CONVERGED, and was that justified?

---

## 5. Notes for the next Claude Code session

- **Start by reading** `CLAUDE.md`, then this file, then the report of whichever run you are given.
- **Rules that matter:**
  - English only in code.
  - Deterministic decisions stay in code.
  - Do not add regex cue lists to fix a misreading: route the case to the judge and the gold set.
  - Do not loosen an evidence rule without gold-set evidence.
  - One fix per commit, each with a test built from the real case.
  - Do not start a live run unless asked.
- **Likely next tasks once pilot8 exists:**
  - Diagnose pilot8 with the questions in 4.4.
  - EM-6 (blocking-test targets), checking query hit counts with `bmira.probe` first.
  - Judge calibration once gold labels exist (Phase 3, one touchpoint at a time).
- **Known soft spots:**
  - The knockout cue list misses notations such as "Slc5a8-null". Such a blocking test is kept but graded lower, and JV-5/JV-15 are meant to replace the cues.
  - The judge's response format is taken from TypeSafe's documentation and not yet confirmed against a live reply. Record a real reply as a test fixture on first use.
