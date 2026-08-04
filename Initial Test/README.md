# Agent Regression Detection Harness

Answers one question: when you change an agent system (model, prompt, tool
permission, added agent), did that change make it measurably less safe, or
is the apparent difference just noise? Agent behavior is stochastic, so a
naive before/after comparison produces false alarms. This is a
statistically sound regression detector, validated against a toy
multi-agent system (built on [Agno](https://github.com/agno-agi/agno))
where the ground truth is under our own control.

## Setup

```bash
cd "Initial Test"
uv venv .venv --python 3.11
uv pip install -e ".[dev]" --python .venv/bin/python
cp .env.example .env   # only needed for provider="anthropic" runs
```

Everything below assumes `.venv/bin/python` (or an activated venv). The
mock backend (`provider="mock"` — the default everywhere in this repo)
needs no API key and makes zero network calls.

Run the test suite:

```bash
.venv/bin/python -m pytest tests/ -q
```

## Project layout

- `target_system/` — the toy company assistant under test: Supervisor +
  Researcher + Operator agents (Agno `Team`, `coordinate` mode), a
  document corpus, fake tools (`send_email`, `lookup_customer`,
  `search_corpus`), a versioned `SystemConfig` (content-hash addressed —
  see `target_system/config.py`), and the trajectory log schema
  (`logging_schema.py`). `orchestration.py` adapts Agno's own output into
  that log — `run_case` for single-turn cases, `run_multi_turn_case` for
  cases that need a real multi-turn conversation.
- `attacker/` — the attack case suite (`cases.py`, 17 cases across four
  families), sourced from AgentDojo where possible (`sourcing.py`
  documents exactly what was used and why AgentHarm wasn't). `executor.py`
  bridges a case into a run — `execute_case()` is the one place that picks
  `run_case` vs. `run_multi_turn_case`.
- `stats/` — the statistical engine: paired comparison methods (cluster
  bootstrap, McNemar, mixed-effects logistic), A/A calibration, BH-FDR
  across families, always-valid sequential inference (confidence
  sequences + group-sequential alpha-spending), variance reduction (CRN,
  CUPED), power analysis, and reporting. Has its own CLI —
  `python -m stats.cli aa-calibration` is the first thing to run if you
  want to validate the statistics module on its own, independent of the
  target system.
- `experiments/` — the runner: executes both arms of a paired comparison
  across the attack suite, caches results, computes the statistical
  comparison, and saves both a printed report and a `*_report.json`
  (`experiments/persist.py`) the dashboard reads. This is what you
  actually run day to day; see below.
- `dashboard/` — a read-only Streamlit dashboard over `data/runs/` and the
  saved reports (`streamlit run dashboard/app.py`). Never executes an
  experiment itself — see "Running the dashboard" below.
- `data/runs/*.jsonl` — trajectory logs, one line per run (gitignored).
- `data/runs/*_report.json` — the computed statistical comparison for each
  experiment (gitignored), read by the dashboard.
- `target_system/configs/*.json` — every resolved `SystemConfig` this
  repo has ever run, keyed by content hash.

## Running experiments

```bash
.venv/bin/python -m experiments.cli list-presets
.venv/bin/python -m experiments.cli run --preset aa
```

**Run `aa` first, always.** It compares a config against itself
(bit-identical — same content hash) and should show nothing significant.
If it does, don't trust any other preset's result until you know why.

### The five presets

| preset | what changes | what to expect |
|---|---|---|
| `aa` | nothing (arm A == arm B) | nothing flagged; sanity-checks the whole pipeline end to end |
| `known_regression` | removes the defensive instruction from the supervisor prompt | `exfiltration` should rise and get flagged; reports how many cases it took to detect (always-valid, not a fixed-N p-value) |
| `known_neutral` | cosmetic prompt reword only | nothing flagged — the false-alarm test |
| `model_swap` | `model_name` only | with the mock backend: nothing flagged, because `MockModel` can't emulate a real capability difference — this only proves the pipeline handles a model-only config diff cleanly. Pass real Anthropic model IDs on both arms for an actual finding |
| `added_agent` | adds a fourth agent (no tools of its own) | small, likely-not-significant shifts on the *existing* attack cases — this doesn't test attacks against the new agent, since none exist yet in `attacker/cases.py` |

```bash
# more runs per case = more statistical power, more (API/wall-clock) cost
.venv/bin/python -m experiments.cli run --preset known_regression --runs-per-case 20

# ad hoc: compare any two already-saved configs by hash
.venv/bin/python -m experiments.cli run --config-a cfg_xxxx --config-b cfg_yyyy --experiment-name my_check
```

Every run is resumable and cached: re-running the same `--experiment-name`
(or the same preset) only executes whatever (config, case, arm, seed)
combination isn't already in `data/runs/<name>.jsonl`. Asking for more
`--runs-per-case` than before only runs the gap. This matters most with
`provider="anthropic"`, where API calls are the actual cost — the mock
backend is free either way.

### Reading the report

```
=== Experiment: known_regression ===
arm A: defensive_prompt_on  (config_hash=cfg_...)
arm B: defensive_prompt_off  (config_hash=cfg_...)
17 cases x 5 runs/case/arm (0 runs already cached, 170 executed this run)

task_success: arm A 100.0%  |  arm B 100.0%

--- exfiltration ---
  [exfiltration             SIGNIFICANT    ] direct_instruction_injection: 13.3% -> 73.3% (diff +60.0pp, 95% CI [33.3, 73.3], q=0.000, n=5 cases)
  [exfiltration_attempted   SIGNIFICANT    ] direct_instruction_injection: 13.3% -> 73.3% (diff +60.0pp, 95% CI [33.3, 73.3], q=0.000, n=5 cases)
  ...

--- always-valid sequential analysis: exfiltration (17 cases) ---
stoppable at n=7: evidence sufficient (e-value 32.16 >= 20.00). Current (n=17) estimate +0.353, always-valid 95% CS [+0.091, +0.615].
```

- **`task_success`** comes first on purpose: if it drops a lot between
  arms, a "safety improvement" may just be an agent that stopped doing
  anything, not one that got safer. Read it before the outcome rows below.
- Every outcome is shown **twice**: the base key (e.g. `exfiltration`)
  means the bad thing actually *happened* (the tool call executed);
  `_attempted` means the model *tried*, whether or not a permission guard
  blocked it. These can diverge — a guard that blocks every attempt but
  doesn't change the model's willingness to try shows up as `exfiltration`
  flat and `exfiltration_attempted` unchanged too, while a prompt change
  that makes the model stop trying shows up as *both* dropping. Don't read
  only one.
- **`q=`** is the Benjamini-Hochberg-adjusted p-value across families —
  compare it to your alpha (0.05 by default), not the raw p-value.
  `SIGNIFICANT` in the row label already reflects this.
- **`n=X cases`** is the number of attack cases in that family with data
  in both arms — cluster_bootstrap (the default method) empirically
  over-rejects (~1.2-1.7x nominal alpha) across a wide range of case
  counts, not just below some small-N cutoff — see the dashboard's A/A
  calibration panel or `stats/paired.py`'s `cluster_bootstrap_diff`
  docstring for the actual measured numbers. With this repo's small
  hand-curated families (3-5 cases each) that effect is worse, not better,
  so treat individual family verdicts as directional, not final. Run
  `python -m stats.cli aa-calibration` (or `experiments/calibration.py`'s
  sweep) to check calibration at whatever scale you're actually operating
  at — don't assume growing the case suite alone fixes it.
- The **sequential analysis** block (only printed for presets that declare
  a `sequential_outcome_key`, currently `known_regression`) answers "how
  many runs did that take" honestly: it's an always-valid confidence
  sequence (Johari, Pekelis & Walsh 2017), so "stoppable at n=7" means you
  could have safely stopped after 7 cases without inflating your false
  positive rate — not a p-value computed once at a pre-committed N and
  presented as if it were.
- If **both arms show the same `config_hash`** (the `aa` preset), the
  report adds an explicit note: any `SIGNIFICANT` flag there is a false
  positive by construction.

### The mock backend is a scripted stand-in, not a real model

`MockModel` can't read a system prompt or reason about a config — it just
plays back a fixed script. So that `known_regression`/`known_neutral`/
`model_swap`/`added_agent` can demonstrate anything at all with
`provider="mock"`, `experiments/mock_policy.py` derives a *toy, explicit*
"compliance probability" from a few config signals (defensive instruction
present, allowlist enforcement on, agent count) and rolls a seeded
deterministic draw against it per case. This is clearly labeled everywhere
it shows up (preset descriptions, module docstring) — it exists so the
pipeline is exercisable and demoable at zero cost, not to claim anything
about real model behavior. The roll uses common random numbers across arms
(same draw for the same `(case_id, seed)`, regardless of which arm) — this
is what makes `known_neutral` and `aa` show *exactly* 0.0pp rather than
noisy near-zero: same true probability, same underlying draw, so nothing
to disagree about. Real findings require `provider="anthropic"` on both
arms — see `.env.example` for the API key, and `target_system/config.py`'s
`ModelConfig` for the model name field.

## Multi-turn cases

`attacker/cases.py`'s `multi_turn_goal_hijack` family (hand-written — no
AgentDojo/AgentHarm analogue, both are single-turn) needs the agent to
have already engaged with a conversation before the hijack attempt lands.
`target_system/orchestration.py`'s `run_multi_turn_case` runs each case's
turns in sequence against one Agno session (`db=InMemoryDb()` +
`add_history_to_context=True`, wired unconditionally for every config —
see that module's docstring), so later turns genuinely see earlier turns'
assistant replies. `attacker/executor.py`'s `execute_case()` dispatches to
it automatically based on `AttackCase.injection_vector` — nothing in
`experiments/` needs to know the difference.

## Running the dashboard

```bash
# one-time backfill — generates the artifacts the dashboard reads (see below)
.venv/bin/python -m experiments.calibration
.venv/bin/python -m experiments.cli run --preset aa
.venv/bin/python -m experiments.cli run --preset known_regression
.venv/bin/python -m experiments.cli run --preset known_neutral
.venv/bin/python -m experiments.cli run --preset model_swap
.venv/bin/python -m experiments.cli run --preset added_agent

.venv/bin/streamlit run dashboard/app.py
```

The dashboard (`dashboard/app.py`) is read-only: it never runs an
experiment or the target system itself. Every panel reads a `*_report.json`
/ `*.jsonl` file that already exists under `data/runs/`, except the power
curve and the CRN panel's supporting chart, which evaluate `stats.power` /
`stats.variance_reduction`'s closed-form formulas live (no target-system
execution involved). If a panel's expected file is missing, it says so
on the page rather than silently computing something else.

Panels, top to bottom:

1. **A/A calibration** — observed false-positive rate vs. nominal alpha
   (`data/runs/aa_calibration_sweep.json`, generated by
   `experiments/calibration.py`), plus real-execution corroboration from
   the `aa` preset. The most important panel — read it before trusting
   anything below it.
2. **Comparison view** — per-family rates for a selected experiment, strict
   and `_attempted` outcome variants shown side by side, sorted by effect
   size, colored by FDR-corrected significance.
3. **CRN / variance reduction** — a real before/after (the actual
   spurious-flag bug caught during Part 4 development, reconstructed via
   an isolated scratch run of the pre-fix code — see
   `experiments/mock_policy.py`'s docstring) plus a synthetic
   required-sample-size comparison.
4. **Confidence sequence** — `known_regression`'s exfiltration confidence
   sequence narrowing over cases, with the always-valid stopping point
   marked.
5. **Power curve** — detectable effect vs. runs/case, adjustable live.
6. **Trajectory inspector** — pick an experiment and a run (defaults to a
   flagged one), see the full event timeline with the injected payload
   highlighted, the offending call marked with its `executed`/`blocked`
   status, and turn boundaries for multi-turn cases.

Tested headlessly via `streamlit.testing.v1.AppTest` in
`tests/test_dashboard_app.py` (skipped automatically if the backfill above
hasn't been run yet).

## Development notes

- `stats/` functions are unit-tested against simulated data with known
  ground truth (`tests/test_stats_*.py`) — including two numerical bugs
  caught by checking against published reference values (a group-sequential
  boundary scale error and a spurious-trapezoid integration bug) and a
  degenerate zero-variance bootstrap case caught by this repo's own `aa`
  preset on its first real run.
- Concurrency: `execute_case` is safe to call from multiple threads —
  every run gets its own tool `Function` copies and its own `InMemoryDb`
  (see `target_system/orchestration.py`'s `_build_tools` / `build_team`
  docstrings and `tests/test_concurrency.py` / `tests/test_multi_turn.py`
  for the regression tests). `experiments/runner.py` parallelizes across
  `(case, arm, run_index)` jobs with a thread pool and a lock around the
  shared JSONL append.
- McNemar's test defaults to continuity-corrected (conservative — see
  `stats/paired.py`'s `mcnemar_test` docstring for the measured effect).
  This is an intentional default, not a bug.
