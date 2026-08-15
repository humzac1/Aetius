# Aetius

Aetius is a security testing tool for AI agent systems. You install it once and then run it as a single command from your terminal.

## Running it

Installation is covered on the project website, so this file assumes the tool is already on your machine. To start it, open a terminal and run:

```bash
aetius
```

That is the whole launch command. It opens an interactive terminal UI (a Textual app) that drives everything else from there.

The first time you run it, it asks for a few credentials before letting you in: an `ANTHROPIC_API_KEY`, and then the connection details for whichever trace source your real agent logs to, either Langfuse or Braintrust. Each value is checked against the live service before anything gets written to disk, and what you enter is saved to a user-level config location outside the repo, so this only happens once per machine. If any of those values are already present as real environment variables, the matching prompt is skipped. You can change any of it later from the Settings menu inside the app.

If you just want to confirm which build is actually installed:

```bash
aetius --version
```

Once you are past setup, everything happens inside the UI: pulling traces from your source, reconstructing an environment out of them, running the attack suite against a single config, or comparing two configs to see whether a change made things worse. Every screen shows its navigation keys in the footer (`b` for back, `h` for home, `q` to quit).

## What this is, and why it exists

When you run an AI agent in production and then change something about it, a newer model, a reworded system prompt, a tool permission you loosened, an extra agent added to the team, you usually want to answer one question honestly: did that change make the system less safe? Is it now more likely to leak data, follow an instruction that was smuggled in through a document, or get talked step by step into doing something it should have refused?

That question is genuinely hard to answer, and the reason is that agents are stochastic. Run the exact same attack twice against the exact same unchanged system and you will often get two different outcomes. So the naive approach, run your attacks before the change, run them again after, and compare the numbers, quietly betrays you. You end up chasing differences that are pure noise, and every so often you wave off a real regression because this time around it happened to look fine.

Aetius is built to answer the question properly, and it works in two parts. The first part reconstructs a structural twin of your real agent system directly from its traces, so the thing under test actually resembles what you are running in production rather than some toy you invented for the occasion. The second part runs a statistically sound attack and regression engine against that twin, one designed specifically to tell a real shift in safety apart from the random variation you would expect regardless. When it reports that a family of attacks got worse, that verdict has already been through multiple-comparison correction and a calibrated false-positive check. When the evidence isn't there yet, it tells you that plainly, and tells you roughly how many more runs it would take to reach a conclusion.

Why bother with all of that? Because "we changed the prompt and it seems fine" is not a safety claim. It is a guess wearing a confident voice. The point of this project is to turn that guess into something you can actually stand behind, or, just as usefully, into an honest admission that you can't stand behind it yet.

## How the repository is laid out

One thing to get out of the way first. The repository is named `Caligula` for historical reasons, and the installable package is called `aetius`, but all of the real code lives inside the `Initial Test/` subdirectory. Every path below is relative to `Initial Test/`.

### `target_system/`

This directory carries two related jobs. The older one is a toy company assistant used as a controlled test bed: a Supervisor, Researcher, and Operator working as an Agno `Team` in coordinate mode (`orchestration.py`, `prompts.py`), a small document corpus in `corpus/` made of fake meeting notes, support tickets, and wiki pages, a handful of stub tools like `send_email`, `lookup_customer`, and `search_corpus` in `tools.py`, a versioned, content-hash-addressed `SystemConfig` in `config.py`, and the trajectory log schema in `logging_schema.py`. Because we built that toy system ourselves, its ground truth is under our control, which is exactly what you need to validate a statistics engine before you trust it on anything real.

The same directory also holds the machinery for running a reconstructed twin of a real system. `reconstructed_execution.py` runs a reconstructed config (a single Agno agent with no real tool implementations behind it) and resolves its tool calls through `tool_synthesis.py` rather than the hand-written toy tools. `provenance.py` records where a reconstructed config came from, `tool_roles.py` classifies each tool into abstract roles so the attacker knows what applies, `factory.py` builds the baseline config and the named preset variants, `mock_model.py` is the scripted stand-in model, and `policy.py` plus `policy.yaml` are the permission layer. The `configs/` subfolder is saved output: every resolved `SystemConfig` this repo has ever produced, keyed by content hash. That folder is deliberately kept out of the built wheel.

### `attacker/`

The attack suite. `cases.py` holds 17 hand-curated cases across four families: direct instruction injection, indirect injection through a document, tool result poisoning, and multi-turn goal hijacking. `sourcing.py` writes down exactly what was borrowed from AgentDojo and why AgentHarm was not used, and `payloads.py` holds the injection templates adapted from AgentDojo. `executor.py` is the single point that turns a case into a run and decides whether it needs single-turn or multi-turn handling.

The rest of this directory exists for reconstructed environments rather than the fixed toy one. `case_generation.py` produces domain-adapted cases for a reconstructed system, `applicability.py` works out which families even make sense given that system's tools, `case_selection.py` picks which suite a given set of configs should be tested with, and `attack_case.py` and `case_store.py` cover the case type and its storage.

### `stats/`

This is the core of the whole thing. `paired.py` implements the paired comparison methods (cluster bootstrap, McNemar, mixed-effects logistic regression). `hierarchical.py` is the hierarchical, partial-pooling Beta-Binomial comparison that is the current default verdict method, paired with a ROPE for deciding practical equivalence. `aa_calibration.py` measures the false-positive rate against a null where nothing actually changed, `multiple_comparisons.py` applies Benjamini-Hochberg FDR control across families, and `sequential.py` provides always-valid sequential inference through confidence sequences and alpha-spending, so you can stop early without cheating. `variance_reduction.py` covers common random numbers and CUPED, `power.py` handles power analysis, `reporting.py` formats the results, and `types.py` holds the shared types. There is also a standalone CLI: `python -m stats.cli aa-calibration` is a good way to sanity-check the statistics on their own, with no target system involved.

### `experiments/`

The runner that ties the pieces together. It executes both arms of a paired comparison across the attack suite, caches every result under `data/runs/`, computes the statistical comparison, and writes out both a printed report and a `*_report.json` that other tools read. `cli.py` is the entry point, `presets.py` defines the five built-in scenarios (`aa`, `known_regression`, `known_neutral`, `model_swap`, `added_agent`), and `mock_policy.py` is the toy compliance model that lets those presets demonstrate something even when running against the free mock backend. `calibration.py` runs the A/A sweep, `persist.py` handles saving, `cost_estimate.py` shows what a batch will cost before you commit to it, and `hierarchical_validation.py` is the full validation sweep behind the hierarchical method in `stats/`. All of this is internal, developer-facing tooling aimed at the toy system, and it is intentionally not reachable from the shipped `aetius` command.

### `tui/`

The interactive Textual application, and the thing `aetius` actually launches (`pyproject.toml` points the `aetius` script at `tui.app:main`). `app.py` is the entry point and the `screens/` folder holds one file per screen: adding an environment, the guided test wizard, verdict screens, config management, past runs, presets, a progress view, credentials, settings, and generated cases. The supporting modules here (`verdict_logic.py`, `execution.py`, `data.py`, `formatting.py`, `run_sizing.py`, `dashboard_link.py`) glue the UI to the real work, but the UI never recomputes a verdict or runs an attack on its own. It always calls into `experiments/`, `stats/`, and `attacker/` for that.

### `ingestion/`

This is where real traces come in and a reconstructed config comes out. `langfuse_client.py` and `braintrust_client.py` pull traces from a live project and cache them to disk, re-pulling only the gap on later runs so you are not hammering the API. `reconstruct.py` and `braintrust_reconstruct.py` turn a cached trace batch into a `SystemConfig`-compatible reconstruction, and they group a project's traces by agent first, since one project can contain several unrelated systems that should not be blended together. A system prompt is never fabricated.

### `scripts/`

Two shell scripts for shipping the tool. `release.sh` builds a versioned wheel and copies it to `release/aetius-latest.whl`. `install_latest.sh` installs that stable-named wheel, working around the fact that pip and pipx reject a wheel whose filename is not a valid PEP 427 name by reading the real name out of the wheel's own metadata and installing under that instead.

### `config/`

User-level configuration that stays independent of both your current directory and wherever the package happened to get installed. `paths.py` resolves the correct per-OS data and config locations through platformdirs, and `credentials.py` validates and stores your credentials, enforcing that exactly one trace source is active at a time.

### `dashboard/`

A read-only Streamlit dashboard over `data/runs/` and the saved reports, launched with `streamlit run dashboard/app.py`. It is an optional extra rather than part of the base `aetius` install, and it never runs an experiment or the target system itself. It only reads and displays artifacts that already exist.

### `data/`

Runtime output, and gitignored. `runs/` holds the trajectory logs (`*.jsonl`, one line per run) alongside the computed comparison reports (`*_report.json`). `traces/` caches the trace batches pulled during ingestion.

### `tests/`

The pytest suite, and it is broad. The statistics are checked against simulated data with known ground truth, the TUI is driven headlessly through Textual's `Pilot`, the dashboard is exercised through Streamlit's `AppTest`, and packaging is verified by inspecting a wheel that was actually built rather than assuming its contents. Ingestion, concurrency, and multi-turn behavior all have their own coverage.

### `dist/` and `release/`

Build artifacts. `dist/` keeps one versioned wheel per release for the record. `release/` holds the stable, version-less `aetius-latest.whl`, whose name stays the same across releases even as its contents change, so a hosted download link never has to be updated. You will also find a leftover `caligula-latest.whl` there from before the tool was renamed.

At the top level, `pyproject.toml` carries the package metadata, the dependency list, the `aetius` entry point, and the optional dashboard and dev extras, and `uv.lock` pins the exact resolved versions of everything.
