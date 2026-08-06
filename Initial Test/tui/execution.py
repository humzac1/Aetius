"""Runs attack cases for the TUI. Comparison checks (two arms) delegate
straight to experiments.runner.run_experiment — there is no paired
statistical logic here, just a deterministic experiment name derived from
the two config hashes so re-running the same pair from the wizard resumes
the same cached file instead of starting a fresh one each time.

Single-config checks have no equivalent in experiments/ (run_experiment is
inherently paired — it always computes a two-arm comparison), so this
module gives them the same execution shape run_experiment uses internally:
CacheIndex for resumability, execute_case as the one dispatch point,
build_mock_scripts for the mock backend, a thread pool with a write lock
around the shared JSONL append. That's reuse of the same primitives
runner.py exports for exactly this reason (see CacheIndex's docstring),
not a second implementation of them.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from attacker.attack_case import AttackCase
from attacker.cases import ATTACK_CASES
from attacker.executor import execute_case
from experiments.mock_policy import build_mock_scripts
from config.credentials import ensure_env_loaded
from experiments.runner import DEFAULT_RUNS_DIR, CacheIndex, ExperimentResult, run_experiment
from target_system.config import ModelConfig, SystemConfig, compute_config_hash, save_config
from target_system.logging_schema import RunRecord, append_run_record
from tui.data import single_config_run_path


def build_anthropic_client() -> Any:
    """Constructs a real anthropic.Anthropic client from ANTHROPIC_API_KEY
    (real env var first, config-file-backed one next — see
    config/credentials.py's ensure_env_loaded) — the one place the TUI
    actually builds this (every lower layer takes anthropic_client as an
    injected parameter, never constructs one itself, so it stays cheaply
    testable without spending real API money — see
    target_system/tool_synthesis.py). Same missing-var discipline as
    ingestion/langfuse_client.py's build_client: raises naming which var is
    missing, never touches its value. Import of the anthropic package is
    local to this function so nothing that never runs a real model needs it
    importable."""
    import anthropic

    ensure_env_loaded()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("missing required credential: ANTHROPIC_API_KEY")
    return anthropic.Anthropic()


def comparison_experiment_name(hash_a: str, hash_b: str) -> str:
    """Deterministic name for an ad hoc (wizard- or menu-driven, not
    preset) two-arm comparison, so re-running the same pair of config
    hashes resumes the same cached JSONL rather than minting a new file
    every time."""
    return f"adhoc_{hash_a}_{hash_b}"


def enforce_reconstructed_provider(config: SystemConfig) -> SystemConfig:
    """Reconstructed environments (config.provenance is not None) are
    real-model-only (Part 5) — ingestion/reconstruct.py's
    _build_model_config already sets provider="anthropic" for every
    reconstruction, so this is a defensive correction, not the primary
    guard (execute_case/run_experiment/run_single_config_check all reject
    a mismatch loudly). It exists so a reconstructed config picked through
    the wizard is never silently handed to the mock backend, no matter how
    it ended up on disk with a different provider."""
    if config.provenance is None or config.model.provider == "anthropic":
        return config
    return config.model_copy(update={"model": ModelConfig(provider="anthropic", model_name=config.model.model_name)})


def peek_n_cached(configs: list[SystemConfig], *, runs_dir: Path = DEFAULT_RUNS_DIR) -> int:
    """How many records already sit in the JSONL this run would append to,
    without executing anything — the same file run_single_config_check
    (len(configs) == 1) or run_comparison_check (len(configs) == 2) would
    write to. Used to ground a pre-execution cost estimate in what's
    actually left to run, not the full batch (see
    experiments/cost_estimate.py's n_cached parameter)."""
    if len(configs) == 1:
        path = single_config_run_path(compute_config_hash(configs[0]), runs_dir=runs_dir)
    elif len(configs) == 2:
        name = comparison_experiment_name(compute_config_hash(configs[0]), compute_config_hash(configs[1]))
        path = runs_dir / f"{name}.jsonl"
    else:
        raise ValueError(f"expected 1 or 2 configs, got {len(configs)}")
    return len(CacheIndex(path).records)


def run_comparison_check(
    config_a: SystemConfig,
    config_b: SystemConfig,
    *,
    cases: list[AttackCase] | None = None,
    n_runs_per_case: int = 5,
    max_workers: int = 8,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    on_progress: Callable[[int, int], None] | None = None,
    anthropic_client: Any = None,
) -> ExperimentResult:
    """Thin pass-through to experiments.runner.run_experiment — the name
    is the only thing this function adds. run_experiment itself rejects a
    reconstructed config (provenance is not None) under provider="mock",
    so that guard doesn't need to be duplicated here."""
    name = comparison_experiment_name(compute_config_hash(config_a), compute_config_hash(config_b))
    return run_experiment(
        config_a,
        config_b,
        experiment_name=name,
        cases=cases,
        n_runs_per_case=n_runs_per_case,
        max_workers=max_workers,
        runs_dir=runs_dir,
        on_progress=on_progress,
        anthropic_client=anthropic_client,
    )


@dataclass
class SingleConfigCheckResult:
    config_hash: str
    n_cases: int
    n_cached: int
    n_executed: int
    records: list[RunRecord] = field(default_factory=list, repr=False)


def run_single_config_check(
    config: SystemConfig,
    *,
    cases: list[AttackCase] | None = None,
    n_runs_per_case: int = 5,
    max_workers: int = 8,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    on_progress: Callable[[int, int], None] | None = None,
    anthropic_client: Any = None,
) -> SingleConfigCheckResult:
    """Runs the attack suite against a single config, no paired arm and no
    statistical comparison — the "test my agent" verdict is a raw
    succeeded/blocked/resisted tally (tui/verdict_logic.compute_single_config_summary),
    not an effect size.

    on_progress has the same contract as run_experiment's: called with
    (0, total) before anything starts, then (completed, total) after each
    finished job, always on the caller's thread.

    Reconstructed environments (config.provenance is not None) are real-
    model-only — same rule as experiments.runner.run_experiment, checked
    up front here for the same reason (a faster, quieter failure than
    letting every job in the pool below reject individually via
    execute_case)."""
    if config.provenance is not None and config.model.provider != "anthropic":
        raise ValueError(
            f"{config.label!r}: reconstructed environments only run under provider='anthropic' — "
            f"got provider={config.model.provider!r}"
        )
    cases = list(cases) if cases is not None else list(ATTACK_CASES)
    config_hash = save_config(config)
    runs_path = single_config_run_path(config_hash, runs_dir=runs_dir)
    cache = CacheIndex(runs_path)
    write_lock = Lock()

    jobs: list[tuple[AttackCase, int]] = []
    for case in cases:
        for run_idx in range(n_runs_per_case):
            if not cache.has(config_hash, case.id, None, run_idx):
                jobs.append((case, run_idx))

    n_cached = len(cache.records)

    if on_progress is not None:
        on_progress(0, len(jobs))

    def _execute(job: tuple[AttackCase, int]) -> RunRecord:
        case, seed = job
        mock_scripts = None
        # build_mock_scripts is toy-system-specific — see the matching
        # comment in experiments/runner.py's _execute. provider == "mock"
        # here always means a toy config: the up-front guard above already
        # rejects a reconstructed config under any provider but
        # "anthropic" before this closure ever runs.
        if config.model.provider == "mock":
            mock_scripts = build_mock_scripts(case, config, seed)
        record = execute_case(config, case, seed=seed, arm=None, mock_scripts=mock_scripts, anthropic_client=anthropic_client)
        with write_lock:
            append_run_record(record, runs_path)
        return record

    if jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_execute, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), start=1):
                cache.add(future.result())
                if on_progress is not None:
                    on_progress(completed, len(jobs))

    return SingleConfigCheckResult(
        config_hash=config_hash,
        n_cases=len(cases),
        n_cached=n_cached,
        n_executed=len(jobs),
        records=cache.records,
    )
