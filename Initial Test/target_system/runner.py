"""Thin wrapper around orchestration.run_case that persists the resulting
RunRecord to <user data dir>/runs/<experiment>.jsonl and ensures the config
that produced it is saved under <user data dir>/configs/ (see
config/paths.py for where that resolves and why)."""

from __future__ import annotations

from pathlib import Path

from config import paths
from target_system.config import DEFAULT_CONFIGS_DIR, SystemConfig, save_config
from target_system.logging_schema import AttackInfo, RunRecord, append_run_record
from target_system.orchestration import MockScripts, run_case
from target_system.policy import TaskContext

DEFAULT_RUNS_DIR = paths.RUNS_DIR


def run_and_record(
    config: SystemConfig,
    task: str,
    *,
    case_id: str,
    task_context: TaskContext,
    seed: int,
    experiment: str,
    arm: str | None = None,
    case_family: str | None = None,
    attack: AttackInfo | None = None,
    mock_scripts: MockScripts | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    configs_dir: Path = DEFAULT_CONFIGS_DIR,
) -> RunRecord:
    save_config(config, configs_dir=configs_dir)
    record = run_case(
        config,
        task,
        case_id=case_id,
        task_context=task_context,
        seed=seed,
        arm=arm,
        case_family=case_family,
        attack=attack,
        mock_scripts=mock_scripts,
    )
    append_run_record(record, runs_dir / f"{experiment}.jsonl")
    return record
