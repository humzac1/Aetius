"""Persists a computed ExperimentResult (and, when applicable, the full
sequential-analysis point sequence) to JSON alongside the raw JSONL
trajectories in data/runs/ — this is what makes "the Part 4 experiment
reports" an actual on-disk artifact the dashboard (Part 5) can read
without recomputing statistics or running anything itself.

FamilyResult and EffectEstimate (stats/) and ConfidenceSequenceResult /
ConfidenceSequencePoint (stats/sequential.py) are all plain dataclasses,
so dataclasses.asdict() handles the nested structure for free — no manual
serialization to keep in sync as those types evolve.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.report import compute_sequential_analysis
from experiments.runner import DEFAULT_RUNS_DIR, ExperimentResult
from target_system.logging_schema import RunRecord


def report_path(experiment_name: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    return runs_dir / f"{experiment_name}_report.json"


def cases_per_family(records: list[RunRecord]) -> dict[str, int]:
    """How many distinct cases this run actually has records for, per
    family. Saved into the report because family_results alone can't
    answer it: a family whose case count is under stats.paired's
    MIN_CASES_FOR_BOOTSTRAP produces no FamilyResult at all
    (compare_families swallows the ValueError), so by the time a verdict
    is computed the evidence for *why* there's nothing to report is
    exactly the thing that got dropped. Counted off the raw records
    rather than the case list so it reflects what genuinely ran, not what
    was requested."""
    by_family: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.case_family:
            by_family[record.case_family].add(record.case_id)
    return {family: len(case_ids) for family, case_ids in sorted(by_family.items())}


def save_experiment_report(
    result: ExperimentResult,
    *,
    sequential_outcome_key: str | None = None,
    tau: float = 0.1,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> Path:
    payload: dict[str, Any] = {
        "name": result.name,
        "arm_a_label": result.arm_a_label,
        "arm_b_label": result.arm_b_label,
        "arm_a_hash": result.arm_a_hash,
        "arm_b_hash": result.arm_b_hash,
        "n_cases": result.n_cases,
        "cases_per_family": cases_per_family(result.records),
        "n_runs_per_case": result.n_runs_per_case,
        "n_cached": result.n_cached,
        "n_executed": result.n_executed,
        "task_success_a": result.task_success_a,
        "task_success_b": result.task_success_b,
        "family_results": {
            outcome_key: [dataclasses.asdict(r) for r in results]
            for outcome_key, results in result.family_results.items()
        },
        "sequential_analysis": None,
    }

    if sequential_outcome_key is not None:
        cs = compute_sequential_analysis(result, sequential_outcome_key, tau=tau)
        if cs is not None:
            payload["sequential_analysis"] = {
                "outcome_key": sequential_outcome_key,
                **dataclasses.asdict(cs),
            }

    path = report_path(result.name, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_experiment_report(experiment_name: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> dict[str, Any] | None:
    path = report_path(experiment_name, runs_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
