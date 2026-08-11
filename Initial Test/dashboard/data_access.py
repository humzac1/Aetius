"""Pure read functions for the dashboard. Reads pre-computed reports and
raw trajectories from disk only — never runs an experiment, never
recomputes statistics that require re-executing the target system.
stats.power's closed-form formulas are the one exception used live
elsewhere (Panel 5): they're pure math, not something that requires new
target-system runs, so "does not run experiments itself" doesn't apply to
them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import paths

DASHBOARD_DIR = Path(__file__).parent
RUNS_DIR = paths.RUNS_DIR

PRESET_ORDER = ["aa", "known_regression", "known_neutral", "model_swap", "added_agent"]


def list_available_reports(runs_dir: Path = RUNS_DIR) -> list[str]:
    """Experiment names with a saved *_report.json on disk, presets first
    in their canonical order, anything else (e.g. the before_crn_fix
    reconstructions) alphabetically after."""
    if not runs_dir.exists():
        return []
    found = {p.name[: -len("_report.json")] for p in runs_dir.glob("*_report.json")}
    ordered = [n for n in PRESET_ORDER if n in found]
    ordered += sorted(found - set(PRESET_ORDER))
    return ordered


def load_report(experiment_name: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any] | None:
    path = runs_dir / f"{experiment_name}_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_calibration_sweep(runs_dir: Path = RUNS_DIR) -> dict[str, Any] | None:
    path = runs_dir / "aa_calibration_sweep.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw_records(experiment_name: str, runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    """Raw RunRecords as plain dicts — deliberately not reconstructed into
    the pydantic RunRecord model. The dashboard only ever reads fields, so
    skipping validation keeps this fast and avoids a dependency on
    target_system's schema just to immediately discard the wrapper."""
    path = runs_dir / f"{experiment_name}.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def flatten_family_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (outcome_key, family) — the shape Panel 2's chart wants.
    Each row carries the fields needed to plot + color + sort without the
    caller re-parsing the nested family_results structure."""
    rows = []
    for outcome_key, family_results in report.get("family_results", {}).items():
        for fr in family_results:
            effect = fr["effect"]
            rows.append(
                {
                    "outcome_key": outcome_key,
                    "family": fr["family"],
                    "q_value": fr["q_value"],
                    "significant": fr["significant_after_correction"],
                    "rate_a": effect["rate_a"],
                    "rate_b": effect["rate_b"],
                    "diff": effect["diff"],
                    "ci_low": effect["ci_low"],
                    "ci_high": effect["ci_high"],
                    "n_cases": effect["n_cases"],
                    "method": effect["method"],
                    "used_fallback": effect["used_fallback"],
                    "fallback_reason": effect["fallback_reason"],
                }
            )
    return rows


def find_flagged_run(
    records: list[dict[str, Any]], outcome_key: str = "exfiltration"
) -> dict[str, Any] | None:
    """First run in `records` whose outcomes[outcome_key] is True — the
    trajectory inspector's default selection ("click a flagged case," so
    it should start on one)."""
    for record in records:
        if record.get("outcomes", {}).get(outcome_key):
            return record
    return None
