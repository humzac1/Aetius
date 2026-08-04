"""Runs stats.aa_calibration's Monte Carlo sweep (simulated against the
statistics module itself, not the target system) across several n_cases
values and saves it to data/runs/aa_calibration_sweep.json so the
dashboard can show a real observed-FPR-vs-nominal-alpha result without
recomputing it on page load.

Sweeps n_cases rather than reporting one point deliberately: an earlier,
single-point version of this sweep (400 trials, n_cases=25) had a CI wide
enough to just barely include nominal alpha and got documented as
"calibrated by 25" — wrong, per a higher-precision re-run (1500+ trials)
that showed BCa cluster_bootstrap actually over-rejects by roughly
1.2-1.7x nominal across a wide n_cases range, not just below some small-N
cutoff (see stats/paired.py's cluster_bootstrap_diff docstring for the
full corrected numbers). Showing the trend rather than a single cherry-
pickable point is what keeps that mistake from happening again.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from experiments.runner import DEFAULT_RUNS_DIR
from stats.aa_calibration import AACalibrationResult, CaseSpec, run_aa_calibration

DEFAULT_SWEEP_PATH = DEFAULT_RUNS_DIR / "aa_calibration_sweep.json"
DEFAULT_N_CASES_SWEEP = (15, 25, 40, 60, 80)


def _case_specs(n_cases: int, min_base_rate: float, max_base_rate: float, seed: int) -> list[CaseSpec]:
    rng = np.random.default_rng(seed)
    return [
        CaseSpec(case_id=f"case_{i}", family="synthetic", base_rate=float(r))
        for i, r in enumerate(rng.uniform(min_base_rate, max_base_rate, n_cases))
    ]


def run_and_save_calibration_sweep(
    *,
    n_cases_sweep: tuple[int, ...] = DEFAULT_N_CASES_SWEEP,
    min_base_rate: float = 0.05,
    max_base_rate: float = 0.35,
    n_runs_per_case: int = 15,
    method: str = "cluster_bootstrap",
    n_trials: int = 1500,
    alpha: float = 0.05,
    seed: int = 0,
    path: Path = DEFAULT_SWEEP_PATH,
) -> Path:
    points = []
    for n_cases in n_cases_sweep:
        case_specs = _case_specs(n_cases, min_base_rate, max_base_rate, seed)
        result: AACalibrationResult = run_aa_calibration(
            case_specs,
            n_runs_per_case=n_runs_per_case,
            method=method,
            n_trials=n_trials,
            alpha=alpha,
            seed=seed,
            method_kwargs={"n_boot": 800},
        )
        payload = dataclasses.asdict(result)
        payload.pop("p_values")  # per-point raw p-values aren't needed for the dashboard and bloat the file
        points.append(payload)

    combined = {
        "method": method,
        "alpha": alpha,
        "n_runs_per_case": n_runs_per_case,
        "n_trials": n_trials,
        "min_base_rate": min_base_rate,
        "max_base_rate": max_base_rate,
        "points": points,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return path


def load_calibration_sweep(path: Path = DEFAULT_SWEEP_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    saved_path = run_and_save_calibration_sweep()
    print(f"Saved calibration sweep: {saved_path}")
