"""The full validation sweep behind stats/hierarchical.py — the Bayesian
counterpart of experiments/calibration.py's A/A sweep for the frequentist
methods. Run it before changing the model, the hyperparameter grid, or
DEFAULT_ROPE_HALF_WIDTH:

    uv run python -m experiments.hierarchical_validation

It simulates repeated experiments with known ground truth against the two
real per-case rate shapes from the A/A run that produced the false
FLAGGED (the same shapes stats/paired.py's guards are pinned to) and
reports, per condition, with Wilson CIs:

  - cov95:    how often the 95% credible interval contains the true
              effect (must sit near 0.95 — higher is vague, lower is
              overconfident);
  - rope_sig: how often the ROPE rule signals (must be ~0 on NULL rows;
              on effect rows this is power);
  - excl0:    how often the interval excludes zero (diagnostic only — the
              live rule is the ROPE, precisely because this over-signals
              on the rare shape).

The committed numbers this sweep must keep reproducing are tabulated in
stats/hierarchical.py's docstring and constants block; the fast
qualitative pins live in tests/test_stats_hierarchical.py. ~15 minutes.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from experiments.runner import DEFAULT_RUNS_DIR
from stats.hierarchical import hierarchical_bayes_diff
from stats.types import CaseObservations, PairedCaseData

DEFAULT_SWEEP_PATH = DEFAULT_RUNS_DIR / "hierarchical_validation_sweep.json"

# The two real measured shapes (see tests/test_bootstrap_calibration_guards.py).
HIGH_CEILING = [0.4805, 1.0, 0.8831, 0.9481, 0.9870]
RARE_FLOOR = [0.0390, 0.0, 0.0, 0.0, 0.0]

DEFAULT_N_TRIALS = 800
# Every run count the live path has actually used, and every case count
# down to the structural floor.
DEFAULT_RUNS_SWEEP = (2, 3, 5, 15, 77)
DEFAULT_CASES_SWEEP = (2, 3, 5)


@dataclasses.dataclass(frozen=True)
class ValidationPoint:
    condition: str
    true_delta: float
    n_cases: int
    n_runs_per_case: int
    n_trials: int
    coverage_95: float
    coverage_95_ci: tuple[float, float]
    rope_signal_rate: float
    rope_signal_rate_ci: tuple[float, float]
    excludes_zero_rate: float

    def summary(self) -> str:
        c_lo, c_hi = self.coverage_95_ci
        s_lo, s_hi = self.rope_signal_rate_ci
        return (
            f"{self.condition:44s} trueD={self.true_delta:+.4f} "
            f"cov95={self.coverage_95:.3f}[{c_lo:.3f},{c_hi:.3f}] "
            f"rope_sig={self.rope_signal_rate:.3f}[{s_lo:.3f},{s_hi:.3f}] "
            f"excl0={self.excludes_zero_rate:.3f}"
        )


def _paired(rates_a: list[float], rates_b: list[float], n_runs: int, rng: np.random.Generator) -> list[PairedCaseData]:
    data = []
    for i, (ra, rb) in enumerate(zip(rates_a, rates_b)):
        a = tuple(int(x) for x in rng.binomial(1, ra, n_runs))
        b = tuple(int(x) for x in rng.binomial(1, rb, n_runs))
        cid = f"case_{i}"
        data.append(PairedCaseData(cid, "sweep", CaseObservations(cid, "sweep", a), CaseObservations(cid, "sweep", b)))
    return data


def _shifted(shape: list[float], delta: float) -> list[float]:
    return [min(1.0, max(0.0, r + delta)) for r in shape]


def run_condition(
    condition: str,
    rates_a: list[float],
    rates_b: list[float],
    *,
    n_runs_per_case: int,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = 0,
) -> ValidationPoint:
    ra, rb = np.asarray(rates_a, float), np.asarray(rates_b, float)
    true_delta = float(np.mean(rb - ra))
    rng = np.random.default_rng(seed)
    covered = signals = excludes = 0
    for _ in range(n_trials):
        data = _paired(list(ra), list(rb), n_runs_per_case, rng)
        est = hierarchical_bayes_diff(data, seed=int(rng.integers(0, 2**31 - 1)))
        covered += est.ci_low <= true_delta <= est.ci_high
        signals += bool(est.extra["rope_signal"])
        excludes += not (est.ci_low <= 0.0 <= est.ci_high)

    def wilson(k: int) -> tuple[float, float]:
        lo, hi = proportion_confint(k, n_trials, alpha=0.05, method="wilson")
        return (float(lo), float(hi))

    return ValidationPoint(
        condition=condition,
        true_delta=true_delta,
        n_cases=len(ra),
        n_runs_per_case=n_runs_per_case,
        n_trials=n_trials,
        coverage_95=covered / n_trials,
        coverage_95_ci=wilson(covered),
        rope_signal_rate=signals / n_trials,
        rope_signal_rate_ci=wilson(signals),
        excludes_zero_rate=excludes / n_trials,
    )


def run_and_save_validation_sweep(
    *,
    runs_sweep: tuple[int, ...] = DEFAULT_RUNS_SWEEP,
    cases_sweep: tuple[int, ...] = DEFAULT_CASES_SWEEP,
    n_trials: int = DEFAULT_N_TRIALS,
    path: Path = DEFAULT_SWEEP_PATH,
) -> Path:
    shapes = [("rare_floor", RARE_FLOOR, +0.15), ("high_ceiling", HIGH_CEILING, -0.10)]
    points: list[ValidationPoint] = []
    seed = 0
    # Run-count sweep at 5 cases, then case-count sweep at the largest run
    # count — the same grid the committed constants were measured on.
    grid = [(5, n_runs) for n_runs in runs_sweep] + [
        (k, n_runs) for k in cases_sweep if k != 5 for n_runs in (5, max(runs_sweep))
    ]
    for k, n_runs in grid:
        for shape_name, shape, effect in shapes:
            base = shape[:k] if k <= len(shape) else [shape[i % len(shape)] for i in range(k)]
            for label, arm_b in [("NULL", base), (f"eff{effect:+.2f}", _shifted(base, effect))]:
                seed += 1
                point = run_condition(
                    f"{shape_name} K={k} runs={n_runs} {label}", base, arm_b,
                    n_runs_per_case=n_runs, n_trials=n_trials, seed=seed,
                )
                print(point.summary(), flush=True)
                points.append(point)

    payload = {"n_trials": n_trials, "points": [dataclasses.asdict(p) for p in points]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    saved = run_and_save_validation_sweep()
    print(f"\nwrote {saved}")
