"""A/A calibration: run the identical configuration as both arms, many
times, and check how often the chosen test spuriously rejects at the
nominal alpha. This is the first thing to run against this whole module —
if a test's empirical false positive rate doesn't match its nominal alpha,
nothing else it reports (effect sizes, CIs, q-values) can be trusted
either.

Each simulated "trial" here is a full synthetic experiment (every case
resampled fresh at its own base rate, identically in both arms since
there's no true effect by construction) — the n_trials rejections are
themselves iid Bernoulli events, so pooling *those* is legitimate; this is
a different level of analysis from pooling runs across cases within one
trial, which case_rate()/paired_rate_diff() in types.py still forbid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from stats.paired import cluster_bootstrap_diff, mcnemar_test, mixed_effects_diff
from stats.types import CaseObservations, EffectEstimate, PairedCaseData

Method = Literal["cluster_bootstrap", "mcnemar", "mixed_effects"]

_METHODS: dict[Method, Callable[..., EffectEstimate]] = {
    "cluster_bootstrap": cluster_bootstrap_diff,
    "mcnemar": mcnemar_test,
    "mixed_effects": mixed_effects_diff,
}


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    family: str
    base_rate: float


@dataclass(frozen=True)
class AACalibrationResult:
    method: str
    alpha: float
    n_trials: int
    n_cases: int
    n_runs_per_case: int
    n_rejections: int
    observed_fpr: float
    fpr_ci_low: float
    fpr_ci_high: float
    well_calibrated: bool
    p_values: list[float] = field(default_factory=list)

    def summary(self) -> str:
        status = "well-calibrated" if self.well_calibrated else "MISCALIBRATED"
        return (
            f"[{status}] {self.method}: observed FPR = {self.observed_fpr:.3f} "
            f"(95% CI [{self.fpr_ci_low:.3f}, {self.fpr_ci_high:.3f}]) against nominal "
            f"alpha = {self.alpha:.3f}, over {self.n_trials} trials "
            f"({self.n_cases} cases x {self.n_runs_per_case} runs/case/arm)."
        )


def simulate_null_paired_data(
    case_specs: list[CaseSpec], n_runs_per_case: int, rng: np.random.Generator
) -> list[PairedCaseData]:
    """Both arms drawn from the SAME base_rate per case — the null/A-A
    scenario. Independent draws per arm (no CRN) unless the caller wants
    to test CRN's effect on calibration too, in which case draw once and
    hand the same outcomes to both arms upstream instead of calling this."""
    data = []
    for spec in case_specs:
        a = tuple(int(x) for x in rng.binomial(1, spec.base_rate, n_runs_per_case))
        b = tuple(int(x) for x in rng.binomial(1, spec.base_rate, n_runs_per_case))
        data.append(
            PairedCaseData(
                spec.case_id,
                spec.family,
                CaseObservations(spec.case_id, spec.family, a),
                CaseObservations(spec.case_id, spec.family, b),
            )
        )
    return data


def run_aa_calibration(
    case_specs: list[CaseSpec],
    *,
    n_runs_per_case: int,
    method: Method = "cluster_bootstrap",
    n_trials: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    method_kwargs: dict | None = None,
) -> AACalibrationResult:
    if method == "mcnemar" and n_runs_per_case != 1:
        raise ValueError("mcnemar calibration requires n_runs_per_case=1 (the simple paired binary design)")

    test_fn = _METHODS[method]
    method_kwargs = dict(method_kwargs or {})
    method_kwargs.setdefault("alpha", alpha)

    rng = np.random.default_rng(seed)
    p_values: list[float] = []
    rejections = 0

    for trial in range(n_trials):
        data = simulate_null_paired_data(case_specs, n_runs_per_case, rng)
        kwargs = dict(method_kwargs)
        if "seed" in test_fn.__code__.co_varnames:
            kwargs["seed"] = int(rng.integers(0, 2**31 - 1))
        result = test_fn(data, **kwargs)
        p_values.append(result.p_value if result.p_value is not None else float("nan"))
        if result.p_value is not None and result.p_value < alpha:
            rejections += 1

    observed_fpr = rejections / n_trials
    ci_low, ci_high = proportion_confint(rejections, n_trials, alpha=0.05, method="wilson")

    return AACalibrationResult(
        method=method,
        alpha=alpha,
        n_trials=n_trials,
        n_cases=len(case_specs),
        n_runs_per_case=n_runs_per_case,
        n_rejections=rejections,
        observed_fpr=observed_fpr,
        fpr_ci_low=float(ci_low),
        fpr_ci_high=float(ci_high),
        well_calibrated=(ci_low <= alpha <= ci_high),
        p_values=p_values,
    )
