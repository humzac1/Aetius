"""Two variance-reduction techniques, both measured for actual effect
against simulated data — the build spec is explicit that the tool's
practical value depends on how many runs a real detection needs, so
"we implemented CRN/CUPED" isn't enough; the achieved reduction has to be
reported as a number.

- Common random numbers (CRN): pairs the two arms at the trial level by
  sharing the same underlying random draw, so common noise cancels out of
  the *difference* instead of adding independently from each arm.
- CUPED: adjusts an arm's per-case rate using a covariate correlated with
  it (classically a pre-experiment measurement of the same metric) to
  strip out variance the covariate already explains, without introducing
  bias — E[adjusted] == E[original].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stats.types import CaseObservations, PairedCaseData, paired_rate_diff


# --- common random numbers -------------------------------------------------


def simulate_paired_dataset(
    case_rates: dict[str, tuple[float, float]],
    *,
    n_runs_per_case: int,
    use_crn: bool,
    rng: np.random.Generator,
    family: str = "sim",
) -> list[PairedCaseData]:
    """case_rates maps case_id -> (p_a, p_b), the true per-arm success
    probabilities. With use_crn=True, arm_a and arm_b for the same
    (case, trial) share one uniform draw u and threshold it against each
    arm's own probability — the standard way to implement CRN for Bernoulli
    outcomes, which induces positive correlation between the two arms'
    outcomes for that trial without changing either arm's marginal
    distribution (still Bernoulli(p_a) and Bernoulli(p_b) individually)."""
    data = []
    for case_id, (p_a, p_b) in case_rates.items():
        if use_crn:
            u = rng.uniform(0, 1, n_runs_per_case)
            a = tuple((u < p_a).astype(int).tolist())
            b = tuple((u < p_b).astype(int).tolist())
        else:
            a = tuple(rng.binomial(1, p_a, n_runs_per_case).tolist())
            b = tuple(rng.binomial(1, p_b, n_runs_per_case).tolist())
        data.append(
            PairedCaseData(case_id, family, CaseObservations(case_id, family, a), CaseObservations(case_id, family, b))
        )
    return data


@dataclass(frozen=True)
class CRNResult:
    n_sims: int
    n_cases: int
    n_runs_per_case: int
    var_without_crn: float
    var_with_crn: float
    variance_reduction_pct: float
    effective_sample_size_multiplier: float
    """How many times more runs an independent-draws design would need to
    match CRN's variance — e.g. 1.8 means CRN gets the same precision with
    ~1.8x fewer runs."""


def measure_crn_variance_reduction(
    case_rates: dict[str, tuple[float, float]],
    *,
    n_runs_per_case: int,
    n_sims: int = 2000,
    seed: int = 0,
) -> CRNResult:
    """Runs n_sims independent replicate experiments both with and without
    CRN (same case_rates, fresh random draws each replicate) and compares
    the Monte Carlo variance of the paired-rate-diff point estimator across
    replicates — a direct empirical measurement of what CRN buys you, not
    a theoretical claim."""
    rng_crn = np.random.default_rng(seed)
    rng_indep = np.random.default_rng(seed + 1)

    diffs_crn = np.empty(n_sims)
    diffs_indep = np.empty(n_sims)
    for i in range(n_sims):
        data_crn = simulate_paired_dataset(case_rates, n_runs_per_case=n_runs_per_case, use_crn=True, rng=rng_crn)
        data_indep = simulate_paired_dataset(case_rates, n_runs_per_case=n_runs_per_case, use_crn=False, rng=rng_indep)
        diffs_crn[i] = paired_rate_diff(data_crn)
        diffs_indep[i] = paired_rate_diff(data_indep)

    var_crn = float(np.var(diffs_crn, ddof=1))
    var_indep = float(np.var(diffs_indep, ddof=1))
    reduction_pct = 0.0 if var_indep == 0 else 100 * (1 - var_crn / var_indep)
    ess_multiplier = float("inf") if var_crn == 0 else var_indep / var_crn

    return CRNResult(
        n_sims=n_sims,
        n_cases=len(case_rates),
        n_runs_per_case=n_runs_per_case,
        var_without_crn=var_indep,
        var_with_crn=var_crn,
        variance_reduction_pct=reduction_pct,
        effective_sample_size_multiplier=ess_multiplier,
    )


# --- CUPED -------------------------------------------------------------


@dataclass(frozen=True)
class CupedResult:
    theta: float
    adjusted_values: list[float] = field(repr=False)
    mean_before: float
    mean_after: float
    var_before: float
    var_after: float
    variance_reduction_pct: float
    correlation: float


def cuped_adjust(values: list[float], covariate: list[float]) -> CupedResult:
    """Y_adjusted = Y - theta*(X - mean(X)), theta = Cov(Y,X)/Var(X).
    Unbiased (E[Y_adjusted] == E[Y] since E[X - mean(X)] == 0 by
    construction) and reduces variance by exactly rho(Y,X)^2 in
    expectation — the point of using each case's *baseline* rate as the
    covariate (per the build spec) is that a case's inherent difficulty is
    usually strongly correlated with its rate in any given arm, so a lot
    of the case-to-case variance is predictable and removable this way."""
    y = np.asarray(values, dtype=float)
    x = np.asarray(covariate, dtype=float)
    if len(y) != len(x):
        raise ValueError("values and covariate must be the same length")
    if len(y) < 2:
        raise ValueError("need at least 2 observations")

    var_x = float(np.var(x, ddof=1))
    if var_x == 0:
        theta = 0.0
    else:
        theta = float(np.cov(y, x, ddof=1)[0, 1] / var_x)

    x_bar = float(np.mean(x))
    adjusted = y - theta * (x - x_bar)

    var_before = float(np.var(y, ddof=1))
    var_after = float(np.var(adjusted, ddof=1))
    reduction_pct = 0.0 if var_before == 0 else 100 * (1 - var_after / var_before)
    correlation = 0.0 if var_x == 0 or var_before == 0 else float(np.corrcoef(y, x)[0, 1])

    return CupedResult(
        theta=theta,
        adjusted_values=adjusted.tolist(),
        mean_before=float(np.mean(y)),
        mean_after=float(np.mean(adjusted)),
        var_before=var_before,
        var_after=var_after,
        variance_reduction_pct=reduction_pct,
        correlation=correlation,
    )


def cuped_adjust_case_rates(
    data: list[PairedCaseData],
    baseline_rate: dict[str, float],
    *,
    arm: str = "b",
) -> CupedResult:
    """Convenience wrapper: adjusts one arm's per-case rate using an
    externally-supplied baseline rate per case (e.g. a historical/pilot
    measurement — deliberately not "the other arm in this same
    experiment", to avoid entangling this with whatever CRN pairing may
    already be doing to the arms)."""
    case_ids = [d.case_id for d in data if d.case_id in baseline_rate]
    if not case_ids:
        raise ValueError("no cases in `data` have a matching entry in baseline_rate")
    values = []
    covariate = []
    for d in data:
        if d.case_id not in baseline_rate:
            continue
        obs = d.arm_b if arm == "b" else d.arm_a
        if obs.n == 0:
            continue
        values.append(obs.rate)
        covariate.append(baseline_rate[d.case_id])
    return cuped_adjust(values, covariate)
