"""Power analysis for the paired, case-clustered design used throughout
stats/. The number that determines whether this whole tool is economically
viable, per the build spec, so it's surfaced as two directly usable
questions:
  - required_runs_per_case: given a case suite of a certain size, how many
    runs per case per arm do I need to reliably detect a given effect?
  - minimum_detectable_effect: given a fixed run budget, what's the
    smallest effect I could actually detect?

The variance model folds in "cases differ enormously in difficulty"
(build spec) as between-case heterogeneity: the true effect isn't assumed
identical across cases, it's treated as varying with standard deviation
`between_case_sd` around the average effect. That term does NOT shrink as
you add more runs per case — it only shrinks with more cases — so a
regression driven mostly by case heterogeneity (rather than per-case
binomial noise) will show up here as "you need more cases, not more runs,"
which required_runs_per_case reports explicitly via HeterogeneityDominates
rather than silently returning a useless answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


class HeterogeneityDominates(Exception):
    """Raised when between-case variance alone already exceeds what's
    tolerable for the requested power/MDE — no number of runs per case can
    fix this; more (or less heterogeneous) cases can."""


@dataclass(frozen=True)
class PowerInputs:
    baseline_rate: float
    mde: float
    n_cases: int
    power: float = 0.8
    alpha: float = 0.05
    between_case_sd: float = 0.0


def _per_run_variance(baseline_rate: float, mde: float) -> float:
    p1 = baseline_rate
    p2 = min(0.999, max(0.001, baseline_rate + mde))
    return p1 * (1 - p1) + p2 * (1 - p2)


def _target_mean_diff_variance(mde: float, n_cases: int, power: float, alpha: float) -> float:
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    return n_cases * (mde / (z_alpha + z_power)) ** 2


def required_runs_per_case(
    baseline_rate: float,
    mde: float,
    n_cases: int,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    between_case_sd: float = 0.0,
) -> int:
    """Runs needed per case, per arm, to detect `mde` (an absolute rate
    difference) with the given power — using the average of per-case rate
    differences over n_cases cases as the estimator (see
    stats.types.paired_rate_diff), the same one cluster_bootstrap_diff /
    mixed_effects_diff target.

    Raises HeterogeneityDominates if between_case_sd alone (independent of
    n) already exceeds the tolerable variance for this n_cases/mde/power
    combination.
    """
    if n_cases < 1:
        raise ValueError("n_cases must be >= 1")
    per_run_var = _per_run_variance(baseline_rate, mde)
    target_var = _target_mean_diff_variance(mde, n_cases, power, alpha)
    between_var = between_case_sd**2

    if between_var >= target_var:
        raise HeterogeneityDominates(
            f"between-case variance ({between_var:.6f}) alone meets or exceeds the "
            f"tolerable variance ({target_var:.6f}) for mde={mde}, n_cases={n_cases}, "
            f"power={power}. No number of runs per case can achieve this power — "
            "you need more cases (or less heterogeneous ones), not more runs."
        )

    n = per_run_var / (target_var - between_var)
    return max(1, math.ceil(n))


def minimum_detectable_effect(
    n_cases: int,
    n_runs_per_case: int,
    baseline_rate: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    between_case_sd: float = 0.0,
) -> float:
    """Inversion: given a fixed budget (n_cases x n_runs_per_case x 2
    arms), the smallest effect detectable at the given power. Solved in
    closed form rather than by searching, since for a fixed mde the
    variance model is linear in the quantities being solved for — but mde
    appears on both sides here (it sets p2 = p1+mde, which sets
    per_run_var), so this iterates a few times to a fixed point rather
    than being a single closed-form expression; converges in a handful of
    iterations because per_run_var is bounded and smooth in mde.
    """
    if n_cases < 1 or n_runs_per_case < 1:
        raise ValueError("n_cases and n_runs_per_case must be >= 1")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    between_var = between_case_sd**2

    mde = 0.1  # initial guess
    for _ in range(50):
        per_run_var = _per_run_variance(baseline_rate, mde)
        var_mean_diff = per_run_var / n_runs_per_case / n_cases + between_var / n_cases
        new_mde = (z_alpha + z_power) * math.sqrt(var_mean_diff)
        if abs(new_mde - mde) < 1e-10:
            mde = new_mde
            break
        mde = new_mde
    return mde


def power_curve(
    n_cases: int,
    baseline_rate: float,
    runs_per_case_grid: list[int],
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    between_case_sd: float = 0.0,
) -> list[tuple[int, float]]:
    """(runs_per_case, minimum_detectable_effect) pairs — the dashboard's
    power-curve panel plots this directly."""
    return [
        (n, minimum_detectable_effect(n_cases, n, baseline_rate, power=power, alpha=alpha, between_case_sd=between_case_sd))
        for n in runs_per_case_grid
    ]
