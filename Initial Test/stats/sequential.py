"""Always-valid inference: two independent ways to look at accumulating
data as many times as you want without inflating the false-positive rate,
which a fixed-N test (repeatedly peeked at) does not give you.

1. mixture_sprt_confidence_sequence — an e-process / confidence sequence
   (Johari, Pekelis & Walsh 2017; the normal-mixture / mSPRT construction).
   Anytime-valid: the interval is valid to read at every n simultaneously,
   not just at one pre-committed n. This is the tool's differentiator —
   "stop now, the evidence is sufficient" said honestly.
2. group_sequential_boundaries — the more classical alternative: Lan-DeMets
   error-spending boundaries (O'Brien-Fleming / Pocock) at a small number
   of pre-specified looks, computed by the standard recursive numerical
   integration over the canonical joint distribution of the sequence of
   standardized statistics (which is Brownian motion in information time).

Both operate on the sequence of per-case rate differences, one entry per
case in accumulation order — the observation unit is a case, not a run,
so this respects clustering the same way the rest of stats/ does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# --- mixture SPRT confidence sequence -------------------------------------


@dataclass(frozen=True)
class ConfidenceSequencePoint:
    n: int
    center: float
    ci_low: float
    ci_high: float
    e_value: float
    always_valid_p: float
    reject_null: bool


@dataclass(frozen=True)
class ConfidenceSequenceResult:
    points: list[ConfidenceSequencePoint]
    alpha: float
    tau: float
    sigma: float
    first_stop_index: int | None

    def can_stop_now(self) -> bool:
        return self.first_stop_index is not None

    def summary(self) -> str:
        if not self.points:
            return "no data"
        last = self.points[-1]
        if self.can_stop_now():
            return (
                f"stoppable at n={self.first_stop_index}: evidence sufficient "
                f"(e-value {self.points[self.first_stop_index - 1].e_value:.2f} >= {1 / self.alpha:.2f}). "
                f"Current (n={last.n}) estimate {last.center:+.3f}, always-valid 95% CS "
                f"[{last.ci_low:+.3f}, {last.ci_high:+.3f}]."
            )
        return (
            f"not yet stoppable at n={last.n}: e-value {last.e_value:.2f} < {1 / self.alpha:.2f}. "
            f"Current estimate {last.center:+.3f}, always-valid 95% CS [{last.ci_low:+.3f}, {last.ci_high:+.3f}]."
        )


def mixture_sprt_confidence_sequence(
    case_diffs: list[float],
    *,
    alpha: float = 0.05,
    tau: float = 0.05,
    sigma: float | None = None,
) -> ConfidenceSequenceResult:
    """Builds the confidence sequence incrementally over case_diffs, in the
    order given (i.e. the order cases actually completed — this is what
    "stop now" means operationally: after however many cases have reported
    so far).

    tau is the analyst's prior scale for a plausible true effect (in rate-
    difference units) — not estimated from data, a genuine design choice;
    the default (5 percentage points) is a reasonable guess for this
    system's outcomes, not a universal constant. sigma is the per-case
    observation SD; if not supplied it's plugged in from case_diffs itself
    (sample SD over all cases seen so far, with a small floor). This is an
    approximation to the theory (which assumes sigma known/fixed in
    advance) — standard practice in real sequential-testing tools, and
    flagged here rather than silently treated as exact.
    """
    if len(case_diffs) < 2:
        raise ValueError("need at least 2 case_diffs to estimate variance for the confidence sequence")

    arr = np.asarray(case_diffs, dtype=float)
    if sigma is None:
        sigma_est = max(float(np.std(arr, ddof=1)), 0.01)
    else:
        sigma_est = sigma

    threshold = 1.0 / alpha
    z_crit = norm.ppf(1 - alpha / 2)

    points: list[ConfidenceSequencePoint] = []
    first_stop: int | None = None
    running_sum = 0.0
    for i, x in enumerate(arr, start=1):
        running_sum += x
        n = i
        var0 = sigma_est**2
        mix_var = var0 + n * tau**2

        # Λ_n: the mixture likelihood ratio (Bayes factor) for H0: mean=0
        # vs. a N(0, tau^2) mixture of alternatives, given a N(0, sigma^2)
        # per-observation model. E[Λ_n] = 1 under H0 for every n, making it
        # a nonnegative martingale — Ville's inequality gives the anytime-
        # valid guarantee (Johari, Pekelis & Walsh 2017, eq. 3-4).
        log_e = 0.5 * math.log(var0 / mix_var) + (tau**2 * running_sum**2) / (2 * var0 * mix_var)
        # math.exp raises OverflowError above ~709 rather than returning
        # inf (unlike e.g. numpy) — a real crash with only a few cases of
        # a clear-cut effect (caught via a dashboard-backfill report with
        # just 2 cases). The martingale has unambiguously diverged well
        # before log_e gets anywhere near that threshold, so clamping to
        # inf there is not an approximation of the math, just a
        # representation fix for a value that's already "certain" evidence.
        e_value = math.inf if log_e > 700 else math.exp(log_e)
        always_valid_p = min(1.0, 1.0 / e_value)
        reject = e_value >= threshold

        center = running_sum / n
        # Confidence sequence half-width: invert Λ_n(θ0) = 1/alpha for θ0 —
        # see module docstring / build notes for the closed-form derivation.
        log_arg = math.log(threshold * math.sqrt(mix_var / var0))
        if log_arg <= 0:
            half_width = float("inf")
        else:
            half_width = (1.0 / n) * math.sqrt((2 * var0 * mix_var / tau**2) * log_arg)

        points.append(
            ConfidenceSequencePoint(
                n=n,
                center=center,
                ci_low=center - half_width,
                ci_high=center + half_width,
                e_value=e_value,
                always_valid_p=always_valid_p,
                reject_null=reject,
            )
        )
        if reject and first_stop is None:
            first_stop = n

    return ConfidenceSequenceResult(points=points, alpha=alpha, tau=tau, sigma=sigma_est, first_stop_index=first_stop)


# --- group-sequential alpha spending ---------------------------------------


def _obrien_fleming_spending(t: float, alpha: float) -> float:
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    z = norm.ppf(1 - alpha / 2)
    return float(2 * (1 - norm.cdf(z / math.sqrt(t))))


def _pocock_spending(t: float, alpha: float) -> float:
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    return alpha * math.log(1 + (math.e - 1) * t)


_SPENDING_FUNCTIONS = {"obrien_fleming": _obrien_fleming_spending, "pocock": _pocock_spending}


@dataclass(frozen=True)
class GroupSequentialDesign:
    spending: str
    alpha: float
    information_fractions: list[float]
    boundaries: list[float]  # z-scale critical values, one per look

    def evaluate(self, z_stats: list[float]) -> "GroupSequentialResult":
        if len(z_stats) > len(self.boundaries):
            raise ValueError("more z_stats than planned looks")
        stop_at: int | None = None
        for i, (z, b) in enumerate(zip(z_stats, self.boundaries), start=1):
            if abs(z) >= b:
                stop_at = i
                break
        return GroupSequentialResult(design=self, z_stats=list(z_stats), stop_at_look=stop_at)


@dataclass(frozen=True)
class GroupSequentialResult:
    design: GroupSequentialDesign
    z_stats: list[float]
    stop_at_look: int | None

    def can_stop_now(self) -> bool:
        return self.stop_at_look is not None


def group_sequential_boundaries(
    information_fractions: list[float],
    *,
    alpha: float = 0.05,
    spending: Literal["obrien_fleming", "pocock"] = "obrien_fleming",
    grid_half_width: float = 8.0,
    grid_points: int = 4001,
) -> GroupSequentialDesign:
    """Lan-DeMets error-spending boundaries via the standard recursive
    numerical integration (Armitage, McPherson & Rowe 1969; Lan & DeMets
    1983): the sequence of cumulative-sum statistics B(t_k) (t_k =
    information fraction at look k) behaves as Brownian motion, so each
    boundary is found by integrating the "still in the continuation
    region" density forward one independent increment at a time and
    solving for the b_k that spends exactly the next increment of alpha.

    Validated against Jennison & Turnbull's published reference values for
    K=5 equally-spaced O'Brien-Fleming boundaries (approx. 4.56, 3.23,
    2.63, 2.28, 2.03) in tests/test_sequential.py.
    """
    if information_fractions != sorted(information_fractions):
        raise ValueError("information_fractions must be increasing")
    if information_fractions[-1] != 1.0:
        raise ValueError("final information_fraction must be 1.0")

    spend_fn = _SPENDING_FUNCTIONS[spending]
    grid = np.linspace(-grid_half_width, grid_half_width, grid_points)
    dx = grid[1] - grid[0]

    boundaries: list[float] = []
    # density (unnormalized: only the continuation-region mass) of B(t_k)
    density = None
    prev_t = 0.0
    cum_alpha_spent = 0.0

    for t in information_fractions:
        dt = t - prev_t
        if density is None:
            density = norm.pdf(grid, loc=0.0, scale=math.sqrt(t))
        else:
            # Convolve the previous (truncated to its continuation region)
            # density with the independent N(0, dt) increment.
            kernel = norm.pdf(grid, loc=0.0, scale=math.sqrt(dt))
            density = np.convolve(density, kernel, mode="same") * dx

        target_cum = spend_fn(t, alpha)
        target_increment = target_cum - cum_alpha_spent

        def tail_mass(b: float, density=density) -> float:
            # Two disjoint regions (x <= -b and x >= b) — integrate each
            # separately. A single trapz() call over a boolean-masked
            # concatenation of both would silently bridge the gap between
            # them with one spurious trapezoid spanning the entire
            # continuation region, wildly overstating the tail mass
            # (caught via a reference-value check against Jennison &
            # Turnbull's published O'Brien-Fleming boundaries).
            left_mask = grid <= -b
            right_mask = grid >= b
            left = float(np.trapezoid(density[left_mask], grid[left_mask])) if left_mask.any() else 0.0
            right = float(np.trapezoid(density[right_mask], grid[right_mask])) if right_mask.any() else 0.0
            return left + right

        if target_increment <= 0:
            b = grid_half_width
        elif tail_mass(0.0) <= target_increment:
            # Spending this increment would consume essentially the whole
            # remaining mass — boundary collapses to 0 (immediate stop).
            b = 0.0
        else:
            try:
                b = brentq(lambda bb: tail_mass(bb) - target_increment, 0.0, grid_half_width - dx)
            except ValueError:
                b = grid_half_width

        # Truncate density to the continuation region for the next step —
        # must happen on the raw B(t) scale, before converting b to the
        # standardized Z(t) = B(t)/sqrt(t) scale used for reporting.
        density = np.where(np.abs(grid) < b, density, 0.0)
        boundaries.append(b / math.sqrt(t))
        cum_alpha_spent = target_cum
        prev_t = t

    return GroupSequentialDesign(spending=spending, alpha=alpha, information_fractions=list(information_fractions), boundaries=boundaries)
