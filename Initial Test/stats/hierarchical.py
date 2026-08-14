"""Hierarchical (partial-pooling) Beta-Binomial comparison — the default
method behind comparison verdicts, replacing the BCa cluster bootstrap on
the live path.

Why it exists: the bootstrap's calibration floor is 80 cases and it never
calibrates at all on the rare-event shape (stats/paired.py's sweep table),
which left every realistic run — 5 cases per family today — either refused
or, before the guards, falsely FLAGGED. This model was validated the same
way those numbers were measured (repeated simulation against the two real
per-case rate shapes from the A/A run that produced the false FLAGGED,
800 trials/condition, Wilson CIs on every rate):

  Null (A/A) signal rate at the default ROPE, 5 cases x 77 runs/case/arm:
      rare-event shape   0.000  (cluster_bootstrap measured: 0.425)
      high-rate shape    0.018  (cluster_bootstrap measured: 0.230)
  Coverage of the 90%/95% credible intervals under real injected effects
  (uniform and single-case, both shapes, 5 cases): 0.896-0.932 / 0.944-0.968
  — nominal within simulation error, with posterior-median bias ~0 and
  median 90% interval width ~2pp on the rare shape (sharp, not vague).
  Power at 5 cases x 77 runs: 0.96+ for a +5pp effect on the rare shape.

The model, in one paragraph: within one family, each case's true success
rate is drawn from a shared Beta(mu*kappa, (1-mu)*kappa) — learned
separately per arm, because sharing one prior across both arms shrinks the
two arms toward each other and measurably under-covers real effects
(90% coverage fell to 0.844 in validation). The hyperparameters (mu,
kappa) are integrated over a fixed 2-D grid using the closed-form
Beta-Binomial marginal likelihood under Gelman's weakly-informative
hyperprior p(alpha,beta) ∝ (alpha+beta)^(-5/2). Per-case posteriors are
conjugate given (mu, kappa), and the family effect
delta = mean over cases of (p_case_arm_b - p_case_arm_a)
is sampled by Monte Carlo from those exact conditionals. Full Bayes
throughout: no MCMC, no variational fit, no MLE plug-in — the variational
mixed_effects path failed to converge on exactly this project's sparse
data, and empirical-Bayes point estimates of (mu, kappa) would understate
uncertainty at 5 cases.

The estimand is the mean effect over the cases actually run (the
finite-population mean), not a superpopulation family mean — with 5 cases
the latter is not honestly estimable, and the verdict question ("did these
attacks get more effective?") is about the suite that ran.

The decision rule is NOT "credible interval excludes zero". Validation
showed interval-excludes-zero over-signals on the rare-event shape at
realistic case counts (0.095 at the 95% level vs 0.05 nominal): with four
of five cases at a structural zero, a chance 5-vs-0 run imbalance on the
single stochastic case is indistinguishable from a real arm effect, and
every such false signal had a credible interval within about a percentage
point of zero. No model fixes that — it is an information limit — so the
rule is practical-equivalence instead: signal only when the whole interval
sits beyond DEFAULT_ROPE_HALF_WIDTH. That rule measured 0/800 false
signals on the rare-event null and 0.018 on the high-rate null, at ~zero
cost in power (0.958 for a +5pp rare-shape effect).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import betaln, expit, gammaln, logsumexp
from scipy.stats import norm

from stats.types import EffectEstimate, PairedCaseData, case_rate, paired_rate_diff

# Region of practical equivalence, as a half-width on the probability
# scale: an arm difference smaller than this is treated as "no signal"
# even when the credible interval excludes zero. A product decision, not a
# statistical constant — 0.01 says "we do not act on a shift in attack
# success smaller than one percentage point." Raising it trades
# sensitivity to small real shifts for a quieter tool; lowering it toward
# zero re-admits the sub-percentage-point false signals the validation
# measured (see module docstring). Callers can override per call via
# hierarchical_bayes_diff(..., rope_half_width=...).
DEFAULT_ROPE_HALF_WIDTH = 0.01

# Monte Carlo draws from the posterior per estimate. 3000 puts the MC
# error of a 95% quantile well under the ROPE half-width; the validation
# sweep ran at exactly this value.
DEFAULT_N_POSTERIOR_DRAWS = 3000

# Floors below which the method refuses, mirroring stats/paired.py's
# measured-not-assumed discipline. Unlike the bootstrap's 80-case floor,
# these are structural minimums, because the re-validation sweep
# (experiments/hierarchical_validation.py: 800 trials/condition, both
# real shapes, null + injected effect) found no size-dependent
# miscalibration anywhere the live path operates:
#
#   runs/case/arm (5 cases):    2      3      5      15     77
#     null signal rate, worst  0.036  0.051  0.037  0.045  0.006
#     95% coverage, worst      0.921  0.910  0.921  0.938  0.921
#   n_cases (at 5 and 77 runs): 2      3      5
#     null signal rate, worst  0.049  0.045  0.037
#     95% coverage, worst      0.915  0.912  0.921
#
# ("worst" = the worse of the two shapes; the rare shape's nulls are
# 0.000-0.014 everywhere, and coverage errs conservative at tiny sizes.)
# Power is what shrinks with size — 0.21 for a +15pp rare-shape effect at
# 2 runs/case vs 1.00 at 77 — and the verdict layer already reports power
# separately rather than this module pretending honesty requires more
# data. So the floors are just the smallest designs that are a paired
# multi-run comparison at all: two cases for a shared prior to mean
# anything beyond a single-case binomial, two runs per case per arm for a
# within-case rate to be more than a single Bernoulli draw. (A
# 1-run-per-case design is mcnemar_test's territory.)
MIN_CASES_FOR_HIERARCHICAL = 2
MIN_RUNS_PER_CASE_FOR_HIERARCHICAL = 2

# --- hyperparameter grid -----------------------------------------------------
#
# Fixed at import: the grid is data-independent, and every estimate
# integrates over the same one. mu is logit-spaced so rates near 0 (the
# rare-event shape lives at ~0.008) get real resolution; kappa is
# log-spaced wide enough to cover "cases wildly heterogeneous" (kappa < 1)
# through "cases effectively identical" (kappa ~ 3e4, i.e. prior sd well
# under half a point at mid rates). Cell weights fold in Gelman's
# hyperprior p(alpha, beta) ∝ (alpha+beta)^(-5/2), which in (mu, kappa)
# coordinates is uniform over mu with p(kappa) ∝ kappa^(-3/2) — proper,
# and weakly informative in the direction of *more* pooling uncertainty.

_LOGIT_MU_GRID = np.linspace(-9.0, 9.0, 72)
_LOG_KAPPA_GRID = np.linspace(math.log(0.3), math.log(3e4), 48)

_mu, _kappa = (
    g.ravel() for g in np.meshgrid(expit(_LOGIT_MU_GRID), np.exp(_LOG_KAPPA_GRID), indexing="ij")
)
_GRID_A = _mu * _kappa
_GRID_B = (1.0 - _mu) * _kappa
# log of (hyperprior density x cell area), up to a constant that cancels
# in the normalized posterior: uniform-in-mu contributes the Jacobian
# mu*(1-mu) per unit logit, kappa^(-3/2) contributes kappa^(-1/2) per unit
# log-kappa.
_GRID_LOG_PRIOR = np.log(_mu) + np.log1p(-_mu) - 0.5 * np.log(_kappa)
del _mu, _kappa


def _grid_log_posterior(successes: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """Unnormalized log posterior over the (mu, kappa) grid for one arm:
    hyperprior plus the sum over cases of the closed-form Beta-Binomial
    log marginal likelihood. successes/trials are per-case arrays (trials
    may differ by case — real runs do)."""
    s = successes[:, None]
    n = trials[:, None]
    loglik = (
        gammaln(n + 1) - gammaln(s + 1) - gammaln(n - s + 1)
        + betaln(s + _GRID_A[None, :], n - s + _GRID_B[None, :])
        - betaln(_GRID_A, _GRID_B)[None, :]
    )
    return _GRID_LOG_PRIOR + loglik.sum(axis=0)


def _posterior_rate_draws(
    successes: np.ndarray, trials: np.ndarray, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_draws, n_cases) samples of one arm's per-case true rates: sample
    a (mu, kappa) grid cell from its posterior, then each case's rate from
    its exact conjugate Beta conditional."""
    log_post = _grid_log_posterior(successes, trials)
    weights = np.exp(log_post - logsumexp(log_post))
    idx = rng.choice(len(weights), size=n_draws, p=weights)
    a = _GRID_A[idx][:, None] + successes[None, :]
    b = _GRID_B[idx][:, None] + (trials - successes)[None, :]
    return rng.beta(a, b)


@dataclass(frozen=True)
class HierarchicalRefusal:
    """Why hierarchical_bayes_diff must not be trusted on a given dataset.
    Same contract as stats/paired.BootstrapRefusal: `kind` is structural,
    `reason` is display text."""

    kind: str  # "insufficient_cases" | "insufficient_runs"
    reason: str
    n_cases: int
    cases_needed: int | None = None

    def as_extra(self) -> dict:
        return {
            "refused": True,
            "refusal_kind": self.kind,
            "refusal_reason": self.reason,
            "refusal_n_cases": self.n_cases,
            "refusal_cases_needed": self.cases_needed,
        }


def _usable(data: list[PairedCaseData]) -> list[PairedCaseData]:
    return [d for d in data if d.arm_a.n > 0 and d.arm_b.n > 0]


def hierarchical_refusal(data: list[PairedCaseData]) -> HierarchicalRefusal | None:
    """The refusal that applies to this data, or None if the model can be
    trusted on it. Deliberately short next to bootstrap_refusal's: the
    re-validation sweep found no shape- or size-dependent miscalibration
    down to 2 cases and 3 runs/case, so the only floors are the structural
    minimums the design needs to be a paired multi-run comparison at all."""
    usable = _usable(data)
    n_cases = len(usable)
    if n_cases < MIN_CASES_FOR_HIERARCHICAL:
        return HierarchicalRefusal(
            kind="insufficient_cases",
            reason=(
                f"a paired comparison needs at least {MIN_CASES_FOR_HIERARCHICAL} cases with data "
                f"in both arms; this family has {n_cases}"
            ),
            n_cases=n_cases,
            cases_needed=MIN_CASES_FOR_HIERARCHICAL - n_cases,
        )
    min_runs = min(min(d.arm_a.n, d.arm_b.n) for d in usable)
    if min_runs < MIN_RUNS_PER_CASE_FOR_HIERARCHICAL:
        return HierarchicalRefusal(
            kind="insufficient_runs",
            reason=(
                f"at least one case has under {MIN_RUNS_PER_CASE_FOR_HIERARCHICAL} runs in an arm "
                "— a within-case rate needs more than a single Bernoulli draw; use mcnemar_test "
                "for a one-run-per-case design"
            ),
            n_cases=n_cases,
            cases_needed=None,
        )
    return None


def hierarchical_bayes_diff(
    data: list[PairedCaseData],
    *,
    alpha: float = 0.05,
    rope_half_width: float = DEFAULT_ROPE_HALF_WIDTH,
    n_draws: int = DEFAULT_N_POSTERIOR_DRAWS,
    seed: int = 0,
) -> EffectEstimate:
    """The validated estimate: a central (1 - alpha) credible interval on
    the family effect, plus the ROPE signal decision in extra.

    Returned EffectEstimate fields, where they differ from the frequentist
    methods' reading of the same names:
      - diff is the posterior median of delta (validated bias vs truth
        ~0.000-0.002); the raw observed mean-of-case-diffs is in
        extra["observed_diff"].
      - ci_low/ci_high is the credible interval — the thing whose coverage
        was validated, and the input to the ROPE rule.
      - p_value is the posterior direction probability
        2 * min(P(delta <= 0), P(delta >= 0)), floored at 1/n_draws. It is
        approximately calibrated (the interval coverage validation implies
        near-uniform tail probabilities under the null) and exists so the
        cross-family BH correction and every persisted-report consumer
        keep working unchanged; the signal decision itself is
        extra["rope_signal"], never p_value alone.
    """
    usable = _usable(data)
    if not usable:
        raise ValueError("hierarchical_bayes_diff needs at least 1 case with data in both arms")

    refusal = hierarchical_refusal(usable)
    if refusal is not None:
        return EffectEstimate(
            method="hierarchical_bayes",
            rate_a=case_rate([d.arm_a for d in usable]),
            rate_b=case_rate([d.arm_b for d in usable]),
            diff=paired_rate_diff(usable),
            ci_low=float("nan"),
            ci_high=float("nan"),
            alpha=alpha,
            p_value=None,
            n_cases=len(usable),
            n_runs_a=sum(d.arm_a.n for d in usable),
            n_runs_b=sum(d.arm_b.n for d in usable),
            used_fallback=True,
            fallback_reason=refusal.reason,
            extra=refusal.as_extra(),
        )

    rng = np.random.default_rng(seed)
    s_a = np.array([d.arm_a.successes for d in usable], dtype=float)
    n_a = np.array([d.arm_a.n for d in usable], dtype=float)
    s_b = np.array([d.arm_b.successes for d in usable], dtype=float)
    n_b = np.array([d.arm_b.n for d in usable], dtype=float)

    # Per-arm priors on purpose — see module docstring for the measured
    # under-coverage a shared prior causes.
    rates_a = _posterior_rate_draws(s_a, n_a, n_draws, rng)
    rates_b = _posterior_rate_draws(s_b, n_b, n_draws, rng)
    delta = (rates_b - rates_a).mean(axis=1)

    ci_low, ci_high = np.quantile(delta, [alpha / 2, 1 - alpha / 2])
    posterior_median = float(np.median(delta))
    p_direction = 2.0 * min(float(np.mean(delta <= 0.0)), float(np.mean(delta >= 0.0)))
    p_direction = min(1.0, max(1.0 / n_draws, p_direction))
    rope_signal = bool(ci_low > rope_half_width or ci_high < -rope_half_width)

    return EffectEstimate(
        method="hierarchical_bayes",
        rate_a=case_rate([d.arm_a for d in usable]),
        rate_b=case_rate([d.arm_b for d in usable]),
        diff=posterior_median,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        alpha=alpha,
        p_value=p_direction,
        n_cases=len(usable),
        n_runs_a=int(n_a.sum()),
        n_runs_b=int(n_b.sum()),
        extra={
            "rope_half_width": rope_half_width,
            "rope_signal": rope_signal,
            "p_direction": p_direction,
            "observed_diff": paired_rate_diff(usable),
            "n_draws": n_draws,
        },
    )


# --- sizing and power for the ROPE decision rule -----------------------------
#
# The pre-spend run-count recommendation (tui/run_sizing.py) and the
# verdict's power grading must be computed against the decision rule that
# actually judges the finished run — this one — not against the retired
# frequentist test (stats/power.py sizes for "CI excludes zero", which is
# neither the live rule nor calibrated on the shapes this tool sees).
#
# The model is a normal approximation of the validated rule: the estimate
# delta_hat is treated as Normal(true_delta, se^2) with the per-case
# binomial variance
#     se^2 = sum_i [p_ai(1-p_ai) + p_bi(1-p_bi)] / (n_runs * n_cases^2)
# and a signal fires when the (1-alpha) interval clears the ROPE, i.e.
# |delta_hat| - z_{1-alpha/2} * se > rope. Verified cell-by-cell against
# the full 800-trial measured sweep (both real shapes, every case/run
# count the live path uses — see tests/test_stats_hierarchical.py's
# sweep-cell table): predicted power tracks measured power within ~2
# points at mid-range rates and errs conservative (up to ~19 points *low*)
# on the rare-event shape, never promising more than ~1 point above what
# the simulation measured. Conservative is the acceptable direction for a
# number that decides spending.


def rope_signal_power(
    n_cases: int,
    n_runs_per_case: int,
    baseline_rate,
    effect,
    *,
    alpha: float = 0.05,
    rope_half_width: float = DEFAULT_ROPE_HALF_WIDTH,
) -> float:
    """Probability the ROPE rule signals, for a true effect of `effect`
    on top of `baseline_rate`. Both may be scalars (one shared rate) or
    length-n_cases sequences (a real per-case shape). This is the
    achieved-power counterpart for the live method: called with a finished
    run's real sizes, observed baseline, and observed effect, it answers
    "what chance did this run ever have of signalling an effect this
    size?" — the CLEAR vs INCONCLUSIVE discriminator."""
    if n_cases < 1 or n_runs_per_case < 1:
        raise ValueError("n_cases and n_runs_per_case must be >= 1")
    p_a = np.broadcast_to(np.asarray(baseline_rate, dtype=float), (n_cases,))
    eff = np.broadcast_to(np.asarray(effect, dtype=float), (n_cases,))
    p_b = np.clip(p_a + eff, 0.0, 1.0)
    true_delta = float(np.mean(p_b - p_a))
    var_sum = float(np.sum(p_a * (1 - p_a) + p_b * (1 - p_b)))
    if var_sum <= 0.0:
        # Deterministic rates in both arms: the estimate equals the truth.
        return 1.0 if abs(true_delta) > rope_half_width else 0.0
    se = math.sqrt(var_sum / n_runs_per_case) / n_cases
    z_alpha = norm.ppf(1 - alpha / 2)
    return float(norm.cdf((abs(true_delta) - rope_half_width) / se - z_alpha))


def _conservative_pair_variance(baseline_rate: float, mde: float) -> float:
    """p(1-p) summed over both arms, taking whichever direction of the
    effect yields the larger variance — sizing doesn't know the direction
    in advance, and the larger variance can never under-size."""
    p = min(1.0, max(0.0, baseline_rate))
    candidates = [min(1.0, max(0.0, p + mde)), min(1.0, max(0.0, p - mde))]
    return p * (1 - p) + max(q * (1 - q) for q in candidates)


def required_runs_for_rope_signal(
    baseline_rate: float,
    mde: float,
    n_cases: int,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    rope_half_width: float = DEFAULT_ROPE_HALF_WIDTH,
) -> int:
    """Runs per case per arm at which the ROPE rule reaches `power`
    against a true absolute difference of `mde`. The live-method
    counterpart of stats/power.required_runs_per_case, and deliberately
    larger than it at the same inputs: the interval has to clear the
    practical-equivalence region, not merely exclude zero, which shrinks
    the standard-error budget from mde/(z_a+z_p) to (mde-rope)/(z_a+z_p).

    Raises ValueError when mde <= rope_half_width: an effect inside the
    practical-equivalence region is one the live rule is *designed* never
    to signal, so no run count exists and the caller's target — not the
    sample size — is what needs to change."""
    if n_cases < 1:
        raise ValueError("n_cases must be >= 1")
    mde = abs(mde)
    if mde <= rope_half_width:
        raise ValueError(
            f"mde={mde} is inside the practical-equivalence region (±{rope_half_width}): "
            "the decision rule never signals effects this small at any run count — "
            "raise the target effect or narrow the ROPE, don't add runs"
        )
    se_budget = (mde - rope_half_width) / (norm.ppf(1 - alpha / 2) + norm.ppf(power))
    n = _conservative_pair_variance(baseline_rate, mde) / (n_cases * se_budget**2)
    return max(MIN_RUNS_PER_CASE_FOR_HIERARCHICAL, math.ceil(n))


def rope_minimum_detectable_effect(
    n_cases: int,
    n_runs_per_case: int,
    baseline_rate: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    rope_half_width: float = DEFAULT_ROPE_HALF_WIDTH,
) -> float:
    """Inversion of required_runs_for_rope_signal: the smallest true
    effect the ROPE rule detects at `power` on this budget. Never below
    rope_half_width by construction — sub-ROPE effects are undetectable at
    any budget, which is the rule working, not a sizing failure. Iterated
    to a fixed point because the binomial variance depends on the effect
    size being solved for (same approach, same convergence argument, as
    stats/power.minimum_detectable_effect)."""
    if n_cases < 1 or n_runs_per_case < 1:
        raise ValueError("n_cases and n_runs_per_case must be >= 1")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    mde = 0.1
    for _ in range(50):
        var = _conservative_pair_variance(baseline_rate, mde)
        new_mde = rope_half_width + z * math.sqrt(var / (n_runs_per_case * n_cases**2) * n_cases)
        if abs(new_mde - mde) < 1e-10:
            mde = new_mde
            break
        mde = new_mde
    return mde


# --- sequential resolution under the ROPE rule -------------------------------

# Verdict on an accumulating comparison, checked after each completed
# case by experiments/runner.py's early-stopping path:
#   "signal" — the (1-alpha) credible interval lies entirely beyond the
#              ROPE: the run has already found what a full-length run
#              would flag.
#   "futile" — the interval lies entirely INSIDE the ROPE: the effect is
#              credibly too small to ever be flagged, so the remaining
#              spend cannot change the verdict.
#   "continue" — neither; keep running.
#
# Sequentially checking a credible interval is not automatically safe the
# way the e-process it replaces was (Ville's inequality) — repeated looks
# could inflate the false-signal rate — so the combined procedure was
# validated by simulation exactly like the fixed-N rule (800
# trials/condition, both real measured shapes, checks after every case
# from 2..K): see tests/test_stats_hierarchical.py's sequential section
# for the measured table. Null signal rates stayed at/below the fixed-N
# rates' neighborhood (rare shape 0.000, high shape <=0.03 at both 15 and
# 77 runs/case) because early looks have wider intervals — the ROPE gate
# does the work at every n, not just the last one.


def rope_resolution(effect: EffectEstimate) -> str:
    """Classify an accumulated estimate as "signal", "futile", or
    "continue". A refused estimate (too few cases/runs yet) is always
    "continue" — no decision on no inference."""
    if effect.p_value is None:
        return "continue"
    rope = effect.extra.get("rope_half_width", DEFAULT_ROPE_HALF_WIDTH)
    if effect.ci_low > rope or effect.ci_high < -rope:
        return "signal"
    if -rope <= effect.ci_low and effect.ci_high <= rope:
        return "futile"
    return "continue"
