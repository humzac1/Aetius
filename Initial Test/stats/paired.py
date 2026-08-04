"""Paired comparison of two arms over the same set of attack cases.

Three methods, in increasing order of modeling assumptions:
  - cluster_bootstrap_diff: fewest assumptions, resamples cases (not runs).
  - mcnemar_test: exact/asymptotic test for the simple one-run-per-arm-per-
    case design.
  - mixed_effects_diff: a logistic model with a random intercept per case,
    for when you want a model-based effect size and have enough cases for
    it to be stable — falls back to the bootstrap otherwise rather than
    reporting a fragile fit.

All three return an EffectEstimate on the same scale (rate_b - rate_a, a
probability-scale difference) so reporting.py can treat them uniformly.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar

from stats.types import EffectEstimate, PairedCaseData, case_rate, paired_rate_diff

DEFAULT_N_BOOT = 5000


def _usable(data: list[PairedCaseData]) -> list[PairedCaseData]:
    return [d for d in data if d.arm_a.n > 0 and d.arm_b.n > 0]


def _bca_correction(diffs: np.ndarray, boot_means: np.ndarray, point: float) -> tuple[float, float]:
    """Bias-correction (z0) and acceleration (a) for the BCa bootstrap
    (Efron & Tibshirani 1993, ch. 14). Plain percentile-bootstrap CIs are
    known to under-cover when the bootstrap distribution is biased or
    skewed — confirmed for this exact function via aa_calibration.py,
    which caught the plain percentile version rejecting at ~9-10% against
    a nominal 5% alpha (n_cases=15) before this correction was added. BCa
    fixes that by adjusting which percentiles of the bootstrap distribution
    the interval endpoints are drawn from, based on how biased (z0) and
    how skewed the statistic's sensitivity to each case is (a, via
    jackknife)."""
    n = len(diffs)
    z0 = norm.ppf(max(1e-6, min(1 - 1e-6, float(np.mean(boot_means < point)))))

    jackknife = np.array([np.delete(diffs, i).mean() for i in range(n)])
    jack_mean = jackknife.mean()
    num = np.sum((jack_mean - jackknife) ** 3)
    den = 6 * (np.sum((jack_mean - jackknife) ** 2) ** 1.5)
    a = 0.0 if den == 0 else num / den
    return z0, a


def _bca_z_to_percentile(z0: float, a: float, z: float) -> float:
    denom = 1 - a * (z0 + z)
    if abs(denom) < 1e-9:
        denom = 1e-9 if denom >= 0 else -1e-9
    adjusted = z0 + (z0 + z) / denom
    return float(norm.cdf(adjusted))


def cluster_bootstrap_diff(
    data: list[PairedCaseData],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> EffectEstimate:
    """BCa (bias-corrected and accelerated) cluster bootstrap over cases.
    The resampling unit is the case, not the run — a bootstrap replicate is
    built by drawing len(data) cases *with replacement* and averaging
    their rate_diffs, exactly mirroring how the point estimate itself is
    computed. This is what makes it respect the nested structure: a case
    that got resampled twice contributes its whole (arm_a, arm_b) pair
    twice, never a run drawn independently of which case it came from.

    Uses BCa rather than the plain percentile bootstrap specifically
    because aa_calibration.py caught the plain version under-covering
    (empirical FPR ~9-10% against nominal 5% at n_cases=15) — a
    textbook-known weakness of the percentile method in small-to-moderate
    samples, not something to just document and ship anyway given
    calibration is this module's primary self-check.

    BCa is asymptotically well-calibrated but, per Efron & Tibshirani, is
    still first-order-accurate rather than exact. An earlier pass at this
    docstring claimed "calibrated by n_cases=25 and staying there through
    60" — that was wrong, based on a 400-trial sweep whose own FPR
    estimate was too noisy to trust (its CI happened to just barely
    include nominal alpha). Re-run at 1500 trials per point (tight enough
    that the FPR estimate's own CI is a few points wide, not 5): n_cases=15
    ~10-11% FPR, n_cases=25 ~8.7% (CI [7.3, 10.2] — still clearly above
    5%), n_cases=40 ~5.9% (CI [4.8, 7.2] — the one point that actually
    included nominal alpha), n_cases=60 ~6.9%, n_cases=80 ~6.3% (both still
    slightly above). Read that as "inflated by roughly 1.2-1.7x nominal
    across this whole range, shrinking but not cleanly vanishing by 80,"
    not "calibrated past some threshold." Don't trust this method's
    p-values/CIs at case counts materially below the case suite you're
    actually about to run with — run aa_calibration.py (or
    experiments/calibration.py's sweep) at your actual scale before
    trusting a result; mcnemar_test or mixed_effects_diff may be
    better-behaved there instead (mixed_effects already declines to fit
    and falls back below 5 cases for the same reason).
    """
    usable = _usable(data)
    if len(usable) < 2:
        raise ValueError("cluster_bootstrap_diff needs at least 2 cases with data in both arms")

    rng = np.random.default_rng(seed)
    diffs = np.array([d.rate_diff for d in usable])
    n = len(usable)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    point = float(diffs.mean())

    if float(np.std(boot_means)) < 1e-12:
        # Degenerate: every bootstrap resample gives the exact same value
        # — e.g. every case has an identical arm_a/arm_b outcome sequence.
        # Caught via the Part 4 A/A preset feeding byte-identical
        # mock-scripted arms: BCa's z0/acceleration machinery divides by
        # quantities that vanish here, feeding norm.ppf values near 0/1
        # and producing a spuriously tiny p-value (every family reported
        # "SIGNIFICANT" despite a literal, exact 0.0pp difference). With
        # no resampling variance at all, the honest answer is a point CI
        # at the observed diff and a p-value reflecting only whether that
        # point equals the null value (0) or not.
        p_value = 0.0 if abs(point) > 1e-12 else 1.0
        return EffectEstimate(
            method="cluster_bootstrap",
            rate_a=case_rate([d.arm_a for d in usable]),
            rate_b=case_rate([d.arm_b for d in usable]),
            diff=point,
            ci_low=point,
            ci_high=point,
            alpha=alpha,
            p_value=p_value,
            n_cases=n,
            n_runs_a=sum(d.arm_a.n for d in usable),
            n_runs_b=sum(d.arm_b.n for d in usable),
            extra={"n_boot": n_boot, "degenerate_zero_variance": True},
        )

    z0, a = _bca_correction(diffs, boot_means, point)
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    pct_lo = _bca_z_to_percentile(z0, a, z_lo)
    pct_hi = _bca_z_to_percentile(z0, a, z_hi)
    ci_low, ci_high = np.percentile(boot_means, [100 * pct_lo, 100 * pct_hi])

    # BCa-consistent two-sided p-value: invert the same percentile
    # transform to find what nominal alpha would place a BCa boundary
    # exactly at 0, so "p < alpha" and "BCa CI excludes 0" agree — unlike
    # pairing an uncorrected percentile p-value with a BCa CI, which can
    # disagree right at the decision boundary.
    p0 = float(np.mean(boot_means <= 0))
    Z = norm.ppf(max(1e-6, min(1 - 1e-6, p0)))
    denom = 1 + a * (Z - z0)
    denom = denom if abs(denom) > 1e-9 else (1e-9 if denom >= 0 else -1e-9)
    z_p = (Z - z0) / denom - z0
    p_one_sided = float(norm.cdf(z_p))
    p_value = min(1.0, 2 * min(p_one_sided, 1 - p_one_sided))

    return EffectEstimate(
        method="cluster_bootstrap",
        rate_a=case_rate([d.arm_a for d in usable]),
        rate_b=case_rate([d.arm_b for d in usable]),
        diff=point,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        alpha=alpha,
        p_value=p_value,
        n_cases=n,
        n_runs_a=sum(d.arm_a.n for d in usable),
        n_runs_b=sum(d.arm_b.n for d in usable),
        extra={"n_boot": n_boot},
    )


def mcnemar_test(
    data: list[PairedCaseData], *, alpha: float = 0.05, exact: bool | None = None, correction: bool = True
) -> EffectEstimate:
    """McNemar's test for the simple paired binary design: exactly one run
    per arm per case. This is not a general-purpose reducer for
    multi-run-per-case data — average or otherwise summarize first (or use
    cluster_bootstrap_diff / mixed_effects_diff, which are built for that)
    if your design has more than one run per arm per case.

    exact=None (default) picks the exact binomial test when discordant
    pairs are few (<25) and the asymptotic chi-square test otherwise —
    statsmodels' own rule of thumb. `correction` (continuity correction,
    default True, matching R's mcnemar.test default; ignored when the
    exact test is used) is known to make the asymptotic test conservative
    — empirical FPR meaningfully below nominal alpha, confirmed against
    this exact codepath by aa_calibration.py at ~0.03 observed vs 0.05
    nominal in testing. That's a documented property of the correction,
    not miscalibration to silently work around: it trades power for never
    inflating the false-positive rate, the safer failure mode for a
    regression-detection tool. Pass correction=False to trade that
    conservatism for power.
    """
    usable = _usable(data)
    if not usable:
        raise ValueError("mcnemar_test needs at least 1 case with data in both arms")
    for d in usable:
        if d.arm_a.n != 1 or d.arm_b.n != 1:
            raise ValueError(
                f"mcnemar_test requires exactly 1 run per arm per case; case {d.case_id!r} "
                f"has {d.arm_a.n} (arm a) / {d.arm_b.n} (arm b). Use cluster_bootstrap_diff or "
                "mixed_effects_diff for multi-run-per-case designs."
            )

    a = np.array([d.arm_a.outcomes[0] for d in usable])
    b = np.array([d.arm_b.outcomes[0] for d in usable])
    n = len(usable)

    n11 = int(np.sum((a == 1) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))  # a-only
    n01 = int(np.sum((a == 0) & (b == 1)))  # b-only
    n00 = int(np.sum((a == 0) & (b == 0)))

    discordant = n10 + n01
    use_exact = discordant < 25 if exact is None else exact
    result = _sm_mcnemar([[n11, n10], [n01, n00]], exact=use_exact, correction=correction)

    diff = (n01 - n10) / n
    # Wald SE for the difference between two matched (dependent) proportions
    # — see e.g. Fleiss, Levin & Paik, Statistical Methods for Rates and
    # Proportions, 3rd ed., section on paired proportions.
    var = max(0.0, (discordant - (n01 - n10) ** 2 / n)) / n**2
    se = math.sqrt(var)
    z = float(norm.ppf(1 - alpha / 2))

    return EffectEstimate(
        method="mcnemar",
        rate_a=(n11 + n10) / n,
        rate_b=(n11 + n01) / n,
        diff=diff,
        ci_low=diff - z * se,
        ci_high=diff + z * se,
        alpha=alpha,
        p_value=float(result.pvalue),
        n_cases=n,
        n_runs_a=n,
        n_runs_b=n,
        extra={
            "n11": n11, "n10": n10, "n01": n01, "n00": n00,
            "exact": use_exact, "correction": correction and not use_exact,
            "statistic": float(result.statistic),
        },
    )


_MIN_CASES_FOR_MIXED_MODEL = 5
_MAX_RELIABLE_FE_SD = 5.0  # log-odds scale; above this the posterior is too diffuse to trust


def mixed_effects_diff(
    data: list[PairedCaseData],
    *,
    alpha: float = 0.05,
    n_mc_draws: int = 5000,
    seed: int = 0,
) -> EffectEstimate:
    """Logistic mixed model: outcome ~ arm + (1 | case). Fit via
    statsmodels' BinomialBayesMixedGLM (variational Bayes — the only
    statsmodels estimator that handles a binomial mixed model without
    needing many groups the way a full ML MixedLM would). Returns the
    effect on the probability scale via a Monte Carlo delta-method
    transform of the fitted fixed-effect posterior (Intercept, arm), which
    is an approximation — it ignores the random-intercept variance's
    contribution to the marginal CI and treats the two fixed-effect
    posteriors as independent normals. Documented, not hidden.

    Falls back to cluster_bootstrap_diff (with used_fallback=True and a
    fallback_reason) when the fit doesn't converge or the posterior on the
    arm coefficient is too diffuse to trust — both are realistic outcomes
    at the case counts this tool will often be run with, and reporting a
    fragile fit as if it were solid would be worse than falling back.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    usable = _usable(data)
    n_cases = len(usable)

    def fallback(reason: str) -> EffectEstimate:
        est = cluster_bootstrap_diff(data, alpha=alpha, seed=seed)
        return EffectEstimate(
            method="mixed_effects_logistic",
            rate_a=est.rate_a,
            rate_b=est.rate_b,
            diff=est.diff,
            ci_low=est.ci_low,
            ci_high=est.ci_high,
            alpha=alpha,
            p_value=est.p_value,
            n_cases=est.n_cases,
            n_runs_a=est.n_runs_a,
            n_runs_b=est.n_runs_b,
            used_fallback=True,
            fallback_reason=reason,
            extra={"fallback_method": "cluster_bootstrap"},
        )

    if n_cases < _MIN_CASES_FOR_MIXED_MODEL:
        return fallback(f"only {n_cases} cases with data in both arms (<{_MIN_CASES_FOR_MIXED_MODEL}); mixed models are unstable with this few groups")

    rows = []
    for d in usable:
        for y in d.arm_a.outcomes:
            rows.append({"case_id": d.case_id, "arm": 0, "y": y})
        for y in d.arm_b.outcomes:
            rows.append({"case_id": d.case_id, "arm": 1, "y": y})
    df = pd.DataFrame(rows)

    try:
        model = BinomialBayesMixedGLM.from_formula("y ~ arm", vc_formulas={"case_id": "0 + C(case_id)"}, data=df)
        result = model.fit_vb()
    except Exception as exc:  # noqa: BLE001 - any fit failure means "fall back", not "crash"
        return fallback(f"model fit raised {type(exc).__name__}: {exc}")

    if not result.optim_retvals.get("success", False):
        return fallback("optimizer did not report success")

    fe_mean, fe_sd = result.fe_mean, result.fe_sd
    if len(fe_mean) < 2 or not np.all(np.isfinite(fe_sd)) or fe_sd[1] > _MAX_RELIABLE_FE_SD:
        return fallback(f"posterior SD on the arm coefficient is {fe_sd[1] if len(fe_sd) > 1 else float('nan'):.2f} (log-odds scale) — too diffuse to trust")

    rng = np.random.default_rng(seed)
    intercept_draws = rng.normal(fe_mean[0], fe_sd[0], n_mc_draws)
    arm_draws = rng.normal(fe_mean[1], fe_sd[1], n_mc_draws)
    p_a_draws = 1 / (1 + np.exp(-intercept_draws))
    p_b_draws = 1 / (1 + np.exp(-(intercept_draws + arm_draws)))
    diff_draws = p_b_draws - p_a_draws

    point = float(np.mean(diff_draws))
    ci_low, ci_high = np.percentile(diff_draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Wald p-value on the log-odds arm coefficient (the model's native
    # scale) — separate from the Monte Carlo risk-difference CI above.
    z_stat = fe_mean[1] / fe_sd[1]
    p_value = float(2 * (1 - norm.cdf(abs(z_stat))))

    return EffectEstimate(
        method="mixed_effects_logistic",
        rate_a=case_rate([d.arm_a for d in usable]),
        rate_b=case_rate([d.arm_b for d in usable]),
        diff=point,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        alpha=alpha,
        p_value=p_value,
        n_cases=n_cases,
        n_runs_a=sum(d.arm_a.n for d in usable),
        n_runs_b=sum(d.arm_b.n for d in usable),
        extra={
            "log_odds_ratio": float(fe_mean[1]),
            "log_odds_ratio_sd": float(fe_sd[1]),
            "odds_ratio": float(np.exp(fe_mean[1])),
        },
    )
