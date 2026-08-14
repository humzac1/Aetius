"""Formats results from every other stats/ module into the sentence shape
the build spec asks for — effect sizes with intervals, never a bare
p-value: "attack success rate rose from 2.1% to 4.8%, difference 2.7
points, 95% CI [1.1, 4.3], BH-adjusted q = 0.02."
"""

from __future__ import annotations

from stats.aa_calibration import AACalibrationResult
from stats.multiple_comparisons import FamilyResult
from stats.power import HeterogeneityDominates, minimum_detectable_effect, required_runs_per_case
from stats.types import EffectEstimate
from stats.variance_reduction import CRNResult, CupedResult


def format_effect(effect: EffectEstimate, *, q_value: float | None = None, label: str = "attack success rate") -> str:
    # Vocabulary tracks the method that produced the numbers: the live
    # hierarchical method's interval is a credible interval and its
    # "p_value" is a posterior direction probability, not a frequentist
    # tail probability — labelling them CI/p would misdescribe them.
    bayesian = "rope_signal" in effect.extra
    direction = "rose" if effect.diff > 0 else "fell" if effect.diff < 0 else "was unchanged"
    parts = [
        f"{label} {direction} from {effect.rate_a * 100:.1f}% to {effect.rate_b * 100:.1f}%",
        f"difference {effect.diff * 100:+.1f} points",
        f"{int((1 - effect.alpha) * 100)}% {'CrI' if bayesian else 'CI'} "
        f"[{effect.ci_low * 100:.1f}, {effect.ci_high * 100:.1f}]",
    ]
    if q_value is not None:
        parts.append(f"BH-adjusted q = {q_value:.3f}")
    elif effect.p_value is not None:
        parts.append(f"posterior direction probability = {effect.p_value:.3f}" if bayesian else f"p = {effect.p_value:.3f}")
    sentence = ", ".join(parts) + f" ({effect.method}, n={effect.n_cases} cases)."
    if effect.used_fallback:
        sentence += f" [fell back from the requested method: {effect.fallback_reason}]"
    return sentence


def format_family_results(results: list[FamilyResult], *, label: str = "attack success rate") -> str:
    lines = []
    for r in results:
        if "rope_signal" in r.effect.extra:
            flag = "FLAGGED (BH + ROPE)" if r.significant_after_correction else "not flagged"
        else:
            flag = "SIGNIFICANT (BH-corrected)" if r.significant_after_correction else "not significant"
        lines.append(f"[{flag}] {r.family}: {format_effect(r.effect, q_value=r.q_value, label=label)}")
    return "\n".join(lines)


def format_aa_calibration(result: AACalibrationResult) -> str:
    return result.summary()


def format_crn_result(result: CRNResult) -> str:
    return (
        f"Common random numbers reduced variance of the paired-diff estimator by "
        f"{result.variance_reduction_pct:.1f}% ({result.var_without_crn:.6f} -> {result.var_with_crn:.6f}) "
        f"over {result.n_sims} simulated replicate experiments "
        f"({result.n_cases} cases x {result.n_runs_per_case} runs/case/arm) — "
        f"equivalent to needing {result.effective_sample_size_multiplier:.2f}x fewer runs for the same precision."
    )


def format_cuped_result(result: CupedResult) -> str:
    return (
        f"CUPED adjustment (correlation with baseline covariate = {result.correlation:.2f}) reduced variance by "
        f"{result.variance_reduction_pct:.1f}% ({result.var_before:.6f} -> {result.var_after:.6f}); "
        f"mean unchanged ({result.mean_before:.4f} -> {result.mean_after:.4f}, as CUPED guarantees)."
    )


def format_power_report(
    baseline_rate: float,
    mde: float,
    n_cases: int,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    between_case_sd: float = 0.0,
) -> str:
    try:
        n = required_runs_per_case(baseline_rate, mde, n_cases, power=power, alpha=alpha, between_case_sd=between_case_sd)
        return (
            f"To detect a {mde * 100:.1f}-point rise from a {baseline_rate * 100:.1f}% baseline with "
            f"{int(power * 100)}% power at alpha={alpha}, using {n_cases} cases, you need "
            f"~{n} runs per case per arm ({n * n_cases * 2} runs total)."
        )
    except HeterogeneityDominates as exc:
        return f"Not achievable by adding runs: {exc}"


def format_mde_report(
    n_cases: int, n_runs_per_case: int, baseline_rate: float, *, power: float = 0.8, alpha: float = 0.05, between_case_sd: float = 0.0
) -> str:
    mde = minimum_detectable_effect(n_cases, n_runs_per_case, baseline_rate, power=power, alpha=alpha, between_case_sd=between_case_sd)
    total_runs = n_cases * n_runs_per_case * 2
    return (
        f"With {n_cases} cases x {n_runs_per_case} runs/case/arm ({total_runs} runs total), the smallest "
        f"effect detectable at {int(power * 100)}% power (alpha={alpha}) from a {baseline_rate * 100:.1f}% "
        f"baseline is {mde * 100:.1f} points."
    )
