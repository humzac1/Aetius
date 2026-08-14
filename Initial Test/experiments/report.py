"""Assembles a full experiment report from an ExperimentResult, on top of
stats/reporting.py's per-effect sentence formatting — this module adds
the experiment-level framing (both arms' config hashes, cache stats, a
task_success sanity check, and each outcome's strict/_attempted variants
shown side by side) that stats/reporting.py has no way to know about on
its own, since it only ever sees one EffectEstimate at a time.
"""

from __future__ import annotations

from pathlib import Path

from experiments.runner import ExperimentResult, build_paired_data
from stats.variance_reduction import CupedResult, cuped_adjust_case_rates
from target_system.logging_schema import read_run_records

TASK_SUCCESS_GAP_WARNING_THRESHOLD = 0.15


def format_experiment_report(result: ExperimentResult) -> str:
    lines = [
        f"=== Experiment: {result.name} ===",
        f"arm A: {result.arm_a_label}  (config_hash={result.arm_a_hash})",
        f"arm B: {result.arm_b_label}  (config_hash={result.arm_b_hash})",
        f"{result.n_cases} cases x {result.n_runs_per_case} runs/case/arm "
        f"({result.n_cached} runs already cached, {result.n_executed} executed this run)",
        "",
    ]

    gap = abs(result.task_success_a - result.task_success_b)
    warning = ""
    if gap > TASK_SUCCESS_GAP_WARNING_THRESHOLD:
        warning = "  [WARNING: large gap — a change here may just be making the agent less useful, not safer]"
    lines.append(
        f"task_success: arm A {result.task_success_a * 100:.1f}%  |  "
        f"arm B {result.task_success_b * 100:.1f}%{warning}"
    )
    lines.append("")

    for base_key in ["exfiltration", "unauthorized_lookup"]:
        attempted_key = f"{base_key}_attempted"
        lines.append(f"--- {base_key} ---")
        executed_by_family = {r.family: r for r in result.family_results.get(base_key, [])}
        attempted_by_family = {r.family: r for r in result.family_results.get(attempted_key, [])}
        families = sorted(set(executed_by_family) | set(attempted_by_family))

        if not families:
            lines.append("  (no cases with data in both arms for this outcome)")

        for family in families:
            for key, by_family in [(base_key, executed_by_family), (attempted_key, attempted_by_family)]:
                r = by_family.get(family)
                if r is None:
                    continue
                # Wording tracks what was actually computed: the live
                # hierarchical method's interval is a credible interval and
                # its decision is the ROPE rule; older frequentist methods
                # keep the CI/significance vocabulary their numbers mean.
                bayesian = "rope_signal" in r.effect.extra
                if bayesian:
                    flag = "FLAGGED" if r.significant_after_correction else "not flagged"
                    interval_label = "CrI"
                else:
                    flag = "SIGNIFICANT" if r.significant_after_correction else "not significant"
                    interval_label = "CI"
                lines.append(
                    f"  [{key:24s} {flag:15s}] {family}: "
                    f"{r.effect.rate_a * 100:.1f}% -> {r.effect.rate_b * 100:.1f}% "
                    f"(diff {r.effect.diff * 100:+.1f}pp, {int((1 - r.effect.alpha) * 100)}% {interval_label} "
                    f"[{r.effect.ci_low * 100:.1f}, {r.effect.ci_high * 100:.1f}], q={r.q_value:.3f}, "
                    f"n={r.effect.n_cases} cases)"
                    + (f" [fell back: {r.effect.fallback_reason}]" if r.effect.used_fallback else "")
                )
        lines.append("")

    if result.arm_a_hash == result.arm_b_hash:
        lines.append(
            "NOTE: both arms resolved to the same config_hash (bit-identical config) — this is an "
            "A/A comparison. Any flagged row above is a false positive; expect at most roughly "
            "alpha's worth of families to be flagged by chance."
        )

    return "\n".join(lines)


def compute_sequential_analysis(result: ExperimentResult, outcome_key: str, *, tau: float = 0.1, alpha: float = 0.05):
    """Feeds this experiment's case-by-case diffs (pooled across all
    families, in the order cases happen to appear in the cached records —
    not re-randomized) into stats.sequential's always-valid confidence
    sequence. Returns None if there aren't enough cases with data in both
    arms (need >= 2). Split out from format_sequential_analysis so
    experiments/persist.py can save the full point-by-point sequence
    (the dashboard's confidence-sequence plot needs every point, not just
    the summary sentence)."""
    from stats.sequential import mixture_sprt_confidence_sequence

    paired = build_paired_data(result.records, result.arm_a_hash, result.arm_a_label, result.arm_b_hash, result.arm_b_label, outcome_key)
    diffs = [d.rate_diff for d in paired if d.arm_a.n > 0 and d.arm_b.n > 0]
    if len(diffs) < 2:
        return None
    return mixture_sprt_confidence_sequence(diffs, alpha=alpha, tau=tau)


def historical_case_rates(
    config_hash: str,
    arm_label: str,
    outcome_key: str,
    *,
    runs_dir: Path,
    exclude_experiment: str | None = None,
) -> dict[str, float]:
    """Per-case success rate for one arm, measured from runs that happened
    in *previous* experiments — the covariate CUPED needs.

    Sourced from other experiment files, never from the arm being
    adjusted in the current one. That restriction is
    variance_reduction.cuped_adjust_case_rates' own ("deliberately not
    'the other arm in this same experiment'"), and it's what keeps the
    adjustment from feeding this run's noise back into itself: theta is
    then estimated against an independent measurement of each case's
    inherent difficulty, so the adjustment stays unbiased rather than
    partially regressing the outcome on itself."""
    hits: dict[str, list[int]] = {}
    for path in sorted(runs_dir.glob("*.jsonl")):
        if exclude_experiment is not None and path.stem == exclude_experiment:
            continue
        for record in read_run_records(path):
            if record.config_hash != config_hash or record.arm != arm_label:
                continue
            if outcome_key not in record.outcomes:
                continue
            hits.setdefault(record.case_id, []).append(int(bool(record.outcomes[outcome_key])))
    return {case_id: sum(v) / len(v) for case_id, v in hits.items() if v}


def compute_cuped_analysis(
    result: ExperimentResult,
    outcome_key: str,
    *,
    runs_dir: Path,
) -> CupedResult | None:
    """CUPED variance reduction on arm B's per-case rates, using each
    case's historically-observed baseline rate as the covariate.

    Exists because common random numbers — the variance reduction this
    project validated for the mock backend — cannot work on real model
    calls: the Anthropic API exposes no sampling seed (see
    target_system/config.py's ModelConfig.seed note), so the paired seeds
    run_experiment assigns are matched labels, not matched randomness.
    CUPED needs no control over sampling at all; it removes the part of
    each case's outcome that its historical difficulty already predicts.

    Returns None when there's no usable history (a first run against these
    configs), or when fewer than 2 cases overlap — never a fabricated
    adjustment."""
    baseline = historical_case_rates(
        result.arm_a_hash, result.arm_a_label, outcome_key, runs_dir=runs_dir, exclude_experiment=result.name
    )
    if len(baseline) < 2:
        return None
    paired = build_paired_data(result.records, result.arm_a_hash, result.arm_a_label, result.arm_b_hash, result.arm_b_label, outcome_key)
    overlapping = [d for d in paired if d.case_id in baseline and d.arm_b.n > 0]
    if len(overlapping) < 2:
        return None
    return cuped_adjust_case_rates(overlapping, baseline, arm="b")


def format_sequential_analysis(result: ExperimentResult, outcome_key: str, *, tau: float = 0.1, alpha: float = 0.05) -> str:
    """How many cases it took to cross the always-valid stopping boundary
    — the concrete answer to "how many runs did that take," honest because
    it's anytime-valid rather than a p-value computed once at a
    pre-committed N and presented as if it were."""
    paired = build_paired_data(result.records, result.arm_a_hash, result.arm_a_label, result.arm_b_hash, result.arm_b_label, outcome_key)
    n_diffs = sum(1 for d in paired if d.arm_a.n > 0 and d.arm_b.n > 0)
    header = f"--- always-valid sequential analysis: {outcome_key} ({n_diffs} cases) ---"

    cs = compute_sequential_analysis(result, outcome_key, tau=tau, alpha=alpha)
    if cs is None:
        return f"{header}\nnot enough cases with data in both arms to run this."
    return f"{header}\n{cs.summary()}"
