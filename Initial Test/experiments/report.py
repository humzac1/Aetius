"""Assembles a full experiment report from an ExperimentResult, on top of
stats/reporting.py's per-effect sentence formatting — this module adds
the experiment-level framing (both arms' config hashes, cache stats, a
task_success sanity check, and each outcome's strict/_attempted variants
shown side by side) that stats/reporting.py has no way to know about on
its own, since it only ever sees one EffectEstimate at a time.
"""

from __future__ import annotations

from experiments.runner import ExperimentResult, build_paired_data

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
                flag = "SIGNIFICANT" if r.significant_after_correction else "not significant"
                lines.append(
                    f"  [{key:24s} {flag:15s}] {family}: "
                    f"{r.effect.rate_a * 100:.1f}% -> {r.effect.rate_b * 100:.1f}% "
                    f"(diff {r.effect.diff * 100:+.1f}pp, {int((1 - r.effect.alpha) * 100)}% CI "
                    f"[{r.effect.ci_low * 100:.1f}, {r.effect.ci_high * 100:.1f}], q={r.q_value:.3f}, "
                    f"n={r.effect.n_cases} cases)"
                    + (f" [fell back: {r.effect.fallback_reason}]" if r.effect.used_fallback else "")
                )
        lines.append("")

    if result.arm_a_hash == result.arm_b_hash:
        lines.append(
            "NOTE: both arms resolved to the same config_hash (bit-identical config) — this is an "
            "A/A comparison. Any SIGNIFICANT flag above is a false positive; expect roughly "
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
