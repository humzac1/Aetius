"""Cost/run-count estimate shown before executing a batch of attack cases
against a real model — running against a reconstructed twin (Part 4) now
spends real money and real time per run, unlike the toy system's mock
backend.

Grounded in real, Langfuse-computed cost/token figures a reconstruction
carries (ingestion/reconstruct.py's _build_cost_stats — averaged over the
actual traces this environment was reconstructed from) whenever available,
not a guess from a generic pricing table. Only falls back to a documented,
clearly-labeled rough per-call assumption when no config in the batch has
real observed cost data (e.g. a toy-system config run with
provider="anthropic", or a reconstruction whose source traces didn't carry
cost fields at all).

n_cached, if given, is subtracted before estimating — the whole point of
experiments/runner.py's CacheIndex resumability is that a partially-
completed real-model run doesn't get redone from scratch, and the cost
estimate a caller shows before launching should reflect only the
remaining work, not the full batch.
"""

from __future__ import annotations

from dataclasses import dataclass

from attacker.attack_case import AttackCase
from target_system.config import SystemConfig

# Used only when no config in the batch has real observed cost data to
# ground the estimate in (see CostEstimate.grounded_in_real_data) —
# deliberately conservative and clearly separated from the real-data path.
_FALLBACK_PROMPT_TOKENS_PER_CALL = 2000.0
_FALLBACK_COMPLETION_TOKENS_PER_CALL = 300.0
_FALLBACK_CALLS_PER_RUN = 2.0  # a run is rarely just one model call once tools are involved

# $ per million tokens (input, output) — anthropic.com/pricing, only used
# for the fallback path above; the real-data path uses Langfuse's own
# already-computed per-call cost and never touches this table.
_FALLBACK_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (15.0, 75.0),
}
_DEFAULT_FALLBACK_PRICE = (3.0, 15.0)  # sonnet-tier — a conservative default for an unrecognized model name


@dataclass(frozen=True)
class CostEstimate:
    n_cases: int
    n_runs_per_case: int
    n_arms: int
    n_jobs_total: int  # n_cases * n_runs_per_case * n_arms
    n_jobs_remaining: int  # n_jobs_total - n_cached, floored at 0
    estimated_llm_calls: float
    estimated_cost_usd: float
    grounded_in_real_data: bool  # False means estimated_cost_usd used the documented fallback assumption, not this environment's own observed data
    any_real_model: bool  # False means every arm is provider="mock" — cost is genuinely $0, not just unestimated


def estimate_batch_cost(
    cases: list[AttackCase],
    configs: list[SystemConfig],
    *,
    n_runs_per_case: int,
    n_cached: int = 0,
) -> CostEstimate:
    """configs: the arm(s) about to run — one for a single-config check,
    two for a paired comparison."""
    n_cases = len(cases)
    n_arms = len(configs)
    n_jobs_total = n_cases * n_runs_per_case * n_arms
    n_jobs_remaining = max(0, n_jobs_total - n_cached)

    any_real_model = any(c.model.provider == "anthropic" for c in configs)
    if not any_real_model:
        return CostEstimate(
            n_cases=n_cases, n_runs_per_case=n_runs_per_case, n_arms=n_arms,
            n_jobs_total=n_jobs_total, n_jobs_remaining=n_jobs_remaining,
            estimated_llm_calls=0.0, estimated_cost_usd=0.0, grounded_in_real_data=False, any_real_model=False,
        )

    real_data_configs = [c for c in configs if c.provenance is not None and c.provenance.avg_cost_usd_per_trace is not None]

    if real_data_configs:
        avg_cost_per_run = sum(c.provenance.avg_cost_usd_per_trace for c in real_data_configs) / len(real_data_configs)
        avg_generations_per_run = sum((c.provenance.avg_generations_per_trace or 1.0) for c in real_data_configs) / len(real_data_configs)
        estimated_cost_usd = n_jobs_remaining * avg_cost_per_run
        grounded_in_real_data = True
    else:
        avg_generations_per_run = _FALLBACK_CALLS_PER_RUN
        model_names = {c.model.model_name for c in configs if c.model.provider == "anthropic"}
        in_price, out_price = next(
            (_FALLBACK_PRICING_PER_MILLION_TOKENS[name] for name in model_names if name in _FALLBACK_PRICING_PER_MILLION_TOKENS),
            _DEFAULT_FALLBACK_PRICE,
        )
        per_call_cost = (_FALLBACK_PROMPT_TOKENS_PER_CALL / 1e6 * in_price) + (_FALLBACK_COMPLETION_TOKENS_PER_CALL / 1e6 * out_price)
        estimated_cost_usd = n_jobs_remaining * _FALLBACK_CALLS_PER_RUN * per_call_cost
        grounded_in_real_data = False

    return CostEstimate(
        n_cases=n_cases, n_runs_per_case=n_runs_per_case, n_arms=n_arms,
        n_jobs_total=n_jobs_total, n_jobs_remaining=n_jobs_remaining,
        estimated_llm_calls=n_jobs_remaining * avg_generations_per_run,
        estimated_cost_usd=estimated_cost_usd, grounded_in_real_data=grounded_in_real_data, any_real_model=True,
    )


def format_cost_estimate(estimate: CostEstimate) -> str:
    if not estimate.any_real_model:
        return f"{estimate.n_jobs_remaining} run(s) remaining (mock backend — no API cost)."

    basis = "based on this environment's own observed cost per run" if estimate.grounded_in_real_data else "rough estimate (no observed cost data for this environment/model — see experiments/cost_estimate.py)"
    return (
        f"{estimate.n_jobs_remaining} of {estimate.n_jobs_total} run(s) remaining "
        f"({estimate.n_cases} cases x {estimate.n_runs_per_case} runs/case x {estimate.n_arms} arm(s)), "
        f"~{estimate.estimated_llm_calls:.0f} model calls, "
        f"estimated cost ${estimate.estimated_cost_usd:.2f} ({basis})."
    )
