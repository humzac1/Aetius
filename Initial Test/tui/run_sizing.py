"""Sizing a run *before* it executes: how many runs per case does this
comparison actually need to be able to answer the question?

Why this exists. Until now the wizard ran a hardcoded 5 runs/case
(wizard.py's DEFAULT_N_RUNS_PER_CASE, never exposed anywhere in the UI)
and only reported achieved power *after* spending the money — which on a
real reconstructed environment meant paying for a run that came back
"INCONCLUSIVE — not enough data to tell" at 2.5% achieved power. This
module asks the power question in advance, so the number and its price
are on screen before anything runs.

Sized against the live decision rule, not the retired one. The counts
here come from stats/hierarchical.required_runs_for_rope_signal — the
normal-approximation power model of the ROPE rule that actually judges
the finished run, verified cell-by-cell against the 800-trial validation
sweep (see that module's sizing section). An earlier version of this
module sized with stats/power.required_runs_per_case, which answers "when
does a CI exclude zero?" for the retired frequentist test — a rule the
verdict no longer applies, computed by a method that refused these case
counts outright. A recommendation must be priced against the rule that
will grade the run it recommends.

The limiting family, not the suite. Power is evaluated per attack family
and the *worst* family decides the verdict (see verdict_logic's
`worst = min(powers, key=achieved_power)`), so sizing to the average
family would still land INCONCLUSIVE on the weakest one. This recommends
the max over families — i.e. the count at which every applicable family
clears target power — and names the family that drove it, because the
honest answer to "why so many?" is usually "because one family only has
three applicable cases."

Baseline rate. Sizing needs a baseline success rate, and before a run
exists there isn't one. Prior runs of the same arm combination are used
when present (the real observed rate_a per family); otherwise this
assumes CONSERVATIVE_BASELINE_RATE = 0.5, which maximises binomial
variance and therefore can never recommend too few runs. Which of the
two happened is carried on the result (baseline_source) so the screen
can state the assumption rather than present a guess as a fact.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from attacker.attack_case import AttackCase
from experiments.persist import load_experiment_report
from experiments.runner import DEFAULT_RUNS_DIR
from stats.hierarchical import required_runs_for_rope_signal, rope_minimum_detectable_effect
from target_system.config import SystemConfig, compute_config_hash
from target_system.logging_schema import read_run_records
from tui.data import single_config_run_name
from tui.execution import comparison_experiment_name
from tui.verdict_logic import DEFAULT_TARGET_POWER

# Absolute rate difference (10 percentage points) the recommendation aims
# to be able to detect. Chosen against this project's own accumulated
# results rather than picked off a convention: across every effect
# estimate in the shipped run reports, the nonzero observed effects run
# 6.7-34.0 points with a median of 15.0, and the only effect that ever
# reached significance after correction was 34.0 points. 10 points
# therefore sits below what this tool actually sees when something real is
# happening, without sizing for effects smaller than any it has yet found
# meaningful — verdict_logic's DEFAULT_MDE_FLOOR of 0.05 answers a
# different question (it's the fallback target when a *completed* run
# observed an effect of ~exactly zero and the "run N more" advice needs
# some target to size around), and using it here would roughly quadruple
# the recommended count and the bill.
DEFAULT_MDE = 0.10

# Used when no prior run exists to observe a rate from. 0.5 maximises
# p(1-p), so it is the choice that cannot under-power: any true baseline
# makes the real requirement smaller than this, never larger.
CONSERVATIVE_BASELINE_RATE = 0.5

# Only for the wall-clock estimate, and only when there are no prior
# records to measure. Deliberately separate from the cost estimate's own
# fallback (experiments/cost_estimate.py) — this one is timing, not money.
FALLBACK_WALL_SECONDS_PER_RUN = 3.0


@dataclass(frozen=True)
class FamilyRequirement:
    """What one attack family needs on its own."""

    family: str
    n_cases: int
    baseline_rate: float
    # None when no run count can reach the target: the requested mde sits
    # inside the ROPE, so the decision rule is designed never to signal it.
    # Unreachable at the module's own DEFAULT_MDE (0.10 >> the ROPE), kept
    # for callers passing a custom target.
    required_runs_per_case: int | None


@dataclass(frozen=True)
class RunCountRecommendation:
    recommended_runs_per_case: int
    mde: float
    target_power: float
    baseline_source: Literal["observed", "assumed"]
    limiting: FamilyRequirement
    per_family: tuple[FamilyRequirement, ...]

    @property
    def unachievable(self) -> bool:
        """True when at least one family can't reach target power at any
        run count — the target effect is inside the practical-equivalence
        region, so the rule that grades the run would never signal it.
        Surfaced so a screen can say that instead of quoting a number
        that won't help."""
        return any(f.required_runs_per_case is None for f in self.per_family)


def _observed_baseline_rates(
    configs: list[SystemConfig], *, runs_dir: Path, case_ids: set[str] | None = None
) -> dict[str, float]:
    """Per-family observed baseline (arm A) success rate from the prior
    report for this exact arm combination, if one exists. Keyed by family;
    the max across outcome keys is taken, since sizing wants the family's
    hardest row, not its friendliest.

    Discarded entirely when the prior run faced a different case suite than
    the one about to run. A rate is only a baseline for the cases that
    produced it, and this is not hypothetical: the E-Commerce environment's
    first comparison ran the hand-authored suite, which that environment
    can't engage with at all, and recorded rate_a = 0.0. Reusing that as
    the baseline for a run of domain-adapted cases (whose real rate is
    nonzero) sized the run at 15 runs/case instead of what the true rate
    needs — under-powering a run using a number measured on a different
    experiment. Falling back to the conservative assumption is the correct
    answer when the suites differ."""
    hashes = [compute_config_hash(c) for c in configs]
    if len(hashes) == 1:
        name = single_config_run_name(hashes[0])
    elif len(hashes) == 2:
        name = comparison_experiment_name(hashes[0], hashes[1])
    else:
        raise ValueError(f"expected 1 or 2 configs, got {len(configs)}")

    report = load_experiment_report(name, runs_dir=runs_dir)
    if not report:
        return {}

    if case_ids is not None:
        # The report doesn't record which cases ran, but its JSONL does.
        prior_case_ids = {r.case_id for r in read_run_records(runs_dir / f"{name}.jsonl")}
        if prior_case_ids and prior_case_ids != case_ids:
            return {}

    rates: dict[str, float] = {}
    for entries in (report.get("family_results") or {}).values():
        for entry in entries:
            effect = entry.get("effect") or {}
            family, rate = entry.get("family"), effect.get("rate_a")
            if family is None or not isinstance(rate, (int, float)):
                continue
            rates[family] = max(rates.get(family, 0.0), float(rate))
    return rates


def recommend_runs_per_case(
    cases: list[AttackCase],
    configs: list[SystemConfig],
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    mde: float = DEFAULT_MDE,
    target_power: float = DEFAULT_TARGET_POWER,
) -> RunCountRecommendation | None:
    """Runs per case per arm at which every applicable family reaches
    target_power for an effect of `mde`. None when `cases` is empty (there
    is nothing to size).

    Returns the max across families, not the mean: the weakest family is
    the one that decides the verdict."""
    by_family: dict[str, int] = {}
    for case in cases:
        by_family[case.family] = by_family.get(case.family, 0) + 1
    if not by_family:
        return None

    observed = _observed_baseline_rates(configs, runs_dir=runs_dir, case_ids={c.id for c in cases})
    baseline_source: Literal["observed", "assumed"] = "observed" if observed else "assumed"

    requirements: list[FamilyRequirement] = []
    for family, n_cases in sorted(by_family.items()):
        baseline = observed.get(family, CONSERVATIVE_BASELINE_RATE)
        try:
            required = required_runs_for_rope_signal(baseline, mde, n_cases, power=target_power)
        except ValueError:
            # mde inside the ROPE: no run count can make the live rule
            # signal an effect that small — only reachable with a custom
            # mde below the default.
            required = None
        requirements.append(
            FamilyRequirement(family=family, n_cases=n_cases, baseline_rate=baseline, required_runs_per_case=required)
        )

    achievable = [f for f in requirements if f.required_runs_per_case is not None]
    # All-blocked is only reachable with a custom sub-ROPE mde, which
    # nothing passes today — fall back to the first family so the caller
    # still gets a well-formed result to render the "unachievable"
    # message from, rather than a None it has to special-case twice.
    limiting = max(achievable, key=lambda f: f.required_runs_per_case) if achievable else requirements[0]

    return RunCountRecommendation(
        recommended_runs_per_case=limiting.required_runs_per_case or 1,
        mde=mde,
        target_power=target_power,
        baseline_source=baseline_source,
        limiting=limiting,
        per_family=tuple(requirements),
    )


def detectable_effect_at(
    n_runs_per_case: int, recommendation: RunCountRecommendation, *, target_power: float | None = None
) -> float:
    """The smallest effect the limiting family could actually detect at
    `n_runs_per_case`. This is what makes a smaller-than-recommended
    choice an informed one rather than a shrug: it converts "fewer runs"
    into "you will not see anything below N points." Same power model as
    the recommendation itself (the ROPE rule's), so the two numbers on
    the screen can't disagree about what a run count buys."""
    return rope_minimum_detectable_effect(
        recommendation.limiting.n_cases,
        max(1, n_runs_per_case),
        recommendation.limiting.baseline_rate,
        power=target_power if target_power is not None else recommendation.target_power,
    )


def observed_wall_seconds_per_run(configs: list[SystemConfig], *, runs_dir: Path = DEFAULT_RUNS_DIR) -> tuple[float, bool]:
    """(seconds per run, grounded_in_real_data). Median wall time of the
    records already in the file this run would append to — the same
    "measure it, don't guess it" discipline experiments/cost_estimate.py
    applies to money. Falls back to a documented constant when there's
    nothing to measure yet."""
    hashes = [compute_config_hash(c) for c in configs]
    if len(hashes) == 1:
        path = runs_dir / f"{single_config_run_name(hashes[0])}.jsonl"
    elif len(hashes) == 2:
        path = runs_dir / f"{comparison_experiment_name(hashes[0], hashes[1])}.jsonl"
    else:
        raise ValueError(f"expected 1 or 2 configs, got {len(configs)}")

    if not path.exists():
        return FALLBACK_WALL_SECONDS_PER_RUN, False
    times = [r["wall_time_seconds"] for r in _raw_records(path) if isinstance(r.get("wall_time_seconds"), (int, float))]
    if not times:
        return FALLBACK_WALL_SECONDS_PER_RUN, False
    return statistics.median(times), True


def _raw_records(path: Path) -> list[dict[str, Any]]:
    return [dict(r) for r in read_run_records(path)]


# Matches tui.execution.run_comparison_check's max_workers default — runs
# execute concurrently, so a wall-clock estimate that multiplies jobs by
# per-run seconds without dividing by this overstates duration ~8x (the
# bug this constant exists to fix: the sizing screen showed "~1.5 hr" for
# what executes in ~10 minutes).
DEFAULT_MAX_WORKERS = 8


def estimated_wall_seconds(
    n_jobs: int, seconds_per_run: float, *, max_workers: int = DEFAULT_MAX_WORKERS
) -> float:
    """Wall-clock estimate for n_jobs executed by a pool of max_workers:
    jobs run in waves, so duration is per-run time x number of waves.
    Ignores inter-case join barriers on the sequential path — measured
    per-run medians (~5s) dwarf them."""
    if n_jobs <= 0:
        return 0.0
    return seconds_per_run * math.ceil(n_jobs / max_workers)


@dataclass(frozen=True)
class BudgetSizedOption:
    """The largest run count that fits a user-stated cost/time ceiling,
    with the effect size that budget can actually resolve — computed from
    the same ROPE power model and the same real cost/timing estimators as
    everything else on the sizing screen, so 'what $2 buys' is a
    statement about detection ability, not just a run count."""

    max_usd: float | None
    max_minutes: float | None
    n_runs_per_case: int  # 0 when infeasible
    total_runs: int
    estimated_cost_usd: float
    estimated_wall_seconds: float
    detectable_effect: float | None  # None when infeasible
    feasible: bool
    # Which ceiling stopped n+1 from fitting: "cost", "time", or "none"
    # (the search cap was reached — effectively unconstrained).
    binding: str


def size_for_budget(
    cases: list[AttackCase],
    configs: list[SystemConfig],
    recommendation: RunCountRecommendation,
    *,
    max_usd: float | None = None,
    max_minutes: float | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> BudgetSizedOption:
    """Largest n_runs_per_case whose real cost estimate and wall-clock
    estimate both fit the given ceilings (either may be None). Uses the
    real estimators — experiments.cost_estimate for money, the measured
    per-run median for time — never a hand-assumed per-run price.

    Infeasible (n below the validated 2-run floor) is reported as such
    with n_runs_per_case=0 rather than silently clamping up to a count
    the budget cannot pay for."""
    from experiments.cost_estimate import estimate_batch_cost
    from stats.hierarchical import MIN_RUNS_PER_CASE_FOR_HIERARCHICAL
    from tui.execution import peek_n_cached

    n_cached = peek_n_cached(configs, cases=cases, runs_dir=runs_dir)
    wall_per_run, _ = observed_wall_seconds_per_run(configs, runs_dir=runs_dir)

    def measures(n: int) -> tuple[float, float]:
        est = estimate_batch_cost(cases, configs, n_runs_per_case=n, n_cached=n_cached)
        return est.estimated_cost_usd, estimated_wall_seconds(est.n_jobs_remaining, wall_per_run, max_workers=max_workers)

    def fits(n: int) -> bool:
        cost, wall = measures(n)
        if max_usd is not None and cost > max_usd:
            return False
        if max_minutes is not None and wall > max_minutes * 60:
            return False
        return True

    floor = MIN_RUNS_PER_CASE_FOR_HIERARCHICAL
    cap = max(recommendation.recommended_runs_per_case * 4, 400)

    if not fits(floor):
        cost, wall = measures(floor)
        return BudgetSizedOption(
            max_usd=max_usd, max_minutes=max_minutes, n_runs_per_case=0,
            total_runs=len(cases) * floor * 2,
            estimated_cost_usd=cost, estimated_wall_seconds=wall,
            detectable_effect=None, feasible=False,
            binding="cost" if (max_usd is not None and cost > max_usd) else "time",
        )

    lo, hi = floor, floor
    while hi < cap and fits(hi * 2 if hi * 2 <= cap else cap):
        hi = hi * 2 if hi * 2 <= cap else cap
        if hi == cap:
            break
    # binary search the largest fitting n in (lo..cap]
    lo, search_hi = hi, min(hi * 2, cap)
    while lo < search_hi:
        mid = (lo + search_hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            search_hi = mid - 1
    best = lo

    cost, wall = measures(best)
    if best >= cap:
        binding = "none"
    else:
        over_cost, over_wall = measures(best + 1)
        binding = "cost" if (max_usd is not None and over_cost > max_usd) else "time"
    return BudgetSizedOption(
        max_usd=max_usd, max_minutes=max_minutes, n_runs_per_case=best,
        total_runs=len(cases) * best * 2,
        estimated_cost_usd=cost, estimated_wall_seconds=wall,
        detectable_effect=detectable_effect_at(best, recommendation),
        feasible=True, binding=binding,
    )
