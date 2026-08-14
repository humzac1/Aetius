"""Executes both arms of a paired comparison across an attack suite,
writes trajectories, runs the Part 3 statistical comparison per outcome
key, and returns a report-ready bundle.

Caches aggressively and is safe to interrupt and re-run: before executing
anything, it reads whatever RunRecords already exist in this experiment's
JSONL file and skips any (config_hash, case_id, arm, seed) already
present. API calls are the binding cost — more so now that multi-turn
cases mean several calls per case — so re-running an experiment after an
interruption (or to add more runs_per_case) only executes what's missing.

Per-run isolation (fresh tool copies, fresh InMemoryDb — see
target_system/orchestration.py) already makes execute_case safe to call
concurrently; this module parallelizes across (case, arm, run_index) jobs
with a thread pool and guards the shared JSONL append with a lock, the
same discipline established in tests/test_concurrency.py and
tests/test_multi_turn.py.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal

from attacker.attack_case import AttackCase
from attacker.cases import ATTACK_CASES
from attacker.executor import execute_case
from config import paths
from experiments.mock_policy import build_mock_scripts
from experiments.presets import ArmSpec
from stats.hierarchical import hierarchical_bayes_diff, rope_resolution
from stats.multiple_comparisons import FamilyResult, compare_families
from stats.sequential import mixture_sprt_confidence_sequence
from stats.types import CaseObservations, PairedCaseData
from target_system.config import SystemConfig, save_config
from target_system.factory import baseline_config
from target_system.logging_schema import RunRecord, append_run_record, read_run_records

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = paths.RUNS_DIR

OUTCOME_KEYS = ["exfiltration", "exfiltration_attempted", "unauthorized_lookup", "unauthorized_lookup_attempted"]


def resolve_arm(arm: ArmSpec) -> SystemConfig:
    return baseline_config(label=arm.label, **arm.overrides)


@dataclass(frozen=True)
class SequentialStopSpec:
    """Opt-in early stopping for run_experiment. Passing None (the default)
    leaves execution exactly as it was: every job for every case runs.

    Two rules:

    rule="rope" (the live default) — after each completed case, fit the
    live hierarchical method on every case completed so far and classify
    each monitored outcome via stats.hierarchical.rope_resolution: stop
    once every monitored outcome has resolved, either "signal" (the 95%
    credible interval already clears the practical-equivalence band — a
    longer run would flag the same thing) or "futile" (the interval sits
    entirely inside the band — no remaining spend can change the verdict).
    Repeatedly checking a credible interval is not automatically safe the
    way an e-process is, so the combined procedure was validated by
    simulation at the project's standard rigor (800 trials/condition, both
    real measured shapes, a look after every case): sequential null signal
    rates stayed in the fixed-N rule's neighborhood (rare shape ~0.00-0.02,
    high shape under 0.05) while real effects stopped in roughly half the
    cases at undiminished power — the measured table lives in
    tests/test_stats_hierarchical.py's sequential section.

    rule="mixture_sprt" — the previous rule, retained retired-but-testable:
    stats.sequential.mixture_sprt_confidence_sequence's e-process boundary
    (anytime-valid by Ville's inequality). It tests "mean differs from 0",
    not the ROPE decision the verdict actually applies, so it can stop
    early on a sub-ROPE effect the verdict will then refuse to flag, and
    it can never stop a null run early (no futility). Nothing on the
    product path selects it anymore.

    The observation unit is a case, not a run, under both rules — a case
    is only ever evaluated once every one of its runs, in both arms, has
    completed. Partial cases are never fed to a boundary."""

    outcome_key: str
    alpha: float = 0.05
    tau: float = 0.1
    # Floor before any boundary is consulted. For mixture_sprt this is the
    # minimum the confidence sequence needs to estimate sigma (it raises
    # below 2); for rope it coincides with the hierarchical method's own
    # MIN_CASES_FOR_HIERARCHICAL (the model refuses below it and
    # rope_resolution treats a refusal as "continue" regardless).
    min_cases: int = 2
    rule: Literal["rope", "mixture_sprt"] = "rope"
    # In-loop rope looks use this stricter credible level (99%), not the
    # verdict's 95%. Measured, not assumed: with 95% looks the sequential
    # null signal rate on the high-rate shape at 15 runs/case inflated to
    # 0.076 (fixed-N: 0.026; 800 trials) — repeated peeks at the decision
    # interval are not free. Demanding a 99% interval beyond the ROPE to
    # stop early brought every measured null back to the fixed-N
    # neighborhood at ~zero power cost (table in
    # tests/test_stats_hierarchical.py). An early 99% signal strictly
    # implies the final 95% flag on the same data, so a signal-stop can
    # never contradict the verdict computed afterwards.
    early_alpha: float = 0.01
    # Additional outcome keys the rope rule must also resolve before
    # stopping. The old single-outcome rule watched only exfiltration and
    # would have kept running (or stopped!) oblivious to a real effect on
    # unauthorized_lookup — the exact shape of the one genuine finding in
    # this project's real data so far.
    extra_outcome_keys: tuple[str, ...] = ()

    @property
    def monitored_outcome_keys(self) -> tuple[str, ...]:
        return (self.outcome_key, *self.extra_outcome_keys)


@dataclass(frozen=True)
class SequentialStopOutcome:
    """What early stopping actually did — recorded so a report can state
    the run count honestly rather than implying the full suite ran.

    e_value/always_valid_p belong to the mixture_sprt rule and are None
    under rope; resolution/resolutions belong to the rope rule (the
    primary outcome's classification and the full per-monitored-outcome
    map at the last look) and are None under mixture_sprt. center/ci_low/
    ci_high are the primary outcome's estimate at the last look under
    either rule — a credible interval under rope."""

    outcome_key: str
    stopped_early: bool
    cases_evaluated: int
    cases_planned: int
    first_stop_index: int | None
    e_value: float | None
    always_valid_p: float | None
    center: float | None
    ci_low: float | None
    ci_high: float | None
    rule: str = "mixture_sprt"
    resolution: str | None = None
    resolutions: dict[str, str] | None = None


@dataclass
class ExperimentResult:
    name: str
    arm_a_label: str
    arm_b_label: str
    arm_a_hash: str
    arm_b_hash: str
    n_cases: int
    n_runs_per_case: int
    n_cached: int
    n_executed: int
    family_results: dict[str, list[FamilyResult]]  # outcome_key -> per-family results
    task_success_a: float
    task_success_b: float
    records: list[RunRecord] = field(default_factory=list, repr=False)
    sequential_stop: SequentialStopOutcome | None = None


class CacheIndex:
    """What's already on disk for this experiment, keyed the same way a
    new run would be — the resumability mechanism. Public (not
    underscore-prefixed) because tui/execution.py reuses it directly for
    single-config checks, which run_experiment can't cover (it's
    inherently paired/two-arm) — this is the shared resumability
    primitive both single- and paired-run execution build on, not
    something the TUI reimplements."""

    def __init__(self, path: Path):
        self.records: list[RunRecord] = []
        self._seen: set[tuple[str, str, str | None, int]] = set()
        if path.exists():
            for record in read_run_records(path):
                self._seen.add(self._key(record))
                self.records.append(record)

    @staticmethod
    def _key(record: RunRecord) -> tuple[str, str, str | None, int]:
        return (record.config_hash, record.case_id, record.arm, record.seed)

    def has(self, config_hash: str, case_id: str, arm: str | None, seed: int) -> bool:
        return (config_hash, case_id, arm, seed) in self._seen

    def add(self, record: RunRecord) -> None:
        self._seen.add(self._key(record))
        self.records.append(record)


def build_paired_data(
    records: list[RunRecord], arm_a_hash: str, arm_a_label: str, arm_b_hash: str, arm_b_label: str, outcome_key: str
) -> list[PairedCaseData]:
    """Groups by (case_id, config_hash, arm) — record.arm alone isn't a
    reliable arm identity: reconstruction defaults a config's label (what
    record.arm is set to, see run_experiment) to its workflow_name/
    agent_name, so two genuinely different reconstructions of the same
    real workflow (a legitimate before/after regression check, e.g.
    re-pulled a week apart with real behavior drift) can silently share a
    label — config_hash (content-only, see compute_config_hash) tells
    them apart, since it's guaranteed to differ whenever anything about
    the resolved config actually differs.

    config_hash alone isn't sufficient either: it deliberately excludes
    label (see _canonical_json), so a genuine AA-equivalence check
    comparing a config to itself (see format_experiment_report's "both
    arms resolved to the same config_hash" note — an intentional,
    already-supported case) has arm_a_hash == arm_b_hash by design, and
    hash-only keying would silently re-merge that scenario's two arms
    right back into the same bug this is fixing, just at the hash level
    instead of the label level. (config_hash, label) together correctly
    separates both failure modes while preserving the same-hash AA-check
    case, as long as its two arms are given distinct labels — the
    caller's responsibility, same as it already is for run_experiment's
    arm_a/arm_b needing to be distinguishable at all."""
    by_case_arm: dict[tuple[str, str, str | None], list[RunRecord]] = defaultdict(list)
    family_by_case: dict[str, str] = {}
    for r in records:
        by_case_arm[(r.case_id, r.config_hash, r.arm)].append(r)
        if r.case_family:
            family_by_case[r.case_id] = r.case_family

    paired = []
    for case_id in sorted({cid for cid, _hash, _arm in by_case_arm}):
        recs_a = sorted(by_case_arm.get((case_id, arm_a_hash, arm_a_label), []), key=lambda r: r.seed)
        recs_b = sorted(by_case_arm.get((case_id, arm_b_hash, arm_b_label), []), key=lambda r: r.seed)
        if not recs_a or not recs_b:
            continue
        outcomes_a = tuple(int(bool(r.outcomes.get(outcome_key, False))) for r in recs_a)
        outcomes_b = tuple(int(bool(r.outcomes.get(outcome_key, False))) for r in recs_b)
        family = family_by_case.get(case_id, "unknown")
        paired.append(
            PairedCaseData(
                case_id, family, CaseObservations(case_id, family, outcomes_a), CaseObservations(case_id, family, outcomes_b)
            )
        )
    return paired


ARM_A_SUFFIX = " (arm A)"
ARM_B_SUFFIX = " (arm B)"


def disambiguate_arm_labels(
    config_a: SystemConfig, config_b: SystemConfig, hash_a: str, hash_b: str
) -> tuple[str, str]:
    """The two arms' labels, guaranteed to key distinctly in
    build_paired_data.

    build_paired_data keys an arm on (config_hash, label), and its
    docstring notes the same-hash A/A case only works "as long as its two
    arms are given distinct labels — the caller's responsibility." No
    caller ever did: tui/screens/wizard.py's ConfigPickerScreen happily
    lets the same saved environment be picked twice, and
    tui/execution.py's run_comparison_check passes both configs straight
    through, so both arms arrived with identical hash AND identical label.
    build_paired_data then read both arms out of the same bucket: arm_a
    and arm_b became the *same records*, n doubled, and every case's
    rate_diff was pinned at exactly 0.0 regardless of what the runs did —
    a structural false CLEAR rather than a real A/A result. run_experiment
    also queued both identical jobs, paying twice for one arm's worth of
    real API calls.

    Only the same-hash case is rewritten. Same label with *different*
    hashes is the separate collision build_paired_data's docstring
    describes (two reconstructions of one workflow both defaulting their
    label to the agent name); (hash, label) already separates those
    correctly, and renaming them here would paper over the very condition
    test_run_experiment_separates_same_label_configs_by_config_hash exists
    to keep exercising. Everything else passes through untouched, so
    ordinary comparisons keep matching their already-cached records."""
    if hash_a != hash_b or config_a.label != config_b.label:
        return config_a.label, config_b.label
    return f"{config_a.label}{ARM_A_SUFFIX}", f"{config_b.label}{ARM_B_SUFFIX}"


def _task_success_rate(records: list[RunRecord], config_hash: str, arm_label: str) -> float:
    # (config_hash, arm_label) together, not arm_label alone — see
    # build_paired_data's docstring for why label alone can't tell two
    # arms apart.
    relevant = [r for r in records if r.config_hash == config_hash and r.arm == arm_label]
    if not relevant:
        return float("nan")
    return sum(1 for r in relevant if r.outcomes.get("task_success")) / len(relevant)


class CaseSuiteMismatch(RuntimeError):
    """The run file being appended to was produced by a different set of
    cases than the one about to run."""


def suite_digest(case_ids: set[str] | list[str]) -> str:
    """Short, order-independent fingerprint of a case-id set. Used to give
    a different suite its own run file (see tui/execution.py's
    comparison_experiment_name)."""
    joined = "\n".join(sorted(set(case_ids)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8]


def assert_case_suite_matches(existing: list[RunRecord], cases: list[AttackCase], runs_path: Path) -> None:
    """Refuse to append runs of one case suite to a file recorded with a
    different one.

    Not hypothetical: this is what silently corrupted a real report. The
    experiment name derives from the two config hashes alone, so a
    comparison of the same two arms under a *different* case suite appended
    straight into the previous suite's file. compare_families then averaged
    per-case rates across the union of both suites, and because the older
    suite was one the environment couldn't engage with (every rate 0.0),
    every figure in the saved report came out at roughly half its true
    value — unauthorized_lookup was persisted as 0.433 when the suite
    actually run measured 0.867. Nothing in the report showed that two
    suites were mixed.

    Raising is right rather than silently starting a new file: a caller
    that picked the name itself needs to know its name is wrong, and the
    normal path (comparison_experiment_name) already avoids the collision
    by naming per suite, so reaching this is a bug rather than a routine
    branch."""
    if not existing:
        return
    existing_ids = {r.case_id for r in existing}
    incoming_ids = {c.id for c in cases}
    if existing_ids and existing_ids != incoming_ids:
        only_existing = sorted(existing_ids - incoming_ids)[:3]
        only_incoming = sorted(incoming_ids - existing_ids)[:3]
        raise CaseSuiteMismatch(
            f"{runs_path} holds runs for a different case suite "
            f"({len(existing_ids)} case(s) on disk vs {len(incoming_ids)} about to run). "
            f"Only on disk: {only_existing}. Only incoming: {only_incoming}. "
            "Appending would average both suites together in the report."
        )


def _case_rate_diff(
    records: list[RunRecord],
    case_id: str,
    hash_a: str,
    label_a: str,
    hash_b: str,
    label_b: str,
    outcome_key: str,
) -> float | None:
    """One case's arm_b - arm_a rate difference, or None if either arm has
    no completed runs for it. Deliberately routed through build_paired_data
    rather than counting records inline, so the number fed to the stopping
    boundary is produced by exactly the same pairing logic the final
    analysis uses — a case can't be scored one way for the stop decision
    and another way in the report."""
    paired = build_paired_data([r for r in records if r.case_id == case_id], hash_a, label_a, hash_b, label_b, outcome_key)
    if not paired or paired[0].arm_a.n == 0 or paired[0].arm_b.n == 0:
        return None
    return paired[0].rate_diff


def run_experiment(
    arm_a: ArmSpec | SystemConfig,
    arm_b: ArmSpec | SystemConfig,
    *,
    experiment_name: str,
    cases: list[AttackCase] | None = None,
    n_runs_per_case: int = 5,
    max_workers: int = 8,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    outcome_keys: list[str] | None = None,
    stats_method: str = "hierarchical_bayes",
    alpha: float = 0.05,
    method_kwargs: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    anthropic_client: Any = None,
    sequential_stop: SequentialStopSpec | None = None,
) -> ExperimentResult:
    """Each arm is either an ArmSpec (resolved here via
    factory.baseline_config(**overrides) — the preset path) or an
    already-resolved SystemConfig (the ad hoc CLI path, for comparing two
    configs someone already saved by hash). Either way what follows only
    ever deals with resolved SystemConfigs, so there's one code path, not
    two.

    on_progress, if given, is called with (completed, total) — once with
    (0, total) before any job starts (total may be 0 if everything's
    already cached), then once per completed job as results come back via
    as_completed(). Always called on the same thread run_experiment was
    called from, never from inside a worker thread — callers that need to
    touch UI state (e.g. the TUI, via Textual's call_from_thread) don't
    need to do their own marshaling for this specific callback, only for
    whatever thread they called run_experiment from in the first place.

    anthropic_client: passed straight through to execute_case, which only
    the reconstructed-twin path (Part 4) uses — the tool-response
    synthesizer's generation fallback when no close historical match
    exists (target_system/tool_synthesis.py). None (the default) means
    that fallback is unreachable and every unmatched call resolves to
    response_source="unavailable" instead of an LLM call — fine for a
    toy-system run (which ignores this parameter entirely), a deliberate
    choice for a reconstructed run only if you want to cap it to replay-only."""
    cases = list(cases) if cases is not None else list(ATTACK_CASES)
    outcome_keys = outcome_keys or OUTCOME_KEYS
    method_kwargs = dict(method_kwargs or {})

    config_a = resolve_arm(arm_a) if isinstance(arm_a, ArmSpec) else arm_a
    config_b = resolve_arm(arm_b) if isinstance(arm_b, ArmSpec) else arm_b

    # Fail once, up front, rather than once per job inside the thread pool
    # below — execute_case enforces this same rule per-job regardless (see
    # its docstring), this is purely a faster/quieter failure for a caller
    # about to submit a whole batch of jobs that would all reject anyway.
    for config in (config_a, config_b):
        if config.provenance is not None and config.model.provider != "anthropic":
            raise ValueError(
                f"{config.label!r}: reconstructed environments only run under provider='anthropic' — "
                f"got provider={config.model.provider!r}"
            )

    hash_a = save_config(config_a)
    hash_b = save_config(config_b)
    label_a, label_b = disambiguate_arm_labels(config_a, config_b, hash_a, hash_b)
    logger.info("arm %s -> %s, arm %s -> %s", label_a, hash_a, label_b, hash_b)

    runs_path = runs_dir / f"{experiment_name}.jsonl"
    cache = CacheIndex(runs_path)
    assert_case_suite_matches(cache.records, cases, runs_path)
    write_lock = Lock()

    def _jobs_for(case_list: list[AttackCase]) -> list[tuple[SystemConfig, str, AttackCase, int]]:
        out: list[tuple[SystemConfig, str, AttackCase, int]] = []
        for case in case_list:
            for run_idx in range(n_runs_per_case):
                for arm_label, config, config_hash in [(label_a, config_a, hash_a), (label_b, config_b, hash_b)]:
                    if not cache.has(config_hash, case.id, arm_label, run_idx):
                        out.append((config, arm_label, case, run_idx))
        return out

    jobs = _jobs_for(cases)

    n_cached = len(cache.records)
    logger.info("experiment %r: %d runs already cached, %d to execute", experiment_name, n_cached, len(jobs))

    if on_progress is not None:
        on_progress(0, len(jobs))

    def _execute(job: tuple[SystemConfig, str, AttackCase, int]) -> RunRecord:
        config, arm_label, case, seed = job
        mock_scripts = None
        # build_mock_scripts is toy-system-specific (it scripts a
        # delegate_task_to_member call no reconstructed solo agent has).
        # provider == "mock" here always means a toy config: the up-front
        # guard above already rejects a reconstructed config (provenance
        # is not None) under any provider but "anthropic" before any job
        # is ever built, so that combination can't reach this closure.
        if config.model.provider == "mock":
            mock_scripts = build_mock_scripts(case, config, seed)
        record = execute_case(config, case, seed=seed, arm=arm_label, mock_scripts=mock_scripts, anthropic_client=anthropic_client)
        with write_lock:
            append_run_record(record, runs_path)
        return record

    stop_outcome: SequentialStopOutcome | None = None
    n_executed = len(jobs)

    if sequential_stop is None:
        if jobs:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_execute, job) for job in jobs]
                for completed, future in enumerate(as_completed(futures), start=1):
                    cache.add(future.result())
                    if on_progress is not None:
                        on_progress(completed, len(jobs))
    else:
        # Case-ordered execution. Parallelism is preserved *within* a case
        # (both arms x every run go to the pool at once); what's given up
        # is overlapping the next case with the current one, which is the
        # price of being able to stop before that next case is paid for.
        # Progress is reported against the full planned job count, so the
        # bar ends short rather than quietly redefining 100%.
        completed = 0
        cases_processed = 0
        # mixture_sprt state (retired rule, kept testable)
        diffs: list[float] = []
        cases_evaluated = 0
        cs_result = None
        # rope state
        resolutions: dict[str, str] = {}
        rope_estimate = None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for case in cases:
                case_jobs = _jobs_for([case])
                if case_jobs:
                    futures = [executor.submit(_execute, job) for job in case_jobs]
                    for future in as_completed(futures):
                        cache.add(future.result())
                        completed += 1
                        if on_progress is not None:
                            on_progress(completed, len(jobs))
                cases_processed += 1

                if sequential_stop.rule == "rope":
                    if cases_processed < max(2, sequential_stop.min_cases):
                        continue
                    # Fit the live method on every case completed so far,
                    # once per monitored outcome. Deterministic (seed=0
                    # given the data), so re-running a cached experiment
                    # reproduces the same stop decision.
                    resolutions = {}
                    for key in sequential_stop.monitored_outcome_keys:
                        paired = build_paired_data(cache.records, hash_a, label_a, hash_b, label_b, key)
                        estimate = hierarchical_bayes_diff(paired, alpha=sequential_stop.early_alpha)
                        resolutions[key] = rope_resolution(estimate)
                        if key == sequential_stop.outcome_key:
                            rope_estimate = estimate
                    if resolutions and all(r != "continue" for r in resolutions.values()):
                        logger.info(
                            "experiment %r: stopping early after %d of %d cases (ROPE resolutions: %s)",
                            experiment_name, cases_processed, len(cases), resolutions,
                        )
                        break
                else:
                    # One rate difference for the case just finished, in
                    # completion order — the sequence's observation unit.
                    case_diff = _case_rate_diff(cache.records, case.id, hash_a, label_a, hash_b, label_b, sequential_stop.outcome_key)
                    if case_diff is None:
                        continue
                    diffs.append(case_diff)
                    cases_evaluated += 1

                    if len(diffs) >= max(2, sequential_stop.min_cases):
                        cs_result = mixture_sprt_confidence_sequence(diffs, alpha=sequential_stop.alpha, tau=sequential_stop.tau)
                        if cs_result.can_stop_now():
                            logger.info(
                                "experiment %r: stopping early after %d of %d cases (e-value %.2f >= %.2f)",
                                experiment_name, cases_evaluated, len(cases),
                                cs_result.points[-1].e_value, 1 / sequential_stop.alpha,
                            )
                            break

        if sequential_stop.rule == "rope":
            resolved = bool(resolutions) and all(r != "continue" for r in resolutions.values())
            stop_outcome = SequentialStopOutcome(
                outcome_key=sequential_stop.outcome_key,
                stopped_early=cases_processed < len(cases),
                cases_evaluated=cases_processed,
                cases_planned=len(cases),
                first_stop_index=cases_processed if resolved and cases_processed < len(cases) else None,
                e_value=None,
                always_valid_p=None,
                center=rope_estimate.diff if rope_estimate is not None else None,
                ci_low=rope_estimate.ci_low if rope_estimate is not None else None,
                ci_high=rope_estimate.ci_high if rope_estimate is not None else None,
                rule="rope",
                resolution=resolutions.get(sequential_stop.outcome_key),
                resolutions=dict(resolutions) if resolutions else None,
            )
        else:
            last = cs_result.points[-1] if cs_result is not None and cs_result.points else None
            stop_outcome = SequentialStopOutcome(
                outcome_key=sequential_stop.outcome_key,
                stopped_early=cases_evaluated < len(cases),
                cases_evaluated=cases_evaluated,
                cases_planned=len(cases),
                first_stop_index=cs_result.first_stop_index if cs_result is not None else None,
                e_value=last.e_value if last else None,
                always_valid_p=last.always_valid_p if last else None,
                center=last.center if last else None,
                ci_low=last.ci_low if last else None,
                ci_high=last.ci_high if last else None,
                rule="mixture_sprt",
            )
        n_executed = completed

    family_results: dict[str, list[FamilyResult]] = {}
    for outcome_key in outcome_keys:
        paired = build_paired_data(cache.records, hash_a, label_a, hash_b, label_b, outcome_key)
        family_results[outcome_key] = compare_families(
            paired, method=stats_method, alpha=alpha, method_kwargs=method_kwargs
        )

    return ExperimentResult(
        name=experiment_name,
        arm_a_label=label_a,
        arm_b_label=label_b,
        arm_a_hash=hash_a,
        arm_b_hash=hash_b,
        n_cases=len(cases),
        # The runs/case the statistics actually consumed, not the number
        # requested. They differ on a resumed/over-complete cache: a
        # comparison cached at 77 runs/case re-requested at 5 executes
        # nothing new but computes over all 77 — a report that then said
        # "5 runs/case" would misdescribe its own evidence (a real report
        # did exactly that before this line).
        n_runs_per_case=max(
            [n_runs_per_case]
            + list(Counter((r.case_id, r.arm) for r in cache.records).values())
        ),
        n_cached=n_cached,
        n_executed=n_executed,
        family_results=family_results,
        task_success_a=_task_success_rate(cache.records, hash_a, label_a),
        task_success_b=_task_success_rate(cache.records, hash_b, label_b),
        records=cache.records,
        sequential_stop=stop_outcome,
    )
