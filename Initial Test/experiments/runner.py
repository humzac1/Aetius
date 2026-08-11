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

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from attacker.attack_case import AttackCase
from attacker.cases import ATTACK_CASES
from attacker.executor import execute_case
from config import paths
from experiments.mock_policy import build_mock_scripts
from experiments.presets import ArmSpec
from stats.multiple_comparisons import FamilyResult, compare_families
from stats.types import CaseObservations, PairedCaseData
from target_system.config import SystemConfig, save_config
from target_system.factory import baseline_config
from target_system.logging_schema import RunRecord, append_run_record, read_run_records

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = paths.RUNS_DIR

OUTCOME_KEYS = ["exfiltration", "exfiltration_attempted", "unauthorized_lookup", "unauthorized_lookup_attempted"]


def resolve_arm(arm: ArmSpec) -> SystemConfig:
    return baseline_config(label=arm.label, **arm.overrides)


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
    stats_method: str = "cluster_bootstrap",
    alpha: float = 0.05,
    method_kwargs: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    anthropic_client: Any = None,
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
    write_lock = Lock()

    jobs: list[tuple[SystemConfig, str, AttackCase, int]] = []
    for case in cases:
        for run_idx in range(n_runs_per_case):
            for arm_label, config, config_hash in [(label_a, config_a, hash_a), (label_b, config_b, hash_b)]:
                if not cache.has(config_hash, case.id, arm_label, run_idx):
                    jobs.append((config, arm_label, case, run_idx))

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

    if jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_execute, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), start=1):
                cache.add(future.result())
                if on_progress is not None:
                    on_progress(completed, len(jobs))

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
        n_runs_per_case=n_runs_per_case,
        n_cached=n_cached,
        n_executed=len(jobs),
        family_results=family_results,
        task_success_a=_task_success_rate(cache.records, hash_a, label_a),
        task_success_b=_task_success_rate(cache.records, hash_b, label_b),
        records=cache.records,
    )
