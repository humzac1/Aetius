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

from attacker.attack_case import AttackCase
from attacker.cases import ATTACK_CASES
from attacker.executor import execute_case
from experiments.mock_policy import build_mock_scripts
from experiments.presets import ArmSpec
from stats.multiple_comparisons import FamilyResult, compare_families
from stats.types import CaseObservations, PairedCaseData
from target_system.config import SystemConfig, save_config
from target_system.factory import baseline_config
from target_system.logging_schema import RunRecord, append_run_record, read_run_records

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path(__file__).parent.parent / "data" / "runs"

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


class _CacheIndex:
    """What's already on disk for this experiment, keyed the same way a
    new run would be — the resumability mechanism."""

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


def build_paired_data(records: list[RunRecord], arm_a_label: str, arm_b_label: str, outcome_key: str) -> list[PairedCaseData]:
    by_case_arm: dict[tuple[str, str | None], list[RunRecord]] = defaultdict(list)
    family_by_case: dict[str, str] = {}
    for r in records:
        by_case_arm[(r.case_id, r.arm)].append(r)
        if r.case_family:
            family_by_case[r.case_id] = r.case_family

    paired = []
    for case_id in sorted({cid for cid, _arm in by_case_arm}):
        recs_a = sorted(by_case_arm.get((case_id, arm_a_label), []), key=lambda r: r.seed)
        recs_b = sorted(by_case_arm.get((case_id, arm_b_label), []), key=lambda r: r.seed)
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


def _task_success_rate(records: list[RunRecord], arm_label: str) -> float:
    relevant = [r for r in records if r.arm == arm_label]
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
) -> ExperimentResult:
    """Each arm is either an ArmSpec (resolved here via
    factory.baseline_config(**overrides) — the preset path) or an
    already-resolved SystemConfig (the ad hoc CLI path, for comparing two
    configs someone already saved by hash). Either way what follows only
    ever deals with resolved SystemConfigs, so there's one code path, not
    two."""
    cases = list(cases) if cases is not None else list(ATTACK_CASES)
    outcome_keys = outcome_keys or OUTCOME_KEYS
    method_kwargs = dict(method_kwargs or {})

    config_a = resolve_arm(arm_a) if isinstance(arm_a, ArmSpec) else arm_a
    config_b = resolve_arm(arm_b) if isinstance(arm_b, ArmSpec) else arm_b
    label_a, label_b = config_a.label, config_b.label
    hash_a = save_config(config_a)
    hash_b = save_config(config_b)
    logger.info("arm %s -> %s, arm %s -> %s", label_a, hash_a, label_b, hash_b)

    runs_path = runs_dir / f"{experiment_name}.jsonl"
    cache = _CacheIndex(runs_path)
    write_lock = Lock()

    jobs: list[tuple[SystemConfig, str, AttackCase, int]] = []
    for case in cases:
        for run_idx in range(n_runs_per_case):
            for arm_label, config, config_hash in [(label_a, config_a, hash_a), (label_b, config_b, hash_b)]:
                if not cache.has(config_hash, case.id, arm_label, run_idx):
                    jobs.append((config, arm_label, case, run_idx))

    n_cached = len(cache.records)
    logger.info("experiment %r: %d runs already cached, %d to execute", experiment_name, n_cached, len(jobs))

    def _execute(job: tuple[SystemConfig, str, AttackCase, int]) -> RunRecord:
        config, arm_label, case, seed = job
        mock_scripts = None
        if config.model.provider == "mock":
            mock_scripts = build_mock_scripts(case, config, seed)
        record = execute_case(config, case, seed=seed, arm=arm_label, mock_scripts=mock_scripts)
        with write_lock:
            append_run_record(record, runs_path)
        return record

    if jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_execute, job) for job in jobs]
            for future in as_completed(futures):
                cache.add(future.result())

    family_results: dict[str, list[FamilyResult]] = {}
    for outcome_key in outcome_keys:
        paired = build_paired_data(cache.records, label_a, label_b, outcome_key)
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
        task_success_a=_task_success_rate(cache.records, label_a),
        task_success_b=_task_success_rate(cache.records, label_b),
        records=cache.records,
    )
