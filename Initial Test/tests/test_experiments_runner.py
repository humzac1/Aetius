from pathlib import Path

from attacker.cases import by_family
from experiments.presets import ArmSpec
from experiments.runner import OUTCOME_KEYS, run_experiment
from target_system.logging_schema import read_run_records


def _small_case_subset():
    # Keep tests fast: one case from two different families rather than
    # the full 17-case suite.
    return [by_family("direct_instruction_injection")[0], by_family("tool_result_poisoning")[0]]


def test_run_experiment_identical_arms_share_config_hash(tmp_path: Path):
    arm_a = ArmSpec(label="a", overrides={})
    arm_b = ArmSpec(label="b", overrides={})
    result = run_experiment(
        arm_a, arm_b, experiment_name="test_identical", cases=_small_case_subset(),
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )
    assert result.arm_a_hash == result.arm_b_hash
    assert set(result.family_results.keys()) == set(OUTCOME_KEYS)


def test_run_experiment_writes_expected_number_of_records(tmp_path: Path):
    cases = _small_case_subset()
    n_runs = 2
    arm_a = ArmSpec(label="a", overrides={})
    arm_b = ArmSpec(label="b", overrides={"email_allowlist": ["only-this@ourcompany.example"]})
    result = run_experiment(
        arm_a, arm_b, experiment_name="test_record_count", cases=cases,
        n_runs_per_case=n_runs, max_workers=2, runs_dir=tmp_path,
    )
    expected = len(cases) * n_runs * 2
    assert len(result.records) == expected
    assert result.n_executed == expected
    assert result.n_cached == 0

    on_disk = list(read_run_records(tmp_path / "test_record_count.jsonl"))
    assert len(on_disk) == expected


def test_run_experiment_is_resumable_and_caches(tmp_path: Path):
    cases = _small_case_subset()
    arm_a = ArmSpec(label="a", overrides={})
    arm_b = ArmSpec(label="b", overrides={})

    first = run_experiment(
        arm_a, arm_b, experiment_name="test_resume", cases=cases,
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )
    assert first.n_executed == len(cases) * 2 * 2
    assert first.n_cached == 0

    second = run_experiment(
        arm_a, arm_b, experiment_name="test_resume", cases=cases,
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )
    assert second.n_executed == 0
    assert second.n_cached == first.n_executed
    assert len(second.records) == len(first.records)


def test_run_experiment_partial_cache_only_executes_the_gap(tmp_path: Path):
    cases = _small_case_subset()
    arm_a = ArmSpec(label="a", overrides={})
    arm_b = ArmSpec(label="b", overrides={})

    run_experiment(arm_a, arm_b, experiment_name="test_partial", cases=cases, n_runs_per_case=2, max_workers=2, runs_dir=tmp_path)
    # Ask for MORE runs per case than before -- only the new ones should execute.
    result = run_experiment(arm_a, arm_b, experiment_name="test_partial", cases=cases, n_runs_per_case=4, max_workers=2, runs_dir=tmp_path)
    assert result.n_cached == len(cases) * 2 * 2
    assert result.n_executed == len(cases) * 2 * 2  # the extra 2 runs/case/arm
    assert len(result.records) == len(cases) * 4 * 2


def test_run_experiment_concurrent_writes_produce_no_corrupted_lines(tmp_path: Path):
    cases = _small_case_subset()
    arm_a = ArmSpec(label="a", overrides={})
    arm_b = ArmSpec(label="b", overrides={})
    result = run_experiment(
        arm_a, arm_b, experiment_name="test_concurrent", cases=cases,
        n_runs_per_case=5, max_workers=8, runs_dir=tmp_path,
    )
    on_disk = list(read_run_records(tmp_path / "test_concurrent.jsonl"))  # raises on any malformed JSON line
    assert len(on_disk) == len(cases) * 5 * 2
    assert len(result.records) == len(on_disk)
