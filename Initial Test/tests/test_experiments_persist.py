from pathlib import Path

from attacker.cases import by_family
from experiments.persist import load_experiment_report, save_experiment_report
from experiments.presets import PRESETS
from experiments.runner import run_experiment


def test_save_and_load_report_roundtrip(tmp_path: Path):
    preset = PRESETS["known_regression"]
    cases = by_family("direct_instruction_injection")[:2]
    result = run_experiment(
        preset.arm_a, preset.arm_b, experiment_name="test_persist", cases=cases,
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )
    path = save_experiment_report(result, sequential_outcome_key="exfiltration", runs_dir=tmp_path)
    assert path.exists()

    loaded = load_experiment_report("test_persist", runs_dir=tmp_path)
    assert loaded is not None
    assert loaded["name"] == "test_persist"
    assert loaded["arm_a_hash"] == result.arm_a_hash
    assert loaded["n_cases"] == 2
    assert set(loaded["family_results"].keys()) == set(result.family_results.keys())

    fam = loaded["family_results"]["exfiltration"][0]
    assert "effect" in fam
    assert "diff" in fam["effect"]
    assert "q_value" in fam
    assert "significant_after_correction" in fam


def test_report_without_sequential_key_has_null_sequential_analysis(tmp_path: Path):
    preset = PRESETS["known_neutral"]
    cases = by_family("tool_result_poisoning")[:2]
    result = run_experiment(
        preset.arm_a, preset.arm_b, experiment_name="test_persist_no_seq", cases=cases,
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )
    save_experiment_report(result, sequential_outcome_key=None, runs_dir=tmp_path)
    loaded = load_experiment_report("test_persist_no_seq", runs_dir=tmp_path)
    assert loaded["sequential_analysis"] is None


def test_report_with_sequential_key_has_points(tmp_path: Path):
    preset = PRESETS["known_regression"]
    cases = by_family("direct_instruction_injection")
    result = run_experiment(
        preset.arm_a, preset.arm_b, experiment_name="test_persist_seq", cases=cases,
        n_runs_per_case=3, max_workers=2, runs_dir=tmp_path,
    )
    save_experiment_report(result, sequential_outcome_key="exfiltration", runs_dir=tmp_path)
    loaded = load_experiment_report("test_persist_seq", runs_dir=tmp_path)
    seq = loaded["sequential_analysis"]
    assert seq is not None
    assert seq["outcome_key"] == "exfiltration"
    assert len(seq["points"]) == len(cases)
    assert "n" in seq["points"][0]
    assert "ci_low" in seq["points"][0]


def test_load_missing_report_returns_none(tmp_path: Path):
    assert load_experiment_report("does_not_exist", runs_dir=tmp_path) is None
