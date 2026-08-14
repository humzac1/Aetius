"""A run file must never hold two different case suites.

The dilution this prevents was real and silent. A comparison of the
E-Commerce environment's two arms first ran the hand-authored suite (which
that environment cannot engage with — every rate 0.0), then ran again with
the domain-adapted suite. Both used the same experiment name, because the
name derived from the two config hashes alone, so the second run appended
into the first one's file. compare_families then averaged per-case rates
across the union of both suites and the saved report recorded
unauthorized_lookup at 0.433 when the suite actually run measured 0.867 —
exactly half, because half the case ids in the file were the dead suite.
Nothing in the report indicated two suites were present.
"""

import pytest

from attacker.attack_case import AttackCase
from experiments.runner import CaseSuiteMismatch, assert_case_suite_matches, run_experiment, suite_digest
from target_system.config import compute_config_hash
from target_system.factory import baseline_config
from target_system.logging_schema import RunRecord
from tui.execution import comparison_experiment_name, run_comparison_check


def _cases(ids, family="direct_instruction_injection"):
    return [
        AttackCase(
            id=i, family=family, injection_vector="task_text", success_outcome="exfiltration",
            source="test", benign_task="do x", injected_payload="do y",
        )
        for i in ids
    ]


def _records(case_ids):
    return [
        RunRecord(run_id=f"r{i}", config_hash="cfg_x", case_id=cid, arm="a", seed=0,
                  started_at="t", ended_at="t", wall_time_seconds=1.0, events=[])
        for i, cid in enumerate(case_ids)
    ]


# --- the guard ------------------------------------------------------------------


def test_appending_a_different_suite_is_refused(tmp_path):
    existing = _records(["hand_1", "hand_2"])
    incoming = _cases(["generated_1__cfg_abc", "generated_2__cfg_abc"])
    with pytest.raises(CaseSuiteMismatch, match="different case suite"):
        assert_case_suite_matches(existing, incoming, tmp_path / "run.jsonl")


def test_the_real_dilution_shape_is_refused(tmp_path):
    """The exact mixture that produced 0.433-instead-of-0.867: five dead
    hand-authored cases already on disk, five live domain-adapted ones
    incoming."""
    existing = _records([f"direct_case_{i}" for i in range(5)])
    incoming = _cases([f"direct_case_{i}__cfg_11c5bca10655" for i in range(5)])
    with pytest.raises(CaseSuiteMismatch):
        assert_case_suite_matches(existing, incoming, tmp_path / "run.jsonl")


def test_resuming_the_same_suite_is_allowed(tmp_path):
    ids = ["case_1", "case_2"]
    assert_case_suite_matches(_records(ids), _cases(ids), tmp_path / "run.jsonl")  # no raise


def test_an_empty_file_accepts_any_suite(tmp_path):
    assert_case_suite_matches([], _cases(["case_1"]), tmp_path / "run.jsonl")  # no raise


# --- suite-aware naming keeps the two apart -------------------------------------


def test_different_suites_get_different_run_files(tmp_path):
    a, b = compute_config_hash(baseline_config(label="a")), compute_config_hash(baseline_config(label="b"))
    first = _cases(["hand_1", "hand_2"])
    second = _cases(["generated_1", "generated_2"])

    name_first = comparison_experiment_name(a, b, cases=first, runs_dir=tmp_path)
    (tmp_path / f"{name_first}.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in _records(["hand_1", "hand_2"])), encoding="utf-8"
    )
    name_second = comparison_experiment_name(a, b, cases=second, runs_dir=tmp_path)

    assert name_first != name_second
    assert name_second.endswith(suite_digest({"generated_1", "generated_2"}))


def test_the_same_suite_resumes_the_same_file(tmp_path):
    a, b = compute_config_hash(baseline_config(label="a")), compute_config_hash(baseline_config(label="b"))
    cases = _cases(["case_1", "case_2"])
    name = comparison_experiment_name(a, b, cases=cases, runs_dir=tmp_path)
    (tmp_path / f"{name}.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in _records(["case_1", "case_2"])), encoding="utf-8"
    )
    assert comparison_experiment_name(a, b, cases=cases, runs_dir=tmp_path) == name


def test_end_to_end_two_suites_produce_two_clean_reports(tmp_path):
    """The whole point: run two different suites against the same pair of
    arms and neither result contains the other's cases."""
    config_a = baseline_config(label="a", defensive_instruction=True)
    config_b = baseline_config(label="b", defensive_instruction=False)

    first = run_comparison_check(config_a, config_b, cases=_cases(["s1_c1", "s1_c2"]), n_runs_per_case=1, runs_dir=tmp_path)
    second = run_comparison_check(config_a, config_b, cases=_cases(["s2_c1", "s2_c2"]), n_runs_per_case=1, runs_dir=tmp_path)

    assert first.name != second.name
    assert {r.case_id for r in first.records} == {"s1_c1", "s1_c2"}
    assert {r.case_id for r in second.records} == {"s2_c1", "s2_c2"}
    # and neither one's stats saw the other's cases
    assert first.n_cases == 2 and second.n_cases == 2


def test_a_caller_that_forces_a_colliding_name_is_refused(tmp_path):
    """Naming avoids the collision on the normal path; the guard is what
    makes it impossible for any other path."""
    config_a = baseline_config(label="a", defensive_instruction=True)
    config_b = baseline_config(label="b", defensive_instruction=False)
    run_experiment(config_a, config_b, experiment_name="forced", cases=_cases(["s1_c1"]), n_runs_per_case=1, runs_dir=tmp_path)

    with pytest.raises(CaseSuiteMismatch):
        run_experiment(config_a, config_b, experiment_name="forced", cases=_cases(["s2_c1"]), n_runs_per_case=1, runs_dir=tmp_path)
