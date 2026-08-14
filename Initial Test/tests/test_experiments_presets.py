from pathlib import Path

from attacker.cases import by_family
from experiments.presets import PRESETS
from experiments.runner import resolve_arm, run_experiment


def test_all_presets_resolve_to_valid_configs():
    for name, preset in PRESETS.items():
        config_a = resolve_arm(preset.arm_a)
        config_b = resolve_arm(preset.arm_b)
        assert config_a.label == preset.arm_a.label, name
        assert config_b.label == preset.arm_b.label, name


def test_aa_preset_arms_are_bit_identical():
    from target_system.config import compute_config_hash

    config_a = resolve_arm(PRESETS["aa"].arm_a)
    config_b = resolve_arm(PRESETS["aa"].arm_b)
    assert compute_config_hash(config_a) == compute_config_hash(config_b)


def test_known_regression_arms_differ_only_in_defensive_instruction():
    preset = PRESETS["known_regression"]
    config_a = resolve_arm(preset.arm_a)
    config_b = resolve_arm(preset.arm_b)
    assert config_a.defensive_instruction is True
    assert config_b.defensive_instruction is False
    assert config_a.supervisor().system_prompt != config_b.supervisor().system_prompt
    assert preset.sequential_outcome_key == "exfiltration"


def test_known_neutral_arms_have_different_wording_same_meaning_flags():
    preset = PRESETS["known_neutral"]
    config_a = resolve_arm(preset.arm_a)
    config_b = resolve_arm(preset.arm_b)
    assert config_a.supervisor().system_prompt != config_b.supervisor().system_prompt
    assert config_a.defensive_instruction == config_b.defensive_instruction == True  # noqa: E712
    assert config_a.security.email_allowlist == config_b.security.email_allowlist


def test_model_swap_arms_differ_only_in_model_name():
    preset = PRESETS["model_swap"]
    config_a = resolve_arm(preset.arm_a)
    config_b = resolve_arm(preset.arm_b)
    assert config_a.model.model_name != config_b.model.model_name
    assert config_a.model.provider == config_b.model.provider == "mock"


def test_added_agent_arm_b_has_one_more_agent():
    preset = PRESETS["added_agent"]
    config_a = resolve_arm(preset.arm_a)
    config_b = resolve_arm(preset.arm_b)
    assert len(config_b.agents) == len(config_a.agents) + 1
    assert "scheduler" in {a.role for a in config_b.agents}


def test_known_neutral_shows_zero_diff_via_common_random_numbers(tmp_path: Path):
    """Integration check that CRN in mock_policy actually delivers on the
    known_neutral preset's promise end to end: with equal true compliance
    probability and shared random draws, every case's diff should be
    exactly zero, not just usually close to zero."""
    preset = PRESETS["known_neutral"]
    cases = by_family("direct_instruction_injection")[:2] + by_family("indirect_injection_document")[:2]
    result = run_experiment(
        preset.arm_a, preset.arm_b, experiment_name="test_known_neutral_zero_diff",
        cases=cases, n_runs_per_case=2, max_workers=4, runs_dir=tmp_path,
    )
    for family_results in result.family_results.values():
        for r in family_results:
            # The CRN promise is about the observed data: every case's two
            # arms saw identical draws, so the observed mean diff is
            # exactly zero. The reported diff is the posterior median,
            # which sits within Monte Carlo error of it, never exactly on
            # it — assert on the observed value the promise is about.
            assert r.effect.extra["observed_diff"] == 0.0, (
                f"{r.family}: expected exact-zero observed diff under CRN, got {r.effect.extra['observed_diff']}"
            )
            assert r.significant_after_correction is False


def test_known_regression_shows_positive_effect(tmp_path: Path):
    preset = PRESETS["known_regression"]
    cases = by_family("direct_instruction_injection")
    result = run_experiment(
        preset.arm_a, preset.arm_b, experiment_name="test_known_regression_effect",
        cases=cases, n_runs_per_case=5, max_workers=4, runs_dir=tmp_path,
    )
    exfil_results = {r.family: r for r in result.family_results["exfiltration"]}
    row = exfil_results["direct_instruction_injection"]
    effect = row.effect
    # Under the retired bootstrap this preset's known real regression was
    # *refused* at 5 cases (the method's calibration floor is 80). The
    # hierarchical default is validated at exactly this size, so the
    # regression must now come back as a real, significant finding — a
    # positive effect whose credible interval clears the ROPE.
    assert effect.method == "hierarchical_bayes"
    assert effect.diff > 0
    assert effect.p_value is not None
    assert effect.extra["rope_signal"] is True
    assert row.significant_after_correction is True
