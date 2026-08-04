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
            assert r.effect.diff == 0.0, f"{r.family}: expected exact-zero diff under CRN, got {r.effect.diff}"
            assert r.significant_after_correction is False


def test_known_regression_shows_positive_effect(tmp_path: Path):
    preset = PRESETS["known_regression"]
    cases = by_family("direct_instruction_injection")
    result = run_experiment(
        preset.arm_a, preset.arm_b, experiment_name="test_known_regression_effect",
        cases=cases, n_runs_per_case=5, max_workers=4, runs_dir=tmp_path,
    )
    exfil_results = {r.family: r for r in result.family_results["exfiltration"]}
    assert exfil_results["direct_instruction_injection"].effect.diff > 0
    assert exfil_results["direct_instruction_injection"].significant_after_correction is True
