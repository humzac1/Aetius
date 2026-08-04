from target_system.factory import baseline_config
from target_system.config import compute_config_hash

from attacker.cases import ATTACK_CASES
from attacker.executor import execute_case
from experiments.mock_policy import _complies, build_mock_scripts, compliance_probability


def test_compliance_probability_moves_in_expected_directions():
    base = baseline_config(provider="mock", defensive_instruction=True)
    no_defense = baseline_config(provider="mock", defensive_instruction=False)
    enforced = baseline_config(provider="mock", defensive_instruction=True, enforce_allowlist=True)

    assert compliance_probability(no_defense) > compliance_probability(base)
    assert compliance_probability(enforced) < compliance_probability(base)
    assert 0.0 < compliance_probability(base) < 1.0


def test_compliance_probability_rises_with_agent_count():
    from target_system.config import AgentSpec

    base = baseline_config(provider="mock")
    extra = AgentSpec(role="scheduler", name="Scheduler", system_prompt="x", tools=[])
    with_extra = baseline_config(provider="mock", extra_agents=[extra])
    assert compliance_probability(with_extra) > compliance_probability(base)


def test_roll_uses_common_random_numbers_not_config_hash():
    """Regression test for the exact bug the known_neutral/model_swap
    presets surfaced: two configs with an IDENTICAL compliance_probability
    but different config_hash must still get the SAME roll for a given
    (case, seed), or arm A and arm B get independent draws against the
    same threshold and disagree purely from sampling noise."""
    config_a = baseline_config(provider="mock", label="a", cosmetic_variant=False)
    config_b = baseline_config(provider="mock", label="b", cosmetic_variant=True)
    assert compute_config_hash(config_a) != compute_config_hash(config_b)
    assert compliance_probability(config_a) == compliance_probability(config_b)

    case = ATTACK_CASES[0]
    for seed in range(20):
        assert _complies(case, config_a, seed) == _complies(case, config_b, seed)


def test_roll_differs_when_probability_differs_enough():
    """The flip side: CRN shares the draw, but a real probability gap must
    still be able to flip the outcome for at least some seeds — otherwise
    known_regression couldn't detect anything either."""
    config_a = baseline_config(provider="mock", defensive_instruction=True)
    config_b = baseline_config(provider="mock", defensive_instruction=False)
    case = ATTACK_CASES[0]
    results_a = [_complies(case, config_a, seed) for seed in range(50)]
    results_b = [_complies(case, config_b, seed) for seed in range(50)]
    assert results_a != results_b
    assert sum(results_b) > sum(results_a)  # no defensive instruction -> more compliant


def test_build_mock_scripts_matches_complies_roll_for_every_case_and_several_seeds():
    config = baseline_config(provider="mock", email_allowlist=["billing@ourcompany.example"])
    for case in ATTACK_CASES:
        for seed in range(5):
            scripts = build_mock_scripts(case, config, seed)
            record = execute_case(config, case, seed=seed, arm="test", mock_scripts=scripts)
            assert record.error is None, f"{case.id} seed={seed}: {record.error}"
            assert record.outcomes.get(case.success_outcome) == _complies(case, config, seed), (
                f"{case.id} seed={seed}: script did not produce the intended compliance outcome"
            )
