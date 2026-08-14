import pytest

from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig, compute_config_hash
from target_system.factory import baseline_config
from target_system.provenance import ReconstructionProvenance, ToolBehaviorProfile
from tui.data import single_config_run_path
from tui.execution import comparison_experiment_name, enforce_reconstructed_provider, peek_n_cached, run_comparison_check, run_single_config_check


def _mock_cases(n=2):
    from attacker.cases import ATTACK_CASES

    return list(ATTACK_CASES[:n])


# --- single-config check -----------------------------------------------------


def test_run_single_config_check_executes_all_jobs(tmp_path):
    config = baseline_config(label="smoke")
    cases = _mock_cases(2)
    result = run_single_config_check(config, cases=cases, n_runs_per_case=2, runs_dir=tmp_path)
    assert result.n_executed == 4  # 2 cases x 2 runs
    assert result.n_cached == 0
    assert len(result.records) == 4
    assert result.config_hash == compute_config_hash(config)


def test_run_single_config_check_writes_to_expected_path(tmp_path):
    config = baseline_config(label="smoke")
    cases = _mock_cases(1)
    result = run_single_config_check(config, cases=cases, n_runs_per_case=1, runs_dir=tmp_path)
    assert single_config_run_path(result.config_hash, runs_dir=tmp_path).exists()


def test_run_single_config_check_rerun_is_fully_cached(tmp_path):
    config = baseline_config(label="smoke")
    cases = _mock_cases(2)
    first = run_single_config_check(config, cases=cases, n_runs_per_case=2, runs_dir=tmp_path)
    second = run_single_config_check(config, cases=cases, n_runs_per_case=2, runs_dir=tmp_path)
    assert first.n_executed == 4
    assert second.n_executed == 0
    assert second.n_cached == 4


def test_run_single_config_check_extends_on_higher_runs_per_case(tmp_path):
    config = baseline_config(label="smoke")
    cases = _mock_cases(2)
    run_single_config_check(config, cases=cases, n_runs_per_case=2, runs_dir=tmp_path)
    extended = run_single_config_check(config, cases=cases, n_runs_per_case=3, runs_dir=tmp_path)
    assert extended.n_cached == 4
    assert extended.n_executed == 2  # one extra run per case
    assert len(extended.records) == 6


def test_run_single_config_check_progress_callback_fires_correctly(tmp_path):
    config = baseline_config(label="smoke")
    cases = _mock_cases(2)
    calls = []
    run_single_config_check(config, cases=cases, n_runs_per_case=1, runs_dir=tmp_path, on_progress=lambda c, t: calls.append((c, t)))
    assert calls[0] == (0, 2)
    assert calls[-1] == (2, 2)
    assert len(calls) == 3  # (0, total) + one per completed job


def test_run_single_config_check_records_have_no_arm(tmp_path):
    config = baseline_config(label="smoke")
    result = run_single_config_check(config, cases=_mock_cases(1), n_runs_per_case=1, runs_dir=tmp_path)
    assert all(r.arm is None for r in result.records)


# --- comparison check ---------------------------------------------------------


def test_comparison_experiment_name_is_deterministic_and_order_sensitive():
    assert comparison_experiment_name("cfg_a", "cfg_b") == comparison_experiment_name("cfg_a", "cfg_b")
    assert comparison_experiment_name("cfg_a", "cfg_b") != comparison_experiment_name("cfg_b", "cfg_a")


def test_run_comparison_check_delegates_to_run_experiment(tmp_path):
    config_a = baseline_config(label="on", defensive_instruction=True)
    config_b = baseline_config(label="off", defensive_instruction=False)
    cases = _mock_cases(2)
    result = run_comparison_check(config_a, config_b, cases=cases, n_runs_per_case=2, runs_dir=tmp_path)
    # the name is scoped to the case suite as well as the two arms, so a
    # different suite can't resume — and dilute — this file
    expected_name = comparison_experiment_name(
        compute_config_hash(config_a), compute_config_hash(config_b), cases=cases, runs_dir=tmp_path
    )
    assert result.name == expected_name
    assert result.n_executed == 8  # 2 cases x 2 runs x 2 arms
    assert set(result.family_results.keys()) == {
        "exfiltration",
        "exfiltration_attempted",
        "unauthorized_lookup",
        "unauthorized_lookup_attempted",
    }


def test_run_comparison_check_rerun_resumes_same_file(tmp_path):
    config_a = baseline_config(label="on", defensive_instruction=True)
    config_b = baseline_config(label="off", defensive_instruction=False)
    first = run_comparison_check(config_a, config_b, cases=_mock_cases(2), n_runs_per_case=2, runs_dir=tmp_path)
    second = run_comparison_check(config_a, config_b, cases=_mock_cases(2), n_runs_per_case=2, runs_dir=tmp_path)
    assert first.n_executed == 8
    assert second.n_executed == 0
    assert second.n_cached == 8


# --- reconstructed environments are real-model-only -------------------------


def _reconstructed_config(label, *, tools=("send_invoice",), provider="mock"):
    return SystemConfig(
        label=label,
        model=ModelConfig(provider=provider, model_name="mock-model"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="[unavailable]", system_prompt_source="unavailable", tools=list(tools))],
        security=SecurityConfig(),
        provenance=ReconstructionProvenance(
            project_id="proj-1", source_agent_name="A", trace_count=5, extraction_date="2026-01-01",
            tool_profiles={name: ToolBehaviorProfile(tool_name=name) for name in tools},
        ),
    )


def test_run_single_config_check_rejects_reconstructed_config_under_mock_provider(tmp_path):
    config = _reconstructed_config("a", provider="mock")
    with pytest.raises(ValueError, match="only run under provider='anthropic'"):
        run_single_config_check(config, cases=_mock_cases(1), n_runs_per_case=1, runs_dir=tmp_path)


def test_run_single_config_check_rejection_happens_before_any_job_runs(tmp_path):
    config = _reconstructed_config("a", provider="mock")
    with pytest.raises(ValueError):
        run_single_config_check(config, cases=_mock_cases(1), n_runs_per_case=1, runs_dir=tmp_path)
    assert not single_config_run_path(compute_config_hash(config), runs_dir=tmp_path).exists()


def test_run_comparison_check_rejects_reconstructed_config_under_mock_provider(tmp_path):
    config_a = _reconstructed_config("a", provider="mock")
    config_b = _reconstructed_config("b", provider="mock")
    with pytest.raises(ValueError, match="only run under provider='anthropic'"):
        run_comparison_check(config_a, config_b, cases=_mock_cases(1), n_runs_per_case=1, runs_dir=tmp_path)


# --- enforce_reconstructed_provider / peek_n_cached --------------------------


def test_enforce_reconstructed_provider_corrects_mock_to_anthropic():
    config = _reconstructed_config("a", provider="mock")
    corrected = enforce_reconstructed_provider(config)
    assert corrected.model.provider == "anthropic"
    assert corrected.provenance is config.provenance  # unrelated fields untouched


def test_enforce_reconstructed_provider_leaves_already_anthropic_config_untouched():
    config = _reconstructed_config("a", provider="anthropic")
    assert enforce_reconstructed_provider(config) is config


def test_enforce_reconstructed_provider_leaves_toy_config_untouched():
    config = baseline_config(label="toy")  # provider="mock", provenance=None
    assert enforce_reconstructed_provider(config) is config


def test_peek_n_cached_zero_before_anything_runs(tmp_path):
    config = baseline_config(label="peek")
    assert peek_n_cached([config], runs_dir=tmp_path) == 0


def test_peek_n_cached_single_config_matches_actual_run(tmp_path):
    config = baseline_config(label="peek-single")
    run_single_config_check(config, cases=_mock_cases(2), n_runs_per_case=1, runs_dir=tmp_path)
    assert peek_n_cached([config], runs_dir=tmp_path) == 2


def test_peek_n_cached_comparison_matches_actual_run(tmp_path):
    config_a = baseline_config(label="peek-a")
    config_b = baseline_config(label="peek-b")
    cases = _mock_cases(2)
    run_comparison_check(config_a, config_b, cases=cases, n_runs_per_case=1, runs_dir=tmp_path)
    assert peek_n_cached([config_a, config_b], cases=cases, runs_dir=tmp_path) == 4
