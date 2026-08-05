from experiments.cost_estimate import estimate_batch_cost, format_cost_estimate
from target_system.factory import baseline_config
from target_system.provenance import ReconstructionProvenance


def _cases(n):
    from attacker.cases import ATTACK_CASES

    return list(ATTACK_CASES[:n])


def _reconstructed_config(*, avg_cost=None, avg_generations=None, model_name="claude-x"):
    config = baseline_config(label="reconstructed", provider="anthropic", model_name=model_name)
    provenance = ReconstructionProvenance(
        project_id="proj-1", source_agent_name="A", trace_count=10, extraction_date="2026-01-01",
        avg_cost_usd_per_trace=avg_cost, avg_generations_per_trace=avg_generations,
    )
    return config.model_copy(update={"provenance": provenance})


# --- mock-only batches cost nothing ------------------------------------


def test_mock_only_batch_has_zero_cost():
    mock_config = baseline_config()  # provider="mock" by default
    est = estimate_batch_cost(_cases(3), [mock_config, mock_config], n_runs_per_case=5)
    assert est.any_real_model is False
    assert est.estimated_cost_usd == 0.0
    assert est.n_jobs_total == 3 * 5 * 2


def test_format_mock_only_mentions_no_api_cost():
    mock_config = baseline_config()
    est = estimate_batch_cost(_cases(2), [mock_config], n_runs_per_case=1)
    text = format_cost_estimate(est)
    assert "no API cost" in text


# --- real-data-grounded estimate ---------------------------------------


def test_real_model_with_provenance_uses_observed_cost():
    config = _reconstructed_config(avg_cost=0.05, avg_generations=3.0)
    est = estimate_batch_cost(_cases(2), [config], n_runs_per_case=2)
    assert est.grounded_in_real_data is True
    assert est.any_real_model is True
    n_jobs = 2 * 2 * 1
    assert est.n_jobs_total == n_jobs
    assert est.estimated_cost_usd == n_jobs * 0.05
    assert est.estimated_llm_calls == n_jobs * 3.0


def test_real_data_averaged_across_both_arms_when_both_have_it():
    config_a = _reconstructed_config(avg_cost=0.04)
    config_b = _reconstructed_config(avg_cost=0.06)
    est = estimate_batch_cost(_cases(1), [config_a, config_b], n_runs_per_case=1)
    assert est.grounded_in_real_data is True
    # 2 jobs total (1 case x 1 run x 2 arms), avg cost per run = 0.05
    assert est.estimated_cost_usd == 2 * 0.05


def test_real_data_used_when_only_one_arm_has_it():
    config_a = _reconstructed_config(avg_cost=0.10)
    config_b = baseline_config(provider="anthropic")  # no provenance at all
    est = estimate_batch_cost(_cases(1), [config_a, config_b], n_runs_per_case=1)
    assert est.grounded_in_real_data is True
    assert est.estimated_cost_usd == 2 * 0.10  # only config_a's data used, applied per job


# --- fallback estimate (no real data available) -----------------------


def test_fallback_used_when_no_config_has_provenance_cost_data():
    config = baseline_config(provider="anthropic", model_name="claude-sonnet-5")
    est = estimate_batch_cost(_cases(2), [config], n_runs_per_case=1)
    assert est.grounded_in_real_data is False
    assert est.any_real_model is True
    assert est.estimated_cost_usd > 0


def test_fallback_used_when_provenance_exists_but_cost_field_is_none():
    config = _reconstructed_config(avg_cost=None)
    est = estimate_batch_cost(_cases(1), [config], n_runs_per_case=1)
    assert est.grounded_in_real_data is False


def test_format_fallback_mentions_rough_estimate():
    config = baseline_config(provider="anthropic")
    est = estimate_batch_cost(_cases(1), [config], n_runs_per_case=1)
    text = format_cost_estimate(est)
    assert "rough estimate" in text


# --- resumability: n_cached reduces remaining work, not total ------------


def test_n_cached_reduces_remaining_but_not_total():
    config = _reconstructed_config(avg_cost=0.01)
    est = estimate_batch_cost(_cases(5), [config], n_runs_per_case=10, n_cached=30)
    assert est.n_jobs_total == 50
    assert est.n_jobs_remaining == 20
    assert est.estimated_cost_usd == 20 * 0.01  # cost reflects only the remaining work


def test_n_cached_never_makes_remaining_negative():
    config = _reconstructed_config(avg_cost=0.01)
    est = estimate_batch_cost(_cases(1), [config], n_runs_per_case=1, n_cached=999)
    assert est.n_jobs_remaining == 0
    assert est.estimated_cost_usd == 0.0


def test_format_includes_remaining_and_total_when_partially_cached():
    config = _reconstructed_config(avg_cost=0.01, avg_generations=1.0)
    est = estimate_batch_cost(_cases(2), [config], n_runs_per_case=5, n_cached=5)
    text = format_cost_estimate(est)
    assert "5 of 10" in text
