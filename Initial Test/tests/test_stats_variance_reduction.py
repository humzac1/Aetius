import numpy as np

from stats.variance_reduction import cuped_adjust, measure_crn_variance_reduction


def test_crn_reduces_variance_when_arms_are_close():
    case_rates = {f"c{i}": (p, min(0.95, p + 0.08)) for i, p in enumerate(np.linspace(0.05, 0.35, 15))}
    result = measure_crn_variance_reduction(case_rates, n_runs_per_case=15, n_sims=1500, seed=0)
    assert result.variance_reduction_pct > 20  # meaningful reduction, not just noise
    assert result.var_with_crn < result.var_without_crn
    assert result.effective_sample_size_multiplier > 1


def test_cuped_variance_reduction_matches_correlation_squared_theory():
    rng = np.random.default_rng(1)
    x = rng.normal(0.2, 0.1, 800)
    noise = rng.normal(0, 0.05, 800)
    y = 0.5 * x + noise
    result = cuped_adjust(y.tolist(), x.tolist())
    expected_reduction = 100 * result.correlation**2
    assert abs(result.variance_reduction_pct - expected_reduction) < 1.0


def test_cuped_preserves_the_mean():
    rng = np.random.default_rng(2)
    y = rng.normal(0.3, 0.1, 200)
    x = rng.normal(0.5, 0.2, 200)
    result = cuped_adjust(y.tolist(), x.tolist())
    assert abs(result.mean_after - result.mean_before) < 1e-9


def test_cuped_no_correlation_no_reduction():
    rng = np.random.default_rng(3)
    y = rng.normal(0.3, 0.1, 500)
    x = rng.normal(0.5, 0.2, 500)  # independent of y
    result = cuped_adjust(y.tolist(), x.tolist())
    assert abs(result.variance_reduction_pct) < 5  # near zero, allowing sampling noise
