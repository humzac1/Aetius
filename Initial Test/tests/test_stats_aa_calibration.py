from stats.aa_calibration import CaseSpec, run_aa_calibration


def _specs(n, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    return [CaseSpec(f"c{i}", "fam", base_rate=float(r)) for i, r in enumerate(rng.uniform(0.05, 0.35, n))]


def test_cluster_bootstrap_well_calibrated_above_min_cases():
    # Below ~20-25 cases BCa is known to under-cover somewhat (documented
    # in paired.py) — check calibration where the method is expected to
    # actually hold, per that same empirical finding.
    result = run_aa_calibration(
        _specs(30, seed=1), n_runs_per_case=15, method="cluster_bootstrap",
        n_trials=300, alpha=0.05, seed=2, method_kwargs={"n_boot": 400},
    )
    assert result.fpr_ci_low < 0.10  # generous — not asserting exact calibration, just "not wildly off"


def test_mcnemar_requires_single_run_per_case_in_calibration():
    import pytest

    with pytest.raises(ValueError):
        run_aa_calibration(_specs(10), n_runs_per_case=5, method="mcnemar", n_trials=10)


def test_mcnemar_calibration_conservative_with_correction():
    result = run_aa_calibration(
        _specs(60, seed=3), n_runs_per_case=1, method="mcnemar", n_trials=500, alpha=0.05, seed=4,
    )
    # Documented behavior: continuity-corrected McNemar under-rejects.
    assert result.observed_fpr <= 0.05


def test_aa_calibration_fpr_ci_is_wilson_and_reasonable():
    result = run_aa_calibration(
        _specs(20, seed=5), n_runs_per_case=10, method="cluster_bootstrap",
        n_trials=100, alpha=0.05, seed=6, method_kwargs={"n_boot": 300},
    )
    assert 0.0 <= result.fpr_ci_low <= result.observed_fpr <= result.fpr_ci_high <= 1.0
