import numpy as np
import pytest

from stats.paired import cluster_bootstrap_diff, mcnemar_test, mixed_effects_diff
from stats.types import CaseObservations, PairedCaseData


def _make_paired_data(rng, n_cases, n_per_case, effect, base_low=0.05, base_high=0.25):
    data = []
    for i in range(n_cases):
        p_a = rng.uniform(base_low, base_high)
        p_b = min(0.95, max(0.01, p_a + effect))
        a = tuple(int(x) for x in rng.binomial(1, p_a, n_per_case))
        b = tuple(int(x) for x in rng.binomial(1, p_b, n_per_case))
        data.append(PairedCaseData(f"c{i}", "fam", CaseObservations(f"c{i}", "fam", a), CaseObservations(f"c{i}", "fam", b)))
    return data


def test_cluster_bootstrap_degenerate_identical_arms_not_flagged_significant():
    """Regression test: found via Part 4's A/A preset, where a fully
    deterministic mock backend gave every case byte-identical arm_a/arm_b
    outcomes, making the bootstrap distribution a zero-variance point mass.
    BCa's z0/acceleration machinery divided by quantities that vanish in
    that case, feeding norm.ppf values near 0/1 and reporting p=0.0
    ("significant") despite an exact, literal 0.0pp difference."""
    data = [
        PairedCaseData(f"c{i}", "fam", CaseObservations(f"c{i}", "fam", (1, 0, 1)), CaseObservations(f"c{i}", "fam", (1, 0, 1)))
        for i in range(5)
    ]
    result = cluster_bootstrap_diff(data, seed=1, n_boot=500)
    assert result.diff == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.p_value == 1.0
    assert result.extra.get("degenerate_zero_variance") is True


def test_cluster_bootstrap_degenerate_certain_nonzero_diff_is_flagged():
    data = [
        PairedCaseData(f"c{i}", "fam", CaseObservations(f"c{i}", "fam", (1, 1, 1)), CaseObservations(f"c{i}", "fam", (0, 0, 0)))
        for i in range(5)
    ]
    result = cluster_bootstrap_diff(data, seed=1, n_boot=500)
    assert result.diff == -1.0
    assert result.ci_low == result.ci_high == -1.0
    assert result.p_value == 0.0


def test_cluster_bootstrap_recovers_known_effect():
    rng = np.random.default_rng(42)
    data = _make_paired_data(rng, n_cases=25, n_per_case=25, effect=0.10)
    result = cluster_bootstrap_diff(data, seed=1, n_boot=2000)
    assert 0.06 < result.diff < 0.16
    assert result.ci_low > 0  # excludes zero — the known effect should be detected
    assert result.p_value < 0.05


def test_cluster_bootstrap_null_ci_usually_contains_zero():
    rng = np.random.default_rng(3)
    data = _make_paired_data(rng, n_cases=25, n_per_case=25, effect=0.0)
    result = cluster_bootstrap_diff(data, seed=1, n_boot=2000)
    assert result.ci_low < 0 < result.ci_high


def test_cluster_bootstrap_requires_at_least_two_cases():
    rng = np.random.default_rng(0)
    data = _make_paired_data(rng, n_cases=1, n_per_case=10, effect=0.0)
    with pytest.raises(ValueError):
        cluster_bootstrap_diff(data)


def test_mcnemar_requires_single_run_per_case():
    rng = np.random.default_rng(0)
    data = _make_paired_data(rng, n_cases=10, n_per_case=5, effect=0.0)
    with pytest.raises(ValueError):
        mcnemar_test(data)


def test_mcnemar_table_construction_matches_manual_counts():
    # 4 cases: (a=1,b=1), (a=1,b=0), (a=0,b=1), (a=0,b=0)
    data = [
        PairedCaseData("c1", "f", CaseObservations("c1", "f", (1,)), CaseObservations("c1", "f", (1,))),
        PairedCaseData("c2", "f", CaseObservations("c2", "f", (1,)), CaseObservations("c2", "f", (0,))),
        PairedCaseData("c3", "f", CaseObservations("c3", "f", (0,)), CaseObservations("c3", "f", (1,))),
        PairedCaseData("c4", "f", CaseObservations("c4", "f", (0,)), CaseObservations("c4", "f", (0,))),
    ]
    result = mcnemar_test(data)
    assert result.extra["n11"] == 1
    assert result.extra["n10"] == 1
    assert result.extra["n01"] == 1
    assert result.extra["n00"] == 1
    assert result.n_cases == 4
    assert result.diff == 0.0  # n01 - n10 = 0


def test_mcnemar_correction_true_is_more_conservative_than_false():
    rng = np.random.default_rng(1)
    # many discordant pairs so the asymptotic (not exact) test applies
    a = rng.binomial(1, 0.4, 200)
    b = rng.binomial(1, 0.5, 200)
    data = [
        PairedCaseData(f"c{i}", "f", CaseObservations(f"c{i}", "f", (int(a[i]),)), CaseObservations(f"c{i}", "f", (int(b[i]),)))
        for i in range(200)
    ]
    corrected = mcnemar_test(data, correction=True)
    uncorrected = mcnemar_test(data, correction=False)
    assert corrected.p_value >= uncorrected.p_value


def test_mixed_effects_falls_back_below_min_cases():
    rng = np.random.default_rng(0)
    data = _make_paired_data(rng, n_cases=3, n_per_case=5, effect=0.0)
    result = mixed_effects_diff(data)
    assert result.used_fallback is True
    assert "cases" in result.fallback_reason


def test_mixed_effects_recovers_known_effect_when_stable():
    rng = np.random.default_rng(7)
    data = _make_paired_data(rng, n_cases=20, n_per_case=25, effect=0.12)
    result = mixed_effects_diff(data, seed=1)
    assert result.used_fallback is False
    assert 0.05 < result.diff < 0.20
    assert result.ci_low > 0
    assert result.extra["odds_ratio"] > 1
