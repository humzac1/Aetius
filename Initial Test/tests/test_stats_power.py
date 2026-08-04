import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from stats.power import HeterogeneityDominates, minimum_detectable_effect, power_curve, required_runs_per_case


def test_required_runs_matches_statsmodels_two_proportion_power_at_n_cases_1():
    p1, mde = 0.1, 0.05
    es = proportion_effectsize(p1 + mde, p1)
    n_sm = NormalIndPower().solve_power(effect_size=es, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided")
    n_mine = required_runs_per_case(baseline_rate=p1, mde=mde, n_cases=1, power=0.8, alpha=0.05)
    assert n_mine == pytest.approx(n_sm, rel=0.02)


def test_required_runs_and_mde_are_approximately_inverse():
    n = required_runs_per_case(baseline_rate=0.1, mde=0.05, n_cases=20, power=0.8, alpha=0.05)
    mde_back = minimum_detectable_effect(n_cases=20, n_runs_per_case=n, baseline_rate=0.1, power=0.8, alpha=0.05)
    assert mde_back <= 0.05 + 1e-6  # n was rounded up, so mde_back should be <= the target


def test_more_cases_reduces_required_runs_per_case():
    n_small = required_runs_per_case(baseline_rate=0.1, mde=0.05, n_cases=10, power=0.8)
    n_large = required_runs_per_case(baseline_rate=0.1, mde=0.05, n_cases=40, power=0.8)
    assert n_large < n_small


def test_heterogeneity_dominates_raised_when_between_case_variance_too_large():
    with pytest.raises(HeterogeneityDominates):
        required_runs_per_case(baseline_rate=0.1, mde=0.02, n_cases=5, power=0.8, between_case_sd=0.05)


def test_between_case_sd_increases_minimum_detectable_effect():
    mde_no_het = minimum_detectable_effect(n_cases=20, n_runs_per_case=30, baseline_rate=0.1, between_case_sd=0.0)
    mde_with_het = minimum_detectable_effect(n_cases=20, n_runs_per_case=30, baseline_rate=0.1, between_case_sd=0.05)
    assert mde_with_het > mde_no_het


def test_power_curve_is_monotonically_decreasing():
    curve = power_curve(n_cases=20, baseline_rate=0.1, runs_per_case_grid=[5, 10, 20, 40, 80])
    mdes = [mde for _, mde in curve]
    assert all(a > b for a, b in zip(mdes, mdes[1:]))
