import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from stats.power import (
    HeterogeneityDominates,
    achieved_power,
    minimum_detectable_effect,
    power_curve,
    required_runs_per_case,
)


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


@pytest.mark.parametrize(
    ("n_cases", "n_runs_per_case", "baseline_rate"),
    [(17, 5, 0.15), (4, 15, 0.2), (25, 10, 0.1), (60, 3, 0.3)],
)
def test_achieved_power_round_trips_against_minimum_detectable_effect(n_cases, n_runs_per_case, baseline_rate):
    for target_power in (0.7, 0.8, 0.9):
        mde = minimum_detectable_effect(n_cases, n_runs_per_case, baseline_rate, power=target_power)
        got = achieved_power(n_cases, n_runs_per_case, baseline_rate, mde)
        assert got == pytest.approx(target_power, abs=1e-4)


def test_achieved_power_increases_with_observed_effect_size():
    small = achieved_power(17, 5, 0.15, 0.05)
    large = achieved_power(17, 5, 0.15, 0.20)
    assert 0 < small < large < 1


def test_achieved_power_decreases_with_fewer_cases():
    more_cases = achieved_power(20, 5, 0.15, 0.20)
    fewer_cases = achieved_power(4, 5, 0.15, 0.20)
    assert fewer_cases < more_cases


def test_achieved_power_is_symmetric_in_sign_of_observed_effect_at_centered_baseline():
    # _per_run_variance clamps p2 = baseline_rate + mde to [0.001, 0.999],
    # so at an off-center baseline (e.g. 0.15) a +0.20 and a -0.20 move
    # land on genuinely different points of p(1-p) and are NOT expected to
    # match — that's correct, pre-existing behavior shared by
    # minimum_detectable_effect/required_runs_per_case too, not something
    # to paper over here. At baseline_rate=0.5, p(1-p) is symmetric around
    # the move, so this is where a real symmetry check belongs.
    assert achieved_power(17, 5, 0.5, 0.20) == achieved_power(17, 5, 0.5, -0.20)


def test_achieved_power_rejects_invalid_sample_size():
    with pytest.raises(ValueError):
        achieved_power(0, 5, 0.15, 0.1)
    with pytest.raises(ValueError):
        achieved_power(5, 0, 0.15, 0.1)


def test_achieved_power_bounded_at_one():
    # A huge effect with essentially no per-run variance (baseline near 0
    # so p1(1-p1) and p2(1-p2) are both tiny) shouldn't overflow past 1.0.
    p = achieved_power(50, 50, 0.001, 0.5)
    assert 0.0 <= p <= 1.0
