import math

import numpy as np
import pytest

from stats.sequential import group_sequential_boundaries, mixture_sprt_confidence_sequence


def test_mixture_sprt_does_not_overflow_on_strong_early_evidence():
    """Regression test: found while backfilling a Part 5 dashboard report
    with only 2 cases of a clear-cut effect (small sigma, consistent large
    diffs) — math.exp(log_e) raised OverflowError instead of returning inf
    for a martingale that had unambiguously diverged. The correct behavior
    is e_value=inf (certain evidence), not a crash."""
    diffs = [1.0, 1.0]
    result = mixture_sprt_confidence_sequence(diffs, alpha=0.05, tau=0.1, sigma=0.01)
    assert all(math.isinf(p.e_value) or p.e_value > 0 for p in result.points)
    assert result.points[-1].e_value == math.inf
    assert result.points[-1].always_valid_p == 0.0
    assert result.can_stop_now()


def test_obrien_fleming_boundaries_match_published_reference():
    # Jennison & Turnbull, classical O'Brien-Fleming, K=5 equally-spaced,
    # two-sided alpha=0.05: approx [4.56, 3.23, 2.63, 2.28, 2.03]. The
    # Lan-DeMets error-spending approximation to O'Brien-Fleming computed
    # here won't match exactly (different construction), but should be
    # close — this is the check that caught a real bug (a spurious-
    # trapezoid integration error) during development.
    design = group_sequential_boundaries([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05, spending="obrien_fleming")
    reference = [4.56, 3.23, 2.63, 2.28, 2.03]
    for got, ref in zip(design.boundaries, reference):
        assert got == pytest.approx(ref, rel=0.1)
    # Qualitative shape: strictly decreasing (heavy early conservatism).
    assert all(a > b for a, b in zip(design.boundaries, design.boundaries[1:]))


def test_pocock_boundaries_are_approximately_constant():
    design = group_sequential_boundaries([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05, spending="pocock")
    reference = 2.41
    for b in design.boundaries:
        assert b == pytest.approx(reference, rel=0.05)


def test_group_sequential_design_rejects_bad_information_fractions():
    with pytest.raises(ValueError):
        group_sequential_boundaries([0.5, 0.2, 1.0])  # not increasing
    with pytest.raises(ValueError):
        group_sequential_boundaries([0.3, 0.6])  # doesn't end at 1.0


def test_group_sequential_evaluate_stops_when_boundary_crossed():
    design = group_sequential_boundaries([0.5, 1.0], alpha=0.05, spending="pocock")
    result = design.evaluate([1.0, 5.0])  # first look weak, second look huge
    assert result.can_stop_now()
    assert result.stop_at_look == 2


def test_mixture_sprt_null_control():
    rng = np.random.default_rng(21)
    n_sims = 500
    ever_reject = 0
    for _ in range(n_sims):
        diffs = rng.normal(0, 0.15, 25).tolist()
        result = mixture_sprt_confidence_sequence(diffs, alpha=0.05, tau=0.1, sigma=0.15)
        if result.can_stop_now():
            ever_reject += 1
    # Anytime-valid guarantee is an upper bound (Ville's inequality), so
    # this should be well within alpha, generously bounded to avoid flakes.
    assert ever_reject / n_sims <= 0.08


def test_mixture_sprt_detects_real_effect_eventually():
    rng = np.random.default_rng(22)
    n_sims = 500
    ever_reject = 0
    for _ in range(n_sims):
        diffs = rng.normal(0.10, 0.15, 25).tolist()
        result = mixture_sprt_confidence_sequence(diffs, alpha=0.05, tau=0.1, sigma=0.15)
        if result.can_stop_now():
            ever_reject += 1
    assert ever_reject / n_sims > 0.4  # meaningfully more power than the null rate


def test_mixture_sprt_requires_at_least_two_points():
    with pytest.raises(ValueError):
        mixture_sprt_confidence_sequence([0.1])


def test_mixture_sprt_confidence_sequence_narrows_over_time():
    rng = np.random.default_rng(23)
    diffs = rng.normal(0.05, 0.1, 40).tolist()
    result = mixture_sprt_confidence_sequence(diffs, alpha=0.05, tau=0.1)
    widths = [p.ci_high - p.ci_low for p in result.points if np.isfinite(p.ci_high - p.ci_low)]
    # Not monotonic point-to-point (depends on realized data) but the last
    # finite width should be smaller than the first — the sequence narrows.
    assert widths[-1] < widths[0]
