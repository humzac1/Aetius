import subprocess
import sys

from stats.reporting import format_cuped_result, format_effect
from stats.types import EffectEstimate
from stats.variance_reduction import cuped_adjust


def test_format_effect_never_reports_a_bare_pvalue_without_effect_size():
    effect = EffectEstimate(
        method="cluster_bootstrap", rate_a=0.021, rate_b=0.048, diff=0.027,
        ci_low=0.011, ci_high=0.043, alpha=0.05, p_value=0.001, n_cases=20,
    )
    sentence = format_effect(effect, q_value=0.02)
    assert "2.1%" in sentence
    assert "4.8%" in sentence
    assert "+2.7 points" in sentence
    assert "1.1" in sentence and "4.3" in sentence
    assert "q = 0.020" in sentence
    assert "rose" in sentence


def test_format_effect_reports_fell_for_negative_diff():
    effect = EffectEstimate(
        method="mcnemar", rate_a=0.10, rate_b=0.04, diff=-0.06,
        ci_low=-0.09, ci_high=-0.03, alpha=0.05, p_value=0.01, n_cases=15,
    )
    sentence = format_effect(effect)
    assert "fell" in sentence


def test_format_effect_notes_fallback():
    effect = EffectEstimate(
        method="mixed_effects_logistic", rate_a=0.1, rate_b=0.15, diff=0.05,
        ci_low=-0.02, ci_high=0.12, alpha=0.05, p_value=0.2, n_cases=3,
        used_fallback=True, fallback_reason="only 3 cases",
    )
    sentence = format_effect(effect)
    assert "fell back" in sentence
    assert "only 3 cases" in sentence


def test_format_cuped_result_reports_variance_reduction():
    result = cuped_adjust([0.3, 0.5, 0.2, 0.4, 0.6], [0.25, 0.45, 0.15, 0.35, 0.55])
    sentence = format_cuped_result(result)
    assert "%" in sentence
    assert "CUPED" in sentence


def test_cli_aa_calibration_runs_end_to_end():
    result = subprocess.run(
        [sys.executable, "-m", "stats.cli", "aa-calibration", "--n-cases", "25", "--runs-per-case", "10",
         "--method", "cluster_bootstrap", "--trials", "40", "--n-boot", "200", "--seed", "1"],
        capture_output=True, text=True, cwd=".",
    )
    assert "observed FPR" in result.stdout


def test_cli_power_runs_end_to_end():
    result = subprocess.run(
        [sys.executable, "-m", "stats.cli", "power", "--baseline-rate", "0.1", "--mde", "0.05", "--n-cases", "20"],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "runs per case" in result.stdout
