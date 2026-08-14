"""End-to-end headless tests for the Streamlit dashboard via
streamlit.testing.v1.AppTest — runs the real app script against the real
data/runs/ artifacts (backfilled by experiments/calibration.py and the
preset CLI runs) and checks it renders without exceptions, plus a few
specific behaviors the build spec called out explicitly (flagged-case
default, multi-turn turn boundaries, evidence marking).

These depend on actual backfilled run artifacts existing — skipped if they
don't, rather than failing, since generating them takes real (if cheap,
mock-backend) computation that shouldn't run as a side effect of `pytest`.

The dashboard reads config.paths.RUNS_DIR (a platformdirs user data
location, not this repo), so the guard below checks *that* directory —
checking the repo's own data/runs/ would skip or run based on files the app
under test never opens. To run these against this checkout's backfilled
fixtures, point the app at them:

    AETIUS_DATA_DIR="$PWD/data" pytest tests/test_dashboard_app.py
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config import paths

APP_PATH = Path(__file__).parent.parent / "dashboard" / "app.py"
RUNS_DIR = paths.RUNS_DIR

pytestmark = pytest.mark.skipif(
    not (RUNS_DIR / "known_regression_report.json").exists(),
    reason="dashboard data not backfilled — run experiments/calibration.py and the preset CLIs first",
)


def _run_app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    return at


def test_app_runs_with_no_exceptions():
    at = _run_app()
    assert not at.exception


def test_calibration_panel_shows_fpr_metrics():
    at = _run_app()
    labels = [m.label for m in at.metric]
    assert any("Observed FPR" in label for label in labels)
    assert any("Nominal alpha" in label for label in labels)


def test_comparison_panel_defaults_to_known_regression():
    at = _run_app()
    comparison_selectbox = at.selectbox[0]
    assert comparison_selectbox.value == "known_regression"


def test_comparison_panel_selection_changes_without_error():
    at = _run_app()
    at.selectbox[0].select("known_neutral").run(timeout=60)
    assert not at.exception


def test_crn_panel_shows_before_after_metrics():
    at = _run_app()
    labels = [m.label for m in at.metric]
    assert any("before fix" in label for label in labels)
    assert any("after fix" in label for label in labels)


def test_power_curve_sliders_present_and_adjustable():
    at = _run_app()
    slider_labels = [s.label for s in at.slider]
    assert "Baseline rate" in slider_labels
    assert "Number of cases" in slider_labels
    baseline_slider = next(s for s in at.slider if s.label == "Baseline rate")
    baseline_slider.set_value(0.30).run(timeout=60)
    assert not at.exception


def test_trajectory_inspector_defaults_to_a_flagged_case():
    at = _run_app()
    run_selectbox = at.selectbox[-1]
    assert "FLAGGED" in run_selectbox.value


def test_trajectory_inspector_multi_turn_case_shows_turn_boundaries():
    at = _run_app()
    run_selectbox = at.selectbox[-1]
    multi_turn_option = next((o for o in run_selectbox.options if "multiturn" in o), None)
    if multi_turn_option is None:
        pytest.skip("no multi-turn run in known_regression's cached records")
    run_selectbox.select(multi_turn_option).run(timeout=60)
    assert not at.exception
    turn_headers = [m.value for m in at.markdown if m.value.startswith("##### Turn")]
    assert len(turn_headers) >= 2


def test_trajectory_inspector_marks_evidence_and_status():
    at = _run_app()
    markdown_values = [m.value for m in at.markdown]
    assert any("🎯" in v for v in markdown_values), "expected at least one evidence-marked event on the default flagged run"
    assert any("executed" in v or "blocked" in v for v in markdown_values)
