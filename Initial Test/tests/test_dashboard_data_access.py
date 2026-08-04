import json
from pathlib import Path

from dashboard.data_access import (
    find_flagged_run,
    flatten_family_rows,
    list_available_reports,
    load_calibration_sweep,
    load_raw_records,
    load_report,
)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_list_available_reports_orders_presets_first(tmp_path: Path):
    for name in ["added_agent", "aa", "custom_experiment", "known_regression"]:
        _write_json(tmp_path / f"{name}_report.json", {})
    result = list_available_reports(runs_dir=tmp_path)
    assert result == ["aa", "known_regression", "added_agent", "custom_experiment"]


def test_list_available_reports_empty_dir(tmp_path: Path):
    assert list_available_reports(runs_dir=tmp_path / "nope") == []


def test_load_report_missing_returns_none(tmp_path: Path):
    assert load_report("missing", runs_dir=tmp_path) is None


def test_load_report_roundtrip(tmp_path: Path):
    _write_json(tmp_path / "foo_report.json", {"name": "foo", "n_cases": 5})
    loaded = load_report("foo", runs_dir=tmp_path)
    assert loaded == {"name": "foo", "n_cases": 5}


def test_load_calibration_sweep_missing_returns_none(tmp_path: Path):
    assert load_calibration_sweep(runs_dir=tmp_path) is None


def test_load_calibration_sweep_roundtrip(tmp_path: Path):
    _write_json(tmp_path / "aa_calibration_sweep.json", {"points": [1, 2]})
    assert load_calibration_sweep(runs_dir=tmp_path) == {"points": [1, 2]}


def test_load_raw_records_missing_file_returns_empty(tmp_path: Path):
    assert load_raw_records("missing", runs_dir=tmp_path) == []


def test_load_raw_records_parses_jsonl(tmp_path: Path):
    path = tmp_path / "exp.jsonl"
    path.write_text('{"run_id": "r1"}\n{"run_id": "r2"}\n', encoding="utf-8")
    records = load_raw_records("exp", runs_dir=tmp_path)
    assert [r["run_id"] for r in records] == ["r1", "r2"]


def test_flatten_family_rows():
    report = {
        "family_results": {
            "exfiltration": [
                {
                    "family": "direct_instruction_injection",
                    "q_value": 0.02,
                    "significant_after_correction": True,
                    "effect": {
                        "rate_a": 0.1, "rate_b": 0.4, "diff": 0.3, "ci_low": 0.1, "ci_high": 0.5,
                        "n_cases": 5, "method": "cluster_bootstrap", "used_fallback": False, "fallback_reason": None,
                    },
                }
            ]
        }
    }
    rows = flatten_family_rows(report)
    assert len(rows) == 1
    assert rows[0]["outcome_key"] == "exfiltration"
    assert rows[0]["family"] == "direct_instruction_injection"
    assert rows[0]["diff"] == 0.3
    assert rows[0]["significant"] is True


def test_flatten_family_rows_empty_report():
    assert flatten_family_rows({}) == []


def test_find_flagged_run_returns_first_match():
    records = [
        {"run_id": "r1", "outcomes": {"exfiltration": False}},
        {"run_id": "r2", "outcomes": {"exfiltration": True}},
        {"run_id": "r3", "outcomes": {"exfiltration": True}},
    ]
    flagged = find_flagged_run(records, outcome_key="exfiltration")
    assert flagged["run_id"] == "r2"


def test_find_flagged_run_none_when_nothing_flagged():
    records = [{"run_id": "r1", "outcomes": {"exfiltration": False}}]
    assert find_flagged_run(records) is None


def test_find_flagged_run_empty_records():
    assert find_flagged_run([]) is None
