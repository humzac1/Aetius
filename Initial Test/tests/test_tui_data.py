import dataclasses
import json

from target_system.config import save_config
from target_system.factory import baseline_config
from target_system.logging_schema import RunRecord, append_run_record
from tui.data import (
    diff_configs,
    list_all_runs,
    list_configs,
    load_single_config_records,
    single_config_run_path,
)


def _run_record(*, config_hash, case_id, case_family, arm=None, seed=0, succeeded=False, task_success=True):
    return RunRecord(
        run_id=f"{case_id}-{seed}",
        config_hash=config_hash,
        case_id=case_id,
        case_family=case_family,
        arm=arm,
        seed=seed,
        started_at="2026-08-03T00:00:00+00:00",
        ended_at="2026-08-03T00:00:01+00:00",
        wall_time_seconds=1.0,
        outcomes={"exfiltration": succeeded, "exfiltration_attempted": succeeded, "task_success": task_success},
    )


# --- config listing & diff --------------------------------------------------


def test_list_configs_empty_dir(tmp_path):
    assert list_configs(configs_dir=tmp_path / "nope") == []


def test_list_configs_returns_summaries(tmp_path):
    h1 = save_config(baseline_config(label="one"), configs_dir=tmp_path)
    h2 = save_config(baseline_config(label="two", defensive_instruction=False), configs_dir=tmp_path)
    summaries = {s.config_hash: s for s in list_configs(configs_dir=tmp_path)}
    assert set(summaries) == {h1, h2}
    assert summaries[h1].label == "one"
    assert summaries[h1].defensive_instruction is True
    assert summaries[h2].defensive_instruction is False
    assert summaries[h1].n_agents == 3


def test_diff_configs_identical_is_empty(tmp_path):
    h1 = save_config(baseline_config(label="a"), configs_dir=tmp_path)
    h2 = save_config(baseline_config(label="b"), configs_dir=tmp_path)  # label excluded from hash & from diff
    assert diff_configs(h1, h2, configs_dir=tmp_path) == []


def test_diff_configs_reports_scalar_field_change(tmp_path):
    h1 = save_config(baseline_config(defensive_instruction=True), configs_dir=tmp_path)
    h2 = save_config(baseline_config(defensive_instruction=False), configs_dir=tmp_path)
    entries = diff_configs(h1, h2, configs_dir=tmp_path)
    paths = {e.path for e in entries}
    assert "defensive_instruction" in paths
    entry = next(e for e in entries if e.path == "defensive_instruction")
    assert entry.value_a is True
    assert entry.value_b is False


def test_diff_configs_reports_agent_added_by_role(tmp_path):
    from target_system.config import AgentSpec

    extra = AgentSpec(role="scheduler", name="Scheduler", system_prompt="x", tools=[])
    h1 = save_config(baseline_config(), configs_dir=tmp_path)
    h2 = save_config(baseline_config(extra_agents=[extra]), configs_dir=tmp_path)
    entries = diff_configs(h1, h2, configs_dir=tmp_path)
    matching = [e for e in entries if e.path == "agents[role=scheduler]"]
    assert len(matching) == 1
    assert matching[0].value_a is None
    assert matching[0].value_b is not None


def test_diff_configs_reports_agent_field_change_by_role():
    # different tool lists on the same role should show up scoped to that role, not the whole agents list
    from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig

    def cfg(tools):
        return SystemConfig(
            label="x",
            model=ModelConfig(provider="mock", model_name="m"),
            agents=[AgentSpec(role="operator", name="Operator", system_prompt="p", tools=tools)],
            security=SecurityConfig(),
        )

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        configs_dir = Path(d)
        h1 = save_config(cfg(["send_email"]), configs_dir=configs_dir)
        h2 = save_config(cfg(["send_email", "lookup_customer"]), configs_dir=configs_dir)
        entries = diff_configs(h1, h2, configs_dir=configs_dir)
        assert len(entries) == 1
        assert entries[0].path == "agents[role=operator].tools"


# --- single-config run persistence -------------------------------------------


def test_single_config_run_path_keyed_by_hash(tmp_path):
    p1 = single_config_run_path("cfg_abc123", runs_dir=tmp_path)
    p2 = single_config_run_path("cfg_abc123", runs_dir=tmp_path)
    assert p1 == p2
    assert p1.name == "single_cfg_abc123.jsonl"


def test_load_single_config_records_roundtrip(tmp_path):
    record = _run_record(config_hash="cfg_x", case_id="case-1", case_family="direct_instruction_injection")
    append_run_record(record, single_config_run_path("cfg_x", runs_dir=tmp_path))
    records = load_single_config_records("cfg_x", runs_dir=tmp_path)
    assert len(records) == 1
    assert records[0].case_id == "case-1"


def test_load_single_config_records_missing_file_is_empty(tmp_path):
    assert load_single_config_records("cfg_never_run", runs_dir=tmp_path) == []


# --- unified run listing ------------------------------------------------------


def test_list_all_runs_empty(tmp_path):
    assert list_all_runs(runs_dir=tmp_path / "nope") == []


def test_list_all_runs_includes_single_config_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    configs_dir = tmp_path / "configs"
    # not bit-identical to baseline, so config_label ends up as a real
    # generated description rather than the trivial "baseline (defaults)" case
    config_hash = save_config(baseline_config(label="my-config", defensive_instruction=False), configs_dir=configs_dir)
    append_run_record(
        _run_record(config_hash=config_hash, case_id="c1", case_family="direct_instruction_injection", succeeded=True),
        single_config_run_path(config_hash, runs_dir=runs_dir),
    )
    listings = list_all_runs(runs_dir=runs_dir, configs_dir=configs_dir)
    assert len(listings) == 1
    listing = listings[0]
    assert listing.kind == "single_config"
    assert listing.name == config_hash
    # config_label holds the auto-generated description, not the raw
    # human-chosen SystemConfig.label ("my-config") — see tui.data.describe_config_for_humans
    assert listing.single_summary.config_label == "baseline, but supervisor's defensive instruction removed"
    assert listing.single_summary.total_attacks == 1
    assert listing.single_summary.succeeded == 1


def test_list_all_runs_includes_comparison_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    report = {
        "name": "toy_experiment",
        "arm_a_label": "arm_a",
        "arm_b_label": "arm_b",
        "arm_a_hash": "cfg_a",
        "arm_b_hash": "cfg_b",
        "n_cases": 1,
        "n_runs_per_case": 1,
        "n_cached": 0,
        "n_executed": 1,
        "task_success_a": 1.0,
        "task_success_b": 1.0,
        "family_results": {},
        "sequential_analysis": None,
    }
    (runs_dir / "toy_experiment_report.json").write_text(json.dumps(report), encoding="utf-8")
    listings = list_all_runs(runs_dir=runs_dir, configs_dir=tmp_path / "configs")
    assert len(listings) == 1
    listing = listings[0]
    assert listing.kind == "comparison"
    assert listing.name == "toy_experiment"
    assert listing.comparison_verdict.tier == "INCONCLUSIVE"  # no family_results at all


def test_list_all_runs_sorted_newest_first(tmp_path):
    import os
    import time

    runs_dir = tmp_path / "runs"
    configs_dir = tmp_path / "configs"
    h1 = save_config(baseline_config(label="first"), configs_dir=configs_dir)
    append_run_record(_run_record(config_hash=h1, case_id="c1", case_family="f"), single_config_run_path(h1, runs_dir=runs_dir))
    time.sleep(0.01)
    h2 = save_config(baseline_config(label="second", defensive_instruction=False), configs_dir=configs_dir)
    append_run_record(_run_record(config_hash=h2, case_id="c1", case_family="f"), single_config_run_path(h2, runs_dir=runs_dir))
    os.utime(single_config_run_path(h2, runs_dir=runs_dir), None)  # ensure a distinct, later mtime

    listings = list_all_runs(runs_dir=runs_dir, configs_dir=configs_dir)
    assert [listing.name for listing in listings] == [h2, h1]
