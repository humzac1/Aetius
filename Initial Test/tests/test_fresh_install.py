"""First-use behavior on a genuinely fresh install: every write path must
create its own directory tree on demand, and every read path must treat
an absent directory as empty rather than crashing.

Every test here starts from a directory that does NOT exist (nested under
tmp_path but never created) — not a pre-existing empty one. The
distinction is this project's recurring bug shape: assumptions that held
on every developer machine (where the directories always already
existed) and failed precisely on first real use. Verified end-to-end
against the installed package with AETIUS_DATA_DIR pointed at an absent
location before these tests were written; these pin each path
individually so a regression is caught at the function that broke.
"""

from __future__ import annotations

from pathlib import Path

from attacker.attack_case import AttackCase
from attacker.case_store import load_generated_cases, save_generated_cases
from config import paths
from config.credentials import write_credentials
from experiments.persist import load_experiment_report, save_experiment_report
from experiments.runner import run_experiment
from target_system.config import save_config
from target_system.factory import baseline_config


def _absent(tmp_path: Path, *parts: str) -> Path:
    """A path whose whole tail is guaranteed not to exist — two levels
    deep so parent creation is exercised too, never just the leaf. Each
    call gets its own root so two absent paths in one test can't share a
    parent that the first exercised function legitimately created."""
    p = tmp_path.joinpath(f"never-created-{'-'.join(parts)}", *parts)
    assert not p.exists() and not p.parent.exists()
    return p


def test_credentials_write_creates_absent_parent_dirs(tmp_path):
    env_path = _absent(tmp_path, "config-home") / ".env"
    write_credentials({"ANTHROPIC_API_KEY": "sk-fresh"}, env_path=env_path)
    assert "sk-fresh" in env_path.read_text()


def test_langfuse_pull_creates_absent_cache_dir(tmp_path):
    from ingestion.langfuse_client import load_cached_trace_ids, pull_traces

    class _EmptyPage:
        data = []

    class _Stub:
        class api:  # noqa: N801 — mimics the real client's attribute path
            class trace:  # noqa: N801
                @staticmethod
                def list(**kwargs):
                    return _EmptyPage()

    traces_dir = _absent(tmp_path, "traces")
    assert pull_traces(project_id="p1", client=_Stub(), batch_size=3, traces_dir=traces_dir) == []
    assert (traces_dir / "p1").is_dir()
    # read path on a dir that never existed: empty, not a crash
    assert load_cached_trace_ids("nope", traces_dir=_absent(tmp_path, "other")) == []


def test_braintrust_pull_creates_absent_cache_dir(tmp_path, monkeypatch):
    import ingestion.braintrust_client as bt

    monkeypatch.setattr(bt, "_list_recent_root_span_ids", lambda state, project_name, *, batch_size, page_limit: [])
    traces_dir = _absent(tmp_path, "traces_braintrust")
    assert bt.pull_traces(project_name="p1", client=object(), batch_size=3, traces_dir=traces_dir) == []
    assert (traces_dir / "p1").is_dir()
    assert bt.load_cached_trace_ids("nope", traces_dir=_absent(tmp_path, "other")) == []


def test_config_save_creates_absent_configs_dir(tmp_path):
    configs_dir = _absent(tmp_path, "configs")
    cfg_hash = save_config(baseline_config(label="fresh"), configs_dir=configs_dir)
    assert (configs_dir / f"{cfg_hash}.json").exists()


def test_generated_case_save_creates_absent_store_dir(tmp_path):
    cases_dir = _absent(tmp_path, "generated_cases")
    out = save_generated_cases("cfg_fresh", [], model="claude-sonnet-5", generated_at="2026-08-14T00:00:00Z", cases_dir=cases_dir)
    assert out.exists() and out.parent == cases_dir
    assert load_generated_cases("cfg_missing", cases_dir=_absent(tmp_path, "other")) is None


def test_first_comparison_creates_absent_runs_dir_and_report(tmp_path):
    runs_dir = _absent(tmp_path, "runs")
    cases = [
        AttackCase(
            id=f"c{i}", family="direct_instruction_injection", injection_vector="task_text",
            success_outcome="exfiltration", source="test", benign_task="do x", injected_payload="do y",
        )
        for i in range(2)
    ]
    result = run_experiment(
        baseline_config(label="arm-a", defensive_instruction=True),
        baseline_config(label="arm-b", defensive_instruction=False),
        experiment_name="fresh_first", cases=cases, n_runs_per_case=2, max_workers=2, runs_dir=runs_dir,
    )
    report_path = save_experiment_report(result, runs_dir=runs_dir)
    assert (runs_dir / "fresh_first.jsonl").exists()
    assert report_path.exists()
    assert load_experiment_report("fresh_first", runs_dir=runs_dir) is not None


def test_fresh_launch_listing_paths_treat_absent_dirs_as_empty(tmp_path):
    from dashboard.data_access import list_available_reports
    from tui.data import list_configs

    assert list_configs(configs_dir=_absent(tmp_path, "configs")) == []
    assert list_available_reports(runs_dir=_absent(tmp_path, "runs")) == []


def test_no_default_storage_location_lives_inside_the_package():
    """The original near-data-loss bug: defaults resolving inside the
    installed package tree, wiped on every reinstall. Every default
    storage constant must resolve outside the package root — the
    package-relative locations exist only as read-only LEGACY migration
    sources."""
    package_root = Path(paths.__file__).resolve().parent.parent
    for name in ("DATA_DIR", "CONFIGS_DIR", "RUNS_DIR", "GENERATED_CASES_DIR", "TRACES_DIR", "TRACES_BRAINTRUST_DIR", "CONFIG_DIR", "ENV_PATH"):
        location = Path(getattr(paths, name)).resolve()
        assert not str(location).startswith(str(package_root)), f"{name} resolves inside the package: {location}"
