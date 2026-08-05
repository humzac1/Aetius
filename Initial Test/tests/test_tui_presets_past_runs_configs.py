import json

from textual.widgets import DataTable, Label, ListView

from experiments.presets import PRESETS
from target_system.config import save_config
from target_system.factory import baseline_config
from target_system.logging_schema import append_run_record
from tui.app import HarnessApp
from tui.data import single_config_run_path
from tui.screens.configs import ConfigDiffScreen, ManageConfigsScreen
from tui.screens.past_runs import PastRunsScreen
from tui.screens.presets import PresetMenuScreen, PresetProgressScreen
from tui.screens.verdict import ComparisonVerdictScreen, SingleConfigVerdictScreen
from tests.tui_test_support import run_async
from tests.tui_test_support import wait_until as _wait_until


# --- preset menu ---------------------------------------------------------


def test_preset_menu_lists_all_five_presets():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PresetMenuScreen())
            await pilot.pause()
            menu = app.screen.query_one("#preset-menu", ListView)
            assert len(menu.children) == len(PRESETS)
            assert {item.id for item in menu.children} == set(PRESETS.keys())

    run_async(scenario)


def test_preset_progress_runs_aa_and_lands_on_comparison_verdict(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PresetProgressScreen(preset=PRESETS["aa"], n_runs_per_case=1, runs_dir=tmp_path))
            await _wait_until(pilot, lambda: isinstance(app.screen, ComparisonVerdictScreen))
            assert app.screen.experiment_name == "aa"
            assert (tmp_path / "aa_report.json").exists()

    run_async(scenario)


def test_selecting_a_preset_starts_it_and_eventually_lands_on_its_verdict(tmp_path):
    # the mock backend can finish before the next pilot.pause(), so this checks
    # the flow reaches the right destination rather than catching the
    # transient PresetProgressScreen mid-flight
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PresetMenuScreen(runs_dir=tmp_path))
            await pilot.pause()
            await pilot.press("enter")  # first preset in dict order: "aa"
            await _wait_until(pilot, lambda: isinstance(app.screen, ComparisonVerdictScreen))
            assert app.screen.experiment_name == "aa"

    run_async(scenario)


# --- past runs -------------------------------------------------------------


def _write_comparison_report(runs_dir, name):
    report = {
        "name": name, "arm_a_label": "a", "arm_b_label": "b", "arm_a_hash": "cfg_a", "arm_b_hash": "cfg_b",
        "n_cases": 1, "n_runs_per_case": 1, "n_cached": 0, "n_executed": 1,
        "task_success_a": 1.0, "task_success_b": 1.0, "family_results": {}, "sequential_analysis": None,
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{name}_report.json").write_text(json.dumps(report), encoding="utf-8")
    (runs_dir / f"{name}.jsonl").write_text("", encoding="utf-8")


def _write_single_config_run(runs_dir, configs_dir):
    from target_system.logging_schema import RunRecord

    # not bit-identical to baseline, so config_label ends up as a real
    # generated description rather than the trivial "baseline (defaults)" case
    config_hash = save_config(baseline_config(label="past-run-test", defensive_instruction=False), configs_dir=configs_dir)
    record = RunRecord(
        run_id="r1", config_hash=config_hash, case_id="c1", case_family="direct_instruction_injection", arm=None, seed=0,
        started_at="2026-08-03T00:00:00+00:00", ended_at="2026-08-03T00:00:01+00:00", wall_time_seconds=1.0,
        outcomes={"exfiltration": False, "exfiltration_attempted": False},
    )
    append_run_record(record, single_config_run_path(config_hash, runs_dir=runs_dir))
    return config_hash


def test_past_runs_screen_empty_state(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PastRunsScreen(runs_dir=tmp_path / "nope"))
            await pilot.pause()
            label = app.screen.query_one("#empty-state", Label)
            assert "No runs found" in str(label.render())

    run_async(scenario)


def test_past_runs_screen_lists_comparison_and_single_config_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    configs_dir = tmp_path / "configs"
    _write_comparison_report(runs_dir, "toy_experiment")
    _write_single_config_run(runs_dir, configs_dir)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PastRunsScreen(runs_dir=runs_dir, configs_dir=configs_dir))
            await pilot.pause()
            menu = app.screen.query_one("#past-runs-list", ListView)
            assert len(menu.children) == 2

    run_async(scenario)


def test_selecting_comparison_run_opens_comparison_verdict(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_comparison_report(runs_dir, "toy_experiment")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PastRunsScreen(runs_dir=runs_dir, configs_dir=tmp_path / "configs"))
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ComparisonVerdictScreen)
            assert app.screen.experiment_name == "toy_experiment"

    run_async(scenario)


def test_selecting_single_config_run_opens_single_config_verdict(tmp_path):
    runs_dir = tmp_path / "runs"
    configs_dir = tmp_path / "configs"
    _write_single_config_run(runs_dir, configs_dir)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PastRunsScreen(runs_dir=runs_dir, configs_dir=configs_dir))
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, SingleConfigVerdictScreen)
            # config_label holds the auto-generated description, not the raw
            # human-chosen SystemConfig.label ("past-run-test") — see tui.data.describe_config_for_humans
            assert app.screen.summary.config_label == "baseline, but supervisor's defensive instruction removed"

    run_async(scenario)


# --- manage configs ----------------------------------------------------------


def test_manage_configs_empty_state(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ManageConfigsScreen(configs_dir=tmp_path / "nope"))
            await pilot.pause()
            label = app.screen.query_one("#empty-state", Label)
            assert "No saved configs" in str(label.render())

    run_async(scenario)


def test_manage_configs_lists_saved_configs(tmp_path):
    save_config(baseline_config(label="one"), configs_dir=tmp_path)
    save_config(baseline_config(label="two", defensive_instruction=False), configs_dir=tmp_path)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ManageConfigsScreen(configs_dir=tmp_path))
            await pilot.pause()
            menu = app.screen.query_one("#config-list", ListView)
            assert len(menu.children) == 2

    run_async(scenario)


def test_picking_two_configs_opens_diff_screen(tmp_path):
    save_config(baseline_config(label="one", defensive_instruction=True), configs_dir=tmp_path)
    save_config(baseline_config(label="two", defensive_instruction=False), configs_dir=tmp_path)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ManageConfigsScreen(configs_dir=tmp_path))
            await pilot.pause()
            await pilot.press("enter")  # first pick
            await pilot.pause()
            assert isinstance(app.screen, ManageConfigsScreen)  # still here after one pick
            await pilot.press("down")
            await pilot.press("enter")  # second pick
            await pilot.pause()
            assert isinstance(app.screen, ConfigDiffScreen)

    run_async(scenario)


def test_diff_screen_shows_a_row_for_the_changed_field(tmp_path):
    h1 = save_config(baseline_config(defensive_instruction=True), configs_dir=tmp_path)
    h2 = save_config(baseline_config(defensive_instruction=False), configs_dir=tmp_path)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ConfigDiffScreen(h1, h2, configs_dir=tmp_path))
            await pilot.pause()
            table = app.screen.query_one("#diff-table", DataTable)
            assert table.row_count >= 1

    run_async(scenario)


def test_diff_screen_identical_configs_shows_no_table(tmp_path):
    h1 = save_config(baseline_config(label="a"), configs_dir=tmp_path)
    h2 = save_config(baseline_config(label="b"), configs_dir=tmp_path)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ConfigDiffScreen(h1, h2, configs_dir=tmp_path))
            await pilot.pause()
            assert not app.screen.query("#diff-table")

    run_async(scenario)
