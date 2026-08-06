"""Confirms the toy target system (Supervisor/Researcher/Operator, its 5
presets in experiments/presets.py) has no menu path anywhere in the
shipped TUI — it stays in the repo as an internal regression baseline
(still directly testable, still driven by experiments.cli / pytest), but
a fresh install must never be able to reach it through the UI. See
tui/app.py's module docstring for the full rationale; the individual
removals are in tui/app.py (no "run_preset" menu item), tui/screens/
wizard.py's ConfigPickerScreen (no baseline_config() option), and
tui/screens/configs.py's ManageConfigsScreen (no "diff vs. baseline").
"""

from __future__ import annotations

from target_system.config import save_config
from target_system.factory import baseline_config
from tests.tui_test_support import run_async
from tui.app import MENU_ITEMS, HarnessApp
from tui.screens.configs import ManageConfigsScreen
from tui.screens.presets import PresetMenuScreen
from tui.screens.wizard import ConfigPickerScreen, WizardModeScreen


def test_home_menu_has_no_preset_entry():
    assert "run_preset" not in {key for key, _ in MENU_ITEMS}
    assert not any("preset" in label.lower() for _, label in MENU_ITEMS)


def test_home_menu_never_pushes_the_preset_screen():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            for _ in range(len(MENU_ITEMS)):
                await pilot.press("enter")
                await pilot.pause()
                assert not isinstance(app.screen, PresetMenuScreen)
                await pilot.press("h")
                await pilot.pause()
                await pilot.press("down")

    run_async(scenario)


def test_wizard_config_picker_never_offers_baseline(tmp_path):
    save_config(baseline_config(label="real-one"), configs_dir=tmp_path)

    async def scenario():
        from textual.widgets import ListView

        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(WizardModeScreen(configs_dir=tmp_path))
            await pilot.pause()
            await pilot.press("enter")  # "single"
            await pilot.pause()
            assert isinstance(app.screen, ConfigPickerScreen)
            menu = app.screen.query_one("#config-picker-list", ListView)
            assert all(item.id != "__baseline__" for item in menu.children)

    run_async(scenario)


def test_manage_configs_has_no_diff_vs_baseline_binding():
    binding_keys = {binding.key for binding in ManageConfigsScreen.BINDINGS}
    assert "v" not in binding_keys
    assert not hasattr(ManageConfigsScreen, "action_view_diff")


def test_config_picker_has_no_diff_vs_baseline_binding():
    binding_keys = {binding.key for binding in ConfigPickerScreen.BINDINGS}
    assert "v" not in binding_keys
    assert not hasattr(ConfigPickerScreen, "action_view_diff")
