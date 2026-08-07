"""Headless Textual Pilot tests for the TUI navigation shell — the
Pilot/run_test() analogue of test_dashboard_app.py's AppTest pattern.
Wrapped in asyncio.run() rather than pytest-asyncio so this needs no new
test dependency; each test function is a plain sync def pytest can collect
normally."""

import asyncio

import config.credentials as credentials
from tui.app import HarnessApp, HomeScreen, MENU_ITEMS, PlaceholderScreen
from tui.screens.credentials import AnthropicStepScreen
from tui.screens.presets import PresetMenuScreen
from tui.screens.settings import SettingsScreen


def run_async(coro_fn):
    asyncio.run(coro_fn())


def test_app_starts_on_home_screen():
    async def scenario():
        app = HarnessApp()
        async with app.run_test():
            assert isinstance(app.screen, HomeScreen)

    run_async(scenario)


def test_home_screen_lists_all_five_menu_items():
    async def scenario():
        app = HarnessApp()
        async with app.run_test():
            from textual.widgets import ListView

            menu = app.screen.query_one("#menu", ListView)
            assert len(menu.children) == 5

    run_async(scenario)


def test_no_menu_item_routes_to_the_toy_preset_screen():
    # the toy target system's 5 presets are internal regression-baseline
    # tooling only (see tui/app.py's module docstring) — "run_preset" must
    # never appear as a Home menu destination in the shipped CLI
    assert "run_preset" not in {key for key, _ in MENU_ITEMS}


def test_selecting_menu_item_pushes_a_screen():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            base_depth = len(app.screen_stack)
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.screen_stack) == base_depth + 1

    run_async(scenario)


def test_all_menu_items_resolve_to_real_screens_none_of_them_the_toy_preset_screen():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            for _ in range(5):  # all five items push a real screen now (Settings pushes SettingsScreen)
                await pilot.press("enter")
                await pilot.pause()
                assert not isinstance(app.screen, PlaceholderScreen)
                assert not isinstance(app.screen, PresetMenuScreen)
                await pilot.press("h")
                await pilot.pause()
                await pilot.press("down")

    run_async(scenario)


def test_placeholder_screen_renders_given_title_and_supports_back():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PlaceholderScreen("Some future screen"))
            await pilot.pause()
            assert isinstance(app.screen, PlaceholderScreen)
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    run_async(scenario)


def test_back_binding_returns_to_home():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, HomeScreen)
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    run_async(scenario)


def test_home_binding_returns_to_home_from_any_depth():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, HomeScreen)
            await pilot.press("h")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    run_async(scenario)


def test_settings_menu_item_opens_settings_screen():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            from textual.widgets import ListView

            menu = app.screen.query_one("#menu", ListView)
            menu.index = 4  # "Settings / exit" is the fifth (last) item
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)

    run_async(scenario)


def test_settings_quit_item_exits_app():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            from textual.widgets import ListView

            menu = app.screen.query_one("#menu", ListView)
            menu.index = 4
            await pilot.pause()
            await pilot.press("enter")  # into Settings
            await pilot.pause()
            settings_menu = app.screen.query_one("#settings-menu", ListView)
            settings_menu.index = 1  # "Quit"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.return_code == 0 or not app.is_running

    run_async(scenario)


def test_quit_binding_exits_app():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            await pilot.press("q")
            await pilot.pause()
            assert not app.is_running

    run_async(scenario)


# --- credentials gate on launch --------------------------------------------
#
# Real env vars are cleared (not credentials.missing_keys mocked) so the
# credentials flow's own fresh resolve_all() agrees with what triggered
# the gate -- mocking missing_keys alone would leave conftest's "every
# credential resolved" default env vars in place underneath, so the
# entry screen would immediately route past every step to the terminal
# "already set via environment variables" state instead of the step
# these tests actually mean to exercise.


def test_app_shows_credentials_screen_when_keys_missing(monkeypatch):
    monkeypatch.delenv(credentials.ANTHROPIC_KEY, raising=False)

    async def scenario():
        app = HarnessApp()
        async with app.run_test():
            assert isinstance(app.screen, AnthropicStepScreen)
            assert app.screen.first_run is True

    run_async(scenario)


def test_home_is_unreachable_while_credentials_screen_is_up(monkeypatch):
    monkeypatch.delenv(credentials.ANTHROPIC_KEY, raising=False)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            assert isinstance(app.screen, AnthropicStepScreen)
            base_depth = len(app.screen_stack)
            await pilot.press("b")  # disabled on first_run — see _FormStepScreen.action_go_back
            await pilot.pause()
            assert isinstance(app.screen, AnthropicStepScreen)
            await pilot.press("h")  # disabled on first_run — see _FormStepScreen.action_go_home
            await pilot.pause()
            assert isinstance(app.screen, AnthropicStepScreen)
            assert len(app.screen_stack) == base_depth

    run_async(scenario)


def test_credentials_complete_routes_to_add_environment_with_home_underneath_and_pops_every_step():
    # Onboarding's actual first action is pulling traces / building the
    # first reconstructed environment, not an empty Home menu -- but Home
    # still goes on the stack underneath, not skipped, so 'b'/'h' from
    # Add Environment land somewhere real. Also the regression this once
    # was: a multi-step flow can push several screens before completing
    # (Anthropic -> source picker -> fields), so completion must pop all
    # of them, not just one -- simulated here by pushing two placeholder
    # screens before calling _credentials_complete directly.
    from tui.screens.add_environment import AddEnvironmentScreen

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PlaceholderScreen("step 1"))
            app.push_screen(PlaceholderScreen("step 2"))
            await pilot.pause()
            app._credentials_complete()
            await pilot.pause()
            assert isinstance(app.screen, AddEnvironmentScreen)
            # exactly [_default, HomeScreen, AddEnvironmentScreen] -- both
            # placeholder steps popped, nothing left over underneath
            assert len(app.screen_stack) == 3
            assert isinstance(app.screen_stack[1], HomeScreen)
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    run_async(scenario)


# --- --version -------------------------------------------------------------


def test_version_flag_prints_version_and_never_launches_the_app(monkeypatch, capsys):
    import sys

    import tui.app as app_module

    def _fail_if_constructed(*args, **kwargs):
        raise AssertionError("--version must not construct/run HarnessApp")

    monkeypatch.setattr(app_module, "HarnessApp", _fail_if_constructed)
    monkeypatch.setattr(sys, "argv", ["caligula", "--version"])

    app_module.main()

    out = capsys.readouterr().out
    assert out.strip() == f"caligula {app_module._package_version()}"


def test_version_flag_falls_back_gracefully_when_not_installed_as_a_package(monkeypatch, capsys):
    import sys
    from importlib.metadata import PackageNotFoundError

    import tui.app as app_module

    def _raise_not_found(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(app_module, "_installed_version", _raise_not_found)
    monkeypatch.setattr(sys, "argv", ["caligula", "--version"])

    app_module.main()

    out = capsys.readouterr().out
    assert "caligula" in out
    assert "unknown" in out.lower()
