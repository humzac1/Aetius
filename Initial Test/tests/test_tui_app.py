"""Headless Textual Pilot tests for the TUI navigation shell — the
Pilot/run_test() analogue of test_dashboard_app.py's AppTest pattern.
Wrapped in asyncio.run() rather than pytest-asyncio so this needs no new
test dependency; each test function is a plain sync def pytest can collect
normally."""

import asyncio

import config.credentials as credentials
from tui.app import HarnessApp, HomeScreen, MENU_ITEMS, PlaceholderScreen
from tui.screens.credentials import CredentialsScreen
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


def test_app_shows_credentials_screen_when_keys_missing(monkeypatch):
    monkeypatch.setattr(credentials, "missing_keys", lambda **kwargs: ["ANTHROPIC_API_KEY"])

    async def scenario():
        app = HarnessApp()
        async with app.run_test():
            assert isinstance(app.screen, CredentialsScreen)
            assert app.screen.first_run is True

    run_async(scenario)


def test_home_is_unreachable_while_credentials_screen_is_up(monkeypatch):
    monkeypatch.setattr(credentials, "missing_keys", lambda **kwargs: ["ANTHROPIC_API_KEY"])

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            assert isinstance(app.screen, CredentialsScreen)
            base_depth = len(app.screen_stack)
            await pilot.press("b")  # disabled on first_run — see CredentialsScreen.action_go_back
            await pilot.pause()
            assert isinstance(app.screen, CredentialsScreen)
            await pilot.press("h")  # disabled on first_run — see CredentialsScreen.action_go_home
            await pilot.pause()
            assert isinstance(app.screen, CredentialsScreen)
            assert len(app.screen_stack) == base_depth

    run_async(scenario)


def test_app_completing_credentials_screen_routes_to_add_environment_with_home_underneath(monkeypatch):
    # Onboarding's actual first action is pulling traces / building the
    # first reconstructed environment, not an empty Home menu -- but Home
    # still goes on the stack underneath, not skipped, so 'b'/'h' from
    # Add Environment land somewhere real (see tui/app.py's
    # _credentials_complete).
    from tui.screens.add_environment import AddEnvironmentScreen

    monkeypatch.setattr(credentials, "missing_keys", lambda **kwargs: ["ANTHROPIC_API_KEY"])

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            assert isinstance(app.screen, CredentialsScreen)
            app.screen.on_complete()
            await pilot.pause()
            assert isinstance(app.screen, AddEnvironmentScreen)
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    run_async(scenario)
