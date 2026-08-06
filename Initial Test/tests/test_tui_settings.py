"""SettingsScreen: the door to re-editing credentials after first run (Part
"packaging + credentials onboarding"). "Edit credentials" reuses
CredentialsScreen with first_run=False, so b/h aren't disabled the way
they are during the first-run gate — see tui/screens/credentials.py's
action_go_back/action_go_home.
"""

from __future__ import annotations

from textual.widgets import ListView

from tests.tui_test_support import run_async
from tui.app import HarnessApp
from tui.screens.credentials import CredentialsScreen
from tui.screens.settings import SettingsScreen


def test_edit_credentials_opens_credentials_screen_not_first_run():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(SettingsScreen())
            await pilot.pause()
            menu = app.screen.query_one("#settings-menu", ListView)
            menu.index = 0  # "Edit credentials"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, CredentialsScreen)
            assert app.screen.first_run is False

    run_async(scenario)


def test_back_binding_works_on_re_edit_credentials_screen():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(SettingsScreen())
            await pilot.pause()
            await pilot.press("enter")  # "Edit credentials"
            await pilot.pause()
            assert isinstance(app.screen, CredentialsScreen)
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)

    run_async(scenario)


def test_settings_quit_exits_app():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(SettingsScreen())
            await pilot.pause()
            menu = app.screen.query_one("#settings-menu", ListView)
            menu.index = 1  # "Quit"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.return_code == 0 or not app.is_running

    run_async(scenario)
