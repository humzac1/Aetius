"""SettingsScreen: the door to re-editing credentials after first run (Part
"packaging + credentials onboarding"). "Edit credentials" reuses the same
3-step flow (Anthropic -> source picker -> fields) with first_run=False,
so b/h aren't disabled the way they are during the first-run gate — see
tui/screens/credentials.py's _FormStepScreen.action_go_back/action_go_home.

With conftest's default env vars (Anthropic + Langfuse fully resolved via
"real" env vars), every step is env-sourced, so re-editing lands straight
on the terminal "already set via environment variables" state at
FieldsStepScreen -- that's a real, valid state to test on its own (see
below), not a test artifact to work around.
"""

from __future__ import annotations

from textual.widgets import ListView

import config.paths as paths
from config.credentials import write_credentials
from tests.tui_test_support import run_async
from tui.app import HarnessApp
from tui.screens.credentials import FieldsStepScreen, SourceStepScreen
from tui.screens.settings import SettingsScreen


def test_edit_credentials_with_everything_env_resolved_shows_terminal_message():
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
            assert isinstance(app.screen, FieldsStepScreen)
            assert app.screen.first_run is False
            assert app.screen.editable_fields == []

    run_async(scenario)


def test_back_binding_works_on_re_edit_credentials_screen():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(SettingsScreen())
            await pilot.pause()
            await pilot.press("enter")  # "Edit credentials"
            await pilot.pause()
            assert isinstance(app.screen, FieldsStepScreen)
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)

    run_async(scenario)


def test_edit_credentials_reopens_the_source_picker_when_source_is_file_sourced_not_env(monkeypatch):
    # a previously-configured (file-based, not env) source must be
    # reachable for switching, not silently skipped the way an
    # env-sourced one correctly is -- see tui/screens/credentials.py's
    # module docstring on why the two behave differently.
    monkeypatch.delenv("TRACE_SOURCE", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_PROJECT_ID", raising=False)
    write_credentials(
        {
            "TRACE_SOURCE": "langfuse",
            "LANGFUSE_SECRET_KEY": "sk-lf",
            "LANGFUSE_PUBLIC_KEY": "pk-lf",
            "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
            "LANGFUSE_PROJECT_ID": "proj-1",
        },
        env_path=paths.ENV_PATH,
    )

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(SettingsScreen())
            await pilot.pause()
            await pilot.press("enter")  # "Edit credentials"
            await pilot.pause()
            assert isinstance(app.screen, SourceStepScreen)

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
