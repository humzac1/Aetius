"""Entry point: `caligula` (console script, see pyproject.toml's
[project.scripts]) or `python -m tui.app`. This module owns only
navigation scaffolding (the App subclass, BaseScreen's back/home/quit
bindings, and the top-level menu) — every screen it pushes does its own
work by calling into tui/data.py, tui/execution.py, and
tui/verdict_logic.py, never by recomputing anything itself.

Navigation is Textual's own screen stack (push_screen/pop_screen), not a
hand-rolled state machine: BaseScreen.action_go_back pops one level,
action_go_home pops back to the HomeScreen, and every pushed screen
inherits both for free.

The toy target system (target_system/, its 5 presets in
experiments/presets.py, tui/screens/presets.py) is a retired internal
regression baseline, not part of the shipped product — there is
deliberately no menu item routing to it. It stays fully testable directly
(pytest, experiments.cli) and its screen module still exists so its own
tests keep exercising it in isolation; it's just never reachable from
HomeScreen. See tui/screens/wizard.py's ConfigPickerScreen (no more
baseline_config() option) and tui/screens/configs.py (no more "diff vs.
baseline") for the same boundary applied to the wizard/config-management
screens.

Credentials are gated before Home is ever reachable — see on_mount below
and tui/screens/credentials.py.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView

from config import credentials

APP_TITLE = "Caligula"

MENU_ITEMS = [
    ("test_agent", "Test my agent (wizard)"),
    ("add_environment", "Add environment (from Langfuse)"),
    ("view_runs", "View past runs"),
    ("manage_configs", "Manage configs"),
    ("settings", "Settings / exit"),
]


class BaseScreen(Screen):
    """Every screen in the app gets back/home/quit for free by
    subclassing this instead of textual.screen.Screen directly."""

    BINDINGS = [
        Binding("b", "go_back", "Back"),
        Binding("h", "go_home", "Home"),
        Binding("q", "quit_app", "Quit"),
    ]

    def action_go_back(self) -> None:
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()

    def action_go_home(self) -> None:
        # Pop everything above HomeScreen, not everything down to Textual's
        # implicit base screen — the stack is [_default, HomeScreen, ...],
        # so "pop until len == 1" would land on _default, not the menu.
        while len(self.app.screen_stack) > 1 and not isinstance(self.app.screen, HomeScreen):
            self.app.pop_screen()

    def action_quit_app(self) -> None:
        self.app.exit()


class PlaceholderScreen(BaseScreen):
    """Stands in for a menu destination that isn't built yet, so the app
    is always launchable and every menu item is always clickable, even
    mid-build. Each entry below is removed as its real screen lands."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label(self._title, classes="title"),
            Label("Not implemented yet.", classes="subtitle"),
            Label("Press b to go back, h for home.", classes="hint"),
        )
        yield Footer()


class HomeScreen(BaseScreen):
    BINDINGS = [Binding("q", "quit_app", "Quit")]  # no back/home at the root

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label(APP_TITLE, classes="title"),
            ListView(*(ListItem(Label(label), id=key) for key, label in MENU_ITEMS), id="menu"),
            classes="home-body",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        key = event.item.id
        if key == "test_agent":
            self.app.push_screen(self._wizard_screen())
        elif key == "add_environment":
            self.app.push_screen(self._add_environment_screen())
        elif key == "view_runs":
            self.app.push_screen(self._past_runs_screen())
        elif key == "manage_configs":
            self.app.push_screen(self._manage_configs_screen())
        elif key == "settings":
            self.app.push_screen(self._settings_screen())

    # Each of these is a deferred import so HomeScreen never breaks while a
    # later screen module is still being built — swapped for the real
    # screen class as each one lands (tasks #53-#55).
    def _wizard_screen(self) -> Screen:
        try:
            from tui.screens.wizard import WizardModeScreen

            return WizardModeScreen()
        except ImportError:
            return PlaceholderScreen("Test my agent (wizard)")

    def _add_environment_screen(self) -> Screen:
        try:
            from tui.screens.add_environment import AddEnvironmentScreen

            return AddEnvironmentScreen()
        except ImportError:
            return PlaceholderScreen("Add environment (from Langfuse)")

    def _settings_screen(self) -> Screen:
        try:
            from tui.screens.settings import SettingsScreen

            return SettingsScreen()
        except ImportError:
            return PlaceholderScreen("Settings")

    def _past_runs_screen(self) -> Screen:
        try:
            from tui.screens.past_runs import PastRunsScreen

            return PastRunsScreen()
        except ImportError:
            return PlaceholderScreen("View past runs")

    def _manage_configs_screen(self) -> Screen:
        try:
            from tui.screens.configs import ManageConfigsScreen

            return ManageConfigsScreen()
        except ImportError:
            return PlaceholderScreen("Manage configs")


class HarnessApp(App):
    TITLE = APP_TITLE
    CSS = """
    .title {
        text-style: bold;
        padding: 1 0;
    }

    .subtitle {
        color: $text-muted;
    }

    .hint {
        color: $text-muted;
        padding-top: 1;
    }

    .home-body {
        align: center middle;
        padding: 2 4;
    }

    ListView#menu {
        width: 48;
        height: auto;
        border: round $primary;
    }

    .wizard-body {
        padding: 2 4;
    }

    .verdict-body {
        padding: 1 2;
    }

    .tier-flagged {
        text-style: bold;
        color: $error;
        padding: 1 0;
    }

    .tier-clear {
        text-style: bold;
        color: $success;
        padding: 1 0;
    }

    .tier-inconclusive {
        text-style: bold;
        color: $warning;
        padding: 1 0;
    }

    .breakdown {
        color: $text-muted;
        padding-top: 1;
    }

    .disclaimer {
        background: $panel;
        color: $text-muted;
        padding: 1 2;
        margin-top: 1;
        border-top: solid $primary;
    }

    DataTable {
        height: auto;
        max-height: 20;
    }
    """

    def on_mount(self) -> None:
        if credentials.missing_keys():
            from tui.screens.credentials import CredentialsScreen

            self.push_screen(CredentialsScreen(first_run=True, on_complete=self._credentials_complete))
        else:
            self.push_screen(HomeScreen())

    def _credentials_complete(self) -> None:
        # First real action after setup is pulling traces and building the
        # first reconstructed environment, not an empty Home menu with
        # nothing in it yet -- but Home still goes on the stack underneath
        # (not skipped) so 'b'/'h' from Add Environment land somewhere real,
        # same invariant every other screen in this app relies on.
        from tui.screens.add_environment import AddEnvironmentScreen

        self.pop_screen()
        self.push_screen(HomeScreen())
        self.push_screen(AddEnvironmentScreen())


def main() -> None:
    HarnessApp().run()


if __name__ == "__main__":
    main()
