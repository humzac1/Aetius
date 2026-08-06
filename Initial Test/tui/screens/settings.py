"""Settings: currently just the door to re-editing credentials (Part
"packaging + credentials onboarding") plus quit. Edit credentials reuses
CredentialsScreen exactly as the first-run gate does -- same fields, same
validate-before-write discipline -- just with first_run=False so b/h work
normally and on_complete pops back here instead of proceeding to Home.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView

from tui.app import BaseScreen


class SettingsScreen(BaseScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label("Settings", classes="title"),
            ListView(
                ListItem(Label("Edit credentials"), id="edit_credentials"),
                ListItem(Label("Quit"), id="quit"),
                id="settings-menu",
            ),
            classes="wizard-body",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "edit_credentials":
            from tui.screens.credentials import CredentialsScreen

            self.app.push_screen(CredentialsScreen(first_run=False, on_complete=self.app.pop_screen))
        elif event.item.id == "quit":
            self.app.exit()
