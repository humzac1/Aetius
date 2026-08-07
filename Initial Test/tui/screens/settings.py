"""Settings: currently just the door to re-editing credentials (Part
"packaging + credentials onboarding") plus quit. Edit credentials reuses
the exact same 3-step flow the first-run gate does -- same steps, same
validate-before-write discipline -- just with first_run=False so b/h work
normally between steps, and on_complete pops back here instead of
proceeding to Add Environment.
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
            from tui.screens.credentials import credentials_flow_entry_screen

            self.app.push_screen(credentials_flow_entry_screen(first_run=False, on_complete=self._credentials_edit_complete))
        elif event.item.id == "quit":
            self.app.exit()

    def _credentials_edit_complete(self) -> None:
        # Pops everything the credentials flow pushed (1-3 screens,
        # depending how many steps actually showed) back down to this
        # exact SettingsScreen instance -- a single pop_screen() would
        # only undo the last step, not the whole flow.
        while len(self.app.screen_stack) > 1 and self.app.screen is not self:
            self.app.pop_screen()
