"""Manage configs: list every saved SystemConfig by hash, pick any two for
a readable field-by-field diff. All data comes from tui/data.py
(list_configs/diff_configs/describe_config_for_humans) — this module only
lays out the two screens.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Label, ListItem, ListView

from target_system.config import DEFAULT_CONFIGS_DIR
from tui.app import BaseScreen
from tui.data import ConfigSummary, describe_config_for_humans, diff_configs, list_configs
from tui.formatting import diff_field_label, format_config_list_label


class ManageConfigsScreen(BaseScreen):
    def __init__(self, *, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> None:
        super().__init__()
        self.configs_dir = configs_dir
        self.summaries: list[ConfigSummary] = []
        self._first_pick: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        self.summaries = list_configs(configs_dir=self.configs_dir)
        if not self.summaries:
            yield Vertical(
                Label("Manage configs", classes="title"),
                Label("No saved configs yet.", id="empty-state", classes="subtitle"),
                classes="wizard-body",
            )
        else:
            yield Vertical(
                Label("Manage configs", classes="title"),
                Label(
                    "Pick two to diff against each other (or b/h to go back).",
                    id="picker-hint",
                    classes="subtitle",
                ),
                ListView(
                    *(ListItem(Label(format_config_list_label(s.description, s.config_hash)), id=s.config_hash) for s in self.summaries),
                    id="config-list",
                ),
                classes="wizard-body",
            )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if self._first_pick is None:
            self._first_pick = event.item.id
            first_description = describe_config_for_humans(self._first_pick, configs_dir=self.configs_dir)
            self.query_one("#picker-hint", Label).update(f"First: {first_description}. Now pick the second config.")
        else:
            second_pick = event.item.id
            self.app.push_screen(ConfigDiffScreen(self._first_pick, second_pick, configs_dir=self.configs_dir))
            self._first_pick = None
            self.query_one("#picker-hint", Label).update("Pick two to diff against each other (or b/h to go back).")


class ConfigDiffScreen(BaseScreen):
    def __init__(self, hash_a: str, hash_b: str, *, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> None:
        super().__init__()
        self.hash_a = hash_a
        self.hash_b = hash_b
        self.configs_dir = configs_dir

    def compose(self) -> ComposeResult:
        yield Header()
        entries = diff_configs(self.hash_a, self.hash_b, configs_dir=self.configs_dir)
        desc_a = describe_config_for_humans(self.hash_a, configs_dir=self.configs_dir)
        desc_b = describe_config_for_humans(self.hash_b, configs_dir=self.configs_dir)
        with Vertical(classes="wizard-body"):
            yield Label(f"Diff — {desc_a} vs. {desc_b}", classes="title")
            yield Label(f"{self.hash_a}  vs.  {self.hash_b}", classes="hint")
            if not entries:
                yield Label("These configs resolve to identical content.", classes="subtitle")
            else:
                table = DataTable(id="diff-table")
                table.add_columns("Field", desc_a, desc_b)
                for entry in entries:
                    table.add_row(diff_field_label(entry), str(entry.value_a), str(entry.value_b))
                yield table
        yield Footer()
