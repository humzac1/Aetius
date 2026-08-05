"""View past runs: one row per run already on disk (comparison experiments
and single-config checks alike), each with its verdict already computed by
tui.data.list_all_runs (which itself calls tui.verdict_logic — no
recomputation happens here). Selecting a row opens exactly the verdict
screen a fresh run would land on.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView

from experiments.runner import DEFAULT_RUNS_DIR
from target_system.config import DEFAULT_CONFIGS_DIR
from target_system.logging_schema import read_run_records
from tui.app import BaseScreen
from tui.data import RunListing, describe_comparison_for_humans, list_all_runs, single_config_run_path
from tui.screens.verdict import ComparisonVerdictScreen, SingleConfigVerdictScreen


def _describe(listing: RunListing, *, configs_dir: Path) -> str:
    if listing.kind == "comparison":
        name = describe_comparison_for_humans(listing.comparison_report, listing.name, configs_dir=configs_dir)
        return f"Comparison: {name} — {listing.comparison_verdict.tier}"
    summary = listing.single_summary
    return (
        f"Single-config check: {summary.config_label} — "
        f"{summary.total_attacks} attacks, {summary.succeeded} succeeded, "
        f"{summary.blocked} blocked, {summary.resisted} resisted"
    )


class PastRunsScreen(BaseScreen):
    def __init__(self, *, runs_dir: Path = DEFAULT_RUNS_DIR, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> None:
        super().__init__()
        self.runs_dir = runs_dir
        self.configs_dir = configs_dir
        self.listings: list[RunListing] = []

    def compose(self) -> ComposeResult:
        yield Header()
        self.listings = list_all_runs(runs_dir=self.runs_dir, configs_dir=self.configs_dir)
        items = [
            ListItem(Label(_describe(listing, configs_dir=self.configs_dir)), id=f"run-{i}")
            for i, listing in enumerate(self.listings)
        ]
        if not items:
            yield Vertical(
                Label("View past runs", classes="title"),
                Label("No runs found under data/runs/ yet.", id="empty-state", classes="subtitle"),
                classes="wizard-body",
            )
        else:
            yield Vertical(
                Label("View past runs", classes="title"),
                Label(f"{len(self.listings)} run(s), newest first.", classes="subtitle"),
                ListView(*items, id="past-runs-list"),
                classes="wizard-body",
            )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        listing = self.listings[int(event.item.id.removeprefix("run-"))]
        if listing.kind == "comparison":
            records = [r.model_dump(mode="json") for r in read_run_records(self.runs_dir / f"{listing.name}.jsonl")]
            self.app.push_screen(
                ComparisonVerdictScreen(
                    listing.comparison_verdict, listing.comparison_report, listing.name, records, configs_dir=self.configs_dir
                )
            )
        else:
            path = single_config_run_path(listing.single_summary.config_hash, runs_dir=self.runs_dir)
            records = [r.model_dump(mode="json") for r in read_run_records(path)]
            self.app.push_screen(SingleConfigVerdictScreen(listing.single_summary, records=records, configs_dir=self.configs_dir))
