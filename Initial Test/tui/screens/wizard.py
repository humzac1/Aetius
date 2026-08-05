"""The "Test my agent" wizard: pick a mode, pick config(s), watch progress,
land on the verdict screen. Execution goes through tui/execution.py
exactly as any other caller would use it — the only things this module
adds are the guided config-selection flow and threading run_experiment /
run_single_config_check's on_progress callback onto the UI thread via
Textual's call_from_thread, per run_experiment's documented contract that
the callback always fires on the caller's thread.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView

from attacker.attack_case import AttackCase
from attacker.applicability import applicable_cases_for_configs
from attacker.cases import ATTACK_CASES
from experiments.cost_estimate import CostEstimate, estimate_batch_cost, format_cost_estimate
from experiments.persist import save_experiment_report
from experiments.runner import DEFAULT_RUNS_DIR
from target_system.config import DEFAULT_CONFIGS_DIR, SystemConfig, load_config
from target_system.factory import baseline_config
from tui.app import BaseScreen
from tui.data import describe_config_for_humans, ensure_baseline_saved, list_configs
from tui.execution import build_anthropic_client, enforce_reconstructed_provider, peek_n_cached, run_comparison_check, run_single_config_check
from tui.formatting import format_config_list_label
from tui.screens.configs import ConfigDiffScreen
from tui.screens.progress import WorkerProgressScreen
from tui.screens.verdict import ComparisonVerdictScreen, SingleConfigVerdictScreen
from tui.verdict_logic import compute_comparison_verdict, compute_single_config_summary

DEFAULT_N_RUNS_PER_CASE = 5
_BASELINE_ITEM_ID = "__baseline__"
_ORDINAL_LETTERS = ["A", "B", "C", "D"]


class WizardModeScreen(BaseScreen):
    def __init__(self, *, runs_dir: Path = DEFAULT_RUNS_DIR, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> None:
        super().__init__()
        self.runs_dir = runs_dir
        self.configs_dir = configs_dir

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label("Test my agent", classes="title"),
            Label("Choose what you want to check.", classes="subtitle"),
            ListView(
                ListItem(Label("Test a single config (no comparison)"), id="single"),
                ListItem(Label("Compare two configs"), id="comparison"),
                id="wizard-mode-menu",
            ),
            classes="wizard-body",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "single":
            self.app.push_screen(ConfigPickerScreen(n_needed=1, on_chosen=self._start, configs_dir=self.configs_dir))
        elif event.item.id == "comparison":
            self.app.push_screen(ConfigPickerScreen(n_needed=2, on_chosen=self._start, configs_dir=self.configs_dir))

    def _start(self, configs: list[SystemConfig]) -> None:
        # Reconstructed environments are real-model-only (Part 5) — force
        # provider="anthropic" here unconditionally, no matter what's
        # saved on disk, so this dispatch point can never hand one to the
        # mock backend (see enforce_reconstructed_provider's docstring).
        configs = [enforce_reconstructed_provider(c) for c in configs]
        mode = "single" if len(configs) == 1 else "comparison"
        # Filters out cases whose delivery vector or outcome needs a tool
        # role this environment doesn't have (attacker/applicability.py) —
        # a no-op for the toy system (its tools cover every role), load-
        # bearing for a reconstructed environment (e.g. no untrusted-
        # content-entry-point tool means corpus_document cases would
        # otherwise fail outright — see execute_case's ValueError).
        cases = applicable_cases_for_configs(list(ATTACK_CASES), configs)
        n_cached = peek_n_cached(configs, runs_dir=self.runs_dir)
        estimate = estimate_batch_cost(cases, configs, n_runs_per_case=DEFAULT_N_RUNS_PER_CASE, n_cached=n_cached)

        def _proceed() -> None:
            self.app.push_screen(
                WizardProgressScreen(
                    mode=mode, configs=configs, cases=cases, runs_dir=self.runs_dir, configs_dir=self.configs_dir
                )
            )

        if estimate.any_real_model:
            # Never a free/instant mock option for a run that touches a
            # real model — the estimate must be shown and explicitly
            # confirmed before anything executes.
            self.app.push_screen(CostConfirmScreen(estimate=estimate, on_confirm=_proceed))
        else:
            _proceed()


class CostConfirmScreen(BaseScreen):
    """Blocks on an explicit choice before any run that touches a real
    model — reconstructed environments (Part 5) have no free/instant mock
    path, so this is the one place that stands between "configs picked"
    and "money and time spent." 'b' (inherited from BaseScreen) cancels
    back to the mode screen without proceeding; only picking "Proceed"
    calls on_confirm."""

    def __init__(self, *, estimate: CostEstimate, on_confirm: Callable[[], None]) -> None:
        super().__init__()
        self.estimate = estimate
        self.on_confirm = on_confirm

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label("Confirm before running", classes="title"),
            Label(format_cost_estimate(self.estimate), classes="subtitle"),
            Label("This run calls a real model and spends real money.", classes="hint"),
            ListView(
                ListItem(Label("Proceed"), id="proceed"),
                ListItem(Label("Cancel"), id="cancel"),
                id="cost-confirm-menu",
            ),
            classes="wizard-body",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "proceed":
            on_confirm = self.on_confirm
            self.app.pop_screen()
            on_confirm()
        else:
            self.app.pop_screen()


def _ordinal_label(index: int, total: int) -> str:
    if total == 1:
        return "a config"
    return f"config {_ORDINAL_LETTERS[index]}"


class ConfigPickerScreen(BaseScreen):
    """Picks n_needed configs one at a time from what's already saved
    (tui.data.list_configs), always offering a fresh baseline_config() so
    the wizard works on an install with nothing saved yet. Calls
    on_chosen(configs) and pops itself once enough have been picked.

    Every row's primary label is tui.data's auto-generated description
    (never the bare SystemConfig.label a human never named for this
    purpose) — pressing 'v' opens that config's diff against baseline in
    Manage Configs' existing diff screen rather than rendering a second
    diff view here."""

    BINDINGS = BaseScreen.BINDINGS + [Binding("v", "view_diff", "View diff")]

    def __init__(self, *, n_needed: int, on_chosen: Callable[[list[SystemConfig]], None], configs_dir: Path = DEFAULT_CONFIGS_DIR) -> None:
        super().__init__()
        self.n_needed = n_needed
        self.on_chosen = on_chosen
        self.configs_dir = configs_dir
        self.chosen: list[SystemConfig] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label(f"Pick {_ordinal_label(0, self.n_needed)}", id="picker-title", classes="title"),
            ListView(*self._build_items(), id="config-picker-list"),
            classes="wizard-body",
        )
        yield Footer()

    def _build_items(self) -> list[ListItem]:
        items = [ListItem(Label("New: baseline (defaults)"), id=_BASELINE_ITEM_ID)]
        for summary in list_configs(configs_dir=self.configs_dir):
            items.append(ListItem(Label(format_config_list_label(summary.description, summary.config_hash)), id=summary.config_hash))
        return items

    def action_view_diff(self) -> None:
        list_view = self.query_one("#config-picker-list", ListView)
        highlighted = list_view.highlighted_child
        if highlighted is None or highlighted.id is None or highlighted.id == _BASELINE_ITEM_ID:
            return  # nothing picked yet, or it's the baseline itself — nothing to diff
        baseline_hash = ensure_baseline_saved(configs_dir=self.configs_dir)
        self.app.push_screen(ConfigDiffScreen(baseline_hash, highlighted.id, configs_dir=self.configs_dir))

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        key = event.item.id
        config = baseline_config(label="baseline") if key == _BASELINE_ITEM_ID else load_config(key, configs_dir=self.configs_dir)
        self.chosen.append(config)
        if len(self.chosen) >= self.n_needed:
            on_chosen, chosen = self.on_chosen, self.chosen
            self.app.pop_screen()
            on_chosen(chosen)
        else:
            self.query_one("#picker-title", Label).update(f"Pick {_ordinal_label(len(self.chosen), self.n_needed)}")
            list_view = self.query_one("#config-picker-list", ListView)
            await list_view.clear()  # clear() returns an awaitable — appending before it lands races and can duplicate IDs
            for item in self._build_items():
                await list_view.append(item)
            list_view.index = 0
            list_view.focus()


class WizardProgressScreen(WorkerProgressScreen):
    """Runs the check in a background thread so the UI stays responsive,
    then lands on the appropriate verdict screen. mode is "single" (one
    config, tui.execution.run_single_config_check) or "comparison" (two
    configs, tui.execution.run_comparison_check — a thin pass-through to
    experiments.runner.run_experiment)."""

    title_text = "Running attack suite..."

    def __init__(
        self,
        *,
        mode: str,
        configs: list[SystemConfig],
        cases: list[AttackCase] | None = None,
        n_runs_per_case: int = DEFAULT_N_RUNS_PER_CASE,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.configs = configs
        self.cases = cases
        self.n_runs_per_case = n_runs_per_case
        self.runs_dir = runs_dir
        self.configs_dir = configs_dir

    def _execute(self) -> None:
        # Only reconstructed/real-model configs need a live client — build
        # it here (not earlier) so a toy-only run never requires
        # ANTHROPIC_API_KEY to be set at all.
        anthropic_client = build_anthropic_client() if any(c.model.provider == "anthropic" for c in self.configs) else None
        if self.mode == "single":
            result = run_single_config_check(
                self.configs[0], cases=self.cases, n_runs_per_case=self.n_runs_per_case, runs_dir=self.runs_dir,
                on_progress=self._on_progress, anthropic_client=anthropic_client,
            )
            records = [r.model_dump(mode="json") for r in result.records]
            summary = compute_single_config_summary(
                records,
                config_label=describe_config_for_humans(result.config_hash, configs_dir=self.configs_dir),
                config_hash=result.config_hash,
            )
            self.app.call_from_thread(self._land_single, summary, records)
        else:
            config_a, config_b = self.configs
            result = run_comparison_check(
                config_a, config_b, cases=self.cases, n_runs_per_case=self.n_runs_per_case, runs_dir=self.runs_dir,
                on_progress=self._on_progress, anthropic_client=anthropic_client,
            )
            report_path = save_experiment_report(result, runs_dir=self.runs_dir)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            verdict = compute_comparison_verdict(report)
            records = [r.model_dump(mode="json") for r in result.records]
            self.app.call_from_thread(self._land_comparison, verdict, report, result.name, records)

    def _land_single(self, summary, records) -> None:
        self.app.pop_screen()
        self.app.push_screen(SingleConfigVerdictScreen(summary, records=records, configs_dir=self.configs_dir))

    def _land_comparison(self, verdict, report, name, records) -> None:
        self.app.pop_screen()
        self.app.push_screen(ComparisonVerdictScreen(verdict, report, name, records, configs_dir=self.configs_dir))
