""""Run a specific experiment" — a menu wrapper around the 5 existing
presets in experiments/presets.py. Execution is exactly what
experiments/cli.py's `run --preset` does: experiments.runner.run_experiment
with the preset's own ArmSpec pair and preset.name as the experiment name
(not an ad hoc adhoc_<hash>_<hash> name — presets have real, checked-in
names other tooling already expects), followed by
experiments.persist.save_experiment_report. No experiment-running logic is
reimplemented here.
"""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView

from experiments.persist import save_experiment_report
from experiments.presets import PRESETS, ExperimentPreset
from experiments.runner import DEFAULT_RUNS_DIR, run_experiment
from target_system.config import DEFAULT_CONFIGS_DIR
from tui.app import BaseScreen
from tui.formatting import preset_display_name, preset_verdict_hint
from tui.screens.progress import WorkerProgressScreen
from tui.screens.verdict import ComparisonVerdictScreen
from tui.verdict_logic import compute_comparison_verdict

DEFAULT_N_RUNS_PER_CASE = 5


def _preset_item_label(name: str, preset: ExperimentPreset) -> str:
    base = f"{preset_display_name(name)} — {preset.description}"
    hint = preset_verdict_hint(name)
    return f"{base} — {hint}" if hint else base


class PresetMenuScreen(BaseScreen):
    def __init__(self, *, runs_dir: Path = DEFAULT_RUNS_DIR, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> None:
        super().__init__()
        self.runs_dir = runs_dir
        self.configs_dir = configs_dir

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label("Run a specific experiment", classes="title"),
            ListView(
                *(ListItem(Label(_preset_item_label(name, preset)), id=name) for name, preset in PRESETS.items()),
                id="preset-menu",
            ),
            classes="wizard-body",
        )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        preset = PRESETS[event.item.id]
        self.app.push_screen(PresetProgressScreen(preset=preset, runs_dir=self.runs_dir, configs_dir=self.configs_dir))


class PresetProgressScreen(WorkerProgressScreen):
    title_text = "Running preset experiment..."

    def __init__(
        self,
        *,
        preset: ExperimentPreset,
        n_runs_per_case: int = DEFAULT_N_RUNS_PER_CASE,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
    ) -> None:
        super().__init__()
        self.preset = preset
        self.n_runs_per_case = n_runs_per_case
        self.runs_dir = runs_dir
        self.configs_dir = configs_dir

    def _execute(self) -> None:
        result = run_experiment(
            self.preset.arm_a,
            self.preset.arm_b,
            experiment_name=self.preset.name,
            n_runs_per_case=self.n_runs_per_case,
            runs_dir=self.runs_dir,
            on_progress=self._on_progress,
        )
        report_path = save_experiment_report(
            result, sequential_outcome_key=self.preset.sequential_outcome_key, runs_dir=self.runs_dir
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        verdict = compute_comparison_verdict(report)
        records = [r.model_dump(mode="json") for r in result.records]
        self.app.call_from_thread(self._land, verdict, report, result.name, records)

    def _land(self, verdict, report, name, records) -> None:
        self.app.pop_screen()
        self.app.push_screen(ComparisonVerdictScreen(verdict, report, name, records, configs_dir=self.configs_dir))
