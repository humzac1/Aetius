"""Shared "run a blocking call in a worker thread, show a progress bar,
land somewhere when done" screen base. The wizard (tui/screens/wizard.py)
and the preset runner (tui/screens/presets.py) both subclass this instead
of each reimplementing the same threading/progress-bar boilerplate — the
only thing that differs between them is what _execute calls and where it
lands when done.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, ProgressBar

from tui.app import BaseScreen


class WorkerProgressScreen(BaseScreen):
    title_text = "Running attack suite..."

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label(self.title_text, classes="title"),
            ProgressBar(id="progress", show_eta=False),
            Label("", id="progress-label", classes="subtitle"),
            classes="wizard-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._execute, thread=True, exclusive=True)

    def _on_progress(self, completed: int, total: int) -> None:
        # run_experiment/run_single_config_check call on_progress on the
        # calling (worker) thread — marshal back to the UI thread before
        # touching any widget.
        self.app.call_from_thread(self._update_progress, completed, total)

    def _update_progress(self, completed: int, total: int) -> None:
        bar = self.query_one("#progress", ProgressBar)
        label = self.query_one("#progress-label", Label)
        if total == 0:
            bar.update(total=1, progress=1)
            label.update("Everything already cached.")
        else:
            bar.update(total=total, progress=completed)
            label.update(f"{completed}/{total} runs")

    def _execute(self) -> None:
        """Runs on a worker thread (see on_mount). Subclasses call the
        blocking experiment/execution function here, passing
        self._on_progress, and finish with self.app.call_from_thread(...)
        to land on whatever screen comes next."""
        raise NotImplementedError
