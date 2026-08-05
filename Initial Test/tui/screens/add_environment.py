"""'Add environment': connect the Langfuse project configured in .env,
pull/reuse a cached trace batch, let the user pick which agent_name group
becomes a new reconstructed SystemConfig, and save it. After this, the
reconstructed environment is just another saved config — pickable through
the exact same ConfigPickerScreen flow (tui/screens/wizard.py) as any
hand-authored one, no special-casing needed there.

Never hits the Langfuse API to select a group (ingestion/reconstruct.py's
Part 2 revision) — only the explicit "Pull traces" action does that.
Picking a group reconstructs straight from whatever's already cached on
disk via ingestion.reconstruct.reconstruct_from_cache.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView, ProgressBar

from ingestion.langfuse_client import DEFAULT_BATCH_SIZE, DEFAULT_TRACES_DIR, load_cached_trace_ids, load_cached_traces, pull_traces
from ingestion.reconstruct import GroupSummary, group_traces, reconstruct_from_cache, summarize_groups
from target_system.config import DEFAULT_CONFIGS_DIR, SystemConfig, save_config
from tui.app import BaseScreen

_NONE_GROUP_LABEL = "(no agent_name tag)"


def _group_item_label(summary: GroupSummary) -> str:
    name = summary.agent_name or _NONE_GROUP_LABEL
    return f"{name} — {summary.trace_count} trace(s)"


class AddEnvironmentScreen(BaseScreen):
    """Entry point: resolves the .env-configured project, offers whatever
    is already cached if anything is, and always offers a fresh pull."""

    def __init__(self, *, traces_dir: Path = DEFAULT_TRACES_DIR, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> None:
        super().__init__()
        self.traces_dir = traces_dir
        self.configs_dir = configs_dir
        self.project_id: str | None = None
        self.error: str | None = None
        self.n_cached = 0

    def compose(self) -> ComposeResult:
        yield Header()
        try:
            from ingestion.langfuse_client import default_project_id

            self.project_id = default_project_id()
            self.n_cached = len(load_cached_trace_ids(self.project_id, traces_dir=self.traces_dir))
        except RuntimeError as exc:
            self.error = str(exc)

        if self.error is not None:
            yield Vertical(
                Label("Add environment", classes="title"),
                Label(f"Langfuse isn't configured: {self.error}", classes="subtitle"),
                Label("Set the required vars in .env and try again (b to go back).", classes="hint"),
                classes="wizard-body",
            )
        else:
            items = []
            if self.n_cached:
                items.append(ListItem(Label(f"Use {self.n_cached} cached trace(s)"), id="use_cache"))
            items.append(ListItem(Label(f"Pull traces from Langfuse (up to {DEFAULT_BATCH_SIZE})"), id="pull"))
            yield Vertical(
                Label("Add environment", classes="title"),
                Label(f"Project: {self.project_id}", classes="subtitle"),
                ListView(*items, id="add-env-menu"),
                classes="wizard-body",
            )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "use_cache":
            self.app.push_screen(
                AddEnvironmentGroupsScreen(project_id=self.project_id, traces_dir=self.traces_dir, configs_dir=self.configs_dir)
            )
        elif event.item.id == "pull":
            self.app.push_screen(
                PullTracesScreen(project_id=self.project_id, traces_dir=self.traces_dir, configs_dir=self.configs_dir)
            )


class PullTracesScreen(BaseScreen):
    """Only screen in this flow that talks to the Langfuse API. Runs on a
    worker thread so the UI stays responsive; pull_traces has no
    per-trace progress callback (it's a small, idempotent per-trace cache
    fill — see ingestion/langfuse_client.py), so this shows an
    indeterminate bar rather than a fabricated completed/total count."""

    title_text = "Pulling traces from Langfuse..."

    def __init__(
        self,
        *,
        project_id: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        traces_dir: Path = DEFAULT_TRACES_DIR,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
    ) -> None:
        super().__init__()
        self.project_id = project_id
        self.batch_size = batch_size
        self.traces_dir = traces_dir
        self.configs_dir = configs_dir

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label(self.title_text, classes="title"),
            Label(f"Project: {self.project_id} (up to {self.batch_size} traces; already-cached ones are skipped)", classes="subtitle"),
            ProgressBar(id="progress", show_eta=False),
            classes="wizard-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._pull, thread=True, exclusive=True)

    def _pull(self) -> None:
        try:
            pull_traces(project_id=self.project_id, batch_size=self.batch_size, traces_dir=self.traces_dir)
        except Exception as exc:  # noqa: BLE001 - network/auth failures surfaced to the user, not swallowed
            self.app.call_from_thread(self._land_error, f"{type(exc).__name__}: {exc}")
            return
        self.app.call_from_thread(self._land_success)

    def _land_success(self) -> None:
        self.app.pop_screen()
        self.app.push_screen(
            AddEnvironmentGroupsScreen(project_id=self.project_id, traces_dir=self.traces_dir, configs_dir=self.configs_dir)
        )

    def _land_error(self, message: str) -> None:
        self.app.pop_screen()
        self.app.push_screen(PullErrorScreen(message))


class PullErrorScreen(BaseScreen):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label("Pull failed", classes="title"),
            Label(self.message, classes="subtitle"),
            Label("Press b to go back and try again.", classes="hint"),
            classes="wizard-body",
        )
        yield Footer()


class AddEnvironmentGroupsScreen(BaseScreen):
    """Groups whatever's cached for this project client-side (no API
    call — see module docstring) and lets the user pick which group
    becomes a new reconstructed SystemConfig. Every group actually found
    is shown (name, trace count), sorted by trace count descending, same
    as ingestion.reconstruct.summarize_groups's own ordering."""

    def __init__(
        self, *, project_id: str, traces_dir: Path = DEFAULT_TRACES_DIR, configs_dir: Path = DEFAULT_CONFIGS_DIR
    ) -> None:
        super().__init__()
        self.project_id = project_id
        self.traces_dir = traces_dir
        self.configs_dir = configs_dir
        self.summaries: list[GroupSummary] = []

    def compose(self) -> ComposeResult:
        yield Header()
        traces = load_cached_traces(self.project_id, traces_dir=self.traces_dir)
        groups = group_traces(traces)
        self.summaries = summarize_groups(groups)
        if not self.summaries:
            yield Vertical(
                Label("Add environment", classes="title"),
                Label("No real (non-test-noise) groups found in the cached batch.", classes="subtitle"),
                Label("Press b to go back and pull a larger batch.", classes="hint"),
                classes="wizard-body",
            )
        else:
            yield Vertical(
                Label("Add environment", classes="title"),
                Label("Pick which group to reconstruct as a new environment.", classes="subtitle"),
                ListView(
                    # Textual widget ids must be identifier-like (no
                    # spaces) -- real agent_name values ("Invoice
                    # Generation Assistant") aren't, so items are keyed
                    # positionally and mapped back through self.summaries,
                    # not by embedding the name itself in the id.
                    *(ListItem(Label(_group_item_label(s)), id=f"group-{i}") for i, s in enumerate(self.summaries)),
                    id="group-list",
                ),
                classes="wizard-body",
            )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = int(event.item.id.removeprefix("group-"))
        agent_name = self.summaries[index].agent_name
        config = reconstruct_from_cache(project_id=self.project_id, agent_name=agent_name, traces_dir=self.traces_dir)
        config_hash = save_config(config, configs_dir=self.configs_dir)
        self.app.push_screen(AddEnvironmentDoneScreen(config, config_hash))


class AddEnvironmentDoneScreen(BaseScreen):
    """Confirms what got saved — trace count, tools, and any reconstruction
    warnings (e.g. multi-model drift) surfaced right here rather than left
    for the verdict screen to be the first place fidelity caveats appear."""

    def __init__(self, config: SystemConfig, config_hash: str) -> None:
        super().__init__()
        self.config = config
        self.config_hash = config_hash

    def compose(self) -> ComposeResult:
        yield Header()
        provenance = self.config.provenance
        agent = self.config.supervisor()
        lines = [
            Label("Environment saved", classes="title"),
            Label(f"{self.config.label}  ({self.config_hash})", classes="subtitle"),
            Label(f"Reconstructed from {provenance.trace_count} trace(s), model={self.config.model.model_name}", classes="hint"),
            Label(f"Tools observed: {', '.join(agent.tools) or '(none — conversational only)'}", classes="hint"),
        ]
        if provenance.warnings:
            lines.append(Label("Warnings: " + "; ".join(provenance.warnings), classes="hint"))
        if provenance.other_groups_found:
            others = ", ".join(f"{g.agent_name or _NONE_GROUP_LABEL} ({g.trace_count})" for g in provenance.other_groups_found)
            lines.append(Label(f"Other groups also found in this batch: {others}", classes="hint"))
        lines.append(Label("This environment is real-model-only — pick it from the wizard like any saved config.", classes="hint"))
        yield Vertical(*lines, classes="wizard-body")
        yield Footer()
