"""'Add environment': connects whichever trace source is actually
configured (TRACE_SOURCE -- see config/credentials.py) and pulls/reuses a
cached trace batch, groups it, and reconstructs a SystemConfig from
whichever group is picked. Dispatches per source rather than assuming
Langfuse throughout -- this fixes a real, confirmed bug: this screen used
to check Langfuse configuration unconditionally regardless of which
source the credentials screen actually validated, so a
Braintrust-configured install showed "Langfuse isn't configured:
LANGFUSE_PROJECT_ID not set" even though Braintrust was correctly set up
and validated. Nothing in this screen's logic hardcodes Langfuse as the
source anymore -- every branch reads self.source (resolved fresh from
config.credentials.resolve_source()) and dispatches to the matching
ingestion client and reconstruction module.

Reconstruction itself (ingestion/reconstruct.py for Langfuse,
ingestion/braintrust_reconstruct.py for Braintrust) is genuinely
different per source -- confirmed real, structural differences during
Braintrust's investigation (grouping key: root span metadata["workflow_
name"], not "agent_name" the way Langfuse's is; tool-call shape: flat
input dict and JSON-encoded-string output.result.content, not Langfuse's
{"tool","inputs"} wrapper; cost aggregation: only llm-type spans may be
summed, confirmed task-level spans can exactly mirror -- i.e. double-
count -- a child's cost), not just relabeled field names. This screen
only ever normalizes each module's own GroupSummary (agent_name for
Langfuse, workflow_name for Braintrust) into a plain (name, trace_count)
tuple for display -- it never assumes one module's shape fits the other.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView, ProgressBar

import config.credentials as credentials
import ingestion.braintrust_client as bt
import ingestion.braintrust_reconstruct as bt_reconstruct
import ingestion.langfuse_client as lf
import ingestion.reconstruct as lf_reconstruct
from target_system.config import DEFAULT_CONFIGS_DIR, SystemConfig, save_config
from tui.app import BaseScreen

_NONE_GROUP_LABEL = "(no name tag)"


def _group_item_label(name: str | None, trace_count: int) -> str:
    return f"{name or _NONE_GROUP_LABEL} — {trace_count} trace(s)"


def _source_label(source: str | None) -> str:
    return credentials.TRACE_SOURCE_LABELS.get(source, "Trace source")


class AddEnvironmentScreen(BaseScreen):
    """Entry point: resolves whichever trace source is actually
    configured (self.source, set fresh in compose() every time this
    screen is shown -- never assumed/cached across visits), offers
    whatever's already cached if anything is, and always offers a fresh
    pull."""

    def __init__(
        self,
        *,
        langfuse_traces_dir: Path = lf.DEFAULT_TRACES_DIR,
        braintrust_traces_dir: Path = bt.DEFAULT_TRACES_DIR,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
    ) -> None:
        super().__init__()
        self.langfuse_traces_dir = langfuse_traces_dir
        self.braintrust_traces_dir = braintrust_traces_dir
        self.configs_dir = configs_dir
        self.source: str | None = None
        self.project_id: str | None = None
        self.error: str | None = None
        self.n_cached = 0

    @property
    def traces_dir(self) -> Path:
        return self.braintrust_traces_dir if self.source == "braintrust" else self.langfuse_traces_dir

    def compose(self) -> ComposeResult:
        yield Header()
        self.source = credentials.resolve_source().value
        try:
            if self.source == "langfuse":
                self.project_id = lf.default_project_id()
                self.n_cached = len(lf.load_cached_trace_ids(self.project_id, traces_dir=self.langfuse_traces_dir))
            elif self.source == "braintrust":
                self.project_id = bt.default_project_name()
                self.n_cached = len(bt.load_cached_trace_ids(self.project_id, traces_dir=self.braintrust_traces_dir))
            else:
                raise RuntimeError("no trace source configured -- set one up from Settings")
        except RuntimeError as exc:
            self.error = str(exc)

        if self.error is not None:
            yield Vertical(
                Label("Add environment", classes="title"),
                Label(f"{_source_label(self.source)} isn't configured: {self.error}", classes="subtitle"),
                Label("Fix it from Settings and try again (b to go back).", classes="hint"),
                classes="wizard-body",
            )
        else:
            items = []
            if self.n_cached:
                items.append(ListItem(Label(f"Use {self.n_cached} cached trace(s)"), id="use_cache"))
            batch_size = bt.DEFAULT_BATCH_SIZE if self.source == "braintrust" else lf.DEFAULT_BATCH_SIZE
            items.append(ListItem(Label(f"Pull traces from {_source_label(self.source)} (up to {batch_size})"), id="pull"))
            yield Vertical(
                Label("Add environment", classes="title"),
                Label(f"Source: {_source_label(self.source)} — Project: {self.project_id}", classes="subtitle"),
                ListView(*items, id="add-env-menu"),
                classes="wizard-body",
            )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "use_cache":
            self.app.push_screen(
                AddEnvironmentGroupsScreen(source=self.source, project_id=self.project_id, traces_dir=self.traces_dir, configs_dir=self.configs_dir)
            )
        elif event.item.id == "pull":
            self.app.push_screen(
                PullTracesScreen(source=self.source, project_id=self.project_id, traces_dir=self.traces_dir, configs_dir=self.configs_dir)
            )


class PullTracesScreen(BaseScreen):
    """Only screen in this flow that talks to a trace-source API. Runs on
    a worker thread so the UI stays responsive; neither client's
    pull_traces has a progress callback -- Langfuse's is still a small,
    idempotent per-trace cache fill (ingestion/langfuse_client.py),
    Braintrust's is one bulk fetch followed by writing every trace's
    cache file at once (ingestion/braintrust_client.py -- per-trace calls
    were the real cause of a real, reproduced 429 at ordinary batch
    sizes, fixed by batching) -- either way, this shows an indeterminate
    bar rather than a fabricated completed/total count."""

    def __init__(
        self,
        *,
        source: str,
        project_id: str,
        batch_size: int | None = None,
        traces_dir: Path,
        configs_dir: Path = DEFAULT_CONFIGS_DIR,
    ) -> None:
        super().__init__()
        self.source = source
        self.project_id = project_id
        self.batch_size = batch_size or (bt.DEFAULT_BATCH_SIZE if source == "braintrust" else lf.DEFAULT_BATCH_SIZE)
        self.traces_dir = traces_dir
        self.configs_dir = configs_dir

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label(f"Pulling traces from {_source_label(self.source)}...", classes="title"),
            Label(f"Project: {self.project_id} (up to {self.batch_size} traces; already-cached ones are skipped)", classes="subtitle"),
            ProgressBar(id="progress", show_eta=False),
            classes="wizard-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._pull, thread=True, exclusive=True)

    def _pull(self) -> None:
        try:
            if self.source == "braintrust":
                bt.pull_traces(project_name=self.project_id, batch_size=self.batch_size, traces_dir=self.traces_dir)
            else:
                lf.pull_traces(project_id=self.project_id, batch_size=self.batch_size, traces_dir=self.traces_dir)
        except Exception as exc:  # noqa: BLE001 - network/auth failures surfaced to the user, not swallowed
            self.app.call_from_thread(self._land_error, f"{type(exc).__name__}: {exc}")
            return
        self.app.call_from_thread(self._land_success)

    def _land_success(self) -> None:
        self.app.pop_screen()
        self.app.push_screen(
            AddEnvironmentGroupsScreen(source=self.source, project_id=self.project_id, traces_dir=self.traces_dir, configs_dir=self.configs_dir)
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
    as each reconstruction module's own summarize_groups ordering.

    Dispatches on self.source for both loading (lf vs bt client) and
    reconstruction (lf_reconstruct vs bt_reconstruct) -- see this module's
    docstring on why the two reconstruction modules are genuinely
    different, not just relabeled. self.summaries is normalized to a
    plain list[tuple[name, trace_count]] right here so the rest of this
    screen (rendering, picking) never needs to know which source's
    GroupSummary shape it came from."""

    def __init__(
        self, *, source: str, project_id: str, traces_dir: Path, configs_dir: Path = DEFAULT_CONFIGS_DIR
    ) -> None:
        super().__init__()
        self.source = source
        self.project_id = project_id
        self.traces_dir = traces_dir
        self.configs_dir = configs_dir
        self.summaries: list[tuple[str | None, int]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        if self.source == "braintrust":
            traces = bt.load_cached_traces(self.project_id, traces_dir=self.traces_dir)
            groups = bt_reconstruct.group_traces(traces)
            raw_summaries = bt_reconstruct.summarize_groups(groups)
            self.summaries = [(s.workflow_name, s.trace_count) for s in raw_summaries]
        else:
            traces = lf.load_cached_traces(self.project_id, traces_dir=self.traces_dir)
            groups = lf_reconstruct.group_traces(traces)
            raw_summaries = lf_reconstruct.summarize_groups(groups)
            self.summaries = [(s.agent_name, s.trace_count) for s in raw_summaries]

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
                    # spaces) -- real names ("Invoice Generation
                    # Assistant", "homepilot-ticket-analysis") aren't
                    # guaranteed to be, so items are keyed positionally
                    # and mapped back through self.summaries, not by
                    # embedding the name itself in the id.
                    *(ListItem(Label(_group_item_label(name, count)), id=f"group-{i}") for i, (name, count) in enumerate(self.summaries)),
                    id="group-list",
                ),
                classes="wizard-body",
            )
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = int(event.item.id.removeprefix("group-"))
        name, _count = self.summaries[index]
        if self.source == "braintrust":
            config = bt_reconstruct.reconstruct_from_cache(project_id=self.project_id, workflow_name=name, traces_dir=self.traces_dir)
        else:
            config = lf_reconstruct.reconstruct_from_cache(project_id=self.project_id, agent_name=name, traces_dir=self.traces_dir)
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
