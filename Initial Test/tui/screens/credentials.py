"""Credentials: a 3-step flow -- Anthropic key, trace source picker,
selected source's fields -- shown before anything else on first run (Home
is unreachable until it completes) and reachable again any time
afterward from Settings for re-editing. Same screens, same
validate-before-write discipline, both times; only `first_run` (blocks
'b'/'h' on every step) and what `on_complete` does next (proceed to Add
Environment vs. pop back to Settings) differ.

Every step uses the exact same visibility rule, uniformly: show it
(pre-filled if file-sourced) unless it's resolved by a real environment
variable, in which case it's skipped entirely -- never prompted for,
never re-validated, never written back to the file (config/credentials.py's
resolve_one/resolve_all doc this). That single rule is what makes first
run and re-edit the same flow: first run always has *something*
non-env-resolved to show (it only appears when config.credentials.
missing_keys() is non-empty); re-edit reaches the same router but starts
from a fresh resolve_all(), so it naturally shows whatever isn't
env-locked -- including the source picker again, which is how "switch
source" works, not a separate re-edit-only code path.

One active trace source at a time, by design (see config/credentials.py's
module docstring for the reasoning) -- picking a new source in the
picker step re-resolves *that* source's own fields fresh (never mixes in
a previously-configured different source's fields), and the final write
only ever includes Anthropic + TRACE_SOURCE + the chosen source's fields,
so switching cleanly drops the old source's fields from the file.

Never renders a credential value anywhere except inside the Input widgets
themselves (masked for secret fields via Input(password=True)) -- error
text is built only from static strings and exception type names (see
config/credentials.py's validate_anthropic/validate_langfuse/
validate_braintrust), never from the values being validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView

from config import credentials
from config.credentials import (
    ANTHROPIC_FIELD,
    ANTHROPIC_KEY,
    CredentialField,
    ResolvedCredential,
    TRACE_SOURCE_KEY,
    TRACE_SOURCE_LABELS,
    TRACE_SOURCES,
)
from config import paths
from tui.app import BaseScreen


def _input_id(key: str) -> str:
    return f"cred-input-{key}"


def credentials_flow_entry_screen(
    *,
    first_run: bool,
    on_complete: Callable[[], None],
    env_path: Path | None = None,
) -> BaseScreen:
    """The one entry point tui/app.py and tui/screens/settings.py call --
    resolves current state fresh and routes to whichever step actually
    has something to show. env_path resolved here, at call time, via the
    `paths` module (not a `from config.paths import ENV_PATH` name copied
    once at import time) so monkeypatching config.paths.ENV_PATH (e.g. in
    tests) actually takes effect -- same reason config/credentials.py's
    own functions do this, see that module's resolve_one() docstring."""
    if env_path is None:
        env_path = paths.ENV_PATH
    resolved = credentials.resolve_all(env_path=env_path)
    return _route(resolved=resolved, values={}, first_run=first_run, on_complete=on_complete, env_path=env_path)


def _route(
    *,
    resolved: dict[str, ResolvedCredential],
    values: dict[str, str],
    first_run: bool,
    on_complete: Callable[[], None],
    env_path: Path,
) -> BaseScreen:
    if resolved[ANTHROPIC_KEY].source != "env" and ANTHROPIC_KEY not in values:
        return AnthropicStepScreen(resolved=resolved, values=values, first_run=first_run, on_complete=on_complete, env_path=env_path)
    if resolved[TRACE_SOURCE_KEY].source != "env" and TRACE_SOURCE_KEY not in values:
        return SourceStepScreen(resolved=resolved, values=values, first_run=first_run, on_complete=on_complete, env_path=env_path)
    source = values.get(TRACE_SOURCE_KEY) or resolved[TRACE_SOURCE_KEY].value
    return FieldsStepScreen(source=source, resolved=resolved, values=values, first_run=first_run, on_complete=on_complete, env_path=env_path)


class _FormStepScreen(BaseScreen):
    """Shared shell for a step that collects one or more text fields:
    renders each non-env-sourced field as a labeled Input (pre-filled if
    file-sourced), a Save/Next button, and a ctrl+s binding that mirrors
    it (shown in the footer, same as every other screen's bindings).
    Subclasses implement _on_collected(values) for what happens once
    every visible field has a non-empty value -- move to the next step
    (AnthropicStepScreen) or actually validate-and-write
    (FieldsStepScreen)."""

    # No ctrl+s binding declared here -- each concrete subclass declares
    # its own with a description matching what it actually does ("Next"
    # vs. "Save"; action_submit_step below is shared regardless of which
    # subclass's BINDINGS entry triggered it). Confirmed by actually
    # running the flow in a real terminal: a shared "Next" description
    # showed up in the footer on the final Save step too, which read as
    # a real inconsistency against that screen's own "Save credentials
    # (ctrl+s)" button label, not just a cosmetic nit.

    def __init__(
        self,
        *,
        fields: tuple[CredentialField, ...],
        resolved: dict[str, ResolvedCredential],
        first_run: bool,
        step_title: str,
        intro: str | None,
        submit_label: str,
    ) -> None:
        super().__init__()
        self.resolved = resolved
        self.first_run = first_run
        self.step_title = step_title
        self.intro = intro
        self.submit_label = submit_label
        self.editable_fields = [f for f in fields if resolved[f.key].source != "env"]
        self.env_sourced_fields = [f for f in fields if resolved[f.key].source == "env"]

    def action_go_back(self) -> None:
        if not self.first_run:
            super().action_go_back()

    def action_go_home(self) -> None:
        if not self.first_run:
            super().action_go_home()

    def on_mount(self) -> None:
        # VerticalScroll (needed so this fits a short terminal, see
        # module docstring) is itself focusable and otherwise grabs
        # initial focus ahead of the first Input, so a user landing here
        # can't just start typing -- same fix as SourceStepScreen's
        # on_mount, found by debugging a real "nothing happens" case.
        if self.editable_fields:
            self.query_one(f"#{_input_id(self.editable_fields[0].key)}", Input).focus()

    def compose(self) -> ComposeResult:
        yield Header()
        rows = [Label(self.step_title, classes="title")]
        if self.intro:
            rows.append(Label(self.intro, classes="subtitle"))

        if not self.editable_fields:
            rows.append(Label("Already set via environment variables.", classes="subtitle"))
            rows.append(Label("(press b/h to go back)", classes="hint"))
        else:
            for field in self.editable_fields:
                resolved = self.resolved[field.key]
                rows.append(Label(f"{field.label} — {field.hint}", classes="hint"))
                rows.append(
                    Input(value=resolved.value or "", password=field.secret, placeholder=field.label, id=_input_id(field.key))
                )
            for field in self.env_sourced_fields:
                rows.append(Label(f"Using {field.key} from your environment.", classes="subtitle"))
            rows.append(Label("", id="cred-error", classes="subtitle"))
            rows.append(Button(self.submit_label, id="submit", variant="primary"))
        yield VerticalScroll(*rows, classes="wizard-body")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._try_submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self._try_submit()

    def action_submit_step(self) -> None:
        self._try_submit()

    def _try_submit(self) -> None:
        if not self.editable_fields:
            return
        collected: dict[str, str] = {}
        for field in self.editable_fields:
            value = self.query_one(f"#{_input_id(field.key)}", Input).value.strip()
            if not value:
                self._show_error(f"{field.label} is required.")
                return
            collected[field.key] = value
        self._on_collected(collected)

    def _show_error(self, message: str) -> None:
        self.query_one("#cred-error", Label).update(message)

    def _on_collected(self, collected: dict[str, str]) -> None:
        raise NotImplementedError


class AnthropicStepScreen(_FormStepScreen):
    BINDINGS = BaseScreen.BINDINGS + [Binding("ctrl+s", "submit_step", "Next")]

    def __init__(
        self,
        *,
        resolved: dict[str, ResolvedCredential],
        values: dict[str, str],
        first_run: bool,
        on_complete: Callable[[], None],
        env_path: Path,
    ) -> None:
        super().__init__(
            fields=(ANTHROPIC_FIELD,),
            resolved=resolved,
            first_run=first_run,
            step_title="Set up credentials" if first_run else "Credentials",
            intro=(
                "Caligula needs an Anthropic key to run attacks, and a trace source to pull real agent traces "
                "from. Values are validated against the real service, then saved locally -- never sent anywhere else."
                if first_run
                else None
            ),
            submit_label="Next (ctrl+s)",
        )
        self.values = values
        self.on_complete = on_complete
        self.env_path = env_path

    def _on_collected(self, collected: dict[str, str]) -> None:
        values = {**self.values, **collected}
        next_screen = _route(resolved=self.resolved, values=values, first_run=self.first_run, on_complete=self.on_complete, env_path=self.env_path)
        self.app.push_screen(next_screen)


class SourceStepScreen(BaseScreen):
    def __init__(
        self,
        *,
        resolved: dict[str, ResolvedCredential],
        values: dict[str, str],
        first_run: bool,
        on_complete: Callable[[], None],
        env_path: Path,
    ) -> None:
        super().__init__()
        self.resolved = resolved
        self.values = values
        self.first_run = first_run
        self.on_complete = on_complete
        self.env_path = env_path

    def action_go_back(self) -> None:
        if not self.first_run:
            super().action_go_back()

    def action_go_home(self) -> None:
        if not self.first_run:
            super().action_go_home()

    def _current_source(self) -> str | None:
        source = self.values.get(TRACE_SOURCE_KEY) or self.resolved[TRACE_SOURCE_KEY].value
        return source if source in TRACE_SOURCES else None

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(
            Label("Choose a trace source" if self.first_run else "Trace source", classes="title"),
            Label("Which system are your agent's real traces logged to?", classes="subtitle"),
            ListView(*(ListItem(Label(label), id=key) for key, label in TRACE_SOURCE_LABELS.items()), id="source-list"),
            classes="wizard-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        # VerticalScroll (needed so this fits a short terminal, see
        # module docstring) is itself focusable and otherwise grabs
        # initial focus ahead of the ListView, so Enter/arrow keys never
        # reach it without this -- confirmed by debugging a real "Enter
        # does nothing" failure, not a guess.
        list_view = self.query_one("#source-list", ListView)
        list_view.focus()
        current = self._current_source()
        if current is not None:
            list_view.index = list(TRACE_SOURCE_LABELS.keys()).index(current)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        source = event.item.id
        values = {**self.values, TRACE_SOURCE_KEY: source}
        resolved = dict(self.resolved)
        resolved.update(credentials.resolve_source_fields(source, env_path=self.env_path))
        next_screen = _route(resolved=resolved, values=values, first_run=self.first_run, on_complete=self.on_complete, env_path=self.env_path)
        self.app.push_screen(next_screen)


class FieldsStepScreen(_FormStepScreen):
    BINDINGS = BaseScreen.BINDINGS + [Binding("ctrl+s", "submit_step", "Save")]

    def __init__(
        self,
        *,
        source: str,
        resolved: dict[str, ResolvedCredential],
        values: dict[str, str],
        first_run: bool,
        on_complete: Callable[[], None],
        env_path: Path,
    ) -> None:
        source_label = TRACE_SOURCE_LABELS[source]
        super().__init__(
            fields=TRACE_SOURCES[source],
            resolved=resolved,
            first_run=first_run,
            step_title=f"{source_label} credentials",
            intro=(
                f"Last step -- fill in your {source_label} details, then press ctrl+s to save." if first_run else None
            ),
            submit_label="Save credentials (ctrl+s)",
        )
        self.source = source
        self.values = values
        self.on_complete = on_complete
        self.env_path = env_path

    def _on_collected(self, collected: dict[str, str]) -> None:
        values = {**self.values, **collected}
        self._show_error("Validating...")
        self.run_worker(lambda: self._validate_and_write(values), thread=True, exclusive=True)

    def _validate_and_write(self, values: dict[str, str]) -> None:
        relevant_keys = [ANTHROPIC_KEY, TRACE_SOURCE_KEY] + [field.key for field in TRACE_SOURCES[self.source]]
        effective = {key: values.get(key, self.resolved[key].value) for key in relevant_keys}

        if self.resolved[ANTHROPIC_KEY].source != "env":
            ok, message = credentials.validate_anthropic(effective[ANTHROPIC_KEY])
            if not ok:
                self.app.call_from_thread(self._show_error, f"Anthropic: {message}")
                return

        source_label = TRACE_SOURCE_LABELS[self.source]
        if self.source == "langfuse":
            ok, message = credentials.validate_langfuse(
                secret_key=effective["LANGFUSE_SECRET_KEY"],
                public_key=effective["LANGFUSE_PUBLIC_KEY"],
                base_url=effective["LANGFUSE_BASE_URL"],
                project_id=effective["LANGFUSE_PROJECT_ID"],
            )
        else:
            ok, message = credentials.validate_braintrust(
                api_key=effective["BRAINTRUST_API_KEY"],
                project_name=effective["BRAINTRUST_PROJECT_NAME"],
            )
        if not ok:
            self.app.call_from_thread(self._show_error, f"{source_label}: {message}")
            return

        # Scoped to exactly Anthropic + TRACE_SOURCE + this source's own
        # fields -- never anything left over in self.resolved from a
        # different source the user was previously configured with or
        # briefly considered before switching (see module docstring).
        to_write = {
            key: effective[key] for key in relevant_keys if self.resolved.get(key, ResolvedCredential(key, None, None)).source != "env"
        }
        credentials.write_credentials(to_write, env_path=self.env_path)
        self.app.call_from_thread(self.on_complete)
