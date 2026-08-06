"""Credentials screen: shown before anything else on first run (Home is
unreachable until it completes) and reachable again any time afterward
from Settings for re-editing. Same component, same validate-before-write
discipline, both times -- only `first_run` (blocks 'b'/'h', and lands on
Add Environment instead of Home on success -- see on_complete's callers
in tui/app.py and tui/screens/settings.py) differs.

Five fields plus labels/hints don't fit a standard 24-row terminal, so the
field list is wrapped in VerticalScroll (not the plain Vertical every
other screen in this app uses) -- confirmed by actually launching this
screen in a real 24-row terminal during development: with a plain
Vertical, the last two fields and the Save button rendered completely
off-screen with no way to reach them. ctrl+s (shown in the Footer, same
as every other screen's bindings) is the documented way to submit;
pressing Enter in any field submits too, as a bonus, but isn't the
primary documented mechanism since it's easy to trigger by accident
mid-form.

A field only gets an editable Input if it isn't already resolved by a
real environment variable -- config/credentials.py's resolve_all() checks
os.environ before the config file, and an env-sourced value is used
directly, never prompted for and never written to the file (see
config/credentials.py's docstring). Validation and the write only cover
the fields actually shown here; a fully env-resolved service (e.g. every
LANGFUSE_* var exported in the shell) is never re-validated on submit.

Never renders a credential value anywhere except inside the Input widgets
themselves (masked for secret fields via Input(password=True)) -- error
text is built only from static strings and exception type names (see
config/credentials.py's validate_anthropic/validate_langfuse), never from
the values being validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label

from config import credentials
from config.credentials import CREDENTIAL_FIELDS, LANGFUSE_KEYS, ResolvedCredential
from config.paths import ENV_PATH
from tui.app import BaseScreen


def _input_id(key: str) -> str:
    return f"cred-input-{key}"


class CredentialsScreen(BaseScreen):
    BINDINGS = BaseScreen.BINDINGS + [Binding("ctrl+s", "submit_credentials", "Save")]

    def __init__(
        self,
        *,
        first_run: bool,
        on_complete: Callable[[], None],
        env_path: Path = ENV_PATH,
    ) -> None:
        super().__init__()
        self.first_run = first_run
        self.on_complete = on_complete
        self.env_path = env_path
        self.resolved: dict[str, ResolvedCredential] = {}
        self.editable_fields = []

    # Home isn't reachable yet on first run -- there's nothing to go back
    # or home to, so those bindings are simply disabled rather than
    # popping into a blank default screen.
    def action_go_back(self) -> None:
        if not self.first_run:
            super().action_go_back()

    def action_go_home(self) -> None:
        if not self.first_run:
            super().action_go_home()

    def compose(self) -> ComposeResult:
        yield Header()
        self.resolved = credentials.resolve_all(env_path=self.env_path)
        self.editable_fields = [f for f in CREDENTIAL_FIELDS if self.resolved[f.key].source != "env"]

        rows = [Label("Set up credentials" if self.first_run else "Credentials", classes="title")]
        if self.first_run:
            rows.append(
                Label(
                    "Caligula needs these before it can run. Values are validated against the real "
                    "service, then saved locally -- never sent anywhere else.",
                    classes="subtitle",
                )
            )

        if not self.editable_fields:
            rows.append(Label("Every credential is already set via environment variables.", classes="subtitle"))
            rows.append(Label("(press b/h to go back)", classes="hint"))
        else:
            rows.append(Label("Fill in every field below, then press ctrl+s to save (scroll for more).", classes="subtitle"))
            for field in CREDENTIAL_FIELDS:
                resolved = self.resolved[field.key]
                rows.append(Label(f"{field.label} — {field.hint}", classes="hint"))
                if resolved.source == "env":
                    rows.append(Label(f"Using {field.key} from your environment.", classes="subtitle"))
                else:
                    rows.append(
                        Input(
                            value=resolved.value or "",
                            password=field.secret,
                            placeholder=field.label,
                            id=_input_id(field.key),
                        )
                    )
            rows.append(Label("", id="cred-error", classes="subtitle"))
            rows.append(Button("Save credentials (ctrl+s)", id="submit", variant="primary"))
        # VerticalScroll, not the plain Vertical every other screen uses --
        # five fields' worth of labels+inputs don't fit a standard 24-row
        # terminal, and without this the last fields and the Save button
        # render completely unreachable (confirmed by actually running
        # this screen in one — see module docstring).
        yield VerticalScroll(*rows, classes="wizard-body")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self._submit()

    def action_submit_credentials(self) -> None:
        self._submit()

    def _submit(self) -> None:
        if not self.editable_fields:
            return
        values: dict[str, str] = {}
        for field in self.editable_fields:
            value = self.query_one(f"#{_input_id(field.key)}", Input).value.strip()
            if not value:
                self._show_error(f"{field.label} is required.")
                return
            values[field.key] = value
        self._show_error("Validating...")
        self.run_worker(lambda: self._validate_and_write(values), thread=True, exclusive=True)

    def _validate_and_write(self, values: dict[str, str]) -> None:
        effective = {key: values.get(key, cred.value) for key, cred in self.resolved.items()}

        if self.resolved["ANTHROPIC_API_KEY"].source != "env":
            ok, message = credentials.validate_anthropic(effective["ANTHROPIC_API_KEY"])
            if not ok:
                self.app.call_from_thread(self._show_error, f"Anthropic: {message}")
                return

        if any(self.resolved[key].source != "env" for key in LANGFUSE_KEYS):
            ok, message = credentials.validate_langfuse(
                secret_key=effective["LANGFUSE_SECRET_KEY"],
                public_key=effective["LANGFUSE_PUBLIC_KEY"],
                base_url=effective["LANGFUSE_BASE_URL"],
                project_id=effective["LANGFUSE_PROJECT_ID"],
            )
            if not ok:
                self.app.call_from_thread(self._show_error, f"Langfuse: {message}")
                return

        credentials.write_credentials(values, env_path=self.env_path)
        self.app.call_from_thread(self._land_success)

    def _show_error(self, message: str) -> None:
        self.query_one("#cred-error", Label).update(message)

    def _land_success(self) -> None:
        self.on_complete()
