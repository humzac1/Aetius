"""Headless Textual Pilot tests for the 3-step credentials flow
(AnthropicStepScreen -> SourceStepScreen -> FieldsStepScreen) — the
first-run gate and the Settings "Edit credentials" flow share these exact
screens (see tui/screens/credentials.py's module docstring). Covers:
field display (env-sourced fields/steps aren't prompted for), masked
secret input, validation failure with retry (nothing written),
write-only-on-success for both sources, switching sources, and that no
credential value ever appears in captured terminal output during a
deliberately failed validation.
"""

from __future__ import annotations

import anthropic
import httpx

import config.credentials as credentials
import config.paths as paths
import ingestion.braintrust_client as bt
import ingestion.langfuse_client as lf
from tests.tui_test_support import run_async
from tests.tui_test_support import wait_until as _wait_until
from tui.app import HarnessApp
from tui.screens.credentials import (
    AnthropicStepScreen,
    FieldsStepScreen,
    SourceStepScreen,
    _input_id,
    credentials_flow_entry_screen,
)


def _clear_all(monkeypatch) -> None:
    monkeypatch.delenv(credentials.ANTHROPIC_KEY, raising=False)
    monkeypatch.delenv(credentials.TRACE_SOURCE_KEY, raising=False)
    for fields in credentials.TRACE_SOURCES.values():
        for field in fields:
            monkeypatch.delenv(field.key, raising=False)


def _entry(**kwargs):
    on_complete_calls = []
    screen = credentials_flow_entry_screen(on_complete=lambda: on_complete_calls.append(True), **kwargs)
    return screen, on_complete_calls


def _succeed_anthropic(monkeypatch):
    class _FakeModels:
        def list(self, limit=None):
            return ["ok"]

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)


def _fail_anthropic(monkeypatch):
    fake_response = httpx.Response(401, request=httpx.Request("GET", "https://api.anthropic.com/v1/models"))

    class _FakeModels:
        def list(self, limit=None):
            raise anthropic.AuthenticationError("bad key", response=fake_response, body=None)

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)


class _FakeProject:
    def __init__(self, id):
        self.id = id


class _FakeLfProjectsResponse:
    def __init__(self, data):
        self.data = data


def _succeed_langfuse(monkeypatch, *, project_ids=("proj-test",)):
    class _FakeProjectsClient:
        def get(self):
            return _FakeLfProjectsResponse([_FakeProject(pid) for pid in project_ids])

    class _FakeApi:
        projects = _FakeProjectsClient()

    class _FakeLfClient:
        api = _FakeApi()

    monkeypatch.setattr(lf, "build_client", lambda **kwargs: _FakeLfClient())


def _fail_langfuse(monkeypatch):
    class _FakeProjectsClient:
        def get(self):
            return _FakeLfProjectsResponse([])

    class _FakeApi:
        projects = _FakeProjectsClient()

    class _FakeLfClient:
        api = _FakeApi()

    monkeypatch.setattr(lf, "build_client", lambda **kwargs: _FakeLfClient())


class _FakeBtResponse:
    def __init__(self, names):
        self._payload = {"objects": [{"name": n} for n in names]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _succeed_braintrust(monkeypatch, *, project_names=("proj-test",)):
    class _FakeConn:
        def get(self, path):
            return _FakeBtResponse(project_names)

    class _FakeState:
        def api_conn(self):
            return _FakeConn()

    monkeypatch.setattr(bt, "build_client", lambda **kwargs: _FakeState())


def _fail_braintrust(monkeypatch):
    class _FakeConn:
        def get(self, path):
            return _FakeBtResponse([])

    class _FakeState:
        def api_conn(self):
            return _FakeConn()

    monkeypatch.setattr(bt, "build_client", lambda **kwargs: _FakeState())


def _fill(screen, values: dict[str, str]) -> None:
    from textual.widgets import Input

    for key, value in values.items():
        screen.query_one(f"#{_input_id(key)}", Input).value = value


_LANGFUSE_VALUES = {
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
    "LANGFUSE_PROJECT_ID": "proj-test",
}
_BRAINTRUST_VALUES = {
    "BRAINTRUST_API_KEY": "bt-key-test",
    "BRAINTRUST_PROJECT_NAME": "proj-test",
}


# --- step 1: Anthropic --------------------------------------------------------


def test_anthropic_step_shown_when_unresolved(monkeypatch):
    _clear_all(monkeypatch)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, _ = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            assert isinstance(app.screen, AnthropicStepScreen)
            assert app.screen.query_one(f"#{_input_id('ANTHROPIC_API_KEY')}")

    run_async(scenario)


def test_anthropic_step_skipped_when_env_sourced_routes_to_source_picker(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, _ = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            assert isinstance(app.screen, SourceStepScreen)

    run_async(scenario)


def test_anthropic_input_is_masked():
    from textual.widgets import Input

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = AnthropicStepScreen(
                resolved={credentials.ANTHROPIC_KEY: credentials.ResolvedCredential(credentials.ANTHROPIC_KEY, None, None)},
                values={},
                first_run=True,
                on_complete=lambda: None,
                env_path=paths.ENV_PATH,
            )
            app.push_screen(screen)
            await pilot.pause()
            assert app.screen.query_one(f"#{_input_id('ANTHROPIC_API_KEY')}", Input).password is True

    run_async(scenario)


def test_anthropic_step_next_moves_to_source_picker_without_validating(monkeypatch):
    _clear_all(monkeypatch)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, _ = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            _fill(app.screen, {"ANTHROPIC_API_KEY": "sk-ant-unvalidated"})
            app.screen._try_submit()
            await pilot.pause()
            assert isinstance(app.screen, SourceStepScreen)  # no network call made -- see _succeed/_fail helpers unused here

    run_async(scenario)


# --- step 2: source picker ----------------------------------------------------


def test_source_step_lists_both_sources(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")

    async def scenario():
        from textual.widgets import ListView

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, _ = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            menu = app.screen.query_one("#source-list", ListView)
            assert {item.id for item in menu.children} == {"langfuse", "braintrust"}

    run_async(scenario)


def test_picking_braintrust_routes_to_braintrust_fields_step(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, _ = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("down")  # braintrust is the second item
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, FieldsStepScreen)
            assert app.screen.source == "braintrust"
            assert {f.key for f in app.screen.editable_fields} == set(_BRAINTRUST_VALUES)

    run_async(scenario)


def test_picking_langfuse_routes_to_langfuse_fields_step_not_braintrusts(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, _ = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")  # langfuse is the first item
            await pilot.pause()
            assert isinstance(app.screen, FieldsStepScreen)
            assert app.screen.source == "langfuse"
            assert {f.key for f in app.screen.editable_fields} == set(_LANGFUSE_VALUES)

    run_async(scenario)


# --- step 3: fields ------------------------------------------------------------


def test_langfuse_fields_masking(monkeypatch):
    from textual.widgets import Input

    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, _ = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")  # langfuse
            await pilot.pause()
            assert app.screen.query_one(f"#{_input_id('LANGFUSE_SECRET_KEY')}", Input).password is True
            assert app.screen.query_one(f"#{_input_id('LANGFUSE_PUBLIC_KEY')}", Input).password is False
            assert app.screen.query_one(f"#{_input_id('LANGFUSE_BASE_URL')}", Input).password is False
            assert app.screen.query_one(f"#{_input_id('LANGFUSE_PROJECT_ID')}", Input).password is False

    run_async(scenario)


def test_braintrust_fields_masking(monkeypatch):
    from textual.widgets import Input

    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, _ = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("enter")  # braintrust
            await pilot.pause()
            assert app.screen.query_one(f"#{_input_id('BRAINTRUST_API_KEY')}", Input).password is True
            assert app.screen.query_one(f"#{_input_id('BRAINTRUST_PROJECT_NAME')}", Input).password is False

    run_async(scenario)


def test_langfuse_success_writes_file_and_calls_on_complete(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")
    _succeed_langfuse(monkeypatch)
    env_path = paths.ENV_PATH

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")  # langfuse
            await pilot.pause()
            _fill(app.screen, _LANGFUSE_VALUES)
            app.screen._try_submit()
            await _wait_until(pilot, lambda: bool(completed))
            assert completed == [True]

    run_async(scenario)

    content = env_path.read_text(encoding="utf-8")
    assert "TRACE_SOURCE=langfuse" in content
    assert "LANGFUSE_SECRET_KEY=sk-lf-test" in content
    assert "ANTHROPIC_API_KEY" not in content  # env-sourced this run -- never written


def test_braintrust_success_writes_file_and_calls_on_complete(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")
    _succeed_braintrust(monkeypatch)
    env_path = paths.ENV_PATH

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("enter")  # braintrust
            await pilot.pause()
            _fill(app.screen, _BRAINTRUST_VALUES)
            app.screen._try_submit()
            await _wait_until(pilot, lambda: bool(completed))
            assert completed == [True]

    run_async(scenario)

    content = env_path.read_text(encoding="utf-8")
    assert "TRACE_SOURCE=braintrust" in content
    assert "BRAINTRUST_API_KEY=bt-key-test" in content
    assert "LANGFUSE_SECRET_KEY" not in content


def test_anthropic_validated_at_final_step_not_at_step_one(monkeypatch):
    # step 1 never calls validate_anthropic -- only the final submit does,
    # together with the source's own validation, so one failure surfaces
    # everything at once rather than dribbling out across steps
    _clear_all(monkeypatch)
    _fail_anthropic(monkeypatch)
    _succeed_langfuse(monkeypatch)
    env_path = paths.ENV_PATH

    async def scenario():
        from textual.widgets import Label

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            _fill(app.screen, {"ANTHROPIC_API_KEY": "sk-ant-bad"})
            app.screen._try_submit()
            await pilot.pause()
            assert isinstance(app.screen, SourceStepScreen)  # moved on freely, not validated yet
            await pilot.press("enter")  # langfuse
            await pilot.pause()
            _fill(app.screen, _LANGFUSE_VALUES)
            app.screen._try_submit()
            await _wait_until(pilot, lambda: "Anthropic" in str(app.screen.query_one("#cred-error", Label).render()))
            assert completed == []

    run_async(scenario)

    assert not env_path.exists()


def test_source_validation_failure_does_not_write_and_allows_retry(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")
    _fail_langfuse(monkeypatch)
    env_path = paths.ENV_PATH

    async def scenario():
        from textual.widgets import Label

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")  # langfuse
            await pilot.pause()
            fields_screen = app.screen
            _fill(fields_screen, _LANGFUSE_VALUES)
            fields_screen._try_submit()
            await _wait_until(pilot, lambda: "Langfuse" in str(fields_screen.query_one("#cred-error", Label).render()))
            assert completed == []
            assert not env_path.exists()
            assert app.screen is fields_screen  # retry: still here, values intact
            assert fields_screen.query_one(f"#{_input_id('LANGFUSE_SECRET_KEY')}").value == "sk-lf-test"

    run_async(scenario)

    assert not env_path.exists()


def test_braintrust_project_name_mismatch_fails_clearly(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")
    _succeed_braintrust(monkeypatch, project_names=("some-other-project",))
    env_path = paths.ENV_PATH

    async def scenario():
        from textual.widgets import Label

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("enter")  # braintrust
            await pilot.pause()
            _fill(app.screen, _BRAINTRUST_VALUES)  # BRAINTRUST_PROJECT_NAME="proj-test", not in project_names above
            app.screen._try_submit()
            await _wait_until(pilot, lambda: "project" in str(app.screen.query_one("#cred-error", Label).render()).lower())
            assert completed == []

    run_async(scenario)

    assert not env_path.exists()


def test_empty_required_field_blocks_submit(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")
    _succeed_langfuse(monkeypatch)
    env_path = paths.ENV_PATH

    async def scenario():
        from textual.widgets import Label

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")  # langfuse
            await pilot.pause()
            values = dict(_LANGFUSE_VALUES)
            values["LANGFUSE_BASE_URL"] = ""
            _fill(app.screen, values)
            app.screen._try_submit()
            await pilot.pause()
            assert "required" in str(app.screen.query_one("#cred-error", Label).render())
            assert completed == []

    run_async(scenario)

    assert not env_path.exists()


def test_no_credential_value_in_terminal_output_during_failed_validation(monkeypatch, capsys):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")
    _fail_langfuse(monkeypatch)
    secret_marker = "sk-lf-leak-marker-should-never-print"
    values = dict(_LANGFUSE_VALUES)
    values["LANGFUSE_SECRET_KEY"] = secret_marker

    async def scenario():
        from textual.widgets import Label

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _entry(first_run=True)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")  # langfuse
            await pilot.pause()
            _fill(app.screen, values)
            fields_screen = app.screen
            fields_screen._try_submit()
            await _wait_until(pilot, lambda: "Langfuse" in str(fields_screen.query_one("#cred-error", Label).render()))

    run_async(scenario)

    out, err = capsys.readouterr()
    assert secret_marker not in out
    assert secret_marker not in err
    assert not paths.ENV_PATH.exists()


# --- switching sources ---------------------------------------------------------


def test_switching_from_langfuse_to_braintrust_drops_langfuse_fields_from_the_file(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-env")
    credentials.write_credentials(
        {"TRACE_SOURCE": "langfuse", **_LANGFUSE_VALUES},
        env_path=paths.ENV_PATH,
    )
    _succeed_braintrust(monkeypatch)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _entry(first_run=False)
            app.push_screen(screen)
            await pilot.pause()
            assert isinstance(app.screen, SourceStepScreen)  # file-sourced, so reachable for switching
            await pilot.press("down")
            await pilot.press("enter")  # switch to braintrust
            await pilot.pause()
            assert isinstance(app.screen, FieldsStepScreen)
            assert app.screen.source == "braintrust"
            _fill(app.screen, _BRAINTRUST_VALUES)
            app.screen._try_submit()
            await _wait_until(pilot, lambda: bool(completed))

    run_async(scenario)

    content = paths.ENV_PATH.read_text(encoding="utf-8")
    assert "TRACE_SOURCE=braintrust" in content
    assert "BRAINTRUST_API_KEY=bt-key-test" in content
    for field in credentials.LANGFUSE_FIELDS:
        assert field.key not in content


# --- genuine end-to-end: real keypresses, real screen transition ------------


def test_end_to_end_typing_through_all_three_steps_reaches_add_environment(monkeypatch):
    """Drives the full flow the way a real user does -- focuses each
    Input, presses individual character keys, navigates the source
    picker with real key presses, presses the documented ctrl+s binding
    -- and asserts the app lands on AddEnvironmentScreen via HarnessApp's
    real _credentials_complete routing."""
    _clear_all(monkeypatch)
    _succeed_anthropic(monkeypatch)
    _succeed_langfuse(monkeypatch, project_ids=("proj-e2e",))
    typed_anthropic = "sk-ant-e2e"
    typed_langfuse = {
        "LANGFUSE_PUBLIC_KEY": "pk-lf-e2e",
        "LANGFUSE_SECRET_KEY": "sk-lf-e2e",
        "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
        "LANGFUSE_PROJECT_ID": "proj-e2e",
    }

    async def scenario():
        from textual.widgets import Input

        from tui.screens.add_environment import AddEnvironmentScreen

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = credentials_flow_entry_screen(first_run=True, on_complete=app._credentials_complete)
            app.push_screen(screen)
            await pilot.pause()

            assert isinstance(app.screen, AnthropicStepScreen)
            app.screen.query_one(f"#{_input_id('ANTHROPIC_API_KEY')}", Input).focus()
            await pilot.pause()
            await pilot.press(*list(typed_anthropic))
            await pilot.press("ctrl+s")
            await pilot.pause()

            assert isinstance(app.screen, SourceStepScreen)
            await pilot.press("enter")  # langfuse (first item)
            await pilot.pause()

            assert isinstance(app.screen, FieldsStepScreen)
            for key, value in typed_langfuse.items():
                app.screen.query_one(f"#{_input_id(key)}", Input).focus()
                await pilot.pause()
                await pilot.press(*list(value))
            await pilot.press("ctrl+s")
            await _wait_until(pilot, lambda: isinstance(app.screen, AddEnvironmentScreen))

    run_async(scenario)

    content = paths.ENV_PATH.read_text(encoding="utf-8")
    assert "TRACE_SOURCE=langfuse" in content
    assert f"ANTHROPIC_API_KEY={typed_anthropic}" in content
    for key, value in typed_langfuse.items():
        assert f"{key}={value}" in content, f"{key} did not end up with its typed value in the written file"
