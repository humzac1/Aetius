"""Headless Textual Pilot tests for CredentialsScreen — the first-run gate
and the Settings "Edit credentials" flow share this exact screen (see its
module docstring). Covers: field display (env-sourced fields aren't
prompted for), masked secret input, validation failure with retry
(nothing written), write-only-on-success, and that no credential value
ever appears in captured terminal output during a deliberately failed
validation.
"""

from __future__ import annotations

import anthropic
import httpx

import config.credentials as credentials
import ingestion.langfuse_client as lf
from tests.tui_test_support import run_async
from tests.tui_test_support import wait_until as _wait_until
from tui.app import HarnessApp
from tui.screens.credentials import CredentialsScreen, _input_id


def _push(app, **kwargs):
    on_complete_calls = []
    screen = CredentialsScreen(on_complete=lambda: on_complete_calls.append(True), **kwargs)
    app.push_screen(screen)
    return screen, on_complete_calls


def _succeed_anthropic(monkeypatch):
    class _FakeModels:
        def list(self, limit=None):
            return ["ok"]

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)


class _FakeProject:
    def __init__(self, id):
        self.id = id


class _FakeProjectsResponse:
    def __init__(self, data):
        self.data = data


def _succeed_langfuse(monkeypatch, *, project_ids=("proj-test",)):
    class _FakeProjectsClient:
        def get(self):
            return _FakeProjectsResponse([_FakeProject(pid) for pid in project_ids])

    class _FakeApi:
        projects = _FakeProjectsClient()

    class _FakeLfClient:
        api = _FakeApi()

    monkeypatch.setattr(lf, "build_client", lambda **kwargs: _FakeLfClient())


def _fail_anthropic(monkeypatch):
    fake_response = httpx.Response(401, request=httpx.Request("GET", "https://api.anthropic.com/v1/models"))

    class _FakeModels:
        def list(self, limit=None):
            raise anthropic.AuthenticationError("bad key", response=fake_response, body=None)

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)


def _fail_langfuse(monkeypatch):
    class _FakeProjectsClient:
        def get(self):
            return _FakeProjectsResponse([])  # no accessible projects -> auth rejected

    class _FakeApi:
        projects = _FakeProjectsClient()

    class _FakeLfClient:
        api = _FakeApi()

    monkeypatch.setattr(lf, "build_client", lambda **kwargs: _FakeLfClient())


# --- field display -----------------------------------------------------------


def test_shows_all_five_fields_with_descriptions_when_nothing_resolved(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            _push(app, first_run=True, env_path=tmp_path / ".env")
            await pilot.pause()
            assert len(credentials.CREDENTIAL_FIELDS) == 5
            for field in credentials.CREDENTIAL_FIELDS:
                assert app.screen.query_one(f"#{_input_id(field.key)}")
            # regression: LANGFUSE_PROJECT_ID was originally missing from
            # this screen entirely even though "Add environment" hard-fails
            # without it (ingestion/langfuse_client.py's default_project_id)
            assert app.screen.query_one(f"#{_input_id('LANGFUSE_PROJECT_ID')}")

    run_async(scenario)


def test_env_sourced_field_is_not_prompted_for(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-real-env")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            _push(app, first_run=True, env_path=tmp_path / ".env")
            await pilot.pause()
            assert not app.screen.query(f"#{_input_id('ANTHROPIC_API_KEY')}")
            labels = " ".join(str(w.render()) for w in app.screen.query("Label"))
            assert "ANTHROPIC_API_KEY" in labels and "environment" in labels

    run_async(scenario)


def test_secret_fields_are_masked_non_secret_fields_are_not(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)

    async def scenario():
        from textual.widgets import Input

        app = HarnessApp()
        async with app.run_test() as pilot:
            _push(app, first_run=True, env_path=tmp_path / ".env")
            await pilot.pause()
            assert app.screen.query_one(f"#{_input_id('ANTHROPIC_API_KEY')}", Input).password is True
            assert app.screen.query_one(f"#{_input_id('LANGFUSE_SECRET_KEY')}", Input).password is True
            assert app.screen.query_one(f"#{_input_id('LANGFUSE_PUBLIC_KEY')}", Input).password is False
            assert app.screen.query_one(f"#{_input_id('LANGFUSE_BASE_URL')}", Input).password is False

    run_async(scenario)


# --- validation / write discipline -------------------------------------------


def _fill_all_fields(screen, values: dict[str, str]) -> None:
    from textual.widgets import Input

    for key, value in values.items():
        try:
            screen.query_one(f"#{_input_id(key)}", Input).value = value
        except Exception:
            pass  # env-sourced field with no Input — nothing to fill


_ALL_VALUES = {
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
    "LANGFUSE_PROJECT_ID": "proj-test",
}


def test_successful_validation_writes_file_and_calls_on_complete(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    _succeed_anthropic(monkeypatch)
    _succeed_langfuse(monkeypatch)
    env_path = tmp_path / ".env"

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _push(app, first_run=True, env_path=env_path)
            await pilot.pause()
            _fill_all_fields(screen, _ALL_VALUES)
            screen._submit()
            await _wait_until(pilot, lambda: completed or app.screen is not screen)
            assert completed == [True]

    run_async(scenario)

    content = env_path.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=sk-ant-test" in content
    assert "LANGFUSE_BASE_URL=https://cloud.langfuse.com" in content


def test_anthropic_failure_does_not_write_and_allows_retry(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    _fail_anthropic(monkeypatch)
    _succeed_langfuse(monkeypatch)
    env_path = tmp_path / ".env"

    async def scenario():
        from textual.widgets import Label

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _push(app, first_run=True, env_path=env_path)
            await pilot.pause()
            _fill_all_fields(screen, _ALL_VALUES)
            screen._submit()
            await _wait_until(pilot, lambda: "Anthropic" in str(screen.query_one("#cred-error", Label).render()))
            assert completed == []
            assert not env_path.exists()
            # retry: still on the same screen, inputs still there to fix and resubmit
            assert app.screen is screen
            assert screen.query_one(f"#{_input_id('ANTHROPIC_API_KEY')}").value == "sk-ant-test"

    run_async(scenario)

    assert not env_path.exists()


def test_langfuse_failure_does_not_write_and_allows_retry(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    _succeed_anthropic(monkeypatch)
    _fail_langfuse(monkeypatch)
    env_path = tmp_path / ".env"

    async def scenario():
        from textual.widgets import Label

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _push(app, first_run=True, env_path=env_path)
            await pilot.pause()
            _fill_all_fields(screen, _ALL_VALUES)
            screen._submit()
            await _wait_until(pilot, lambda: "Langfuse" in str(screen.query_one("#cred-error", Label).render()))
            assert completed == []
            assert not env_path.exists()

    run_async(scenario)

    assert not env_path.exists()


def test_langfuse_project_id_mismatch_fails_clearly_and_does_not_write(tmp_path, monkeypatch):
    # regression: the auth keys can be perfectly valid while
    # LANGFUSE_PROJECT_ID names a project this key can't see (typo, wrong
    # project) — that must fail validation on its own, distinct from an
    # outright auth failure, not silently pass through as an unchecked field.
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    _succeed_anthropic(monkeypatch)
    _succeed_langfuse(monkeypatch, project_ids=("some-other-project",))
    env_path = tmp_path / ".env"

    async def scenario():
        from textual.widgets import Label

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _push(app, first_run=True, env_path=env_path)
            await pilot.pause()
            _fill_all_fields(screen, _ALL_VALUES)  # LANGFUSE_PROJECT_ID="proj-test", not in project_ids above
            screen._submit()
            await _wait_until(pilot, lambda: "project" in str(screen.query_one("#cred-error", Label).render()).lower())
            assert completed == []
            assert not env_path.exists()

    run_async(scenario)

    assert not env_path.exists()


def test_empty_required_field_blocks_submit_no_partial_write(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    _succeed_anthropic(monkeypatch)
    _succeed_langfuse(monkeypatch)
    env_path = tmp_path / ".env"
    values = dict(_ALL_VALUES)
    values["LANGFUSE_BASE_URL"] = ""  # left blank

    async def scenario():
        from textual.widgets import Label

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _push(app, first_run=True, env_path=env_path)
            await pilot.pause()
            _fill_all_fields(screen, values)
            screen._submit()
            await pilot.pause()
            assert "required" in str(screen.query_one("#cred-error", Label).render())
            assert completed == []
            assert not env_path.exists()

    run_async(scenario)

    assert not env_path.exists()


def test_no_credential_value_in_terminal_output_during_failed_validation(tmp_path, monkeypatch, capsys):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    _fail_anthropic(monkeypatch)
    _succeed_langfuse(monkeypatch)
    env_path = tmp_path / ".env"
    secret_marker = "sk-ant-leak-marker-should-never-print"
    values = dict(_ALL_VALUES)
    values["ANTHROPIC_API_KEY"] = secret_marker

    async def scenario():
        from textual.widgets import Label

        app = HarnessApp()
        async with app.run_test() as pilot:
            screen, completed = _push(app, first_run=True, env_path=env_path)
            await pilot.pause()
            _fill_all_fields(screen, values)
            screen._submit()
            await _wait_until(pilot, lambda: "Anthropic" in str(screen.query_one("#cred-error", Label).render()))

    run_async(scenario)

    out, err = capsys.readouterr()
    assert secret_marker not in out
    assert secret_marker not in err
    assert not env_path.exists()


# --- genuine end-to-end: real keypresses, real screen transition ------------


def test_end_to_end_typing_all_five_fields_and_pressing_ctrl_s_reaches_add_environment(tmp_path, monkeypatch):
    """Drives the screen the way a real user does -- focuses each Input,
    presses individual character keys (not screen.query_one(...).value =
    "..."), presses the documented ctrl+s binding -- and asserts the app
    actually lands on AddEnvironmentScreen via HarnessApp's real
    _credentials_complete routing, not a hand-rolled substitute. This is
    the test that would have caught bug 3 (no way to submit) even if the
    Save button had never existed at all, since it never touches the
    button or _submit() directly."""
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    _succeed_anthropic(monkeypatch)
    _succeed_langfuse(monkeypatch, project_ids=("proj-e2e",))
    env_path = tmp_path / ".env"

    typed_values = {
        "ANTHROPIC_API_KEY": "sk-ant-e2e",
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
            screen = CredentialsScreen(first_run=True, on_complete=app._credentials_complete, env_path=env_path)
            app.push_screen(screen)
            await pilot.pause()

            for key, value in typed_values.items():
                screen.query_one(f"#{_input_id(key)}", Input).focus()
                await pilot.pause()
                await pilot.press(*list(value))

            await pilot.press("ctrl+s")
            await _wait_until(pilot, lambda: isinstance(app.screen, AddEnvironmentScreen))

    run_async(scenario)

    # the values that actually got typed character-by-character are what
    # landed in the file -- not just that *something* did
    content = env_path.read_text(encoding="utf-8")
    for key, value in typed_values.items():
        assert f"{key}={value}" in content, f"{key} did not end up with its typed value in the written file"
