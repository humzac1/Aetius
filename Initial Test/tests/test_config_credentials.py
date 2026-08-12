"""Unit tests for config/credentials.py — resolution (env-first, then
file, then missing; source-aware once TRACE_SOURCE resolves), validation
(all three services, success and failure), and write-only-on-success
discipline. No live network calls: validate_*'s underlying client
construction is monkeypatched, never the real SDKs.
"""

from __future__ import annotations

import os

import anthropic
import httpx

import config.credentials as credentials
import ingestion.langfuse_client as lf
import ingestion.braintrust_client as bt


def _write_env_file(path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8")


def _clear_all(monkeypatch) -> None:
    monkeypatch.delenv(credentials.ANTHROPIC_KEY, raising=False)
    monkeypatch.delenv(credentials.TRACE_SOURCE_KEY, raising=False)
    for fields in credentials.TRACE_SOURCES.values():
        for field in fields:
            monkeypatch.delenv(field.key, raising=False)


# --- resolve_all / missing_keys ---------------------------------------------


def test_missing_keys_lists_anthropic_and_trace_source_when_nothing_resolved(tmp_path, monkeypatch):
    _clear_all(monkeypatch)
    assert set(credentials.missing_keys(env_path=tmp_path / ".env")) == {credentials.ANTHROPIC_KEY, credentials.TRACE_SOURCE_KEY}


def test_missing_keys_does_not_check_a_sources_fields_until_source_is_resolved(tmp_path, monkeypatch):
    # TRACE_SOURCE unresolved -> we can't know which fields matter yet,
    # so nothing beyond ANTHROPIC_API_KEY/TRACE_SOURCE should be reported
    _clear_all(monkeypatch)
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {credentials.ANTHROPIC_KEY: "sk-ant"})
    assert credentials.missing_keys(env_path=env_path) == [credentials.TRACE_SOURCE_KEY]


def test_missing_keys_checks_langfuse_fields_once_source_is_langfuse(tmp_path, monkeypatch):
    _clear_all(monkeypatch)
    env_path = tmp_path / ".env"
    _write_env_file(
        env_path,
        {
            credentials.ANTHROPIC_KEY: "sk-ant",
            credentials.TRACE_SOURCE_KEY: "langfuse",
            "LANGFUSE_SECRET_KEY": "sk-lf",
            "LANGFUSE_PUBLIC_KEY": "pk-lf",
        },
    )
    # LANGFUSE_BASE_URL and LANGFUSE_PROJECT_ID resolved by neither env nor file
    assert credentials.missing_keys(env_path=env_path) == ["LANGFUSE_BASE_URL", "LANGFUSE_PROJECT_ID"]


def test_missing_keys_checks_braintrust_fields_not_langfuse_when_source_is_braintrust(tmp_path, monkeypatch):
    _clear_all(monkeypatch)
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {credentials.ANTHROPIC_KEY: "sk-ant", credentials.TRACE_SOURCE_KEY: "braintrust"})
    assert set(credentials.missing_keys(env_path=env_path)) == {"BRAINTRUST_API_KEY", "BRAINTRUST_PROJECT_NAME"}
    assert "LANGFUSE_SECRET_KEY" not in credentials.missing_keys(env_path=env_path)


def test_missing_keys_empty_once_source_and_its_fields_all_resolved(tmp_path, monkeypatch):
    _clear_all(monkeypatch)
    env_path = tmp_path / ".env"
    _write_env_file(
        env_path,
        {
            credentials.ANTHROPIC_KEY: "sk-ant",
            credentials.TRACE_SOURCE_KEY: "braintrust",
            "BRAINTRUST_API_KEY": "bt-key",
            "BRAINTRUST_PROJECT_NAME": "my-project",
        },
    )
    assert credentials.missing_keys(env_path=env_path) == []


def test_unrecognized_trace_source_value_treated_as_unresolved(tmp_path, monkeypatch):
    # a corrupted/foreign value on disk shouldn't be silently trusted
    _clear_all(monkeypatch)
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {credentials.ANTHROPIC_KEY: "sk-ant", credentials.TRACE_SOURCE_KEY: "not-a-real-source"})
    resolved = credentials.resolve_source(env_path=env_path)
    assert resolved.value is None
    assert credentials.TRACE_SOURCE_KEY in credentials.missing_keys(env_path=env_path)


# --- resolve_source_fields ----------------------------------------------------


def test_resolve_source_fields_resolves_the_given_source_regardless_of_trace_source_on_disk(tmp_path, monkeypatch):
    # simulates switching: TRACE_SOURCE on disk still says langfuse, but
    # the caller wants to check braintrust's own fields (e.g. the source
    # picker step, after the user just picked a different source)
    _clear_all(monkeypatch)
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {credentials.TRACE_SOURCE_KEY: "langfuse", "BRAINTRUST_API_KEY": "bt-key"})
    resolved = credentials.resolve_source_fields("braintrust", env_path=env_path)
    assert set(resolved.keys()) == {"BRAINTRUST_API_KEY", "BRAINTRUST_PROJECT_NAME"}
    assert resolved["BRAINTRUST_API_KEY"].value == "bt-key"
    assert resolved["BRAINTRUST_PROJECT_NAME"].value is None


# --- env takes priority over the config file --------------------------------


def test_real_env_var_takes_priority_over_config_file_value(tmp_path, monkeypatch):
    monkeypatch.setenv(credentials.ANTHROPIC_KEY, "sk-from-real-env")
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {credentials.ANTHROPIC_KEY: "sk-from-file-should-be-shadowed"})
    resolved = credentials.resolve_all(env_path=env_path)
    assert resolved[credentials.ANTHROPIC_KEY].value == "sk-from-real-env"
    assert resolved[credentials.ANTHROPIC_KEY].source == "env"


def test_file_value_used_only_when_env_var_absent(tmp_path, monkeypatch):
    monkeypatch.delenv(credentials.ANTHROPIC_KEY, raising=False)
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {credentials.ANTHROPIC_KEY: "sk-from-file"})
    resolved = credentials.resolve_all(env_path=env_path)
    assert resolved[credentials.ANTHROPIC_KEY].value == "sk-from-file"
    assert resolved[credentials.ANTHROPIC_KEY].source == "file"


def test_resolve_all_never_mutates_os_environ(tmp_path, monkeypatch):
    monkeypatch.delenv(credentials.ANTHROPIC_KEY, raising=False)
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {credentials.ANTHROPIC_KEY: "sk-from-file"})
    credentials.resolve_all(env_path=env_path)
    assert credentials.ANTHROPIC_KEY not in os.environ  # resolve_all is read-only; ensure_env_loaded is the side-effecting one


# --- validate_anthropic ------------------------------------------------------


def test_validate_anthropic_succeeds_when_models_list_succeeds(monkeypatch):
    class _FakeModels:
        def list(self, limit=None):
            return ["model-a"]

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    ok, message = credentials.validate_anthropic("sk-whatever")
    assert ok is True
    assert message == ""


def test_validate_anthropic_fails_on_authentication_error(monkeypatch):
    fake_response = httpx.Response(401, request=httpx.Request("GET", "https://api.anthropic.com/v1/models"))

    class _FakeModels:
        def list(self, limit=None):
            raise anthropic.AuthenticationError("bad key", response=fake_response, body=None)

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    ok, message = credentials.validate_anthropic("sk-bad")
    assert ok is False
    assert "Anthropic" in message


def test_validate_anthropic_error_message_never_contains_the_key(monkeypatch, capsys):
    secret = "sk-super-secret-leak-marker-12345"

    class _LeakyError(Exception):
        pass

    class _FakeModels:
        def list(self, limit=None):
            raise _LeakyError(f"upstream rejected key={secret}")

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    ok, message = credentials.validate_anthropic(secret)
    assert ok is False
    assert secret not in message
    out, err = capsys.readouterr()
    assert secret not in out
    assert secret not in err


# --- validate_langfuse --------------------------------------------------------


class _FakeProject:
    def __init__(self, id):
        self.id = id


class _FakeProjectsResponse:
    def __init__(self, data):
        self.data = data


class _FakeProjectsClient:
    def __init__(self, project_ids):
        self._project_ids = project_ids

    def get(self):
        return _FakeProjectsResponse([_FakeProject(pid) for pid in self._project_ids])


class _FakeApi:
    def __init__(self, project_ids):
        self.projects = _FakeProjectsClient(project_ids)


class _FakeLangfuseClient:
    def __init__(self, project_ids):
        self.api = _FakeApi(project_ids)


def test_validate_langfuse_succeeds_when_project_id_matches_an_accessible_project(monkeypatch):
    monkeypatch.setattr(lf, "build_client", lambda **kwargs: _FakeLangfuseClient(["proj-1", "proj-2"]))
    ok, message = credentials.validate_langfuse(secret_key="s", public_key="p", base_url="https://x", project_id="proj-2")
    assert ok is True
    assert message == ""


def test_validate_langfuse_fails_when_no_projects_returned(monkeypatch):
    monkeypatch.setattr(lf, "build_client", lambda **kwargs: _FakeLangfuseClient([]))
    ok, message = credentials.validate_langfuse(secret_key="s", public_key="p", base_url="https://x", project_id="proj-1")
    assert ok is False
    assert "Langfuse" in message


def test_validate_langfuse_fails_when_project_id_does_not_match_any_accessible_project(monkeypatch):
    monkeypatch.setattr(lf, "build_client", lambda **kwargs: _FakeLangfuseClient(["proj-1", "proj-2"]))
    ok, message = credentials.validate_langfuse(secret_key="s", public_key="p", base_url="https://x", project_id="proj-does-not-exist")
    assert ok is False
    assert "project" in message.lower()


def test_validate_langfuse_error_message_never_contains_the_key(monkeypatch, capsys):
    secret = "sk-lf-secret-leak-marker-98765"

    class _LeakyError(Exception):
        pass

    def _fake_build_client(**kwargs):
        raise _LeakyError(f"upstream rejected secret_key={secret}")

    monkeypatch.setattr(lf, "build_client", _fake_build_client)
    ok, message = credentials.validate_langfuse(secret_key=secret, public_key="p", base_url="https://x", project_id="proj-1")
    assert ok is False
    assert secret not in message
    out, err = capsys.readouterr()
    assert secret not in out
    assert secret not in err


# --- validate_braintrust ------------------------------------------------------


class _FakeBraintrustResponse:
    def __init__(self, project_names):
        self._payload = {"objects": [{"name": name} for name in project_names]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeBraintrustConn:
    def __init__(self, project_names):
        self._project_names = project_names

    def get(self, path):
        assert path == "v1/project"
        return _FakeBraintrustResponse(self._project_names)


class _FakeBraintrustState:
    def __init__(self, project_names):
        self._conn = _FakeBraintrustConn(project_names)

    def api_conn(self):
        return self._conn


def test_validate_braintrust_succeeds_when_project_name_matches_an_accessible_project(monkeypatch):
    monkeypatch.setattr(bt, "build_client", lambda **kwargs: _FakeBraintrustState(["homepilot", "homepilot-staging"]))
    ok, message = credentials.validate_braintrust(api_key="k", project_name="homepilot")
    assert ok is True
    assert message == ""


def test_validate_braintrust_fails_when_no_projects_returned(monkeypatch):
    monkeypatch.setattr(bt, "build_client", lambda **kwargs: _FakeBraintrustState([]))
    ok, message = credentials.validate_braintrust(api_key="k", project_name="homepilot")
    assert ok is False
    assert "Braintrust" in message


def test_validate_braintrust_fails_when_project_name_does_not_match_any_accessible_project(monkeypatch):
    monkeypatch.setattr(bt, "build_client", lambda **kwargs: _FakeBraintrustState(["homepilot"]))
    ok, message = credentials.validate_braintrust(api_key="k", project_name="does-not-exist")
    assert ok is False
    assert "project" in message.lower()


def test_validate_braintrust_error_message_never_contains_the_key(monkeypatch, capsys):
    secret = "bt-secret-leak-marker-13579"

    class _LeakyError(Exception):
        pass

    def _fake_build_client(**kwargs):
        raise _LeakyError(f"upstream rejected api_key={secret}")

    monkeypatch.setattr(bt, "build_client", _fake_build_client)
    ok, message = credentials.validate_braintrust(api_key=secret, project_name="homepilot")
    assert ok is False
    assert secret not in message
    out, err = capsys.readouterr()
    assert secret not in out
    assert secret not in err


# --- write_credentials --------------------------------------------------------


def test_write_credentials_creates_file_with_given_keys(tmp_path):
    env_path = tmp_path / "nested" / ".env"
    credentials.write_credentials({"ANTHROPIC_API_KEY": "sk-abc"}, env_path=env_path)
    assert env_path.exists()
    assert env_path.read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=sk-abc\n"


def test_write_credentials_only_writes_the_given_keys_not_every_possible_key(tmp_path):
    env_path = tmp_path / ".env"
    credentials.write_credentials({"LANGFUSE_BASE_URL": "https://x"}, env_path=env_path)
    content = env_path.read_text(encoding="utf-8")
    assert "LANGFUSE_BASE_URL" in content
    assert "ANTHROPIC_API_KEY" not in content


def test_write_credentials_overwrite_drops_keys_not_in_the_new_values(tmp_path):
    # the mechanism switching sources relies on (tests/test_tui_credentials.py
    # exercises the actual switch through the screen flow): writing a new
    # set of keys must not leave a previous write's keys lingering.
    env_path = tmp_path / ".env"
    credentials.write_credentials({"LANGFUSE_SECRET_KEY": "sk-lf", "LANGFUSE_BASE_URL": "https://x"}, env_path=env_path)
    credentials.write_credentials({"BRAINTRUST_API_KEY": "bt-key"}, env_path=env_path)
    content = env_path.read_text(encoding="utf-8")
    assert "BRAINTRUST_API_KEY" in content
    assert "LANGFUSE_SECRET_KEY" not in content
    assert "LANGFUSE_BASE_URL" not in content
