"""Unit tests for config/credentials.py — resolution (env-first, then
file, then missing), validation (both services, success and failure), and
write-only-on-success discipline. No live network calls: validate_*'s
underlying client construction is monkeypatched, never the real SDKs.
"""

from __future__ import annotations

import anthropic
import httpx

import config.credentials as credentials
import ingestion.langfuse_client as lf


def _write_env_file(path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8")


# --- resolve_all / missing_keys ---------------------------------------------


def test_missing_keys_lists_everything_when_nothing_resolved(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert set(credentials.missing_keys(env_path=tmp_path / ".env")) == set(credentials.REQUIRED_KEYS)


def test_missing_keys_empty_once_every_key_resolved_via_file(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {k: f"value-{k}" for k in credentials.REQUIRED_KEYS})
    assert credentials.missing_keys(env_path=env_path) == []


def test_missing_keys_reports_only_the_keys_not_resolved_by_either_source(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {"LANGFUSE_SECRET_KEY": "sk-lf", "LANGFUSE_PUBLIC_KEY": "pk-lf"})
    # LANGFUSE_BASE_URL and LANGFUSE_PROJECT_ID resolved by neither env nor file
    assert credentials.missing_keys(env_path=env_path) == ["LANGFUSE_BASE_URL", "LANGFUSE_PROJECT_ID"]


def test_missing_keys_flags_langfuse_project_id_specifically_when_only_it_is_absent(tmp_path, monkeypatch):
    # regression: LANGFUSE_PROJECT_ID is required for "Add environment" to
    # work (ingestion/langfuse_client.py's default_project_id) but was
    # originally missing from the credentials screen entirely — it must be
    # detected as missing on its own, not silently treated as optional.
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_path = tmp_path / ".env"
    other_keys = {k: f"value-{k}" for k in credentials.REQUIRED_KEYS if k != "LANGFUSE_PROJECT_ID"}
    _write_env_file(env_path, other_keys)
    assert credentials.missing_keys(env_path=env_path) == ["LANGFUSE_PROJECT_ID"]


def test_missing_keys_empty_when_all_five_values_present(tmp_path, monkeypatch):
    for key in credentials.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {k: f"value-{k}" for k in credentials.REQUIRED_KEYS})
    assert credentials.missing_keys(env_path=env_path) == []
    assert len(credentials.REQUIRED_KEYS) == 5


# --- env takes priority over the config file --------------------------------


def test_real_env_var_takes_priority_over_config_file_value(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-real-env")
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {"ANTHROPIC_API_KEY": "sk-from-file-should-be-shadowed"})
    resolved = credentials.resolve_all(env_path=env_path)
    assert resolved["ANTHROPIC_API_KEY"].value == "sk-from-real-env"
    assert resolved["ANTHROPIC_API_KEY"].source == "env"


def test_file_value_used_only_when_env_var_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {"ANTHROPIC_API_KEY": "sk-from-file"})
    resolved = credentials.resolve_all(env_path=env_path)
    assert resolved["ANTHROPIC_API_KEY"].value == "sk-from-file"
    assert resolved["ANTHROPIC_API_KEY"].source == "file"


def test_resolve_all_never_mutates_os_environ(tmp_path, monkeypatch):
    import os

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    _write_env_file(env_path, {"ANTHROPIC_API_KEY": "sk-from-file"})
    credentials.resolve_all(env_path=env_path)
    assert "ANTHROPIC_API_KEY" not in os.environ  # resolve_all is read-only; ensure_env_loaded is the side-effecting one


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


# --- write_credentials --------------------------------------------------------


def test_write_credentials_creates_file_with_given_keys(tmp_path):
    env_path = tmp_path / "nested" / ".env"
    credentials.write_credentials({"ANTHROPIC_API_KEY": "sk-abc"}, env_path=env_path)
    assert env_path.exists()
    assert env_path.read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=sk-abc\n"


def test_write_credentials_only_writes_the_given_keys_not_every_required_key(tmp_path):
    env_path = tmp_path / ".env"
    credentials.write_credentials({"LANGFUSE_BASE_URL": "https://x"}, env_path=env_path)
    content = env_path.read_text(encoding="utf-8")
    assert "LANGFUSE_BASE_URL" in content
    assert "ANTHROPIC_API_KEY" not in content
