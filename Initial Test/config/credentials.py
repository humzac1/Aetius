"""Resolves, validates, and persists Caligula's credentials: an always-
required ANTHROPIC_API_KEY, plus TRACE_SOURCE ("langfuse" or "braintrust")
and whichever fields *that* source needs. Real environment variables
always win -- resolve_all/resolve_one check os.environ before ever
looking at the config file, so a shell `export` (dev workflow, CI) is
never shadowed by a stale file and never forces the interactive
credentials screen. dotenv_values() reads the file without mutating
os.environ, so "resolve for display" (tui/screens/credentials.py) and
"load into the environment for the rest of the app to read"
(ensure_env_loaded, called once at startup) are two separate, deliberate
operations.

One active trace source at a time, by design, not a limitation worked
around here -- see tui/screens/credentials.py's module docstring for the
reasoning (nothing downstream can use two configured sources yet).
Switching sources overwrites the previous source's fields out of the
config file entirely (write_credentials replaces the whole file with
exactly what's currently needed) -- not a bug, the intended effect of
"switch," so the file never carries a stale source's fields around.

Never logs or prints a credential value -- validate_anthropic,
validate_langfuse, and validate_braintrust below return messages built
only from static strings and exception *type* names, never str(exc) or
the value itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv

from config import paths

Source = Literal["env", "file", None]

ANTHROPIC_KEY = "ANTHROPIC_API_KEY"
TRACE_SOURCE_KEY = "TRACE_SOURCE"


@dataclass(frozen=True)
class CredentialField:
    key: str
    label: str
    hint: str
    secret: bool  # whether the credentials screen should mask input for this key


ANTHROPIC_FIELD = CredentialField(
    ANTHROPIC_KEY,
    "Anthropic API key",
    "Runs the real Claude model during attack execution. Create one at console.anthropic.com -> API Keys.",
    secret=True,
)

LANGFUSE_FIELDS = (
    CredentialField(
        "LANGFUSE_PUBLIC_KEY",
        "Langfuse public key",
        "Identifies your Langfuse project when pulling traces. Your Langfuse project -> Settings -> API Keys.",
        secret=False,
    ),
    CredentialField(
        "LANGFUSE_SECRET_KEY",
        "Langfuse secret key",
        "Authenticates trace pulls from your Langfuse project. Same page as the public key -- keep this one private.",
        secret=True,
    ),
    CredentialField(
        "LANGFUSE_BASE_URL",
        "Langfuse base URL",
        "The API host for your Langfuse project, e.g. https://cloud.langfuse.com (or your self-hosted URL).",
        secret=False,
    ),
    CredentialField(
        "LANGFUSE_PROJECT_ID",
        "Langfuse project ID",
        "Which Langfuse project to pull traces from. Your Langfuse project -> Settings -> General (not secret, but must match a project your key can access).",
        secret=False,
    ),
)

BRAINTRUST_FIELDS = (
    CredentialField(
        "BRAINTRUST_API_KEY",
        "Braintrust API key",
        "Runs trace pulls against your Braintrust project. Create one at braintrust.dev -> Settings -> API keys.",
        secret=True,
    ),
    CredentialField(
        "BRAINTRUST_PROJECT_NAME",
        "Braintrust project name",
        "Which Braintrust project to pull traces from, exactly as shown in the Braintrust UI (not secret, but must match a project your key can access).",
        secret=False,
    ),
)

TRACE_SOURCES: dict[str, tuple[CredentialField, ...]] = {
    "langfuse": LANGFUSE_FIELDS,
    "braintrust": BRAINTRUST_FIELDS,
}

TRACE_SOURCE_LABELS = {"langfuse": "Langfuse", "braintrust": "Braintrust"}


@dataclass(frozen=True)
class ResolvedCredential:
    key: str
    value: str | None
    source: Source


def resolve_one(key: str, *, env_path: Path | None = None) -> ResolvedCredential:
    """Env-first-then-file-then-missing resolution for a single key. The
    building block resolve_all/resolve_source/resolve_source_fields below
    all share -- pure, never mutates os.environ.

    env_path defaults to paths.ENV_PATH resolved *here*, at call time, not
    baked into the signature as `env_path: Path = paths.ENV_PATH` -- that
    would bind the value once at import time, before any test (or
    anything else) could monkeypatch config.paths.ENV_PATH to point
    somewhere else. Every other function below follows the same pattern
    for the same reason -- confirmed the hard way: without it, this
    suite's tests were silently reading whatever the real user happened
    to have at the real config location, e.g. from a previous manual run,
    rather than the tmp_path each test actually meant to use."""
    if env_path is None:
        env_path = paths.ENV_PATH
    env_value = os.environ.get(key)
    if env_value:
        return ResolvedCredential(key, env_value, "env")
    file_values = dotenv_values(env_path) if env_path.exists() else {}
    file_value = file_values.get(key)
    if file_value:
        return ResolvedCredential(key, file_value, "file")
    return ResolvedCredential(key, None, None)


def resolve_source(*, env_path: Path | None = None) -> ResolvedCredential:
    """TRACE_SOURCE resolved the same env-first-then-file discipline as
    any other key, but only accepted if it's one of TRACE_SOURCES -- a
    corrupted/unrecognized value on disk is treated as unresolved rather
    than silently trusted."""
    resolved = resolve_one(TRACE_SOURCE_KEY, env_path=env_path)
    if resolved.value not in TRACE_SOURCES:
        return ResolvedCredential(TRACE_SOURCE_KEY, None, None)
    return resolved


def resolve_source_fields(source: str, *, env_path: Path | None = None) -> dict[str, ResolvedCredential]:
    """Resolves one source's fields regardless of what TRACE_SOURCE
    currently says on disk -- used when the user is actively picking/
    switching to `source`, not necessarily the one already configured."""
    return {field.key: resolve_one(field.key, env_path=env_path) for field in TRACE_SOURCES[source]}


def resolve_all(*, env_path: Path | None = None) -> dict[str, ResolvedCredential]:
    """ANTHROPIC_API_KEY and TRACE_SOURCE are always resolved; the
    currently-selected source's own fields are resolved too once/if
    TRACE_SOURCE itself resolves to a recognized value -- there's no way
    to know which fields matter before that. Pure, side-effect-free."""
    resolved: dict[str, ResolvedCredential] = {ANTHROPIC_KEY: resolve_one(ANTHROPIC_KEY, env_path=env_path)}
    source_cred = resolve_source(env_path=env_path)
    resolved[TRACE_SOURCE_KEY] = source_cred
    if source_cred.value in TRACE_SOURCES:
        resolved.update(resolve_source_fields(source_cred.value, env_path=env_path))
    return resolved


def missing_keys(*, env_path: Path | None = None) -> list[str]:
    return [key for key, cred in resolve_all(env_path=env_path).items() if cred.value is None]


def ensure_env_loaded(*, env_path: Path | None = None) -> None:
    """Merges file-sourced keys into os.environ without overriding
    anything already set there, so build_client()/build_anthropic_client()
    (and anything else reading os.environ directly) see the same
    env-first precedence resolve_all() computes. Call once at real
    startup, not from resolve_all() itself, since resolve_all() is used
    for display and must stay side-effect-free."""
    if env_path is None:
        env_path = paths.ENV_PATH
    if env_path.exists():
        load_dotenv(env_path, override=False)


def validate_anthropic(api_key: str) -> tuple[bool, str]:
    """Lightest real validation available: models.list(limit=1) hits a
    real endpoint with the given key but costs nothing and generates no
    completion. Never includes str(exc) in the returned message -- only
    the exception type name, which cannot contain the key."""
    import anthropic

    try:
        anthropic.Anthropic(api_key=api_key).models.list(limit=1)
        return True, ""
    except anthropic.AuthenticationError:
        return False, "Anthropic rejected this API key (authentication failed)."
    except anthropic.APIConnectionError:
        return False, "Could not reach the Anthropic API (network/connection error)."
    except Exception as exc:  # noqa: BLE001 - surfaced as a type name only, see module docstring
        return False, f"Anthropic validation failed ({type(exc).__name__})."


def validate_langfuse(*, secret_key: str, public_key: str, base_url: str, project_id: str) -> tuple[bool, str]:
    """Reuses ingestion/langfuse_client.py's build_client (the same
    connection logic Add Environment uses) rather than constructing a
    second Langfuse client here. Calls client.api.projects.get() directly
    rather than the SDK's own auth_check() (which calls the exact same
    endpoint internally but discards the result) so project_id can be
    checked against the projects this key can actually see in the same
    round-trip -- project_id isn't part of authentication itself (it's
    never passed to the Langfuse client), so this is the only way to give
    it real validation rather than just checking it's a non-empty string.
    Never includes str(exc) in the returned message, same discipline as
    validate_anthropic above."""
    from ingestion.langfuse_client import build_client

    try:
        client = build_client(secret_key=secret_key, public_key=public_key, host=base_url, timeout=15.0)
        projects = client.api.projects.get()
    except Exception as exc:  # noqa: BLE001 - surfaced as a type name only, see module docstring
        return False, f"Langfuse validation failed ({type(exc).__name__})."
    if not projects.data:
        return False, "Langfuse rejected these credentials (authentication failed)."
    if project_id not in {project.id for project in projects.data}:
        return False, "Langfuse project ID doesn't match any project this API key can access."
    return True, ""


def validate_braintrust(*, api_key: str, project_name: str) -> tuple[bool, str]:
    """Reuses ingestion/braintrust_client.py's build_client (the same
    connection logic Add Environment will use), same shape as
    validate_langfuse: one real round-trip that both authenticates and
    checks project_name against the projects this key can actually see
    (confirmed live: GET v1/project returns {"objects": [...]},
    project_name matched by Braintrust's own human-readable "name" field,
    not an opaque id -- see ingestion/braintrust_client.py's module
    docstring for the full investigation). Never includes str(exc) in the
    returned message, same discipline as validate_anthropic/
    validate_langfuse above."""
    from ingestion.braintrust_client import build_client

    try:
        state = build_client(api_key=api_key)
        resp = state.api_conn().get("v1/project")
        resp.raise_for_status()
        projects = resp.json().get("objects", [])
    except Exception as exc:  # noqa: BLE001 - surfaced as a type name only, see module docstring
        return False, f"Braintrust validation failed ({type(exc).__name__})."
    if not projects:
        return False, "Braintrust rejected these credentials (authentication failed)."
    if project_name not in {project["name"] for project in projects}:
        return False, "Braintrust project name doesn't match any project this API key can access."
    return True, ""


def write_credentials(values: dict[str, str], *, env_path: Path | None = None) -> None:
    """Writes exactly the given keys to the config file (creating its
    parent dir if needed), each on its own KEY=value line. Callers only
    ever pass this the keys that actually need to live in the file --
    env-sourced keys are never written here, see tui/screens/credentials.py.
    Overwrites the whole file rather than merging, since the caller always
    has the complete resolved set of file-bound keys at write time --
    this is also what makes switching trace sources work correctly: a
    previous source's fields are simply never included in `values` after
    a switch, so they drop out of the file rather than lingering."""
    if env_path is None:
        env_path = paths.ENV_PATH
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in values.items()]
    env_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    try:
        env_path.chmod(0o600)
    except OSError:
        pass  # best-effort; not all platforms (e.g. Windows) support POSIX file modes
