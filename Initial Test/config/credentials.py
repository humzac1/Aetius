"""Resolves, validates, and persists the five credentials Caligula needs
(ANTHROPIC_API_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY,
LANGFUSE_BASE_URL, LANGFUSE_PROJECT_ID -- the last one is what
ingestion/langfuse_client.py's default_project_id() reads to know which
Langfuse project to pull traces from; it isn't used for authentication
itself, but "Add environment" hard-fails without it, so it's required
upfront same as the other four). Real environment variables always win -- resolve_all
checks os.environ before ever looking at the config file, so a shell
`export` (dev workflow, CI) is never shadowed by a stale file and never
forces the interactive credentials screen. dotenv_values() reads the file
without mutating os.environ, so "resolve for display" (tui/screens/
credentials.py) and "load into the environment for the rest of the app to
read" (ensure_env_loaded, called once at startup) are two separate,
deliberate operations.

Never logs or prints a credential value -- validate_anthropic and
validate_langfuse below return messages built only from static strings and
exception *type* names, never str(exc) or the value itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv

from config.paths import ENV_PATH

Source = Literal["env", "file", None]

REQUIRED_KEYS = (
    "ANTHROPIC_API_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_PROJECT_ID",
)

LANGFUSE_KEYS = ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_PROJECT_ID")


@dataclass(frozen=True)
class CredentialField:
    key: str
    label: str
    hint: str
    secret: bool  # whether the credentials screen should mask input for this key


CREDENTIAL_FIELDS = (
    CredentialField(
        "ANTHROPIC_API_KEY",
        "Anthropic API key",
        "Runs the real Claude model during attack execution. Create one at console.anthropic.com -> API Keys.",
        secret=True,
    ),
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


@dataclass(frozen=True)
class ResolvedCredential:
    key: str
    value: str | None
    source: Source


def resolve_all(*, env_path: Path = ENV_PATH) -> dict[str, ResolvedCredential]:
    """Real environment variables first (never shadowed by the file);
    falls back to the config file; None/None if resolved by neither. Pure
    -- never mutates os.environ, so calling this to render the credentials
    screen has no side effect on what the rest of the app sees until
    ensure_env_loaded() is actually called."""
    file_values = dotenv_values(env_path) if env_path.exists() else {}
    resolved: dict[str, ResolvedCredential] = {}
    for key in REQUIRED_KEYS:
        env_value = os.environ.get(key)
        if env_value:
            resolved[key] = ResolvedCredential(key, env_value, "env")
        elif file_values.get(key):
            resolved[key] = ResolvedCredential(key, file_values[key], "file")
        else:
            resolved[key] = ResolvedCredential(key, None, None)
    return resolved


def missing_keys(*, env_path: Path = ENV_PATH) -> list[str]:
    return [key for key, cred in resolve_all(env_path=env_path).items() if cred.value is None]


def ensure_env_loaded(*, env_path: Path = ENV_PATH) -> None:
    """Merges file-sourced keys into os.environ without overriding
    anything already set there, so build_client()/build_anthropic_client()
    (and anything else reading os.environ directly) see the same
    env-first precedence resolve_all() computes. Call once at real
    startup, not from resolve_all() itself, since resolve_all() is used
    for display and must stay side-effect-free."""
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


def write_credentials(values: dict[str, str], *, env_path: Path = ENV_PATH) -> None:
    """Writes exactly the given keys to the config file (creating its
    parent dir if needed), each on its own KEY=value line. Callers only
    ever pass this the keys that actually need to live in the file --
    env-sourced keys are never written here, see tui/screens/credentials.py's
    _submit(). Overwrites the whole file rather than merging, since the
    caller always has the complete resolved set of file-bound keys at
    write time."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in values.items()]
    env_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    try:
        env_path.chmod(0o600)
    except OSError:
        pass  # best-effort; not all platforms (e.g. Windows) support POSIX file modes
