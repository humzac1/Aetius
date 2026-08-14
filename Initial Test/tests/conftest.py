"""Shared pytest fixtures.

Every TUI test that constructs HarnessApp() directly expects it to boot
straight to HomeScreen, same as before the credentials gate existed (see
tui/app.py's on_mount) -- none of them are testing the gate itself and
none should need real, valid credentials just to run. This autouse
fixture sets dummy real env vars for Anthropic + a fixed trace source
(Langfuse, arbitrarily -- any test that cares which source is active
overrides TRACE_SOURCE itself) so config.credentials.missing_keys()
naturally resolves to [] by default — deliberately *not* monkeypatching
missing_keys/resolve_all themselves, since tests/test_config_credentials.py
exercises those functions directly and needs their real behavior. Tests
that need a "something's missing" state (tests/test_tui_app.py's gate
tests, tests/test_config_credentials.py) just monkeypatch.delenv the
relevant key(s) themselves, which layers cleanly on top since it's the
same monkeypatch fixture instance.
"""

from __future__ import annotations

import pytest

import config.paths as paths
from config.credentials import ANTHROPIC_KEY, LANGFUSE_FIELDS, TRACE_SOURCE_KEY


@pytest.fixture(autouse=True)
def _isolate_real_credentials_file(monkeypatch, tmp_path):
    """No test should ever read or write the real per-user credentials
    file (platformdirs.user_config_dir("aetius")/.env) -- confirmed the
    hard way: config/credentials.py's functions default env_path to
    config.paths.ENV_PATH when not given one explicitly, and a stray real
    file from earlier manual testing this session silently made
    tests/test_tui_app.py's gate tests see "everything already resolved"
    regardless of which env vars a test had cleared, since the file
    fallback still resolved them. Redirecting config.paths.ENV_PATH here
    (every function reads it as `paths.ENV_PATH` at call time, never a
    name copied in at import time, specifically so this monkeypatch can
    reach it) makes every default-env_path call in this suite land on an
    empty, test-local file instead, regardless of what's really on disk."""
    monkeypatch.setattr(paths, "ENV_PATH", tmp_path / "isolated-test.env")


@pytest.fixture(autouse=True)
def _credentials_resolved_by_default(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_KEY, f"test-{ANTHROPIC_KEY}")
    monkeypatch.setenv(TRACE_SOURCE_KEY, "langfuse")
    for field in LANGFUSE_FIELDS:
        monkeypatch.setenv(field.key, f"test-{field.key}")
