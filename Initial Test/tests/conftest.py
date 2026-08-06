"""Shared pytest fixtures.

Every TUI test that constructs HarnessApp() directly expects it to boot
straight to HomeScreen, same as before the credentials gate existed (see
tui/app.py's on_mount) -- none of them are testing the gate itself and
none should need real, valid credentials just to run. This autouse
fixture sets dummy real env vars for all required keys so
config.credentials.missing_keys() naturally resolves to [] by default —
deliberately *not* monkeypatching missing_keys/resolve_all themselves,
since tests/test_config_credentials.py exercises those functions directly
and needs their real behavior. Tests that need a "something's missing"
state (tests/test_tui_app.py's gate tests, tests/test_config_credentials.py)
just monkeypatch.delenv the relevant key(s) themselves, which layers
cleanly on top since it's the same monkeypatch fixture instance.
"""

from __future__ import annotations

import pytest

from config.credentials import REQUIRED_KEYS


@pytest.fixture(autouse=True)
def _credentials_resolved_by_default(monkeypatch):
    for key in REQUIRED_KEYS:
        monkeypatch.setenv(key, f"test-{key}")
