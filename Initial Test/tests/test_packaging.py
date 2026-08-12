"""Regression coverage for the `caligula` console-script entry point
itself (pyproject.toml's [project.scripts]) — not just that tui.app.main
is importable and callable, which would pass even if the packaging
metadata were wrong (e.g. a typo in the entry-point target, or
[project.scripts] missing entirely) and `caligula` resolved to "command
not found" in a real terminal after a real `pip install`, as happened
before this test existed. These checks run against whatever environment
pytest itself is running under, so they're only meaningful after that
environment has actually been `pip install -e .`'d (or `pip install`'d
from a wheel/VCS URL) — same precondition as the rest of this suite (see
README's Setup).
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


def _script_path() -> Path:
    script_dir = Path(sysconfig.get_path("scripts"))
    name = "caligula.exe" if sys.platform == "win32" else "caligula"
    return script_dir / name


def test_caligula_is_registered_as_a_console_script_entry_point():
    entry_points = importlib.metadata.entry_points(group="console_scripts")
    matches = [ep for ep in entry_points if ep.name == "caligula"]
    assert matches, "no 'caligula' console_scripts entry point registered — check [project.scripts] in pyproject.toml"
    assert matches[0].value == "tui.app:main"


def test_caligula_script_file_exists_and_is_executable():
    script_path = _script_path()
    assert script_path.exists(), (
        f"expected an installed console script at {script_path} — "
        "was `pip install -e .` (or an install from a wheel/VCS URL) actually run in this environment?"
    )
    if sys.platform != "win32":
        assert os.access(script_path, os.X_OK), f"{script_path} exists but isn't executable"


def test_caligula_version_flag_prints_the_real_installed_version():
    """Unlike the other subprocess test below, `--version` is expected to
    exit on its own (not be terminated by us) -- it must never fall
    through to launching the interactive TUI."""
    script_path = _script_path()
    result = subprocess.run(
        [str(script_path), "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    expected_version = importlib.metadata.version("caligula")
    assert result.stdout.strip() == f"caligula {expected_version}"


def test_caligula_script_actually_launches_a_process_not_command_not_found():
    """The exact failure this reproduces: running the script by path
    (bypassing shell PATH lookup entirely) must start a real OS process.
    If the entry point were missing/misconfigured, this would raise
    FileNotFoundError here — the direct analogue of a shell's "command not
    found" — rather than something caught later inside the app."""
    script_path = _script_path()
    try:
        proc = subprocess.Popen(
            [str(script_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise AssertionError(f"{script_path} did not launch as a process (this is 'command not found'): {exc}") from None

    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass  # still running after 2s — a real, live process; exactly what we're confirming
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def test_wheel_does_not_ship_saved_configs():
    # target_system/configs/ was DEFAULT_CONFIGS_DIR before configs moved
    # to the platformdirs user data dir, so whatever a developer had saved
    # locally got swept into the build: the first 0.1.0 wheel shipped 19,
    # including real reconstructed customer environments (homepilot tool
    # names and observed argument/response profiles from live Braintrust
    # traces). Nothing under that path belongs in a distributed artifact.
    import zipfile
    from pathlib import Path

    wheels = sorted((Path(__file__).parent.parent / "dist").glob("caligula-*.whl"))
    if not wheels:
        pytest.skip("no built wheel in dist/ — run scripts/release.sh first")

    with zipfile.ZipFile(wheels[-1]) as wheel:
        names = wheel.namelist()
    assert not [n for n in names if n.startswith("target_system/configs/")]
    # ...while the genuine read-only package resources are still shipped.
    assert "target_system/policy.yaml" in names
    assert any(n.startswith("target_system/corpus/") for n in names)
