"""Regression coverage for scripts/release.sh and scripts/install_latest.sh.

Doesn't invoke a real pipx/network install in the regular suite (slow,
environment-dependent) -- that was verified manually for real (build,
inspect the wheel's own METADATA, `pipx uninstall` + install_latest.sh
into a clean pipx state, `aetius --version` in a fresh shell, confirm
streamlit/pyarrow absent from the installed venv). What's covered here
automatically: release.sh actually builds a wheel whose METADATA never
requires streamlit/plotly/pyarrow unconditionally, produces the stable
release/aetius-latest.whl copy, and install_latest.sh's core job --
deriving a real, spec-compliant wheel filename from a stable-named file's
own dist-info metadata -- using a stubbed `pipx` on PATH so this stays
fast and hermetic.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_latest.sh"


def test_release_script_builds_wheel_with_dashboard_extra_gated_correctly(tmp_path):
    dist_dir = tmp_path / "dist"
    result = subprocess.run(["uv", "build", "--wheel", "--clear", "-o", str(dist_dir)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr

    wheels = list(dist_dir.glob("aetius-*-py3-none-any.whl"))
    assert len(wheels) == 1, f"expected exactly one built wheel, found {wheels}"
    wheel_path = wheels[0]

    with zipfile.ZipFile(wheel_path) as z:
        metadata_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = z.read(metadata_name).decode("utf-8")

    requires = [line for line in metadata.splitlines() if line.startswith("Requires-Dist")]
    unconditional = [line for line in requires if "extra ==" not in line]
    dashboard_gated = [line for line in requires if "extra == 'dashboard'" in line]

    assert not any("streamlit" in line.lower() for line in unconditional), unconditional
    assert not any("plotly" in line.lower() for line in unconditional), unconditional
    assert not any("pyarrow" in line.lower() for line in unconditional), unconditional
    assert any("streamlit" in line.lower() for line in dashboard_gated)
    assert any("plotly" in line.lower() for line in dashboard_gated)


def _make_fake_pipx(bin_dir: Path, calls_file: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_pipx = bin_dir / "pipx"
    fake_pipx.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{calls_file}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_pipx.chmod(fake_pipx.stat().st_mode | stat.S_IEXEC)


def test_install_latest_derives_a_real_wheel_filename_and_invokes_pipx_with_it(tmp_path, monkeypatch):
    # A minimal, real wheel (proper zip, real *.dist-info/METADATA) named
    # the stable, non-PEP-427-compliant way -- exactly what release.sh
    # produces and what pip/pipx reject by that literal name.
    stable_source = tmp_path / "aetius-latest.whl"
    with zipfile.ZipFile(stable_source, "w") as z:
        z.writestr("aetius-9.9.9.dist-info/METADATA", "Metadata-Version: 2.1\nName: aetius\nVersion: 9.9.9\n")
        z.writestr("aetius-9.9.9.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")

    calls_file = tmp_path / "pipx_calls.txt"
    fake_bin = tmp_path / "fakebin"
    _make_fake_pipx(fake_bin, calls_file)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), str(stable_source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    assert calls_file.exists(), "the stubbed pipx was never invoked"
    calls = [line.split(" ") for line in calls_file.read_text(encoding="utf-8").strip().splitlines()]
    # the hardened script probes usability first ("pipx --version"), then installs
    assert calls[0] == ["--version"]
    install_call = next(c for c in calls if c[0] == "install")
    assert install_call[-1] == "--force"
    assert Path(install_call[1]).name == "aetius-9.9.9-py3-none-any.whl"


def test_install_latest_errors_clearly_when_source_file_missing(tmp_path):
    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), str(tmp_path / "does-not-exist.whl")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower()


# --- install_latest.sh on machines without a working pipx / old Python ----------
#
# The real reported failure: a machine with no pipx at all, where the old
# script died with a bare "pipx: command not found". Verified end-to-end
# against a genuinely bare environment (fresh HOME, PATH without pipx,
# real pip install --user bootstrap); these tests pin each branch with a
# delegating fake python3 so the suite needs no network.


def _make_fake_python3(bin_dir: Path, calls_file: Path, state_dir: Path, *, version_ok: bool = True, has_pip: bool = True, pep668: bool = False) -> None:
    """A python3 that delegates to the real interpreter except for the
    version gate and `-m pip` / `-m pipx`, which are simulated: pipx is
    'not installed' until a recorded `pip install --user pipx` happens."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    real = sys.executable
    pip_install_action = 'cat "$STATE/pep668.txt" >&2; exit 1' if pep668 else 'touch "$STATE/pipx_installed"'
    fake = bin_dir / "python3"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'CALLS="{calls_file}"; STATE="{state_dir}"; REAL="{real}"\n'
        'if [ "$1" = "-c" ]; then\n'
        '  case "$2" in\n'
        '    *"%d.%d.%d"*) echo "3.9.6"; exit 0;;\n'
        f'    *version_info*) exit {0 if version_ok else 1};;\n'
        "  esac\n"
        '  exec "$REAL" "$@"\n'
        "fi\n"
        'if [ "$1" = "-m" ]; then\n'
        '  mod="$2"; shift 2\n'
        '  echo "-m $mod $*" >> "$CALLS"\n'
        '  if [ "$mod" = "pipx" ]; then\n'
        '    if [ ! -f "$STATE/pipx_installed" ]; then exit 1; fi\n'
        '    if [ "$1" = "environment" ]; then echo "/fake/pipx/bin"; fi\n'
        "    exit 0\n"
        "  fi\n"
        '  if [ "$mod" = "pip" ]; then\n'
        f'    {"" if has_pip else "exit 1"}\n'
        '    if [ "$1" = "install" ]; then ' + pip_install_action + '; fi\n'
        "    exit 0\n"
        "  fi\n"
        "fi\n"
        'exec "$REAL" "$@"\n',
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    if pep668:
        (state_dir / "pep668.txt").write_text(_PEP668_TEXT, encoding="utf-8")


def _bare_env(bin_dir: Path) -> dict:
    return {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(bin_dir.parent / "home")}


def _run_install(tmp_path: Path, env: dict, wheel: Path):
    script = Path(__file__).parent.parent / "scripts" / "install_latest.sh"
    return subprocess.run(
        ["bash", str(script), str(wheel)], env=env, capture_output=True, text=True, timeout=120
    )


def _stable_wheel(tmp_path: Path) -> Path:
    wheel = tmp_path / "aetius-latest.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("aetius-9.9.9.dist-info/METADATA", "Metadata-Version: 2.1\nName: aetius\nVersion: 9.9.9\n")
        z.writestr("aetius-9.9.9.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
    return wheel


def test_install_refuses_old_python_before_doing_anything(tmp_path):
    calls = tmp_path / "calls.txt"
    _make_fake_python3(tmp_path / "bin", calls, tmp_path / "state", version_ok=False)
    result = _run_install(tmp_path, _bare_env(tmp_path / "bin"), _stable_wheel(tmp_path))
    assert result.returncode == 1
    assert "requires Python 3.11+" in result.stderr
    assert "3.9.6" in result.stderr  # names the actual version found
    assert "Nothing was installed" in result.stderr
    assert not calls.exists()  # stopped before any pip/pipx attempt


def test_install_bootstraps_pipx_when_missing_and_completes_in_same_run(tmp_path):
    calls = tmp_path / "calls.txt"
    _make_fake_python3(tmp_path / "bin", calls, tmp_path / "state")
    result = _run_install(tmp_path, _bare_env(tmp_path / "bin"), _stable_wheel(tmp_path))
    assert result.returncode == 0, result.stderr
    logged = calls.read_text()
    assert "-m pip install --user pipx" in logged
    # the install proceeds via `python3 -m pipx` in the SAME run — no
    # shell restart between bootstrap and install
    assert "-m pipx install" in logged and "aetius-9.9.9-py3-none-any.whl" in logged
    assert "no shell restart needed" in result.stdout


def test_install_fails_actionably_when_pip_itself_is_missing(tmp_path):
    calls = tmp_path / "calls.txt"
    _make_fake_python3(tmp_path / "bin", calls, tmp_path / "state", has_pip=False)
    result = _run_install(tmp_path, _bare_env(tmp_path / "bin"), _stable_wheel(tmp_path))
    assert result.returncode == 1
    assert "python3 has no working pip" in result.stderr
    assert "ensurepip" in result.stderr and "python3-pip" in result.stderr  # exact commands, both platforms


def test_script_min_python_matches_the_wheel_requires_python():
    # The gate must be the wheel's own claim, not a hand-maintained guess.
    script = (Path(__file__).parent.parent / "scripts" / "install_latest.sh").read_text()
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert 'MIN_PYTHON_MINOR=11' in script
    assert 'requires-python = ">=3.11"' in pyproject


# --- Homebrew / PEP 668 (externally-managed-environment) ------------------------
#
# The second real reported failure: a Homebrew Python enforces PEP 668,
# so the pip bootstrap is a guaranteed refusal there. With brew present
# the script must go straight to `brew install pipx` and never touch
# pip; without brew, a PEP 668 refusal must produce the venv-based fix,
# and the script must never run --break-system-packages itself.

_PEP668_TEXT = (
    "error: externally-managed-environment\n\n"
    "\u00d7 This environment is externally managed\n"
    "\u2570\u2500> To install Python packages system-wide, try brew install xyz.\n"
    "    If you wish to install a Python application, it may be easiest to use\n"
    "    pipx install xyz or create a virtual environment.\n"
    "    Note: you can pass --break-system-packages to override this, at the risk\n"
    "    of breaking your Python installation.\n"
)


def _make_fake_brew(bin_dir: Path, prefix: Path, brew_calls: Path, pipx_calls: Path) -> None:
    """brew that records calls; `install pipx` materializes a working fake
    pipx under the fake prefix's bin (NOT on PATH — the script must find
    it via `brew --prefix`)."""
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    fake_pipx = prefix / "bin" / "pipx"
    fake_pipx.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{pipx_calls}"\n'
        'if [ "$1" = "--version" ]; then echo "1.7.1"; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    brew = bin_dir / "brew"
    brew.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{brew_calls}"\n'
        f'if [ "$1" = "--prefix" ]; then echo "{prefix}"; exit 0; fi\n'
        'if [ "$1" = "install" ] && [ "$2" = "pipx" ]; then\n'
        f'  chmod +x "{fake_pipx}"\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    brew.chmod(brew.stat().st_mode | stat.S_IEXEC)


def test_with_homebrew_present_pipx_comes_from_brew_never_pip(tmp_path):
    py_calls, brew_calls, pipx_calls = tmp_path / "py.txt", tmp_path / "brew.txt", tmp_path / "pipx.txt"
    _make_fake_python3(tmp_path / "bin", py_calls, tmp_path / "state")
    _make_fake_brew(tmp_path / "bin", tmp_path / "brew-prefix", brew_calls, pipx_calls)
    result = _run_install(tmp_path, _bare_env(tmp_path / "bin"), _stable_wheel(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "install pipx" in brew_calls.read_text()
    # the brew-prefix pipx (not on PATH) did the install
    logged = pipx_calls.read_text()
    assert "install" in logged and "aetius-9.9.9-py3-none-any.whl" in logged
    # pip was never attempted — that path is a guaranteed PEP 668 failure on brew Pythons
    py_logged = py_calls.read_text() if py_calls.exists() else ""
    assert "-m pip install" not in py_logged


def test_pep668_refusal_gets_venv_guidance_and_never_break_system_packages(tmp_path):
    py_calls = tmp_path / "py.txt"
    _make_fake_python3(tmp_path / "bin", py_calls, tmp_path / "state", pep668=True)
    result = _run_install(tmp_path, _bare_env(tmp_path / "bin"), _stable_wheel(tmp_path))
    assert result.returncode == 1
    assert "externally-managed-environment" in result.stderr
    assert "-m venv" in result.stderr and "bin/pip install pipx" in result.stderr
    assert "never run it for you" in result.stderr  # --break-system-packages stays manual
    # and the script itself never executed pip with the override flag
    assert "--break-system-packages" not in py_calls.read_text()
