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
    call_args = calls_file.read_text(encoding="utf-8").strip().split(" ")
    assert call_args[0] == "install"
    assert call_args[-1] == "--force"
    installed_path = Path(call_args[1])
    assert installed_path.name == "aetius-9.9.9-py3-none-any.whl"


def test_install_latest_errors_clearly_when_source_file_missing(tmp_path):
    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), str(tmp_path / "does-not-exist.whl")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower()
