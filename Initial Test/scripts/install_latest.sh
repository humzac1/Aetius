#!/usr/bin/env bash
# Installs release/aetius-latest.whl (see scripts/release.sh) via pipx —
# including on a machine that has no working pipx yet, which is a real
# reported failure of the previous version of this script (it assumed
# `pipx` was on PATH and died with a bare "command not found").
#
# Order of operations, and why:
#   1. Python version gate FIRST. The wheel's own metadata declares
#      Requires-Python >=3.11 (verified against the built artifact, and
#      pinned by tests/test_release_scripts.py) — on an older Python the
#      failure would otherwise surface as a confusing pip resolution
#      error long after this script has done half its work.
#   2. pipx must be USABLE, not merely present: `pipx --version` has to
#      actually run. If there's no working pipx and Homebrew exists,
#      `brew install pipx` comes FIRST — Homebrew Pythons enforce PEP 668
#      (externally-managed-environment), so `pip install --user pipx`
#      against them is a guaranteed failure (a real reported one), and
#      brew is Homebrew's own recommended way to get pipx. Only without
#      Homebrew does the pip bootstrap run: `python3 -m pip install
#      --user pipx`, then `python3 -m pipx ...` for the rest of this run
#      — module form works immediately regardless of PATH, so the
#      install completes in THIS shell with no restart needed. If pip
#      itself refuses with the PEP 668 error on a non-Homebrew system,
#      the script stops with the safe venv-based fix spelled out; it
#      never runs --break-system-packages on your behalf — overriding an
#      OS package manager's protection is an informed manual choice.
#   3. PATH is handled explicitly at the end: `pipx ensurepath` for
#      future shells, plus a check whether `aetius` resolves right now —
#      and if it doesn't, the exact one-line export to run, never a
#      shrug.
#
# aetius-latest.whl is a stable *filename* for hosting/linking, but
# pip/pipx require an actual PEP 427 wheel filename to install anything,
# so the file is copied to a spec-compliant temporary name read from the
# wheel's own *.dist-info metadata (never hand-maintained).
#
# Usage: scripts/install_latest.sh [path-or-url-to-aetius-latest.whl]
#   (defaults to release/aetius-latest.whl next to this script)
set -euo pipefail

MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11  # must match Requires-Python in the wheel metadata

die() {
    echo "" >&2
    echo "error: $1" >&2
    shift
    for line in "$@"; do
        echo "  $line" >&2
    done
    exit 1
}

# --- 1. Python prerequisite, checked before anything else --------------------

if ! command -v python3 >/dev/null 2>&1; then
    die "python3 was not found on PATH — aetius needs Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+." \
        "Install it from https://www.python.org/downloads/ (or 'brew install python' on macOS," \
        "'sudo apt install python3' on Debian/Ubuntu), then re-run this script."
fi

PYTHON="$(command -v python3)"
PY_VERSION="$("$PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR}) else 1)"; then
    die "your python3 is ${PY_VERSION}, but aetius requires Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ (the wheel's own Requires-Python)." \
        "Nothing was installed. Install a newer Python first —" \
        "  macOS:         brew install python   (or https://www.python.org/downloads/)" \
        "  Debian/Ubuntu: sudo apt install python3.12" \
        "— then re-run this script with it first on PATH."
fi

# --- 2. locate the wheel ------------------------------------------------------

SRC="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/release/aetius-latest.whl}"

if [[ "$SRC" == http://* || "$SRC" == https://* ]]; then
    TMP_DOWNLOAD="$(mktemp -d)/aetius-latest.whl"
    echo "Downloading $SRC..."
    curl -fsSL "$SRC" -o "$TMP_DOWNLOAD"
    SRC="$TMP_DOWNLOAD"
fi

if [ ! -f "$SRC" ]; then
    die "$SRC not found." \
        "Pass the wheel's path or URL: scripts/install_latest.sh /path/to/aetius-latest.whl"
fi

REAL_NAME="$("$PYTHON" - "$SRC" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as z:
    dist_info = next(n.split("/")[0] for n in z.namelist() if n.endswith(".dist-info/METADATA"))
name_version = dist_info[: -len(".dist-info")]
print(f"{name_version}-py3-none-any.whl")  # this project always builds a universal (pure-Python) wheel
PY
)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
REAL_PATH="$TMP_DIR/$REAL_NAME"
cp "$SRC" "$REAL_PATH"

# --- 3. a working pipx, made if necessary ------------------------------------

PIPX=()
if command -v pipx >/dev/null 2>&1 && pipx --version >/dev/null 2>&1; then
    PIPX=(pipx)
elif "$PYTHON" -m pipx --version >/dev/null 2>&1; then
    # installed as a module but not on PATH — perfectly usable this way
    PIPX=("$PYTHON" -m pipx)
elif command -v brew >/dev/null 2>&1; then
    # Homebrew first, and INSTEAD of pip — brew-managed Pythons enforce
    # PEP 668, so the pip bootstrap below is a guaranteed
    # "externally-managed-environment" failure there (a real reported
    # one). brew install pipx is Homebrew's own recommended path.
    echo "pipx not found — installing it via Homebrew (brew install pipx)..."
    PIPX_READY=0
    if brew install pipx; then
        if command -v pipx >/dev/null 2>&1 && pipx --version >/dev/null 2>&1; then
            PIPX=(pipx); PIPX_READY=1
        else
            # brew succeeded but its bin dir isn't on this shell's PATH —
            # use the absolute path so this run still completes.
            BREW_PIPX="$(brew --prefix)/bin/pipx"
            if [ -x "$BREW_PIPX" ] && "$BREW_PIPX" --version >/dev/null 2>&1; then
                PIPX=("$BREW_PIPX"); PIPX_READY=1
            fi
        fi
    fi
    if [ "$PIPX_READY" -ne 1 ]; then
        die "Homebrew is present but 'brew install pipx' did not produce a working pipx." \
            "Run these yourself, then re-run this script:" \
            "  brew install pipx" \
            "  brew link --overwrite pipx" \
            "(the pip-based fallback is deliberately not attempted here: Homebrew Pythons" \
            "refuse it with PEP 668's externally-managed-environment error.)"
    fi
    echo "pipx ready via Homebrew; continuing."
else
    echo "pipx not found — installing it now ($PYTHON -m pip install --user pipx)..."
    if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
        die "python3 has no working pip, so pipx cannot be installed automatically." \
            "Fix pip first, then re-run this script:" \
            "  Debian/Ubuntu: sudo apt install python3-pip" \
            "  most systems:  $PYTHON -m ensurepip --upgrade" \
            "Or sidestep pip entirely with a virtual environment:" \
            "  $PYTHON -m venv \$HOME/.aetius-bootstrap-venv" \
            "  \$HOME/.aetius-bootstrap-venv/bin/pip install pipx" \
            "  export PATH=\"\$HOME/.aetius-bootstrap-venv/bin:\$PATH\"" \
            "(pip is Python's own package installer; pipx builds on it.)"
    fi
    PIP_LOG="$TMP_DIR/pip-install-pipx.log"
    if ! "$PYTHON" -m pip install --user pipx >"$PIP_LOG" 2>&1; then
        cat "$PIP_LOG" >&2
        if grep -qi "externally-managed-environment" "$PIP_LOG"; then
            die "this Python refuses pip installs into its environment (PEP 668, externally-managed-environment — see pip's message above)." \
                "The safe fix is the one pip's own message points at — install pipx in its own" \
                "virtual environment, then re-run this script:" \
                "  $PYTHON -m venv \$HOME/.aetius-bootstrap-venv" \
                "  \$HOME/.aetius-bootstrap-venv/bin/pip install pipx" \
                "  export PATH=\"\$HOME/.aetius-bootstrap-venv/bin:\$PATH\"" \
                "(pip's --break-system-packages flag would also work, but overriding your OS" \
                "package manager's protection is a choice to make yourself — this script will" \
                "never run it for you.)"
        fi
        die "automatic pipx install failed (see pip's output above)." \
            "Install pipx yourself, then re-run this script:" \
            "  Debian/Ubuntu: sudo apt install pipx" \
            "  any platform:  $PYTHON -m pip install --user pipx" \
            "pipx is what keeps aetius in its own isolated environment instead of" \
            "polluting (or being broken by) your system Python packages."
    fi
    cat "$PIP_LOG"
    if ! "$PYTHON" -m pipx --version >/dev/null 2>&1; then
        die "pipx was installed but '$PYTHON -m pipx --version' still fails — something is wrong with this Python environment." \
            "Try installing pipx via your package manager instead, then re-run this script:" \
            "  Debian/Ubuntu: sudo apt install pipx    macOS: brew install pipx"
    fi
    PIPX=("$PYTHON" -m pipx)
    echo "pipx installed; continuing with '$PYTHON -m pipx' (no shell restart needed for this run)."
fi

# --- 4. install ---------------------------------------------------------------

echo "Installing $REAL_NAME (from $SRC)..."
"${PIPX[@]}" install "$REAL_PATH" --force

# --- 5. PATH: make `aetius` reachable, or say exactly how to make it so ------

"${PIPX[@]}" ensurepath >/dev/null 2>&1 || true  # future shells; harmless if already set

if command -v aetius >/dev/null 2>&1; then
    echo ""
    echo "Done: $(aetius --version) — run 'aetius' to start."
else
    BIN_DIR="$("${PIPX[@]}" environment --value PIPX_BIN_DIR 2>/dev/null || true)"
    echo ""
    echo "Installed, but 'aetius' is not on this shell's PATH yet (pipx has added it for future shells)."
    if [ -n "$BIN_DIR" ] && [ -x "$BIN_DIR/aetius" ]; then
        echo "Either open a new terminal, or run these two commands in this one:"
        echo ""
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        echo "  aetius"
    else
        echo "Open a new terminal and run: aetius"
    fi
fi
