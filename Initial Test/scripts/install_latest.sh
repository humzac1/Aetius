#!/usr/bin/env bash
# Installs release/caligula-latest.whl (see scripts/release.sh) via pipx.
#
# caligula-latest.whl is a stable *filename* for hosting/linking, but
# pip/pipx require an actual PEP 427 wheel filename
# (name-version-pytag-abitag-platformtag.whl) to install anything from --
# verified for real, not assumed: `pipx install .../caligula-latest.whl`
# fails outright with "Invalid wheel filename" (plain `pip install` on the
# same file fails identically, so this isn't a pipx quirk). This script
# copies the stable file to a spec-compliant temporary filename -- read
# from the wheel's own *.dist-info metadata, never hand-maintained, so it
# can't drift from whatever version was actually built -- and installs
# that instead.
#
# Usage: scripts/install_latest.sh [path-or-url-to-caligula-latest.whl]
#   (defaults to release/caligula-latest.whl next to this script)
set -euo pipefail

SRC="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/release/caligula-latest.whl}"

if [[ "$SRC" == http://* || "$SRC" == https://* ]]; then
    TMP_DOWNLOAD="$(mktemp -d)/caligula-latest.whl"
    echo "Downloading $SRC..."
    curl -fsSL "$SRC" -o "$TMP_DOWNLOAD"
    SRC="$TMP_DOWNLOAD"
fi

if [ ! -f "$SRC" ]; then
    echo "error: $SRC not found" >&2
    exit 1
fi

REAL_NAME="$(python3 - "$SRC" <<'PY'
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

echo "Installing $REAL_NAME (from $SRC)..."
pipx install "$REAL_PATH" --force
