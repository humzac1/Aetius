#!/usr/bin/env bash
# Builds a self-hosted wheel for caligula -- no PyPI involved. Produces:
#   dist/caligula-<version>-py3-none-any.whl   (versioned, kept for records)
#   release/caligula-latest.whl                (stable filename -- this is
#                                                the one to actually host/link)
#
# The core wheel only ever requires the base [project.dependencies] list --
# streamlit/plotly (and the pyarrow they drag in, which has broken a plain
# `pipx install` on newer Pythons before -- see pyproject.toml's [dashboard]
# extra comment) are never required by a plain install of this wheel.
#
# Usage: scripts/release.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DIST_DIR="$PROJECT_ROOT/dist"
RELEASE_DIR="$PROJECT_ROOT/release"
STABLE_PATH="$RELEASE_DIR/caligula-latest.whl"

echo "Building wheel from $PROJECT_ROOT..."
uv build --wheel --clear -o "$DIST_DIR"

WHEEL_PATH="$(ls -t "$DIST_DIR"/caligula-*-py3-none-any.whl 2>/dev/null | head -n1)"
if [ -z "$WHEEL_PATH" ]; then
    echo "error: no caligula-*-py3-none-any.whl found in $DIST_DIR after build" >&2
    exit 1
fi

mkdir -p "$RELEASE_DIR"
cp "$WHEEL_PATH" "$STABLE_PATH"

echo
echo "Versioned (kept for records): $WHEEL_PATH"
echo "Stable filename (host this):  $STABLE_PATH"
echo
echo "Install with:  pipx install $STABLE_PATH --force"
echo "Verify with:   caligula --version"
