"""Where Caligula's own local config and user data live, independent of the
current working directory *and* of where the package itself got installed.
A pip-installed `caligula` runs from wherever the user happens to be, so
nothing here is Path(__file__)-relative -- these are genuine user-level
locations, resolved via platformdirs so they're the right OS-appropriate
paths on Linux/macOS/Windows.

Why the data dirs moved here too (they used to be Path(__file__)-relative,
resolving to `<package>/data/...` and `target_system/configs/`):

  Package-relative is not the same problem CWD-relative would be -- it's
  stable across launch directories -- but it's the wrong location for
  *user* data for three reasons that all bit in practice:

    - It writes into site-packages. A pipx/venv install put a user's real
      reconstructed configs, 395MB of cached Braintrust traces, and every
      run's JSONL inside
      `.../pipx/venvs/caligula/lib/python3.14/site-packages/`, which is
      invisible from the source checkout and silently diverges from it.
    - A reinstall or upgrade wipes it. pipx reinstall/upgrade replaces the
      venv wholesale, taking every reconstructed config and cached trace
      with it -- data the user paid real API calls to produce.
    - It's read-only in plenty of real installs (system Python, Nix,
      containers, `pip install --user` under a root-owned prefix), where
      the first save would just fail.

  What stays Path(__file__)-relative, deliberately: target_system/corpus/
  and target_system/policy.yaml. Those are package *resources* shipped
  inside the wheel and never written to -- they belong with the code.

CALIGULA_DATA_DIR overrides the data location entirely (read once, at
import) -- for keeping separate working sets, for CI, and for pointing the
dashboard's end-to-end tests at this repo's own backfilled data/runs/
fixtures instead of the real user data dir.

LEGACY_* below record the old locations purely so migrate_legacy_data()
can bring existing data forward; nothing else should read them.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "caligula"

CONFIG_DIR = Path(user_config_dir(APP_NAME))
ENV_PATH = CONFIG_DIR / ".env"

# Read once, at import: these are module-level constants that other modules
# capture as function defaults, so a later os.environ change wouldn't reach
# them anyway -- better to have one honest resolution point than a value
# that's live in some code paths and stale in others. Set it in the
# environment before launching, not mid-process.
DATA_DIR_ENV_VAR = "CALIGULA_DATA_DIR"
_data_dir_override = os.environ.get(DATA_DIR_ENV_VAR)
DATA_DIR = Path(_data_dir_override).expanduser() if _data_dir_override else Path(user_data_dir(APP_NAME))
CONFIGS_DIR = DATA_DIR / "configs"
RUNS_DIR = DATA_DIR / "runs"
TRACES_DIR = DATA_DIR / "traces"  # Langfuse
TRACES_BRAINTRUST_DIR = DATA_DIR / "traces_braintrust"

# Pre-platformdirs locations, relative to the installed package root.
_PACKAGE_ROOT = Path(__file__).parent.parent
LEGACY_CONFIGS_DIR = _PACKAGE_ROOT / "target_system" / "configs"
LEGACY_RUNS_DIR = _PACKAGE_ROOT / "data" / "runs"
LEGACY_TRACES_DIR = _PACKAGE_ROOT / "data" / "traces"
LEGACY_TRACES_BRAINTRUST_DIR = _PACKAGE_ROOT / "data" / "traces_braintrust"

_MIGRATIONS = (
    (LEGACY_CONFIGS_DIR, CONFIGS_DIR),
    (LEGACY_RUNS_DIR, RUNS_DIR),
    (LEGACY_TRACES_DIR, TRACES_DIR),
    (LEGACY_TRACES_BRAINTRUST_DIR, TRACES_BRAINTRUST_DIR),
)


def migrate_legacy_data() -> list[tuple[Path, Path]]:
    """One-time, non-destructive move of any pre-platformdirs data into the
    new locations. Returns the (source, destination) pairs actually
    migrated, so a caller can report what moved.

    Copies rather than moves, and only ever *adds* files: a destination
    that already exists is left exactly as it is (per-file, not per-
    directory -- so a partially-populated new location still picks up
    whatever the legacy one has that it doesn't). The legacy directory is
    left in place. That asymmetry is deliberate: this runs unattended at
    startup, and the cost of leaving a stale duplicate behind is a few
    hundred MB the user can delete, while the cost of a move going wrong
    is trace data that cost real money to fetch.

    Safe to call when the legacy paths don't exist (a fresh install, or a
    source checkout that never wrote there) -- it's a no-op."""
    migrated: list[tuple[Path, Path]] = []
    for source, destination in _MIGRATIONS:
        if not source.is_dir() or not any(source.iterdir()):
            continue
        copied = _copy_new_files(source, destination)
        if copied:
            migrated.append((source, destination))
    return migrated


def _copy_new_files(source: Path, destination: Path) -> int:
    copied = 0
    for entry in source.rglob("*"):
        if not entry.is_file():
            continue
        target = destination / entry.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry, target)
        copied += 1
    return copied
