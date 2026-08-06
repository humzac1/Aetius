"""Where Caligula's own local config lives, independent of the current
working directory or the repo layout. A pip-installed `caligula` runs from
wherever the user happens to be, so nothing here is Path(__file__)-relative
the way target_system's data dirs are (see ingestion/langfuse_client.py's
DEFAULT_TRACES_DIR) -- this is a genuine user-level location, resolved via
platformdirs so it's the right OS-appropriate path on Linux/macOS/Windows.
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_config_dir

APP_NAME = "caligula"

CONFIG_DIR = Path(user_config_dir(APP_NAME))
ENV_PATH = CONFIG_DIR / ".env"
