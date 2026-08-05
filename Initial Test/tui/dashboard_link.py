"""Opens the existing Streamlit dashboard (Part 5) at the URL for a
specific run, rather than reimplementing any of its charts in the TUI.
Only ever opens a browser tab — never launches `streamlit run` itself,
since spawning a background server behind the user's back is exactly the
kind of side effect this tool should ask for, not just do silently.
"""

from __future__ import annotations

import socket
import webbrowser

DASHBOARD_HOST = "localhost"
DASHBOARD_PORT = 8501


def dashboard_is_running(host: str = DASHBOARD_HOST, port: int = DASHBOARD_PORT, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def dashboard_url_for_run(experiment_name: str) -> str:
    return f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}/?experiment={experiment_name}"


def open_dashboard_for_run(experiment_name: str) -> tuple[bool, str]:
    """Returns (opened, url). Only actually opens a browser tab if the
    dashboard looks like it's already running; otherwise the caller is
    expected to show `url` alongside instructions to start it manually
    (`streamlit run dashboard/app.py`)."""
    url = dashboard_url_for_run(experiment_name)
    if dashboard_is_running():
        webbrowser.open(url)
        return True, url
    return False, url
