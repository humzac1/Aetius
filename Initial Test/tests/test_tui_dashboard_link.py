import socket

import pytest

from tui import dashboard_link


def test_dashboard_url_includes_experiment_query_param():
    url = dashboard_link.dashboard_url_for_run("known_regression")
    assert url == "http://localhost:8501/?experiment=known_regression"


def test_dashboard_is_running_false_when_nothing_listens():
    # an arbitrary high port nothing binds during a test run
    assert dashboard_link.dashboard_is_running(port=59123, timeout=0.05) is False


def test_dashboard_is_running_true_when_something_listens():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert dashboard_link.dashboard_is_running(port=port, timeout=0.5) is True
    finally:
        server.close()


def test_open_dashboard_for_run_does_not_open_browser_when_not_running(monkeypatch):
    monkeypatch.setattr(dashboard_link, "dashboard_is_running", lambda **kwargs: False)
    called = []
    monkeypatch.setattr(dashboard_link.webbrowser, "open", lambda url: called.append(url))
    opened, url = dashboard_link.open_dashboard_for_run("known_regression")
    assert opened is False
    assert called == []
    assert "known_regression" in url


def test_open_dashboard_for_run_opens_browser_when_running(monkeypatch):
    monkeypatch.setattr(dashboard_link, "dashboard_is_running", lambda **kwargs: True)
    called = []
    monkeypatch.setattr(dashboard_link.webbrowser, "open", lambda url: called.append(url))
    opened, url = dashboard_link.open_dashboard_for_run("known_regression")
    assert opened is True
    assert called == [url]
