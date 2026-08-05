import json

import pytest
from textual.widgets import Label, ListView

from target_system.config import list_config_hashes
from tui.app import HarnessApp
from tui.screens.add_environment import (
    AddEnvironmentDoneScreen,
    AddEnvironmentGroupsScreen,
    AddEnvironmentScreen,
    PullErrorScreen,
    PullTracesScreen,
)
from tests.tui_test_support import run_async
from tests.tui_test_support import wait_until as _wait_until


def _write_trace(traces_dir, project_id, trace_id, *, agent_name, observations=None):
    d = traces_dir / project_id
    d.mkdir(parents=True, exist_ok=True)
    metadata = {"agent_name": agent_name} if agent_name is not None else {}
    payload = {"id": trace_id, "metadata": metadata, "observations": observations or []}
    (d / f"{trace_id}.json").write_text(json.dumps(payload), encoding="utf-8")


_TOOL_CALL_OBS = {
    "name": "tool-call-send_invoice",
    "type": "SPAN",
    "input": {"tool": "send_invoice", "inputs": {"invoice_id": "INV-1"}},
    "output": {"status": "sent"},
}


# --- entry screen --------------------------------------------------------


def test_entry_screen_shows_error_when_project_id_missing(tmp_path, monkeypatch):
    import ingestion.langfuse_client as lf

    def _raise():
        raise RuntimeError("LANGFUSE_PROJECT_ID not set in .env")

    monkeypatch.setattr(lf, "default_project_id", _raise)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentScreen(traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            assert app.screen.error is not None
            assert "LANGFUSE_PROJECT_ID" in app.screen.error

    run_async(scenario)


def test_entry_screen_offers_only_pull_when_nothing_cached(tmp_path, monkeypatch):
    import ingestion.langfuse_client as lf

    monkeypatch.setattr(lf, "default_project_id", lambda: "proj-1")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentScreen(traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            menu = app.screen.query_one("#add-env-menu", ListView)
            ids = [item.id for item in menu.children]
            assert ids == ["pull"]

    run_async(scenario)


def test_entry_screen_offers_cache_option_when_traces_already_cached(tmp_path, monkeypatch):
    import ingestion.langfuse_client as lf

    monkeypatch.setattr(lf, "default_project_id", lambda: "proj-1")
    _write_trace(tmp_path, "proj-1", "t1", agent_name="Invoice Generation Assistant")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentScreen(traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            menu = app.screen.query_one("#add-env-menu", ListView)
            ids = [item.id for item in menu.children]
            assert ids == ["use_cache", "pull"]

    run_async(scenario)


# --- pull screen -----------------------------------------------------------


def test_pull_screen_success_lands_on_groups_screen(tmp_path, monkeypatch):
    import tui.screens.add_environment as mod

    def _fake_pull(*, project_id, batch_size, traces_dir):
        _write_trace(traces_dir, project_id, "t1", agent_name="Invoice Generation Assistant")
        return ["t1"]

    monkeypatch.setattr(mod, "pull_traces", _fake_pull)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PullTracesScreen(project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await _wait_until(pilot, lambda: isinstance(app.screen, AddEnvironmentGroupsScreen))

    run_async(scenario)


def test_pull_screen_failure_lands_on_error_screen(tmp_path, monkeypatch):
    import tui.screens.add_environment as mod

    def _fake_pull(*, project_id, batch_size, traces_dir):
        raise RuntimeError("missing required .env vars: LANGFUSE_SECRET_KEY")

    monkeypatch.setattr(mod, "pull_traces", _fake_pull)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PullTracesScreen(project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await _wait_until(pilot, lambda: isinstance(app.screen, PullErrorScreen))
            assert "LANGFUSE_SECRET_KEY" in app.screen.message

    run_async(scenario)


# --- groups screen -----------------------------------------------------------


def test_groups_screen_lists_real_groups_sorted_by_count_and_excludes_noise(tmp_path):
    _write_trace(tmp_path, "proj-1", "a1", agent_name="Invoice Generation Assistant", observations=[_TOOL_CALL_OBS])
    _write_trace(tmp_path, "proj-1", "a2", agent_name="Invoice Generation Assistant", observations=[_TOOL_CALL_OBS])
    _write_trace(tmp_path, "proj-1", "b1", agent_name="HR Onboarding Assistant")
    _write_trace(tmp_path, "proj-1", "n1", agent_name="test")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentGroupsScreen(project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            menu = app.screen.query_one("#group-list", ListView)
            assert len(menu.children) == 2  # noise excluded
            first_label = str(menu.children[0].query_one(Label).render())
            assert "Invoice Generation Assistant" in first_label
            assert "2 trace" in first_label

    run_async(scenario)


def test_groups_screen_handles_missing_agent_name_group(tmp_path):
    _write_trace(tmp_path, "proj-1", "u1", agent_name=None)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentGroupsScreen(project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            menu = app.screen.query_one("#group-list", ListView)
            label = str(menu.children[0].query_one(Label).render())
            assert "no agent_name tag" in label

    run_async(scenario)


def test_selecting_group_with_spaces_in_name_reconstructs_and_saves(tmp_path):
    # Regression: agent_name values contain spaces ("Invoice Generation
    # Assistant") which are not valid Textual widget ids -- items must be
    # keyed positionally, not by embedding the name in the id directly.
    _write_trace(tmp_path, "proj-1", "a1", agent_name="Invoice Generation Assistant", observations=[_TOOL_CALL_OBS])

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentGroupsScreen(project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, AddEnvironmentDoneScreen)
            assert app.screen.config.provenance.source_agent_name == "Invoice Generation Assistant"
            assert app.screen.config.provenance.trace_count == 1
            assert "send_invoice" in app.screen.config.supervisor().tools
            assert app.screen.config_hash in list_config_hashes(configs_dir=tmp_path)

    run_async(scenario)


def test_done_screen_surfaces_other_groups_found(tmp_path):
    _write_trace(tmp_path, "proj-1", "a1", agent_name="Invoice Generation Assistant", observations=[_TOOL_CALL_OBS])
    _write_trace(tmp_path, "proj-1", "b1", agent_name="HR Onboarding Assistant")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentGroupsScreen(project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            await pilot.press("enter")  # picks the first (highest trace count) group
            await pilot.pause()
            assert isinstance(app.screen, AddEnvironmentDoneScreen)
            other_names = {g.agent_name for g in app.screen.config.provenance.other_groups_found}
            assert "HR Onboarding Assistant" in other_names

    run_async(scenario)
