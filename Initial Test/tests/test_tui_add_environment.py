import json

import pytest
from textual.widgets import Label, ListView

import config.credentials as credentials
import ingestion.braintrust_client as bt
import ingestion.langfuse_client as lf
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
    def _raise():
        raise RuntimeError("LANGFUSE_PROJECT_ID not set")

    monkeypatch.setattr(lf, "default_project_id", _raise)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentScreen(langfuse_traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            assert app.screen.error is not None
            assert "LANGFUSE_PROJECT_ID" in app.screen.error
            assert app.screen.source == "langfuse"

    run_async(scenario)


def test_entry_screen_offers_only_pull_when_nothing_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(lf, "default_project_id", lambda: "proj-1")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentScreen(langfuse_traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            menu = app.screen.query_one("#add-env-menu", ListView)
            ids = [item.id for item in menu.children]
            assert ids == ["pull"]

    run_async(scenario)


def test_entry_screen_offers_cache_option_when_traces_already_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(lf, "default_project_id", lambda: "proj-1")
    _write_trace(tmp_path, "proj-1", "t1", agent_name="Invoice Generation Assistant")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentScreen(langfuse_traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            menu = app.screen.query_one("#add-env-menu", ListView)
            ids = [item.id for item in menu.children]
            assert ids == ["use_cache", "pull"]

    run_async(scenario)


# --- source dispatch (regression: this screen used to assume Langfuse) -----


def test_entry_screen_dispatches_to_braintrust_when_braintrust_is_configured(tmp_path, monkeypatch):
    # The actual bug: this screen used to call ingestion.langfuse_client
    # unconditionally, so a Braintrust-configured install saw "Langfuse
    # isn't configured: LANGFUSE_PROJECT_ID not set" even though
    # Braintrust was correctly set up and validated. Confirms the fix:
    # with TRACE_SOURCE=braintrust, this screen never touches Langfuse at
    # all -- no LANGFUSE_* env var, no lf.default_project_id call.
    monkeypatch.setenv(credentials.TRACE_SOURCE_KEY, "braintrust")
    monkeypatch.delenv("LANGFUSE_PROJECT_ID", raising=False)
    monkeypatch.setattr(bt, "default_project_name", lambda: "homepilot")

    def _fail_if_called():
        raise AssertionError("AddEnvironmentScreen must not call langfuse_client.default_project_id when Braintrust is configured")

    monkeypatch.setattr(lf, "default_project_id", _fail_if_called)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentScreen(braintrust_traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            assert app.screen.error is None
            assert app.screen.source == "braintrust"
            assert app.screen.project_id == "homepilot"
            body_text = " ".join(str(label.render()) for label in app.screen.query(Label))
            assert "Langfuse" not in body_text
            assert "LANGFUSE" not in body_text
            assert "Braintrust" in body_text

    run_async(scenario)


def test_entry_screen_error_message_names_the_configured_source_not_always_langfuse(monkeypatch):
    monkeypatch.setenv(credentials.TRACE_SOURCE_KEY, "braintrust")

    def _raise():
        raise RuntimeError("BRAINTRUST_PROJECT_NAME not set")

    monkeypatch.setattr(bt, "default_project_name", _raise)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentScreen())
            await pilot.pause()
            assert app.screen.error is not None
            assert "BRAINTRUST_PROJECT_NAME" in app.screen.error
            body_text = " ".join(str(label.render()) for label in app.screen.query(Label))
            assert "Braintrust isn't configured" in body_text
            assert "Langfuse isn't configured" not in body_text

    run_async(scenario)


def _bt_root(root_id, workflow_name):
    return {
        "span_id": root_id, "root_span_id": root_id, "is_root": True, "span_parents": None,
        "span_attributes": {"type": "task", "name": f"{workflow_name}.arun_stream"},
        "metadata": {"workflow_name": workflow_name}, "input": {}, "output": None, "metrics": {},
        "created": "2026-01-01T00:00:00Z",
    }


def _bt_llm(root_id, span_id, *, model="claude-opus-4-8"):
    return {
        "span_id": span_id, "root_span_id": root_id, "is_root": False, "span_parents": [root_id],
        "span_attributes": {"type": "llm", "name": "Anthropic.aresponse"},
        "metadata": {"model": model, "agent_name": "Real Agent"},
        "input": {"messages": [{"role": "system", "content": "You are a real reconstructed agent."}]},
        "output": None, "metrics": {"estimated_cost": 0.05, "prompt_tokens": 100, "completion_tokens": 20},
        "created": "2026-01-01T00:00:01Z",
    }


def _write_bt_trace(traces_dir, project_name, root_id, spans):
    import json as _json

    d = traces_dir / project_name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{root_id}.json").write_text(_json.dumps(spans), encoding="utf-8")


def test_pulling_braintrust_traces_reconstructs_a_real_config_no_langfuse_involved(tmp_path, monkeypatch):
    monkeypatch.setenv(credentials.TRACE_SOURCE_KEY, "braintrust")

    def _fake_pull(*, project_name, batch_size, traces_dir):
        _write_bt_trace(traces_dir, project_name, "r1", [_bt_root("r1", "homepilot-ticket-analysis"), _bt_llm("r1", "s1")])
        return ["r1"]

    monkeypatch.setattr(bt, "pull_traces", _fake_pull)

    def _fail_if_called(**kwargs):
        raise AssertionError("Braintrust pull dispatch must never call langfuse_client.pull_traces")

    monkeypatch.setattr(lf, "pull_traces", _fail_if_called)

    async def scenario():
        from tui.screens.add_environment import AddEnvironmentDoneScreen, AddEnvironmentGroupsScreen

        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PullTracesScreen(source="braintrust", project_id="homepilot", traces_dir=tmp_path, configs_dir=tmp_path))
            await _wait_until(pilot, lambda: isinstance(app.screen, AddEnvironmentGroupsScreen))
            assert app.screen.source == "braintrust"
            assert app.screen.summaries == [("homepilot-ticket-analysis", 1)]
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, AddEnvironmentDoneScreen)
            assert app.screen.config.provenance.source_agent_name == "homepilot-ticket-analysis"
            assert app.screen.config.supervisor().system_prompt == "You are a real reconstructed agent."
            body_text = " ".join(str(label.render()) for label in app.screen.query(Label))
            assert "Langfuse" not in body_text

    run_async(scenario)


def test_using_cached_braintrust_traces_reconstructs_directly(tmp_path, monkeypatch):
    monkeypatch.setenv(credentials.TRACE_SOURCE_KEY, "braintrust")
    monkeypatch.setattr(bt, "default_project_name", lambda: "homepilot")
    _write_bt_trace(tmp_path, "homepilot", "r1", [_bt_root("r1", "homepilot-ticket-analysis"), _bt_llm("r1", "s1")])

    async def scenario():
        from tui.screens.add_environment import AddEnvironmentGroupsScreen

        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentScreen(braintrust_traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            await pilot.press("enter")  # "Use 1 cached trace(s)" is first when cache exists
            await pilot.pause()
            assert isinstance(app.screen, AddEnvironmentGroupsScreen)
            assert app.screen.source == "braintrust"

    run_async(scenario)


# --- pull screen -----------------------------------------------------------


def test_pull_screen_success_lands_on_groups_screen(tmp_path, monkeypatch):
    def _fake_pull(*, project_id, batch_size, traces_dir):
        _write_trace(traces_dir, project_id, "t1", agent_name="Invoice Generation Assistant")
        return ["t1"]

    monkeypatch.setattr(lf, "pull_traces", _fake_pull)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PullTracesScreen(source="langfuse", project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await _wait_until(pilot, lambda: isinstance(app.screen, AddEnvironmentGroupsScreen))

    run_async(scenario)


def test_pull_screen_failure_lands_on_error_screen(tmp_path, monkeypatch):
    def _fake_pull(*, project_id, batch_size, traces_dir):
        raise RuntimeError("missing required credentials: LANGFUSE_SECRET_KEY")

    monkeypatch.setattr(lf, "pull_traces", _fake_pull)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PullTracesScreen(source="langfuse", project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await _wait_until(pilot, lambda: isinstance(app.screen, PullErrorScreen))
            assert "LANGFUSE_SECRET_KEY" in app.screen.message

    run_async(scenario)


def test_pull_screen_surfaces_clear_message_when_braintrust_rate_limit_persists(tmp_path, monkeypatch):
    # Regression: a real, persistent 429 from Braintrust used to reach
    # this screen as a bare requests.HTTPError -- ingestion.braintrust_
    # client._query_btql now retries transiently and only raises
    # BraintrustRateLimitError once its retry budget is exhausted, with a
    # clear, specific message. This confirms that message reaches the
    # user through PullTracesScreen's existing exception handling, not
    # just that the underlying client function behaves correctly in
    # isolation (see tests/test_ingestion_braintrust_client.py for that).
    def _fake_pull(*, project_name, batch_size, traces_dir):
        raise bt.BraintrustRateLimitError("Braintrust API is rate limited (HTTP 429) — still failing after 5 attempts, retry in 5s.")

    monkeypatch.setattr(bt, "pull_traces", _fake_pull)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PullTracesScreen(source="braintrust", project_id="homepilot", traces_dir=tmp_path, configs_dir=tmp_path))
            await _wait_until(pilot, lambda: isinstance(app.screen, PullErrorScreen))
            assert "rate limited" in app.screen.message
            assert "retry in 5s" in app.screen.message

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
            app.push_screen(AddEnvironmentGroupsScreen(source="langfuse", project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
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
            app.push_screen(AddEnvironmentGroupsScreen(source="langfuse", project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            menu = app.screen.query_one("#group-list", ListView)
            label = str(menu.children[0].query_one(Label).render())
            assert "no name tag" in label

    run_async(scenario)


def test_selecting_group_with_spaces_in_name_reconstructs_and_saves(tmp_path):
    # Regression: agent_name values contain spaces ("Invoice Generation
    # Assistant") which are not valid Textual widget ids -- items must be
    # keyed positionally, not by embedding the name in the id directly.
    _write_trace(tmp_path, "proj-1", "a1", agent_name="Invoice Generation Assistant", observations=[_TOOL_CALL_OBS])

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentGroupsScreen(source="langfuse", project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, AddEnvironmentDoneScreen)
            assert app.screen.config.provenance.source_agent_name == "Invoice Generation Assistant"
            assert app.screen.config.provenance.trace_count == 1
            assert "send_invoice" in app.screen.config.supervisor().tools
            assert app.screen.config_hash in list_config_hashes(configs_dir=tmp_path)

    run_async(scenario)


def test_done_screen_reports_what_is_on_disk_not_what_was_reconstructed(tmp_path):
    """Regression: this screen used to render the in-memory reconstruction
    it was handed, so a save that didn't write reported the unsaved
    config's numbers as if they'd been persisted. It now takes only a
    hash — mutating the saved file behind it must change what it says."""
    from target_system.config import load_config, save_config
    from ingestion.reconstruct import reconstruct_from_cache

    _write_trace(tmp_path, "proj-1", "a1", agent_name="Invoice Generation Assistant", observations=[_TOOL_CALL_OBS])
    config = reconstruct_from_cache(project_id="proj-1", agent_name="Invoice Generation Assistant", traces_dir=tmp_path)
    config_hash = save_config(config, configs_dir=tmp_path)
    assert config.provenance.trace_count == 1

    saved = json.loads((tmp_path / f"{config_hash}.json").read_text(encoding="utf-8"))
    saved["provenance"]["trace_count"] = 9999
    (tmp_path / f"{config_hash}.json").write_text(json.dumps(saved), encoding="utf-8")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentDoneScreen(config_hash, configs_dir=tmp_path))
            await pilot.pause()
            assert app.screen.config.provenance.trace_count == 9999
            assert load_config(config_hash, configs_dir=tmp_path).provenance.trace_count == 9999
            rendered = " ".join(str(label.render()) for label in app.screen.query(Label))
            assert "9999 trace(s)" in rendered
            assert "1 trace(s)" not in rendered

    run_async(scenario)


def test_done_screen_surfaces_other_groups_found(tmp_path):
    _write_trace(tmp_path, "proj-1", "a1", agent_name="Invoice Generation Assistant", observations=[_TOOL_CALL_OBS])
    _write_trace(tmp_path, "proj-1", "b1", agent_name="HR Onboarding Assistant")

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(AddEnvironmentGroupsScreen(source="langfuse", project_id="proj-1", traces_dir=tmp_path, configs_dir=tmp_path))
            await pilot.pause()
            await pilot.press("enter")  # picks the first (highest trace count) group
            await pilot.pause()
            assert isinstance(app.screen, AddEnvironmentDoneScreen)
            other_names = {g.agent_name for g in app.screen.config.provenance.other_groups_found}
            assert "HR Onboarding Assistant" in other_names

    run_async(scenario)
