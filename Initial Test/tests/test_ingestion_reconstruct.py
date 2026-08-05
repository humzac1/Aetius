from ingestion.reconstruct import (
    JUDGE_OBSERVATION_NAME,
    UNAVAILABLE_SYSTEM_PROMPT,
    GroupSummary,
    _build_tool_profiles,
    _extract_tool_call,
    group_traces,
    reconstruct_from_cache,
    reconstruct_system_config,
    summarize_groups,
)


def _generation(name, model="claude-x", input_=None, output=None):
    return {"type": "GENERATION", "name": name, "model": model, "input": input_, "output": output, "startTime": "t"}


def _tool_span(tool_name, arguments, response, *, style="structured"):
    if style == "structured":
        return {"type": "SPAN", "name": f"call-{tool_name}", "input": {"tool": tool_name, "inputs": arguments}, "output": response, "startTime": "t"}
    return {"type": "SPAN", "name": f"tool-call-{tool_name}", "input": arguments, "output": response, "startTime": "t"}


def _trace(agent_name, observations, *, trace_id="t1"):
    return {"id": trace_id, "metadata": {"agent_name": agent_name} if agent_name is not None else {}, "observations": observations}


# --- grouping -----------------------------------------------------------


def test_group_traces_clusters_by_agent_name():
    traces = [_trace("A", []), _trace("A", []), _trace("B", [])]
    groups = group_traces(traces)
    assert set(groups.keys()) == {"A", "B"}
    assert len(groups["A"]) == 2
    assert len(groups["B"]) == 1


def test_group_traces_excludes_configured_noise_names():
    traces = [_trace("A", []), _trace("test", []), _trace("test-fix", [])]
    groups = group_traces(traces)
    assert set(groups.keys()) == {"A"}


def test_group_traces_custom_exclude_set():
    traces = [_trace("A", []), _trace("staging", [])]
    groups = group_traces(traces, exclude_names=frozenset({"staging"}))
    assert set(groups.keys()) == {"A"}


def test_group_traces_keeps_missing_agent_name_as_its_own_group():
    traces = [_trace(None, []), _trace(None, [])]
    groups = group_traces(traces)
    assert None in groups
    assert len(groups[None]) == 2


def test_summarize_groups_sorted_by_trace_count_descending():
    groups = {"small": [_trace("small", [])], "big": [_trace("big", [])] * 5}
    summaries = summarize_groups(groups)
    assert [s.agent_name for s in summaries] == ["big", "small"]
    assert summaries[0].trace_count == 5


def test_summarize_groups_marks_noise_group():
    groups = {"A": [_trace("A", [])], "test": [_trace("test", [])]}
    summaries = {s.agent_name: s for s in summarize_groups(groups)}
    assert summaries["test"].is_noise is True
    assert summaries["A"].is_noise is False


# --- tool call extraction -------------------------------------------------


def test_extract_tool_call_structured_shape():
    obs = _tool_span("send_x", {"a": 1}, {"result": {"ok": True}})
    result = _extract_tool_call(obs)
    assert result == ("send_x", {"a": 1}, {"result": {"ok": True}})


def test_extract_tool_call_name_prefix_fallback():
    obs = _tool_span("send_x", {"a": 1}, {"ok": True}, style="prefixed")
    result = _extract_tool_call(obs)
    assert result[0] == "send_x"
    assert result[2] == {"ok": True}


def test_extract_tool_call_returns_none_for_non_tool_span():
    obs = {"type": "SPAN", "name": "simulation-session", "input": None, "output": {"x": 1}}
    assert _extract_tool_call(obs) is None


def test_extract_tool_call_returns_none_for_generation():
    obs = _generation("agent-turn-1", input_=[{"role": "user", "content": "hi"}])
    assert _extract_tool_call(obs) is None


# --- tool profile building -------------------------------------------------


def test_build_tool_profiles_aggregates_across_traces():
    traces = [
        _trace("A", [_tool_span("lookup_customer", {"customer_id": "CUST-1"}, {"result": {"name": "Bob"}, "success": True})]),
        _trace("A", [_tool_span("lookup_customer", {"customer_id": "CUST-2"}, {"result": {"name": "Amy"}, "success": True})]),
    ]
    profiles = _build_tool_profiles(traces)
    assert set(profiles.keys()) == {"lookup_customer"}
    profile = profiles["lookup_customer"]
    assert profile.n_calls_observed == 2
    assert profile.argument_profiles["customer_id"].distinct_value_count == 2
    assert set(profile.response_key_set) == {"result", "success"}
    assert len(profile.example_calls) == 2


def test_build_tool_profiles_excludes_judge_evaluation():
    traces = [_trace("A", [{"type": "GENERATION", "name": "judge-evaluation", "input": {"tool": "lookup_customer", "inputs": {}}, "output": {}}])]
    # even if a judge observation happened to look tool-call-shaped, it must not be picked up
    profiles = _build_tool_profiles(traces)
    assert profiles == {}


def test_build_tool_profiles_numeric_and_string_ranges():
    traces = [
        _trace("A", [_tool_span("calc", {"hours": 5, "note": "ab"}, {"total": 5})]),
        _trace("A", [_tool_span("calc", {"hours": 15, "note": "abcd"}, {"total": 15})]),
    ]
    profile = _build_tool_profiles(traces)["calc"]
    assert profile.argument_profiles["hours"].numeric_range == (5, 15)
    assert profile.argument_profiles["note"].string_length_range == (2, 4)


# --- full reconstruction ---------------------------------------------------


def test_reconstruct_system_config_single_tool_no_drift():
    traces = [
        _trace(
            "Billing Bot",
            [
                _generation("agent-turn-1", model="claude-x", input_=[{"role": "user", "content": "hi"}], output="ok"),
                _tool_span("send_invoice", {"invoice_id": "I-1"}, {"result": {}, "success": True}),
            ],
        )
    ]
    config = reconstruct_system_config(traces, label="billing-bot", project_id="proj-1", source_agent_name="Billing Bot")
    assert config.model.model_name == "claude-x"
    assert config.model.provider == "anthropic"
    agent = config.agents[0]
    assert agent.role == "supervisor"
    assert agent.tools == ["send_invoice"]
    assert agent.system_prompt == UNAVAILABLE_SYSTEM_PROMPT
    assert agent.system_prompt_source == "unavailable"
    assert config.provenance.trace_count == 1
    assert config.provenance.project_id == "proj-1"
    assert config.provenance.warnings == []


def test_reconstruct_system_config_zero_tools():
    traces = [_trace("Chat Bot", [_generation("agent-turn-1", input_=[{"role": "user", "content": "hi"}], output="ok")])]
    config = reconstruct_system_config(traces, label="chat-bot", project_id="proj-1", source_agent_name="Chat Bot")
    assert config.agents[0].tools == []
    assert config.provenance.tool_profiles == {}


def test_reconstruct_system_config_warns_on_model_drift():
    traces = [
        _trace("A", [_generation("t1", model="claude-old", input_=[{"role": "user", "content": "hi"}], output="ok")]),
        _trace("A", [_generation("t1", model="claude-new", input_=[{"role": "user", "content": "hi"}], output="ok")]),
        _trace("A", [_generation("t1", model="claude-new", input_=[{"role": "user", "content": "hi"}], output="ok")]),
    ]
    config = reconstruct_system_config(traces, label="a", project_id="proj-1", source_agent_name="A")
    assert config.model.model_name == "claude-new"  # most common wins
    assert len(config.provenance.warnings) == 1
    assert "claude-old" in config.provenance.warnings[0]
    assert "claude-new" in config.provenance.warnings[0]


def _generation_with_cost(name, prompt_tokens, completion_tokens, cost, model="claude-x"):
    return {
        "type": "GENERATION", "name": name, "model": model,
        "input": [{"role": "user", "content": "hi"}], "output": "ok",
        "promptTokens": prompt_tokens, "completionTokens": completion_tokens, "calculatedTotalCost": cost,
    }


def test_reconstruct_system_config_computes_real_cost_stats():
    traces = [
        _trace("A", [_generation_with_cost("t1", 100, 20, 0.01), _generation_with_cost("t2", 50, 10, 0.005)]),
        _trace("A", [_generation_with_cost("t1", 200, 40, 0.02)]),
    ]
    config = reconstruct_system_config(traces, label="a", project_id="proj-1", source_agent_name="A")
    p = config.provenance
    assert p.avg_generations_per_trace == 1.5  # (2 + 1) / 2 traces
    assert p.avg_prompt_tokens_per_generation == (100 + 50 + 200) / 3
    assert p.avg_completion_tokens_per_generation == (20 + 10 + 40) / 3
    assert p.avg_cost_usd_per_trace == (0.015 + 0.02) / 2  # per-trace totals, then averaged


def test_reconstruct_system_config_excludes_judge_evaluation_from_cost_stats():
    traces = [
        _trace(
            "A",
            [
                _generation_with_cost("agent-turn-1", 100, 20, 0.01),
                _generation_with_cost(JUDGE_OBSERVATION_NAME, 5000, 1000, 5.0),  # would dominate the average if included
            ],
        )
    ]
    config = reconstruct_system_config(traces, label="a", project_id="proj-1", source_agent_name="A")
    p = config.provenance
    assert p.avg_generations_per_trace == 1
    assert p.avg_prompt_tokens_per_generation == 100
    assert p.avg_cost_usd_per_trace == 0.01


def test_reconstruct_system_config_cost_stats_none_when_no_cost_data():
    traces = [_trace("A", [_generation("t1", input_=[{"role": "user", "content": "hi"}], output="ok")])]
    config = reconstruct_system_config(traces, label="a", project_id="proj-1", source_agent_name="A")
    p = config.provenance
    assert p.avg_prompt_tokens_per_generation is None
    assert p.avg_cost_usd_per_trace is None
    assert p.avg_generations_per_trace == 1  # generation count is still known even without token/cost fields


def test_reconstruct_system_config_records_other_groups_found():
    traces = [_trace("A", [_generation("t1", input_=[{"role": "user", "content": "x"}])])]
    config = reconstruct_system_config(
        traces,
        label="a",
        project_id="proj-1",
        source_agent_name="A",
        other_groups_found=[GroupSummary(agent_name="B", trace_count=5, is_noise=False), GroupSummary(agent_name="test", trace_count=2, is_noise=True)],
    )
    assert config.provenance.other_groups_found == [type(config.provenance.other_groups_found[0])(agent_name="B", trace_count=5)]


def test_reconstruct_system_config_hash_excludes_provenance():
    from target_system.config import compute_config_hash

    traces_a = [_trace("A", [_generation("t1", input_=[{"role": "user", "content": "x"}])])]
    config_a = reconstruct_system_config(traces_a, label="a", project_id="proj-1", source_agent_name="A")
    config_b = reconstruct_system_config(traces_a, label="a", project_id="proj-2", source_agent_name="A")  # different project_id -> different provenance
    assert compute_config_hash(config_a) == compute_config_hash(config_b)


# --- reconstruct_from_cache -------------------------------------------------


def test_reconstruct_from_cache_raises_for_unknown_group(tmp_path, monkeypatch):
    import json

    proj_dir = tmp_path / "proj-1"
    proj_dir.mkdir(parents=True)
    (proj_dir / "t1.json").write_text(json.dumps(_trace("A", [])), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="no cached traces"):
        reconstruct_from_cache(project_id="proj-1", agent_name="DoesNotExist", traces_dir=tmp_path)


def test_reconstruct_from_cache_builds_from_disk(tmp_path):
    import json

    proj_dir = tmp_path / "proj-1"
    proj_dir.mkdir(parents=True)
    (proj_dir / "t1.json").write_text(
        json.dumps(_trace("A", [_tool_span("send_invoice", {"x": 1}, {"ok": True})])), encoding="utf-8"
    )
    config = reconstruct_from_cache(project_id="proj-1", agent_name="A", traces_dir=tmp_path)
    assert config.label == "A"
    assert config.agents[0].tools == ["send_invoice"]
