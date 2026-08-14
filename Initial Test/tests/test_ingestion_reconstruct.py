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


def test_reconstruct_system_config_hash_ignores_extraction_time():
    """Two reconstructions of the same traces from the same source are the
    same environment and must dedupe, even though each carries its own
    extraction_date."""
    from target_system.config import compute_config_hash

    traces_a = [_trace("A", [_generation("t1", input_=[{"role": "user", "content": "x"}])])]
    config_a = reconstruct_system_config(traces_a, label="a", project_id="proj-1", source_agent_name="A")
    config_b = reconstruct_system_config(traces_a, label="different label", project_id="proj-1", source_agent_name="A")
    assert config_a.provenance.extraction_date != config_b.provenance.extraction_date
    assert compute_config_hash(config_a) == compute_config_hash(config_b)


def test_reconstruct_system_config_hash_separates_source_projects():
    """Same agent name, different source project, is a different
    environment — it must not inherit the other project's id."""
    from target_system.config import compute_config_hash

    traces_a = [_trace("A", [_generation("t1", input_=[{"role": "user", "content": "x"}])])]
    config_a = reconstruct_system_config(traces_a, label="a", project_id="proj-1", source_agent_name="A")
    config_b = reconstruct_system_config(traces_a, label="a", project_id="proj-2", source_agent_name="A")
    assert compute_config_hash(config_a) != compute_config_hash(config_b)


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


# --- distinct identity per pull ---------------------------------------------


def _write_cached_traces(proj_dir, count, *, start=0):
    """`count` cached traces for agent "A", all calling the same one tool
    so the tool-name list (which the hash has always covered) is identical
    between batches — the pulls differ only in how much evidence sits
    behind that tool, which is exactly the case that used to collide."""
    import json

    proj_dir.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + count):
        trace = _trace("A", [_tool_span("get_order_status", {"order_id": f"ORD-{i:05d}"}, {"ok": True})], trace_id=f"t{i:05d}")
        (proj_dir / f"t{i:05d}.json").write_text(json.dumps(trace), encoding="utf-8")


def test_re_pull_with_more_traces_saves_a_distinct_config(tmp_path):
    """Regression: a second pull of the same agent with more traces used to
    hash to the first pull's id, so save_config's existence check skipped
    the write and the richer reconstruction was silently discarded while
    the UI reported it saved. Both pulls must now persist as their own
    retrievable config, neither overwriting the other."""
    from target_system.config import compute_config_hash, load_config, save_config

    traces_dir = tmp_path / "traces"
    configs_dir = tmp_path / "configs"
    proj_dir = traces_dir / "proj-1"

    _write_cached_traces(proj_dir, 100)
    first = reconstruct_from_cache(project_id="proj-1", agent_name="A", traces_dir=traces_dir)
    first_hash = save_config(first, configs_dir=configs_dir)

    _write_cached_traces(proj_dir, 100, start=100)  # same cache, now 200 traces deep
    second = reconstruct_from_cache(project_id="proj-1", agent_name="A", traces_dir=traces_dir)
    second_hash = save_config(second, configs_dir=configs_dir)

    assert first.provenance.trace_count == 100
    assert second.provenance.trace_count == 200
    assert first.agents[0].tools == second.agents[0].tools  # the old hash saw only this, and collided
    assert first_hash != second_hash

    assert {p.stem for p in configs_dir.glob("cfg_*.json")} == {first_hash, second_hash}
    reloaded_first = load_config(first_hash, configs_dir=configs_dir)
    reloaded_second = load_config(second_hash, configs_dir=configs_dir)
    assert reloaded_first.provenance.trace_count == 100
    assert reloaded_second.provenance.trace_count == 200
    assert compute_config_hash(reloaded_first) == first_hash
    assert compute_config_hash(reloaded_second) == second_hash


def test_re_pull_of_identical_traces_still_dedupes(tmp_path):
    """The other half of the contract: distinct identity must come from the
    trace data, not from the fact that a save happened twice. Re-saving an
    unchanged pull stays a one-file no-op."""
    from target_system.config import save_config

    traces_dir = tmp_path / "traces"
    configs_dir = tmp_path / "configs"
    _write_cached_traces(traces_dir / "proj-1", 20)

    first = reconstruct_from_cache(project_id="proj-1", agent_name="A", traces_dir=traces_dir)
    second = reconstruct_from_cache(project_id="proj-1", agent_name="A", traces_dir=traces_dir)
    assert first.provenance.extraction_date != second.provenance.extraction_date

    assert save_config(first, configs_dir=configs_dir) == save_config(second, configs_dir=configs_dir)
    assert len(list(configs_dir.glob("cfg_*.json"))) == 1


# --- system prompt extraction -------------------------------------------


def _agent_obs(agent, role, tools_available=(), *, name=None):
    return {
        "type": "AGENT",
        "name": name or agent,
        "startTime": "t",
        "metadata": {"agent": agent, "agent_role": role, "tools_available": list(tools_available)},
    }


def _agent_generation(agent, system_prompt, *, name="gen", model="claude-x"):
    return {
        "type": "GENERATION",
        "name": name,
        "model": model,
        "startTime": "t",
        "metadata": {"agent": agent},
        "input": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "hi"}],
        "output": "ok",
    }


def test_system_prompt_is_extracted_when_traces_carry_it():
    """The bug this replaced: the source was hardcoded "unavailable", so a
    project whose traces do carry the text was still reported as having run
    with no system prompt."""
    traces = [_trace("A", [_generation("g", input_=[{"role": "system", "content": "BEHAVE WELL"}, {"role": "user", "content": "hi"}], output="ok")])]
    config = reconstruct_system_config(traces, label="a", project_id="p", source_agent_name="A")
    assert config.agents[0].system_prompt == "BEHAVE WELL"
    assert config.agents[0].system_prompt_source == "observed"


def test_system_prompt_unavailable_when_traces_really_lack_it():
    traces = [_trace("A", [_generation("g", input_=[{"role": "user", "content": "hi"}], output="ok")])]
    config = reconstruct_system_config(traces, label="a", project_id="p", source_agent_name="A")
    assert config.agents[0].system_prompt == UNAVAILABLE_SYSTEM_PROMPT
    assert config.agents[0].system_prompt_source == "unavailable"


def test_system_prompt_accepts_content_block_list():
    traces = [_trace("A", [_generation("g", input_=[{"role": "system", "content": [{"type": "text", "text": "BLOCK PROMPT"}]}], output="ok")])]
    config = reconstruct_system_config(traces, label="a", project_id="p", source_agent_name="A")
    assert config.agents[0].system_prompt == "BLOCK PROMPT"
    assert config.agents[0].system_prompt_source == "observed"


def test_judge_generation_is_not_a_system_prompt_source():
    traces = [_trace("A", [
        {"type": "GENERATION", "name": JUDGE_OBSERVATION_NAME, "model": "m", "startTime": "t",
         "input": [{"role": "system", "content": "JUDGE RUBRIC"}], "output": "ok"},
        _generation("g", input_=[{"role": "user", "content": "hi"}], output="ok"),
    ])]
    config = reconstruct_system_config(traces, label="a", project_id="p", source_agent_name="A")
    assert config.agents[0].system_prompt_source == "unavailable"


def test_prompt_drift_uses_most_common_and_warns():
    traces = [
        _trace("A", [_generation("g", input_=[{"role": "system", "content": "NEW"}], output="ok")]),
        _trace("A", [_generation("g", input_=[{"role": "system", "content": "NEW"}], output="ok")]),
        _trace("A", [_generation("g", input_=[{"role": "system", "content": "OLD"}], output="ok")]),
    ]
    config = reconstruct_system_config(traces, label="a", project_id="p", source_agent_name="A")
    assert config.agents[0].system_prompt == "NEW"
    assert any("distinct system prompts" in w for w in config.provenance.warnings)


# --- multi-agent decomposition ------------------------------------------


def test_nested_traces_reconstruct_one_agent_per_observed_agent():
    observations = [
        _agent_obs("sup", "supervisor", ["notify"]),
        _agent_obs("looker", "specialist", ["read_it"]),
        _agent_generation("sup", "SUPERVISOR PROMPT"),
        _agent_generation("looker", "LOOKUP PROMPT"),
    ]
    config = reconstruct_system_config([_trace("A", observations)], label="a", project_id="p", source_agent_name="A")
    by_name = {a.name: a for a in config.agents}
    assert set(by_name) == {"sup", "looker"}
    assert by_name["sup"].role == "supervisor"
    assert by_name["sup"].system_prompt == "SUPERVISOR PROMPT"
    assert by_name["looker"].system_prompt == "LOOKUP PROMPT"
    assert by_name["looker"].tools == ["read_it"]
    assert config.supervisor().name == "sup"


def test_flat_traces_keep_the_single_agent_shape():
    """Backwards compatibility: traces with no per-agent metadata must
    reconstruct exactly as before, one supervisor named for the group."""
    traces = [_trace("A", [_generation("g", input_=[{"role": "user", "content": "hi"}], output="ok")])]
    config = reconstruct_system_config(traces, label="a", project_id="p", source_agent_name="A")
    assert len(config.agents) == 1
    assert config.agents[0].role == "supervisor"
    assert config.agents[0].name == "A"


def test_shared_roles_are_disambiguated_so_member_ids_stay_unique():
    observations = [
        _agent_obs("sup", "supervisor"),
        _agent_obs("a1", "specialist"),
        _agent_obs("a2", "specialist"),
        _agent_generation("sup", "S"),
        _agent_generation("a1", "P1"),
        _agent_generation("a2", "P2"),
    ]
    config = reconstruct_system_config([_trace("A", observations)], label="a", project_id="p", source_agent_name="A")
    member_roles = [m.role for m in config.members()]
    assert len(set(member_roles)) == len(member_roles)
    assert any("distinct" in w or "suffixed" in w for w in config.provenance.warnings)


def test_missing_supervisor_role_is_designated_and_warned():
    observations = [
        _agent_obs("a1", "specialist"),
        _agent_obs("a2", "specialist"),
        _agent_generation("a1", "P1"),
        _agent_generation("a2", "P2"),
    ]
    traces = [_trace("A", observations), _trace("A", [_agent_obs("a1", "specialist"), _agent_generation("a1", "P1")], trace_id="t2")]
    config = reconstruct_system_config(traces, label="a", project_id="p", source_agent_name="A")
    assert config.supervisor().name == "a1"  # seen in more traces
    assert any("designated" in w for w in config.provenance.warnings)


def test_per_agent_tools_union_grant_and_use():
    observations = [
        _agent_obs("sup", "supervisor", ["granted_never_called"]),
        _agent_generation("sup", "S"),
        {"type": "SPAN", "name": "tool-call-actually_called", "startTime": "t",
         "metadata": {"agent": "sup"}, "input": {"tool": "actually_called", "inputs": {}}, "output": {"ok": True}},
    ]
    config = reconstruct_system_config([_trace("A", observations)], label="a", project_id="p", source_agent_name="A")
    assert config.agents[0].tools == ["actually_called", "granted_never_called"]
