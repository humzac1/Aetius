"""No live API calls here — trace fixtures shaped exactly like the real
Braintrust span data confirmed live against the "homepilot" account (see
ingestion/braintrust_reconstruct.py's module docstring for the
investigation): a trace is a flat list of spans sharing a root_span_id,
linked by is_root/span_parents, span_attributes["type"] in
{"task","llm","tool","score"}.
"""

from __future__ import annotations

import json

from ingestion.braintrust_reconstruct import (
    UNAVAILABLE_SYSTEM_PROMPT,
    GroupSummary,
    _build_cost_stats,
    _build_model_config,
    _extract_system_prompt,
    _extract_tool_call,
    _is_judge_span,
    group_traces,
    reconstruct_from_cache,
    reconstruct_system_config,
    summarize_groups,
)


def _root_span(root_id, *, workflow_name=None, workflow_id=None, name="wf.arun_stream", extra_metadata=None):
    metadata = {}
    if workflow_name is not None:
        metadata["workflow_name"] = workflow_name
    if workflow_id is not None:
        metadata["workflow_id"] = workflow_id
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "span_id": root_id,
        "root_span_id": root_id,
        "is_root": True,
        "span_parents": None,
        "span_attributes": {"type": "task", "name": name},
        "metadata": metadata,
        "input": {},
        "output": None,
        "metrics": {},
        "created": "2026-01-01T00:00:00Z",
    }


def _llm_span(root_id, span_id, *, model="claude-x", agent_name=None, messages=None, cost=None, prompt_tokens=None, completion_tokens=None, parent=None):
    metadata = {}
    if agent_name is not None:
        metadata["agent_name"] = agent_name
    if model is not None:
        metadata["model"] = model
    metrics = {}
    if cost is not None:
        metrics["estimated_cost"] = cost
    if prompt_tokens is not None:
        metrics["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        metrics["completion_tokens"] = completion_tokens
    return {
        "span_id": span_id,
        "root_span_id": root_id,
        "is_root": False,
        "span_parents": [parent or root_id],
        "span_attributes": {"type": "llm", "name": "Anthropic.aresponse"},
        "metadata": metadata,
        "input": {"messages": messages or []},
        "output": None,
        "metrics": metrics,
        "created": "2026-01-01T00:00:01Z",
    }


def _tool_span(root_id, span_id, tool_name, arguments, response_content, *, parent=None):
    return {
        "span_id": span_id,
        "root_span_id": root_id,
        "is_root": False,
        "span_parents": [parent or root_id],
        "span_attributes": {"type": "tool", "name": f"{tool_name}.aexecute"},
        "metadata": {},
        "input": arguments,
        "output": {"result": {"content": response_content}, "status": "success"},
        "metrics": {},
        "created": "2026-01-01T00:00:02Z",
    }


def _score_span(root_id, span_id, name="Tool Correctness", *, parent=None):
    return {
        "span_id": span_id,
        "root_span_id": root_id,
        "is_root": False,
        "span_parents": [parent or root_id],
        "span_attributes": {"type": "score", "name": name},
        "metadata": {},
        "input": {},
        "output": None,
        "metrics": {},
        "created": "2026-01-01T00:00:03Z",
    }


def _task_span(root_id, span_id, name, *, agent_name=None, model=None, cost=None, parent=None):
    metadata = {}
    if agent_name is not None:
        metadata["agent_name"] = agent_name
    if model is not None:
        metadata["model"] = model
    metrics = {"estimated_cost": cost} if cost is not None else {}
    return {
        "span_id": span_id,
        "root_span_id": root_id,
        "is_root": False,
        "span_parents": [parent or root_id],
        "span_attributes": {"type": "task", "name": name},
        "metadata": metadata,
        "input": {},
        "output": None,
        "metrics": metrics,
        "created": "2026-01-01T00:00:00.5Z",
    }


# --- grouping -----------------------------------------------------------


def test_group_traces_clusters_by_root_workflow_name():
    traces = [
        [_root_span("r1", workflow_name="homepilot-ticket-analysis")],
        [_root_span("r2", workflow_name="homepilot-ticket-analysis")],
        [_root_span("r3", workflow_name="Issue Classification Workflow")],
    ]
    groups = group_traces(traces)
    assert set(groups.keys()) == {"homepilot-ticket-analysis", "Issue Classification Workflow"}
    assert len(groups["homepilot-ticket-analysis"]) == 2


def test_group_traces_falls_back_to_workflow_id_when_no_workflow_name():
    traces = [[_root_span("r1", workflow_id="issue-classification")]]
    groups = group_traces(traces)
    assert "issue-classification" in groups


def test_group_traces_falls_back_to_root_span_name_when_no_workflow_metadata():
    # regression: confirmed real case ("wade-workorder-recommend") — a
    # real system with no workflow_name/workflow_id convention at all,
    # just domain-specific metadata (ticket_id, brand, etc.)
    traces = [[_root_span("r1", name="wade-workorder-recommend", extra_metadata={"ticket_id": "123", "brand": "haven"})]]
    groups = group_traces(traces)
    assert "wade-workorder-recommend" in groups


def test_summarize_groups_sorted_by_trace_count_descending():
    groups = {"small": [[_root_span("r1", workflow_name="small")]], "big": [[_root_span(f"r{i}", workflow_name="big")] for i in range(5)]}
    summaries = summarize_groups(groups)
    assert [s.workflow_name for s in summaries] == ["big", "small"]
    assert summaries[0].trace_count == 5


# --- judge/score exclusion ----------------------------------------------


def test_is_judge_span_true_for_score_type():
    assert _is_judge_span(_score_span("r1", "s1"), {}) is True


def test_is_judge_span_true_for_agent_name_on_the_span_itself():
    span = _llm_span("r1", "s1", agent_name="Staci 2.0 Action-Aware Result Judge")
    assert _is_judge_span(span, {"s1": span}) is True


def test_is_judge_span_false_for_ordinary_agent():
    span = _llm_span("r1", "s1", agent_name="Staci 2.0 Bungalow Tenant Agent")
    assert _is_judge_span(span, {"s1": span}) is False


def test_is_judge_span_resolves_agent_name_from_parent_task_span():
    # regression: a real 15-trace end-to-end pull showed agent_name is
    # NOT reliably present on llm spans themselves (all None in that
    # batch) -- it lives on the parent task span. Without walking up,
    # this silently let a judge's system prompt through as if it were the
    # real agent's.
    root = _root_span("r1", workflow_name="wf")
    judge_task = _task_span("r1", "t1", "Staci 2.0 Judge.arun", agent_name="Staci 2.0 Judge")
    llm = _llm_span("r1", "s1", agent_name=None, parent="t1")  # no agent_name on the leaf itself
    by_id = {s["span_id"]: s for s in [root, judge_task, llm]}
    assert _is_judge_span(llm, by_id) is True


def test_is_judge_span_false_when_parent_chain_has_no_judge_agent():
    root = _root_span("r1", workflow_name="wf")
    agent_task = _task_span("r1", "t1", "Staci 2.0 Bungalow Tenant Agent.arun", agent_name="Staci 2.0 Bungalow Tenant Agent")
    llm = _llm_span("r1", "s1", agent_name=None, parent="t1")
    by_id = {s["span_id"]: s for s in [root, agent_task, llm]}
    assert _is_judge_span(llm, by_id) is False


# --- tool call extraction (real confirmed shape) -------------------------


def test_extract_tool_call_flat_input_and_json_encoded_output_content():
    span = _tool_span("r1", "s1", "get_tenant_current_balance", {"occupancy_id": 11643}, json.dumps({"success": True, "current_balance": 1575.51}))
    result = _extract_tool_call(span)
    assert result == ("get_tenant_current_balance", {"occupancy_id": 11643}, {"success": True, "current_balance": 1575.51})


def test_extract_tool_call_strips_aexecute_suffix():
    span = _tool_span("r1", "s1", "list_ticket_actions", {}, json.dumps({"actions": []}))
    name, _args, _resp = _extract_tool_call(span)
    assert name == "list_ticket_actions"
    assert not name.endswith(".aexecute")


def test_extract_tool_call_falls_back_to_raw_string_when_content_not_json():
    span = _tool_span("r1", "s1", "weird_tool", {}, "not valid json {")
    _name, _args, response = _extract_tool_call(span)
    assert response == "not valid json {"


def test_extract_tool_call_returns_none_for_non_tool_span():
    assert _extract_tool_call(_llm_span("r1", "s1")) is None


# --- model detection -------------------------------------------------------


def test_build_model_config_picks_dominant_model_excluding_judge():
    traces = [
        [
            _root_span("r1", workflow_name="wf"),
            _llm_span("r1", "s1", model="claude-opus-4-8", agent_name="Staci 2.0 Bungalow Tenant Agent"),
            _llm_span("r1", "s2", model="gpt-5.6-luna", agent_name="Staci 2.0 Judge"),
        ]
    ]
    warnings = []
    model = _build_model_config(traces, warnings=warnings)
    assert model.provider == "anthropic"
    assert model.model_name == "claude-opus-4-8"
    assert not warnings  # only one non-judge model observed -- no drift warning


def test_build_model_config_warns_on_multiple_non_judge_models():
    traces = [
        [
            _root_span("r1", workflow_name="wf"),
            _llm_span("r1", "s1", model="claude-opus-4-8"),
            _llm_span("r1", "s2", model="claude-sonnet-5"),
        ]
    ]
    warnings = []
    _build_model_config(traces, warnings=warnings)
    assert any("multiple model names" in w for w in warnings)


# --- cost aggregation: the confirmed double-counting regression ----------


def test_cost_stats_sum_only_llm_spans_never_task_spans():
    # regression: confirmed live that a task span's estimated_cost can be
    # bit-for-bit identical to its single llm child's -- summing both
    # would double it. Only the llm span's cost may ever be counted.
    traces = [
        [
            _root_span("r1", workflow_name="wf"),
            _task_span("r1", "t1", "Agent.arun", cost=0.294),  # mirrors the llm child's cost -- must be ignored
            _llm_span("r1", "s1", cost=0.294, prompt_tokens=15100, completion_tokens=2770, parent="t1"),
        ]
    ]
    stats = _build_cost_stats(traces)
    assert stats.avg_cost_usd_per_trace == 0.294  # not 0.588


def test_cost_stats_excludes_judge_spans():
    traces = [
        [
            _root_span("r1", workflow_name="wf"),
            _llm_span("r1", "s1", cost=0.10, agent_name="Real Agent"),
            _llm_span("r1", "s2", cost=999.0, agent_name="Judge"),
        ]
    ]
    stats = _build_cost_stats(traces)
    assert stats.avg_cost_usd_per_trace == 0.10
    assert stats.avg_generations_per_trace == 1  # judge generation not counted


def test_cost_stats_none_when_no_traces():
    stats = _build_cost_stats([])
    assert stats.avg_cost_usd_per_trace is None


# --- system prompt ---------------------------------------------------------


def test_extract_system_prompt_finds_real_system_message():
    traces = [
        [
            _root_span("r1", workflow_name="wf"),
            _llm_span("r1", "s1", messages=[{"role": "system", "content": "You are Staci."}, {"role": "user", "content": "hi"}]),
        ]
    ]
    prompt, source = _extract_system_prompt(traces)
    assert prompt == "You are Staci."
    assert source == "observed"


def test_extract_system_prompt_ignores_judge_spans():
    traces = [
        [
            _root_span("r1", workflow_name="wf"),
            _llm_span("r1", "s1", agent_name="Judge", messages=[{"role": "system", "content": "You are the judge."}]),
        ]
    ]
    prompt, source = _extract_system_prompt(traces)
    assert source == "unavailable"
    assert prompt == UNAVAILABLE_SYSTEM_PROMPT


def test_extract_system_prompt_unavailable_when_none_found():
    traces = [[_root_span("r1", workflow_name="wf"), _llm_span("r1", "s1", messages=[{"role": "user", "content": "hi"}])]]
    prompt, source = _extract_system_prompt(traces)
    assert source == "unavailable"
    assert prompt == UNAVAILABLE_SYSTEM_PROMPT


# --- end-to-end reconstruction ----------------------------------------------


def test_reconstruct_system_config_end_to_end():
    traces = [
        [
            _root_span("r1", workflow_name="homepilot-ticket-analysis"),
            _llm_span(
                "r1", "s1", model="claude-opus-4-8", agent_name="Staci 2.0 Bungalow Tenant Agent",
                messages=[{"role": "system", "content": "You are Staci."}], cost=0.294, prompt_tokens=15100, completion_tokens=2770,
            ),
            _tool_span("r1", "s2", "get_tenant_current_balance", {"occupancy_id": 11643}, json.dumps({"success": True})),
            _score_span("r1", "s3"),
        ]
    ]
    config = reconstruct_system_config(traces, label="test-env", project_id="homepilot", source_workflow_name="homepilot-ticket-analysis")
    agent = config.supervisor()
    assert agent.system_prompt == "You are Staci."
    assert agent.system_prompt_source == "observed"
    assert agent.tools == ["get_tenant_current_balance"]
    assert config.model.provider == "anthropic"
    assert config.model.model_name == "claude-opus-4-8"
    assert config.provenance.trace_count == 1
    assert config.provenance.avg_cost_usd_per_trace == 0.294


def test_reconstruct_system_config_end_to_end_when_agent_name_only_on_parent_task_span():
    # regression: exactly the real bug a live end-to-end validation found
    # -- agent_name absent from the llm span itself, present only on its
    # parent task span. Without the parent-walk fix, the judge's llm span
    # (and its system prompt) would be mistaken for the real agent's.
    traces = [
        [
            _root_span("r1", workflow_name="homepilot-ticket-analysis"),
            _task_span("r1", "t-agent", "Staci 2.0 Bungalow Tenant Agent.arun", agent_name="Staci 2.0 Bungalow Tenant Agent"),
            _llm_span(
                "r1", "s-agent", model="claude-opus-4-8", agent_name=None, parent="t-agent",
                messages=[{"role": "system", "content": "You are the real Staci agent."}], cost=0.294,
            ),
            _task_span("r1", "t-judge", "Staci 2.0 Judge.arun", agent_name="Staci 2.0 Judge"),
            _llm_span(
                "r1", "s-judge", model="gpt-5.6-luna", agent_name=None, parent="t-judge",
                messages=[{"role": "system", "content": "You are the judge, not the agent."}], cost=999.0,
            ),
        ]
    ]
    config = reconstruct_system_config(traces, label="test-env", project_id="homepilot", source_workflow_name="homepilot-ticket-analysis")
    agent = config.supervisor()
    assert agent.system_prompt == "You are the real Staci agent."
    assert config.model.model_name == "claude-opus-4-8"  # not gpt-5.6-luna, the judge's model
    assert config.provenance.avg_cost_usd_per_trace == 0.294  # not 999.294 -- judge cost excluded


def test_reconstruct_from_cache_raises_clearly_on_unknown_group(tmp_path, monkeypatch):
    import ingestion.braintrust_reconstruct as mod

    monkeypatch.setattr(mod, "load_cached_traces", lambda project_id, *, traces_dir: [[_root_span("r1", workflow_name="known")]])
    try:
        reconstruct_from_cache(project_id="homepilot", workflow_name="unknown", traces_dir=tmp_path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "known" in str(exc)
