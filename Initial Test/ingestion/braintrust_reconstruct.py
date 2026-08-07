"""Builds a SystemConfig-compatible reconstruction from a cached
Braintrust trace batch (ingestion/braintrust_client.py). Never re-hits
the API — reads whatever's already on disk. Parallel in shape to
ingestion/reconstruct.py (Langfuse), but every extraction rule below was
investigated fresh against this project's real "homepilot" account data,
not copied from Langfuse's shape — confirmed genuinely different in
several places, not assumed:

Grouping: Langfuse grouped by trace.metadata["agent_name"]; Braintrust
traces don't carry that at the root. Confirmed against a real 200-root
sample of the "homepilot" project: two of the three real systems present
tag their root span with metadata["workflow_name"] (and a matching
metadata["workflow_id"]) — "homepilot-ticket-analysis" (189/200) and
"Issue Classification Workflow" (6/200) — but a third real system
("wade-workorder-recommend", 5/200) uses no "workflow_name"/"workflow_id"
convention at all, just domain-specific metadata (ticket_id, brand,
etc.) with none of those keys. Braintrust's root span always has its own
span_attributes["name"] though (unlike Langfuse, where an untagged trace
really can carry no identifying tag at all), so the grouping key here is
workflow_name -> workflow_id -> root span_attributes["name"], in that
order — this basically never actually falls through to "no group" the
way Langfuse's sometimes did.

Judge/eval exclusion: confirmed two independent, real signals in this
data — span_attributes["type"] == "score" (Braintrust's own scorer
executions, a structural signal, not name-based) and
metadata["agent_name"] containing "judge" (this project's own
agent-naming convention). Getting the second one right took two passes,
not one: an initial single-example check saw agent_name duplicated onto
an llm span directly and assumed that was reliable; running the real
end-to-end validation (a 15-trace pull) surfaced a genuine bug from that
assumption — agent_name was None on every llm span in that batch, so the
judge filter silently passed judge spans through, and the reconstructed
system_prompt turned out to be the *judge's* prompt ("Independent
fresh-context judge for Staci 2.0 outputs..."), not the real agent's.
Confirmed by checking the parent chain: agent_name reliably lives on the
nearest task-type ancestor (e.g. "Staci 2.0 Judge", "Staci 2.0 Bungalow
Owner Agent"), not consistently on the llm/tool leaf spans themselves.
_resolve_agent_name walks span_parents to find it (checking the span's
own metadata first, so it still works on the rarer span that does carry
it directly); _is_judge_span uses that resolved name, not just the
span's own metadata.

Tool-call extraction: confirmed across 6 distinct real tools
(zendesk_list_user_tickets, get_tenant_current_balance,
get_open_tenant_charges, get_tenant_ledger, search_maintenance_tickets,
list_ticket_actions) that a tool-type span's `input` is the flat
keyword-argument dict directly (no {"tool", "inputs"} wrapper the way
Langfuse's was) and `output` is
{"result": {"content": "<json-encoded string>"}, "status": ...,
"updated_session_state": ...} — the real response is JSON-*encoded text*
inside output.result.content, not already a dict; parsed here with a
defensive fallback to the raw string if it isn't valid JSON. Tool name is
span_attributes["name"] with Agno's own ".aexecute" suffix stripped.

Cost/token aggregation: confirmed a real double-counting risk before
picking a rule, not assumed either way — pulled a real trace and found a
parent task span's metrics.estimated_cost bit-for-bit identical to its
single llm child's (0.29425625, both), and a separate root task span
with metrics all None despite its two task children carrying real,
distinct costs. Task-level metrics are not an independently reliable
figure: sometimes they exactly mirror a child (double-counting risk if
summed alongside it), sometimes they're absent even when children have
real data. Only span_attributes["type"] == "llm" spans are ever summed
here — exactly the same principle ingestion/reconstruct.py already
applies to Langfuse (GENERATION-type observations only, never a
parent/container's own totals), confirmed to also hold for Braintrust,
not a new assumption.

system_prompt: unlike the Langfuse project investigated previously, this
project's data DOES carry full system-prompt text (role="system" chat
messages inside an llm span's input.messages) — confirmed live. Every
reconstructed AgentSpec here gets system_prompt_source="observed" with
the real text when a non-judge llm span in the group actually has one;
falls back to "unavailable" (same disclosure discipline as Langfuse, own
placeholder text) only if none is found across the whole group — never
fabricated either way.

model/provider: provider is always "anthropic" regardless of what was
observed, same as ingestion/reconstruct.py's Langfuse version — this
harness only ever executes reconstructed environments through Anthropic
(target_system/tool_synthesis.py has no other real-model path). This
project's data mixes Anthropic and OpenAI (the judge agents specifically
use OpenAI, e.g. "gpt-5.6-luna") — judge exclusion already removes the
OpenAI-attributed spans from the dominant-model calculation, so this
falls out correctly without separate provider filtering.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingestion.braintrust_client import DEFAULT_TRACES_DIR, load_cached_traces
from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig
from target_system.provenance import ArgumentProfile, ObservedToolCall, OtherGroupFound, ReconstructionProvenance, ToolBehaviorProfile

_MAX_SAMPLE_VALUES = 20
_MAX_EXAMPLE_CALLS = 50
UNAVAILABLE_SYSTEM_PROMPT = "[unavailable — no system prompt observed in source Braintrust traces]"


@dataclass(frozen=True)
class GroupSummary:
    workflow_name: str | None
    trace_count: int


def _trace_group_key(spans: list[dict[str, Any]]) -> str | None:
    root = next((s for s in spans if s.get("is_root")), None)
    if root is None:
        # defensive: a cached "trace" should always include its root span
        # (ingestion/braintrust_client.py fetches every span sharing a
        # root_span_id), but don't crash on a malformed/partial cache entry
        return None
    metadata = root.get("metadata") or {}
    name = metadata.get("workflow_name") or metadata.get("workflow_id")
    if name:
        return name
    span_attrs = root.get("span_attributes") or {}
    return span_attrs.get("name")


def group_traces(traces: list[list[dict[str, Any]]]) -> dict[str | None, list[list[dict[str, Any]]]]:
    """Clusters an already-cached trace batch by _trace_group_key. Unlike
    Langfuse's group_traces, nothing is excluded as "test noise" here —
    no equivalent noise convention (Langfuse's agent_name in {"test",
    "test-fix"}) was found in this account's real data; every group found
    is real and offered."""
    groups: dict[str | None, list[list[dict[str, Any]]]] = defaultdict(list)
    for spans in traces:
        groups[_trace_group_key(spans)].append(spans)
    return dict(groups)


def summarize_groups(groups: dict[str | None, list[list[dict[str, Any]]]]) -> list[GroupSummary]:
    summaries = [GroupSummary(workflow_name=name, trace_count=len(trace_list)) for name, trace_list in groups.items()]
    return sorted(summaries, key=lambda s: s.trace_count, reverse=True)


def _resolve_agent_name(span: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str | None:
    """Walks up span_parents to the nearest ancestor (or the span itself)
    carrying metadata["agent_name"] -- confirmed against a real 15-trace
    pull that agent_name is NOT reliably present on llm-type spans
    themselves (all None in that batch, despite an earlier single-example
    check having seen it duplicated onto one) but IS reliably present on
    their parent task-type span (e.g. "Staci 2.0 Judge", "Staci 2.0
    Bungalow Owner Agent"). Checking the span's own metadata first (before
    walking up) keeps this correct for spans that do carry it directly."""
    current: dict[str, Any] | None = span
    seen: set[str] = set()
    while current is not None:
        agent_name = (current.get("metadata") or {}).get("agent_name")
        if agent_name:
            return agent_name
        span_id = current.get("span_id")
        if span_id is None or span_id in seen:
            break
        seen.add(span_id)
        parents = current.get("span_parents") or []
        current = by_id.get(parents[0]) if parents else None
    return None


def _is_judge_span(span: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> bool:
    span_attrs = span.get("span_attributes") or {}
    if span_attrs.get("type") == "score":
        return True
    agent_name = _resolve_agent_name(span, by_id) or ""
    return "judge" in agent_name.lower()


def _index_by_id(spans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["span_id"]: s for s in spans if s.get("span_id") is not None}


def _strip_tool_suffix(name: str) -> str:
    suffix = ".aexecute"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _extract_tool_call(span: dict[str, Any]) -> tuple[str, dict[str, Any], Any] | None:
    span_attrs = span.get("span_attributes") or {}
    if span_attrs.get("type") != "tool":
        return None
    name = _strip_tool_suffix(span_attrs.get("name") or "")
    arguments = span.get("input") if isinstance(span.get("input"), dict) else {}
    output = span.get("output")
    response: Any = output
    if isinstance(output, dict) and isinstance(output.get("result"), dict):
        content = output["result"].get("content")
        if isinstance(content, str):
            try:
                response = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                response = content
    return name, arguments, response


def _python_type_name(value: Any) -> str:
    return type(value).__name__


def _build_argument_profiles(calls: list[tuple[dict[str, Any], Any]]) -> dict[str, ArgumentProfile]:
    values_by_arg: dict[str, list[Any]] = defaultdict(list)
    for arguments, _response in calls:
        for key, value in arguments.items():
            values_by_arg[key].append(value)

    profiles: dict[str, ArgumentProfile] = {}
    for arg_name, values in values_by_arg.items():
        observed_types = sorted({_python_type_name(v) for v in values})
        try:
            distinct = len({json.dumps(v, sort_keys=True, default=str) for v in values})
        except TypeError:
            distinct = len(values)
        numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        strings = [v for v in values if isinstance(v, str)]
        profiles[arg_name] = ArgumentProfile(
            observed_types=observed_types,
            distinct_value_count=distinct,
            sample_values=values[:_MAX_SAMPLE_VALUES],
            numeric_range=(min(numbers), max(numbers)) if numbers else None,
            string_length_range=(min(len(s) for s in strings), max(len(s) for s in strings)) if strings else None,
        )
    return profiles


def _build_tool_profiles(traces: list[list[dict[str, Any]]]) -> dict[str, ToolBehaviorProfile]:
    calls_by_tool: dict[str, list[tuple[dict[str, Any], Any, str | None]]] = defaultdict(list)
    for spans in traces:
        by_id = _index_by_id(spans)
        for span in spans:
            if _is_judge_span(span, by_id):
                continue
            extracted = _extract_tool_call(span)
            if extracted is None:
                continue
            tool_name, arguments, response = extracted
            calls_by_tool[tool_name].append((arguments, response, span.get("created")))

    profiles: dict[str, ToolBehaviorProfile] = {}
    for tool_name, calls in calls_by_tool.items():
        response_keys: set[str] = set()
        for _args, response, _ts in calls:
            if isinstance(response, dict):
                response_keys |= set(response.keys())
        profiles[tool_name] = ToolBehaviorProfile(
            tool_name=tool_name,
            n_calls_observed=len(calls),
            argument_profiles=_build_argument_profiles([(a, r) for a, r, _ts in calls]),
            response_key_set=sorted(response_keys),
            example_calls=[
                ObservedToolCall(arguments=args, response=response, observed_at=ts) for args, response, ts in calls[:_MAX_EXAMPLE_CALLS]
            ],
        )
    return profiles


def _build_model_config(traces: list[list[dict[str, Any]]], *, warnings: list[str]) -> ModelConfig:
    model_names: Counter[str] = Counter()
    for spans in traces:
        by_id = _index_by_id(spans)
        for span in spans:
            if _is_judge_span(span, by_id):
                continue
            span_attrs = span.get("span_attributes") or {}
            if span_attrs.get("type") != "llm":
                continue
            model = (span.get("metadata") or {}).get("model")
            if model:
                model_names[model] += 1

    if not model_names:
        warnings.append("no llm-type spans with a model name found — model_name defaults to 'unknown'")
        return ModelConfig(provider="anthropic", model_name="unknown", temperature=None)

    if len(model_names) > 1:
        breakdown = ", ".join(f"{name} ({count})" for name, count in model_names.most_common())
        warnings.append(
            f"multiple model names observed across this group's traces: {breakdown} — "
            f"possible drift or multiple app versions in the batch; using the most common"
        )
    dominant_model = model_names.most_common(1)[0][0]
    return ModelConfig(provider="anthropic", model_name=dominant_model, temperature=None)


def _extract_system_prompt(traces: list[list[dict[str, Any]]]) -> tuple[str, str]:
    """(system_prompt, system_prompt_source) — the first real system
    message found in a non-judge llm span's input.messages across the
    group, or the standard "unavailable" disclosure if none exists."""
    for spans in traces:
        by_id = _index_by_id(spans)
        for span in spans:
            if _is_judge_span(span, by_id):
                continue
            span_attrs = span.get("span_attributes") or {}
            if span_attrs.get("type") != "llm":
                continue
            messages = (span.get("input") or {}).get("messages") or []
            for message in messages:
                if message.get("role") == "system":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content, "observed"
    return UNAVAILABLE_SYSTEM_PROMPT, "unavailable"


@dataclass(frozen=True)
class _CostStats:
    avg_generations_per_trace: float | None
    avg_prompt_tokens_per_generation: float | None
    avg_completion_tokens_per_generation: float | None
    avg_cost_usd_per_trace: float | None


def _build_cost_stats(traces: list[list[dict[str, Any]]]) -> _CostStats:
    """Real, Braintrust-computed cost/token figures — never estimated
    here. Only llm-type spans are summed (see module docstring on why
    task-level metrics can't be trusted not to double-count); judge spans
    excluded from every average, same reasoning as Langfuse's judge
    exclusion — eval infrastructure cost, not the reconstructed agent's
    own."""
    if not traces:
        return _CostStats(None, None, None, None)

    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []
    generations_per_trace: list[int] = []
    trace_costs: list[float] = []

    for spans in traces:
        by_id = _index_by_id(spans)
        n_generations = 0
        trace_cost = 0.0
        has_cost = False
        for span in spans:
            if _is_judge_span(span, by_id):
                continue
            span_attrs = span.get("span_attributes") or {}
            if span_attrs.get("type") != "llm":
                continue
            n_generations += 1
            metrics = span.get("metrics") or {}
            if isinstance(metrics.get("prompt_tokens"), (int, float)):
                prompt_tokens.append(metrics["prompt_tokens"])
            if isinstance(metrics.get("completion_tokens"), (int, float)):
                completion_tokens.append(metrics["completion_tokens"])
            cost = metrics.get("estimated_cost")
            if isinstance(cost, (int, float)):
                trace_cost += cost
                has_cost = True
        generations_per_trace.append(n_generations)
        if has_cost:
            trace_costs.append(trace_cost)

    def _avg(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return _CostStats(
        avg_generations_per_trace=_avg(generations_per_trace),
        avg_prompt_tokens_per_generation=_avg(prompt_tokens),
        avg_completion_tokens_per_generation=_avg(completion_tokens),
        avg_cost_usd_per_trace=_avg(trace_costs),
    )


def reconstruct_system_config(
    traces: list[list[dict[str, Any]]],
    *,
    label: str,
    project_id: str,
    source_workflow_name: str | None,
    other_groups_found: list[GroupSummary] | None = None,
) -> SystemConfig:
    """Builds a SystemConfig from one group's traces (see group_traces).
    Single AgentSpec, role="supervisor" — same compatibility-only choice
    ingestion/reconstruct.py made for Langfuse (SystemConfig.supervisor(),
    orchestration.build_team expect it), not a claim that a
    supervisor/subordinate structure was observed. This account's real
    data does show multiple distinct agents per trace (e.g. "Staci 2.0
    Bungalow Tenant Agent" handing off to "Staci 2.0 Action-Aware Result
    Judge") — reconstructing that multi-agent structure faithfully is a
    real, separate improvement not attempted here; this flattens every
    non-judge tool call and the first observed system prompt into one
    agent, same flattening Langfuse's reconstruction already does."""
    warnings: list[str] = []
    tool_profiles = _build_tool_profiles(traces)
    model = _build_model_config(traces, warnings=warnings)
    cost_stats = _build_cost_stats(traces)
    system_prompt, system_prompt_source = _extract_system_prompt(traces)

    agent = AgentSpec(
        role="supervisor",
        name=source_workflow_name or "reconstructed-agent",
        system_prompt=system_prompt,
        system_prompt_source=system_prompt_source,
        tools=sorted(tool_profiles.keys()),
    )

    provenance = ReconstructionProvenance(
        project_id=project_id,
        source_agent_name=source_workflow_name,
        trace_count=len(traces),
        extraction_date=datetime.now(timezone.utc).isoformat(),
        other_groups_found=[OtherGroupFound(agent_name=g.workflow_name, trace_count=g.trace_count) for g in (other_groups_found or [])],
        warnings=warnings,
        tool_profiles=tool_profiles,
        avg_generations_per_trace=cost_stats.avg_generations_per_trace,
        avg_prompt_tokens_per_generation=cost_stats.avg_prompt_tokens_per_generation,
        avg_completion_tokens_per_generation=cost_stats.avg_completion_tokens_per_generation,
        avg_cost_usd_per_trace=cost_stats.avg_cost_usd_per_trace,
    )

    return SystemConfig(
        label=label,
        model=model,
        agents=[agent],
        security=SecurityConfig(),
        defensive_instruction=False,
        provenance=provenance,
    )


def reconstruct_from_cache(
    *,
    project_id: str,
    workflow_name: str | None,
    label: str | None = None,
    traces_dir: Path = DEFAULT_TRACES_DIR,
) -> SystemConfig:
    """End-to-end convenience: load whatever's cached for project_id,
    group it, and reconstruct the named group. Raises ValueError if that
    group isn't in the cached batch (caller should pull a fresh/larger
    batch via braintrust_client.pull_traces() first, not silently fall
    back)."""
    all_traces = load_cached_traces(project_id, traces_dir=traces_dir)
    groups = group_traces(all_traces)
    if workflow_name not in groups:
        available = sorted((name or "<none>") for name in groups)
        raise ValueError(f"no cached traces for workflow_name={workflow_name!r} in project {project_id!r}; available groups: {available}")
    summaries = summarize_groups(groups)
    other_groups = [s for s in summaries if s.workflow_name != workflow_name]
    return reconstruct_system_config(
        groups[workflow_name],
        label=label or (workflow_name or f"{project_id}-unnamed"),
        project_id=project_id,
        source_workflow_name=workflow_name,
        other_groups_found=other_groups,
    )
