"""Builds a SystemConfig-compatible reconstruction from a cached Langfuse
trace batch (ingestion/langfuse_client.py). Never re-hits the API — reads
whatever's already on disk.

Grouping (revised design): a single Langfuse project can contain multiple
unrelated real systems. Confirmed against this project's real 49-trace
batch: trace.metadata["agent_name"] splits it into "Invoice Generation
Assistant" (11 traces, tool-calling), "HR Onboarding Assistant" (33
traces, pure conversational, zero tools), plus smoke-test noise
(agent_name in {"test", "test-fix"} or missing, 5 traces). group_traces()
clusters the already-cached batch by that key entirely client-side — no
second API pull. summarize_groups() reports every group found (name,
trace count) for the "Add environment" flow (Part 5) to list; the caller
picks which group's traces get reconstructed via reconstruct_system_config().
Only pull_traces() (langfuse_client.py) ever talks to the API again, and
only when explicitly asked for a fresh/larger batch.

system_prompt is never fabricated: confirmed live (see the Part 1 report)
that this project's traces don't carry system-prompt text anywhere
(GENERATION.input message roles are only ever user/assistant,
model_parameters is empty, no metadata field carries it). Every
reconstructed AgentSpec gets system_prompt_source="unavailable" and an
explicit placeholder string — never invented behavioral text standing in
for it.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ingestion.langfuse_client import DEFAULT_TRACES_DIR, load_cached_traces
from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig
from target_system.provenance import ArgumentProfile, ObservedToolCall, OtherGroupFound, ReconstructionProvenance, ToolBehaviorProfile

DEFAULT_TEST_NOISE_NAMES: frozenset[str] = frozenset({"test", "test-fix"})
JUDGE_OBSERVATION_NAME = "judge-evaluation"
_MAX_SAMPLE_VALUES = 20
_MAX_EXAMPLE_CALLS = 50
UNAVAILABLE_SYSTEM_PROMPT = "[unavailable — no system prompt observed in source Langfuse traces]"


@dataclass(frozen=True)
class GroupSummary:
    agent_name: str | None
    trace_count: int
    is_noise: bool


def group_traces(
    traces: list[dict[str, Any]], *, exclude_names: frozenset[str] = DEFAULT_TEST_NOISE_NAMES
) -> dict[str | None, list[dict[str, Any]]]:
    """Clusters an already-cached trace batch by metadata["agent_name"].
    Traces whose agent_name is in exclude_names are dropped entirely (not
    even kept in a group) — smoke-test noise, not a real environment to
    ever offer as a reconstruction target. Traces with no agent_name at
    all (None) are kept as their own group rather than dropped, since an
    un-tagged real system is still possibly worth reconstructing — it's
    reported like any other group, just with a None name."""
    groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        metadata = trace.get("metadata") or {}
        agent_name = metadata.get("agent_name") if isinstance(metadata, dict) else None
        if agent_name in exclude_names:
            continue
        groups[agent_name].append(trace)
    return dict(groups)


def summarize_groups(
    groups: dict[str | None, list[dict[str, Any]]], *, exclude_names: frozenset[str] = DEFAULT_TEST_NOISE_NAMES
) -> list[GroupSummary]:
    """Newest-first isn't meaningful here (groups, not traces) — sorted by
    trace_count descending instead, so the dominant/most-evidenced group
    is what a picker naturally shows first."""
    summaries = [
        GroupSummary(agent_name=name, trace_count=len(trace_list), is_noise=name in exclude_names)
        for name, trace_list in groups.items()
    ]
    return sorted(summaries, key=lambda s: s.trace_count, reverse=True)


def _extract_tool_call(obs: dict[str, Any]) -> tuple[str, dict[str, Any], Any] | None:
    """(tool_name, arguments, response) if this observation is a tool call,
    else None. Tries the observed input shape {"tool": str, "inputs":
    {...}} first (real, structured signal); falls back to stripping a
    "tool-call-" name prefix if input doesn't match (seen in this
    project's Invoice system but not guaranteed elsewhere) — never trusts
    the name alone when the input shape already answers the question, so
    this stays robust across differently-instrumented projects."""
    input_ = obs.get("input")
    if isinstance(input_, dict) and isinstance(input_.get("tool"), str) and isinstance(input_.get("inputs"), dict):
        return input_["tool"], input_["inputs"], obs.get("output")
    name = obs.get("name") or ""
    if name.startswith("tool-call-"):
        return name[len("tool-call-") :], {}, obs.get("output")
    return None


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


def _build_tool_profiles(traces: list[dict[str, Any]]) -> dict[str, ToolBehaviorProfile]:
    calls_by_tool: dict[str, list[tuple[dict[str, Any], Any, str | None]]] = defaultdict(list)
    for trace in traces:
        for obs in trace.get("observations") or []:
            if obs.get("name") == JUDGE_OBSERVATION_NAME:
                continue
            extracted = _extract_tool_call(obs)
            if extracted is None:
                continue
            tool_name, arguments, response = extracted
            calls_by_tool[tool_name].append((arguments, response, obs.get("startTime")))

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
                ObservedToolCall(arguments=args, response=response, observed_at=ts)
                for args, response, ts in calls[:_MAX_EXAMPLE_CALLS]
            ],
        )
    return profiles


def _build_model_config(traces: list[dict[str, Any]], *, warnings: list[str]) -> ModelConfig:
    model_names: Counter[str] = Counter()
    for trace in traces:
        for obs in trace.get("observations") or []:
            if obs.get("type") == "GENERATION" and obs.get("name") != JUDGE_OBSERVATION_NAME and obs.get("model"):
                model_names[obs["model"]] += 1

    if not model_names:
        warnings.append("no GENERATION observations with a model name found — model_name defaults to 'unknown'")
        return ModelConfig(provider="anthropic", model_name="unknown", temperature=None)

    if len(model_names) > 1:
        breakdown = ", ".join(f"{name} ({count})" for name, count in model_names.most_common())
        warnings.append(
            f"multiple model names observed across this group's traces: {breakdown} — "
            f"possible drift or multiple app versions in the batch; using the most common"
        )
    dominant_model = model_names.most_common(1)[0][0]
    return ModelConfig(provider="anthropic", model_name=dominant_model, temperature=None)


@dataclass(frozen=True)
class _CostStats:
    avg_generations_per_trace: float | None
    avg_prompt_tokens_per_generation: float | None
    avg_completion_tokens_per_generation: float | None
    avg_cost_usd_per_trace: float | None


def _build_cost_stats(traces: list[dict[str, Any]]) -> _CostStats:
    """Real, Langfuse-computed cost/token figures — never estimated here.
    judge-evaluation is excluded from every average: it's eval
    infrastructure cost, not the reconstructed agent's own, and including
    it would overstate what a real attack run against this twin costs."""
    if not traces:
        return _CostStats(None, None, None, None)

    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []
    generations_per_trace: list[int] = []
    trace_costs: list[float] = []

    for trace in traces:
        n_generations = 0
        trace_cost = 0.0
        has_cost = False
        for obs in trace.get("observations") or []:
            if obs.get("type") != "GENERATION" or obs.get("name") == JUDGE_OBSERVATION_NAME:
                continue
            n_generations += 1
            if isinstance(obs.get("promptTokens"), (int, float)):
                prompt_tokens.append(obs["promptTokens"])
            if isinstance(obs.get("completionTokens"), (int, float)):
                completion_tokens.append(obs["completionTokens"])
            cost = obs.get("calculatedTotalCost")
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
    traces: list[dict[str, Any]],
    *,
    label: str,
    project_id: str,
    source_agent_name: str | None,
    other_groups_found: list[GroupSummary] | None = None,
) -> SystemConfig:
    """Builds a SystemConfig from one group's traces (see group_traces).
    Single AgentSpec, role="supervisor": every trace this project has
    produced (both real groups) shows a flat observation hierarchy under
    one root span, with metadata["agent_name"] as the only role-identity
    signal available — there's no evidence in this data of a multi-agent
    decomposition to reconstruct, so this doesn't guess one. role is set
    to "supervisor" (not, say, "agent") purely for compatibility with
    existing plumbing (SystemConfig.supervisor(), orchestration.build_team)
    that Part 4 will wire a reconstructed config through — it doesn't
    imply a supervisor/subordinate structure was observed."""
    warnings: list[str] = []
    tool_profiles = _build_tool_profiles(traces)
    model = _build_model_config(traces, warnings=warnings)
    cost_stats = _build_cost_stats(traces)

    agent = AgentSpec(
        role="supervisor",
        name=source_agent_name or "reconstructed-agent",
        system_prompt=UNAVAILABLE_SYSTEM_PROMPT,
        system_prompt_source="unavailable",
        tools=sorted(tool_profiles.keys()),
    )

    provenance = ReconstructionProvenance(
        project_id=project_id,
        source_agent_name=source_agent_name,
        trace_count=len(traces),
        extraction_date=datetime.now(timezone.utc).isoformat(),
        other_groups_found=[
            OtherGroupFound(agent_name=g.agent_name, trace_count=g.trace_count) for g in (other_groups_found or []) if not g.is_noise
        ],
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
    agent_name: str | None,
    label: str | None = None,
    traces_dir=DEFAULT_TRACES_DIR,
    exclude_names: frozenset[str] = DEFAULT_TEST_NOISE_NAMES,
) -> SystemConfig:
    """End-to-end convenience: load whatever's cached for project_id, group
    it, and reconstruct the named group. Raises ValueError if that group
    isn't in the cached batch (caller should pull a fresh/larger batch via
    langfuse_client.pull_traces() first, not silently fall back)."""
    all_traces = load_cached_traces(project_id, traces_dir=traces_dir)
    groups = group_traces(all_traces, exclude_names=exclude_names)
    if agent_name not in groups:
        available = sorted((name or "<none>") for name in groups)
        raise ValueError(f"no cached traces for agent_name={agent_name!r} in project {project_id!r}; available groups: {available}")
    summaries = summarize_groups(groups, exclude_names=exclude_names)
    other_groups = [s for s in summaries if s.agent_name != agent_name]
    return reconstruct_system_config(
        groups[agent_name],
        label=label or (agent_name or f"{project_id}-unnamed"),
        project_id=project_id,
        source_agent_name=agent_name,
        other_groups_found=other_groups,
    )
