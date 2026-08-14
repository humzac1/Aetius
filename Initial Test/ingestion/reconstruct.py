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

system_prompt is never fabricated, and is now never assumed absent
either. The original version of this module hardcoded
system_prompt_source="unavailable" for every reconstruction, on the
strength of a live check against this project's first real batch (the
49-trace Invoice/HR pull, whose GENERATION.input message roles really are
only ever user/assistant). Baking that finding in as a constant rather
than a check made it a permanent property of the reconstructor: a later
project whose traces *do* carry the text — confirmed against the
320-trace "E-Commerce Order Support" batch, where all 2126 generations
carry a role="system" message of 2298-4080 chars — was still reported as
having "run with no system prompt at all". _extract_system_prompt() now
looks, per agent, and only falls back to the placeholder when nothing is
there. Same discipline as ingestion/braintrust_reconstruct.py, which had
the working version of this all along.

Multi-agent decomposition was the same mistake twice. The single
role="supervisor" AgentSpec was justified by "every trace this project has
produced shows a flat observation hierarchy under one root span" — true of
the 49-trace batch, false of any nested instrumentation. Traces that carry
per-observation metadata["agent"] (and AGENT-typed spans with
metadata["agent_role"] / ["tools_available"]) are now partitioned into one
AgentSpec per observed agent, each with its own prompt, role and tool
grant. Traces without that signal keep the single-agent shape exactly as
before, so the original batch reconstructs unchanged.
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


def _message_text(content: Any) -> str | None:
    """Text of one chat message. A system prompt is a plain string in most
    instrumentations, but Anthropic-style block lists ([{"type": "text", ...}])
    are accepted too — reading one of those as absent is exactly the failure this
    module already made once."""
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip()
        ]
        return "\n".join(parts) or None
    return None


def _generation_messages(obs: dict[str, Any]) -> list[Any]:
    """Both observed input shapes: a bare message list, or {"messages": [...]}."""
    input_ = obs.get("input")
    if isinstance(input_, list):
        return input_
    if isinstance(input_, dict) and isinstance(input_.get("messages"), list):
        return input_["messages"]
    return []


def _extract_system_prompt(
    observations: list[dict[str, Any]], *, warnings: list[str], agent_label: str
) -> tuple[str, str]:
    """(system_prompt, system_prompt_source) for one agent's observations.

    Takes the most frequently observed system message rather than the first one:
    a batch can span a prompt edit, and the dominant text is the better single
    answer for a config that has to pick one. Any drift is surfaced as a warning
    instead of being silently resolved, because a batch straddling two prompt
    versions is a fact about the evidence the caller should see.
    """
    seen: Counter[str] = Counter()
    for obs in observations:
        if obs.get("type") != "GENERATION" or obs.get("name") == JUDGE_OBSERVATION_NAME:
            continue
        for message in _generation_messages(obs):
            if isinstance(message, dict) and message.get("role") == "system":
                text = _message_text(message.get("content"))
                if text:
                    seen[text] += 1
                break
    if not seen:
        return UNAVAILABLE_SYSTEM_PROMPT, "unavailable"
    if len(seen) > 1:
        lengths = ", ".join(f"{len(text)} chars (x{count})" for text, count in seen.most_common())
        warnings.append(
            f"agent {agent_label!r}: {len(seen)} distinct system prompts observed across this "
            f"batch — {lengths}; using the most common. The batch may straddle a prompt edit."
        )
    return seen.most_common(1)[0][0], "observed"


@dataclass(frozen=True)
class _AgentEvidence:
    """Everything one observed agent's own observations support."""

    name: str
    observed_role: str | None
    tools: list[str]
    n_traces: int
    observations: list[dict[str, Any]]


def _partition_by_agent(traces: list[dict[str, Any]]) -> list[_AgentEvidence]:
    """One evidence bundle per distinct metadata["agent"], ordered by how many
    traces the agent appears in (descending). Returns [] when the traces carry no
    per-agent signal at all, which is the caller's cue to keep the flat
    single-agent shape rather than invent a decomposition."""
    observations_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trace_indexes: dict[str, set[int]] = defaultdict(set)
    roles: dict[str, Counter[str]] = defaultdict(Counter)
    granted: dict[str, set[str]] = defaultdict(set)

    for index, trace in enumerate(traces):
        for obs in trace.get("observations") or []:
            if obs.get("name") == JUDGE_OBSERVATION_NAME:
                continue
            metadata = obs.get("metadata")
            if not isinstance(metadata, dict):
                continue
            agent = metadata.get("agent")
            if not isinstance(agent, str) or not agent.strip():
                continue
            observations_by_agent[agent].append(obs)
            trace_indexes[agent].add(index)
            role = metadata.get("agent_role")
            if isinstance(role, str) and role.strip():
                roles[agent][role] += 1
            for tool_name in metadata.get("tools_available") or []:
                if isinstance(tool_name, str) and tool_name.strip():
                    granted[agent].add(tool_name)

    evidence: list[_AgentEvidence] = []
    for agent, observations in observations_by_agent.items():
        called = {
            extracted[0]
            for obs in observations
            if (extracted := _extract_tool_call(obs)) is not None
        }
        # Grant ∪ use: tools_available is a real observed declaration, so a
        # granted-but-never-called tool is evidence, not a guess. A tool called
        # without appearing in any grant still counts — the call is the stronger
        # signal of the two.
        evidence.append(
            _AgentEvidence(
                name=agent,
                observed_role=roles[agent].most_common(1)[0][0] if roles[agent] else None,
                tools=sorted(called | granted[agent]),
                n_traces=len(trace_indexes[agent]),
                observations=observations,
            )
        )
    return sorted(evidence, key=lambda e: (-e.n_traces, e.name))


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")


def _build_agent_specs(evidence: list[_AgentEvidence], *, warnings: list[str]) -> list[AgentSpec]:
    """One AgentSpec per observed agent.

    Two structural constraints from target_system/config.py and orchestration.py
    have to be satisfied, and neither is guaranteed by trace data:

      - SystemConfig.supervisor() raises unless exactly one agent has
        role == "supervisor", and members() is everything else. So a batch whose
        agents report no supervisor role needs one designated.
      - orchestration._build_member_agent sets Agent(id=spec.role), so two
        members sharing a role (three "specialist"s, say) would collide on id.
        Shared roles are disambiguated with the agent's own name.

    Both adjustments are recorded as warnings; neither invents behavioral text.
    """
    supervisors = [item for item in evidence if (item.observed_role or "").lower() == "supervisor"]
    if supervisors:
        supervisor_name = supervisors[0].name
        if len(supervisors) > 1:
            others = ", ".join(repr(item.name) for item in supervisors[1:])
            warnings.append(
                f"{len(supervisors)} agents reported role 'supervisor'; {supervisor_name!r} "
                f"(seen in the most traces) is the reconstructed supervisor and {others} were "
                f"re-labelled as members, since config.members() excludes every 'supervisor'."
            )
    else:
        supervisor_name = evidence[0].name
        observed = ", ".join(f"{item.name}={item.observed_role or 'unknown'}" for item in evidence)
        warnings.append(
            f"no agent reported role 'supervisor'; {supervisor_name!r} (seen in "
            f"{evidence[0].n_traces} trace(s), more than any other agent) was designated "
            f"supervisor so the reconstructed team has a root. Observed roles: {observed}."
        )

    role_counts = Counter(
        (item.observed_role or "agent").lower() for item in evidence if item.name != supervisor_name
    )
    disambiguated: list[str] = []

    specs: list[AgentSpec] = []
    for item in evidence:
        if item.name == supervisor_name:
            role = "supervisor"
        else:
            base = (item.observed_role or "agent").lower()
            if base == "supervisor":
                base = "member"
            if role_counts[base] > 1:
                role = f"{base}-{_slug(item.name)}"
                disambiguated.append(f"{item.name!r} -> role {role!r}")
            else:
                role = base
        prompt, source = _extract_system_prompt(
            item.observations, warnings=warnings, agent_label=item.name
        )
        specs.append(
            AgentSpec(
                role=role,
                name=item.name,
                system_prompt=prompt,
                system_prompt_source=source,
                tools=item.tools,
            )
        )

    if disambiguated:
        warnings.append(
            "shared observed roles were suffixed with the agent name to keep member ids "
            "distinct: " + "; ".join(disambiguated)
        )

    specs.sort(key=lambda spec: (spec.role != "supervisor", spec.name))
    return specs


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

    Agent shape follows the evidence rather than a fixed assumption. When the
    observations carry per-agent identity (metadata["agent"]), each observed
    agent becomes its own AgentSpec with the prompt, role and tool grant its own
    observations support — that is the multi-agent structure the source system
    really ran, and collapsing it loses both the topology and four of the five
    prompts.

    When there is no such signal — a flat hierarchy with metadata["agent_name"]
    as the only identity key, which is what the original 49-trace batch looks
    like — the single-AgentSpec shape is kept unchanged, including role
    "supervisor" for compatibility with SystemConfig.supervisor() and
    orchestration.build_team. That role still doesn't imply a
    supervisor/subordinate structure was observed. The one thing that changed for
    the flat case is that its system prompt is now extracted if present instead
    of assumed absent."""
    warnings: list[str] = []
    tool_profiles = _build_tool_profiles(traces)
    model = _build_model_config(traces, warnings=warnings)
    cost_stats = _build_cost_stats(traces)

    evidence = _partition_by_agent(traces)
    if evidence:
        # Any per-agent identity at all is enough to prefer this path, including
        # a single agent: its own observed name, role and tool grant are better
        # evidence than the group name and the union of every tool called.
        agents = _build_agent_specs(evidence, warnings=warnings)
    else:
        # Flat (or single-agent) traces: unchanged shape, but look for the prompt
        # rather than declaring it unavailable.
        all_observations = [obs for trace in traces for obs in (trace.get("observations") or [])]
        prompt, prompt_source = _extract_system_prompt(
            all_observations,
            warnings=warnings,
            agent_label=source_agent_name or "reconstructed-agent",
        )
        agents = [
            AgentSpec(
                role="supervisor",
                name=source_agent_name or "reconstructed-agent",
                system_prompt=prompt,
                system_prompt_source=prompt_source,
                tools=sorted(tool_profiles.keys()),
            )
        ]

    # A missing prompt deliberately does NOT add a warning here: AgentSpec
    # carries system_prompt_source="unavailable" structurally and the verdict
    # layer already surfaces that inline, so warning as well would add noise to
    # every reconstruction of a batch that never had the text -- the normal case
    # for flat traces, not an anomaly. Warnings stay for things nothing else
    # records: model drift, prompt drift, and the role adjustments above.

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
        agents=agents,
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
