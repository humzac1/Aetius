"""Executes a reconstructed twin's SystemConfig (a single agent, no real
tool implementations — see ingestion/reconstruct.py) through a real Agno
Agent, with tool calls resolved through target_system/tool_synthesis.py
instead of target_system/tools.py's TOOL_REGISTRY of hand-written toy
implementations.

Parallel to target_system/orchestration.py's team-based build_team/run_case
/run_multi_turn_case, not a modification of them: a reconstructed config has
none of the toy system's fixed 3-agent structure, and its RunOutput has no
member_responses to walk (verified live: a solo Agent's RunOutput has
.tools/.metrics/.content exactly like a Team member's, but no
member_responses attribute at all) — so the event-adaptation logic is
simpler here, not a variant of orchestration.py's. Reuses only the pure,
config-shape-agnostic helpers from there (_iso, _build_model, _tool_error,
_enforce_monotonic_timestamps) rather than duplicating them.

attacker/executor.py's execute_case() is still the one dispatch point:
it routes to this module instead of orchestration.py whenever
config.provenance is not None.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.tools.function import Function

from target_system.config import SystemConfig, compute_config_hash
from target_system.logging_schema import (
    AgentEndEvent,
    AgentStartEvent,
    AgentTokenUsage,
    AttackInfo,
    ErrorEvent,
    Event,
    RunRecord,
    ToolCallEvent,
    TokenUsage,
)
from target_system.orchestration import MockScripts, _build_model, _enforce_monotonic_timestamps, _iso, _tool_error
from target_system.policy import TaskContext, evaluate_outcomes
from target_system.provenance import ToolBehaviorProfile
from target_system.tool_synthesis import DEFAULT_SYNTHESIS_MODEL, synthesize_tool_response

_JSON_SCHEMA_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object", "NoneType": "null"}


@dataclass
class SynthesisLogEntry:
    tool_name: str
    arguments: dict[str, Any]
    response: Any
    source: Literal["replay", "generated", "unavailable"]


@dataclass
class SynthesisCallLog:
    """Parallel to target_system/tools.py's CallLog for the toy system —
    records what target_system/tool_synthesis.py actually did per call, so
    _build_solo_events can attribute response_source to the right
    ToolCallEvent afterward."""

    entries: list[SynthesisLogEntry] = field(default_factory=list)

    def record(self, tool_name: str, arguments: dict[str, Any], response: Any, source: Literal["replay", "generated", "unavailable"]) -> None:
        self.entries.append(SynthesisLogEntry(tool_name, arguments, response, source))


def _json_schema_type(observed_types: list[str]) -> str:
    for t in observed_types:
        if t in _JSON_SCHEMA_TYPES:
            return _JSON_SCHEMA_TYPES[t]
    return "string"


def _build_tool_parameters_schema(profile: ToolBehaviorProfile) -> dict[str, Any]:
    """No field is marked required: ArgumentProfile doesn't track whether
    an argument was present on *every* observed call (only that it was
    observed at all), so requiring it could push the model to supply a
    value where the real tool never needed one."""
    properties = {name: {"type": _json_schema_type(p.observed_types)} for name, p in profile.argument_profiles.items()}
    return {"type": "object", "properties": properties, "required": []}


def _make_synthetic_entrypoint(
    tool_name: str,
    profile: ToolBehaviorProfile,
    *,
    anthropic_client: Any,
    model_name: str,
    synthesis_log: SynthesisCallLog,
    injected_content: str | None,
):
    def entrypoint(**kwargs: Any) -> Any:
        result = synthesize_tool_response(
            tool_name, kwargs, profile, client=anthropic_client, model_name=model_name, injected_content=injected_content
        )
        synthesis_log.record(tool_name, kwargs, result.response, result.source)
        return result.response

    return entrypoint


def _build_synthetic_tools(
    config: SystemConfig,
    *,
    anthropic_client: Any,
    model_name: str,
    synthesis_log: SynthesisCallLog,
    tool_result_poison: dict[str, str] | None,
) -> list[Function]:
    agent_spec = config.supervisor()
    tool_profiles = config.provenance.tool_profiles if config.provenance else {}
    tools = []
    for tool_name in agent_spec.tools:
        profile = tool_profiles.get(tool_name) or ToolBehaviorProfile(tool_name=tool_name)
        injected = (tool_result_poison or {}).get(tool_name)
        tools.append(
            Function(
                name=tool_name,
                description=f"{tool_name} (reconstructed from observed historical calls)",
                parameters=_build_tool_parameters_schema(profile),
                entrypoint=_make_synthetic_entrypoint(
                    tool_name, profile, anthropic_client=anthropic_client, model_name=model_name, synthesis_log=synthesis_log, injected_content=injected
                ),
                skip_entrypoint_processing=True,
            )
        )
    return tools


def build_reconstructed_agent(
    config: SystemConfig,
    *,
    mock_scripts: MockScripts | None = None,
    anthropic_client: Any = None,
    synthesis_model_name: str = DEFAULT_SYNTHESIS_MODEL,
    synthesis_log: SynthesisCallLog,
    tool_result_poison: dict[str, str] | None = None,
    num_history_runs: int = 1,
) -> Agent:
    """system_message is omitted entirely when
    agent_spec.system_prompt_source == "unavailable" — passing the
    disclosure placeholder text itself as the model's literal system
    prompt would be worse than having none: the model would see
    "[unavailable — no system prompt observed...]" as its own instructions
    rather than behaving anything like the real system. db +
    add_history_to_context wired unconditionally, same reasoning as
    orchestration.build_team's — every reconstructed run gets identical
    session handling regardless of arm."""
    agent_spec = config.supervisor()
    model = _build_model(agent_spec.model_override or config.model, agent_spec.role, mock_scripts)
    tools = _build_synthetic_tools(
        config, anthropic_client=anthropic_client, model_name=synthesis_model_name, synthesis_log=synthesis_log, tool_result_poison=tool_result_poison
    )
    system_message = agent_spec.system_prompt if agent_spec.system_prompt_source == "observed" else None
    return Agent(
        id=agent_spec.role,
        name=agent_spec.name,
        model=model,
        system_message=system_message,
        tools=tools,
        store_events=True,
        db=InMemoryDb(),
        add_history_to_context=True,
        num_history_runs=num_history_runs,
    )


def _build_solo_events(result: Any, config: SystemConfig, started: float, ended: float, synthesis_log: SynthesisCallLog) -> list[Event]:
    events: list[Event] = []
    agent_role = config.supervisor().role

    def add(event: Event) -> None:
        event.idx = len(events)
        events.append(event)

    add(AgentStartEvent(idx=0, timestamp=_iso(started), agent=agent_role))

    # Matched in call order per tool name, same lockstep-queue approach as
    # orchestration._CallLogMatcher — synthesis_log entries are appended in
    # the same order Agno invokes the entrypoints, which is the same order
    # they appear in result.tools.
    queues: dict[str, deque[SynthesisLogEntry]] = defaultdict(deque)
    for entry in synthesis_log.entries:
        queues[entry.tool_name].append(entry)

    for t in result.tools or []:
        t_args = t.tool_args or {}
        # None, not a guessed value: if synthesize_tool_response raised
        # (no replay match and no anthropic_client for the generation
        # fallback — see tool_synthesis.py), Agno's own Function.execute
        # catches that exception, logs a warning, and turns it into a
        # tool_call_error on this ToolExecution rather than propagating it
        # — so a missing synthesis_log entry here means synthesis
        # genuinely failed, not that it happened silently. Defaulting to
        # "generated" here previously claimed a successful LLM synthesis
        # for a call that actually errored out — caught by checking real
        # events end to end, not assumed correct from the code alone.
        response_source: Literal["real", "replay", "generated", "unavailable"] | None = None
        queue = queues.get(t.tool_name)
        if queue and queue[0].arguments == t_args:
            response_source = queue.popleft().source
        add(
            ToolCallEvent(
                idx=0,
                timestamp=_iso(t.created_at),
                agent=agent_role,
                tool_name=t.tool_name,
                arguments=t_args,
                result=t.result,
                error=_tool_error(t),
                status="executed",  # no permission-guard mechanism for reconstructed tools in this pass
                response_source=response_source,
            )
        )

    final_answer = result.content if isinstance(result.content, str) else None
    add(AgentEndEvent(idx=0, timestamp=_iso(ended), agent=agent_role, final_answer=final_answer))
    return events


def _build_token_usage_solo(result: Any, config: SystemConfig) -> TokenUsage:
    usage = TokenUsage()
    if result is None or result.metrics is None:
        return usage
    role = config.supervisor().role
    usage.total_prompt_tokens = result.metrics.input_tokens or 0
    usage.total_completion_tokens = result.metrics.output_tokens or 0
    usage.by_agent[role] = AgentTokenUsage(prompt_tokens=usage.total_prompt_tokens, completion_tokens=usage.total_completion_tokens)
    return usage


def run_reconstructed_case(
    config: SystemConfig,
    task: str,
    *,
    case_id: str,
    task_context: TaskContext,
    seed: int,
    arm: str | None = None,
    case_family: str | None = None,
    attack: AttackInfo | None = None,
    mock_scripts: MockScripts | None = None,
    anthropic_client: Any = None,
    synthesis_model_name: str = DEFAULT_SYNTHESIS_MODEL,
    tool_result_poison: dict[str, str] | None = None,
) -> RunRecord:
    """Solo-agent sibling of orchestration.run_case."""
    synthesis_log = SynthesisCallLog()
    agent = build_reconstructed_agent(
        config, mock_scripts=mock_scripts, anthropic_client=anthropic_client, synthesis_model_name=synthesis_model_name,
        synthesis_log=synthesis_log, tool_result_poison=tool_result_poison,
    )
    run_id = f"run_{uuid4().hex[:12]}"

    started = time.time()
    error_message: str | None = None
    result: Any = None
    try:
        result = agent.run(task, session_id=run_id)
    except Exception as exc:  # noqa: BLE001 - captured into the trajectory log, not swallowed
        error_message = f"{type(exc).__name__}: {exc}"
    ended = time.time()

    events = _build_solo_events(result, config, started, ended, synthesis_log) if result is not None else []
    if error_message:
        events.append(ErrorEvent(idx=len(events), timestamp=_iso(ended), agent=None, message=error_message))
    _enforce_monotonic_timestamps(events)

    record = RunRecord(
        run_id=run_id,
        config_hash=compute_config_hash(config),
        case_id=case_id,
        case_family=case_family,
        arm=arm,
        seed=seed,
        started_at=_iso(started),
        ended_at=_iso(ended),
        wall_time_seconds=ended - started,
        events=events,
        token_usage=_build_token_usage_solo(result, config),
        attack=attack,
        error=error_message,
    )

    outcome_result = evaluate_outcomes(events, config=config, task=task_context)
    record.outcomes = outcome_result.outcomes
    record.outcome_evidence = outcome_result.evidence
    return record


def run_reconstructed_multi_turn_case(
    config: SystemConfig,
    turns: list[str],
    *,
    case_id: str,
    task_context: TaskContext,
    seed: int,
    arm: str | None = None,
    case_family: str | None = None,
    attack: AttackInfo | None = None,
    mock_scripts: MockScripts | None = None,
    anthropic_client: Any = None,
    synthesis_model_name: str = DEFAULT_SYNTHESIS_MODEL,
    tool_result_poison: dict[str, str] | None = None,
) -> RunRecord:
    """Solo-agent sibling of orchestration.run_multi_turn_case: runs `turns`
    in sequence against one Agent/session, one RunRecord for the whole
    conversation, stops on the first turn that raises — same contract."""
    if not turns:
        raise ValueError("run_reconstructed_multi_turn_case requires at least one turn")

    synthesis_log = SynthesisCallLog()
    agent = build_reconstructed_agent(
        config, mock_scripts=mock_scripts, anthropic_client=anthropic_client, synthesis_model_name=synthesis_model_name,
        synthesis_log=synthesis_log, tool_result_poison=tool_result_poison, num_history_runs=len(turns),
    )
    run_id = f"run_{uuid4().hex[:12]}"

    started = time.time()
    error_message: str | None = None
    combined_events: list[Event] = []
    turn_token_usages: list[TokenUsage] = []
    log_cursor = 0

    for turn_text in turns:
        turn_started = time.time()
        result: Any = None
        try:
            result = agent.run(turn_text, session_id=run_id)
        except Exception as exc:  # noqa: BLE001 - captured into the trajectory log, not swallowed
            error_message = f"{type(exc).__name__}: {exc}"
        turn_ended = time.time()

        if result is not None:
            turn_log = SynthesisCallLog(entries=synthesis_log.entries[log_cursor:])
            combined_events.extend(_build_solo_events(result, config, turn_started, turn_ended, turn_log))
            turn_token_usages.append(_build_token_usage_solo(result, config))
        log_cursor = len(synthesis_log.entries)

        if error_message:
            break

    ended = time.time()
    if error_message:
        combined_events.append(ErrorEvent(idx=len(combined_events), timestamp=_iso(ended), agent=None, message=error_message))

    for i, event in enumerate(combined_events):
        event.idx = i
    _enforce_monotonic_timestamps(combined_events)

    total_usage = TokenUsage()
    for usage in turn_token_usages:
        total_usage.total_prompt_tokens += usage.total_prompt_tokens
        total_usage.total_completion_tokens += usage.total_completion_tokens
        for agent_name, agent_usage in usage.by_agent.items():
            prev = total_usage.by_agent.get(agent_name, AgentTokenUsage())
            total_usage.by_agent[agent_name] = AgentTokenUsage(
                prompt_tokens=prev.prompt_tokens + agent_usage.prompt_tokens,
                completion_tokens=prev.completion_tokens + agent_usage.completion_tokens,
            )

    record = RunRecord(
        run_id=run_id,
        config_hash=compute_config_hash(config),
        case_id=case_id,
        case_family=case_family,
        arm=arm,
        seed=seed,
        started_at=_iso(started),
        ended_at=_iso(ended),
        wall_time_seconds=ended - started,
        events=combined_events,
        token_usage=total_usage,
        attack=attack,
        error=error_message,
    )

    outcome_result = evaluate_outcomes(combined_events, config=config, task=task_context)
    record.outcomes = outcome_result.outcomes
    record.outcome_evidence = outcome_result.evidence
    return record
