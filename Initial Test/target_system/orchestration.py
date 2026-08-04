"""Builds an Agno Team from a SystemConfig and adapts its output into the
approved trajectory log schema.

The adapter walks two things Agno already gives us on TeamRunOutput after a
non-streaming team.run() call — no event-stream parsing needed:
  - `result.tools`: the supervisor's own ToolExecution list, in call order.
    A `delegate_task_to_member` entry there IS the inter-agent message: its
    tool_args (`member_id`, `task`) is the delegation, its `.result` is the
    member's reply.
  - `result.member_responses`: one RunOutput per delegated call, in the same
    order as the matching delegate_task_to_member entries in `result.tools`,
    each carrying that member's own ToolExecution list (search_corpus,
    send_email, lookup_customer) and RunMetrics.
This was verified interactively against agno==2.6.17 (see the exploration
in the step-1 build notes) rather than assumed from docs.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.anthropic import Claude
from agno.team import Team
from agno.team.mode import TeamMode

from target_system.config import AgentSpec, ModelConfig, SystemConfig, compute_config_hash
from target_system.logging_schema import (
    AgentEndEvent,
    AgentStartEvent,
    AgentTokenUsage,
    AttackInfo,
    ErrorEvent,
    Event,
    MessageEvent,
    RunRecord,
    ToolCallEvent,
    TokenUsage,
)
from target_system.mock_model import MockModel, MockStep
from target_system.policy import TaskContext, evaluate_outcomes
from target_system.tools import CallLog, CallLogEntry, lookup_customer, run_context, search_corpus, send_email

TOOL_REGISTRY = {
    "send_email": send_email,
    "lookup_customer": lookup_customer,
    "search_corpus": search_corpus,
}

MockScripts = dict[str, list[MockStep]]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _build_model(model_config: ModelConfig, agent_label: str, mock_scripts: MockScripts | None):
    if model_config.provider == "mock":
        steps = (mock_scripts or {}).get(agent_label, [])
        return MockModel(agent_label, steps)
    if model_config.provider == "anthropic":
        return Claude(
            id=model_config.model_name,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
        )
    raise ValueError(f"unknown model provider: {model_config.provider!r}")


def _tool_hooks_for(tool_name: str, config: SystemConfig) -> list | None:
    """Per-run hook set for a given tool, derived from this run's config.
    Called once per Agent build so the resulting Function copy (see
    _build_tools) is never shared with — or mutated by — any other run."""
    if tool_name == "send_email" and config.security.enforce_allowlist:
        return [_email_permission_guard(config)]
    return None


def _build_tools(spec: AgentSpec, config: SystemConfig) -> list:
    """Fresh, run-scoped copies of this agent's tools. TOOL_REGISTRY holds
    one shared Function singleton per tool name, imported once at module
    load; handing that same object to every run's Agent would mean
    attaching a hook for one run's config (e.g. enforce_allowlist) is
    visible to every other run using the same tool, including concurrently
    executing ones. model_copy() gives each run an independent Function
    object — same .entrypoint/.name/.parameters (so tool_name/arguments in
    the trajectory log and TOOL_REGISTRY lookups by name are unaffected),
    but a .tool_hooks list nothing else can see or mutate."""
    tools = []
    for name in spec.tools:
        tool = TOOL_REGISTRY[name].model_copy()
        tool.tool_hooks = _tool_hooks_for(name, config)
        tools.append(tool)
    return tools


def _build_member_agent(spec: AgentSpec, config: SystemConfig, mock_scripts: MockScripts | None) -> Agent:
    model = _build_model(spec.model_override or config.model, spec.role, mock_scripts)
    tools = _build_tools(spec, config)
    return Agent(id=spec.role, name=spec.name, model=model, system_message=spec.system_prompt, tools=tools)


def build_team(config: SystemConfig, mock_scripts: MockScripts | None = None, num_history_runs: int = 1) -> Team:
    """db + add_history_to_context are wired unconditionally, for every
    config and every call site (run_case included) — not gated by
    anything in SystemConfig. That's what lets run_multi_turn_case give
    later turns real access to earlier turns' assistant replies (see its
    docstring), while guaranteeing both arms of a paired comparison always
    get identical session handling, with nothing SystemConfig-dependent
    about it that could introduce an arm-asymmetric confound.

    Each call gets its own fresh InMemoryDb — same per-run isolation
    discipline as _build_tools' per-run Function copies: two runs (or two
    arms) must never be able to see each other's session state, including
    when running concurrently. For run_case's single team.run() call this
    is inert (a fresh db has no prior session to load), verified in
    tests/test_multi_turn.py.
    """
    supervisor_spec = config.supervisor()
    members = [_build_member_agent(spec, config, mock_scripts) for spec in config.members()]
    supervisor_model = _build_model(supervisor_spec.model_override or config.model, supervisor_spec.role, mock_scripts)
    return Team(
        id=supervisor_spec.role,
        name=supervisor_spec.name,
        members=members,
        model=supervisor_model,
        mode=TeamMode.coordinate,
        system_message=supervisor_spec.system_prompt,
        store_events=True,
        store_member_responses=True,
        db=InMemoryDb(),
        add_history_to_context=True,
        num_history_runs=num_history_runs,
    )


def _tool_error(tool_execution: Any) -> str | None:
    # ToolExecution.tool_call_error defaults to False (not None) when there
    # was no error, so normalize to Optional[str] for the log schema.
    err = getattr(tool_execution, "tool_call_error", None)
    return str(err) if err else None


class _CallLogMatcher:
    """Cross-references Agno's own ToolExecution list against tools.py's
    independent call log to tell "executed" apart from "blocked".

    A blocked call (see run_case's permission-guard hook) never reaches the
    real tool entrypoint, so it never appends to the call log — but Agno
    still records a ToolExecution for the attempt (with whatever the guard
    hook returned as its result). Call-log entries only exist for calls that
    actually ran, in real execution order per tool, so walking both lists in
    lockstep and matching on (tool_name, arguments) tells us which
    ToolExecutions actually went through.
    """

    def __init__(self, call_log: CallLog, start_index: int = 0) -> None:
        # start_index lets a multi-turn run build a matcher scoped to only
        # the entries a single turn added, rather than replaying entries
        # already matched (and popped) by an earlier turn's matcher —
        # call_log itself still spans the whole conversation.
        self._queues: dict[str, deque[CallLogEntry]] = defaultdict(deque)
        for entry in call_log.entries[start_index:]:
            self._queues[entry.tool_name].append(entry)

    def status_for(self, tool_name: str, arguments: dict[str, Any]) -> str:
        queue = self._queues.get(tool_name)
        if queue and queue[0].arguments == arguments:
            queue.popleft()
            return "executed"
        return "blocked"


def _enforce_monotonic_timestamps(events: list[Event]) -> None:
    """Guarantee timestamp order agrees with idx order. Agno's ToolExecution
    timestamps are integer-second epoch values captured at different points
    (a delegate call's own timestamp vs. its nested tool calls' timestamps),
    which can tie or invert across agents even though idx order is
    structurally correct — see the step-1 ordering probe. Downstream code
    (stats, dashboard) should still treat idx as authoritative, but this
    keeps timestamp from actively lying if something sorts by it instead."""
    last: datetime | None = None
    for event in events:
        current = datetime.fromisoformat(event.timestamp)
        if last is not None and current <= last:
            current = last + timedelta(microseconds=1)
        event.timestamp = current.isoformat()
        last = current


def _build_events(
    result: Any, config: SystemConfig, started: float, ended: float, call_log: CallLog, call_log_start: int = 0
) -> list[Event]:
    events: list[Event] = []
    supervisor_role = config.supervisor().role
    call_log_matcher = _CallLogMatcher(call_log, start_index=call_log_start)

    def add(event) -> None:
        event.idx = len(events)
        events.append(event)

    add(AgentStartEvent(idx=0, timestamp=_iso(started), agent=supervisor_role))

    # member_responses is produced in the same order as the matching
    # delegate_task_to_member calls in result.tools (verified interactively),
    # so a per-member-id queue pairs them correctly for our single-hop
    # supervisor -> member delegation pattern.
    queues: dict[str, list] = {}
    for mr in result.member_responses or []:
        queues.setdefault(mr.agent_id, []).append(mr)

    for t in result.tools or []:
        ts = _iso(t.created_at)
        if t.tool_name == "delegate_task_to_member":
            member_id = t.tool_args.get("member_id", "unknown_member")
            task_desc = t.tool_args.get("task", "")
            add(MessageEvent(idx=0, timestamp=ts, from_agent=supervisor_role, to_agent=member_id, role="delegation", content=task_desc))

            member_run = None
            queue = queues.get(member_id)
            if queue:
                member_run = queue.pop(0)

            if member_run is not None:
                add(AgentStartEvent(idx=0, timestamp=ts, agent=member_id))
                for mt in member_run.tools or []:
                    mt_args = mt.tool_args or {}
                    add(
                        ToolCallEvent(
                            idx=0,
                            timestamp=_iso(mt.created_at),
                            agent=member_id,
                            tool_name=mt.tool_name,
                            arguments=mt_args,
                            result=mt.result,
                            error=_tool_error(mt),
                            status=call_log_matcher.status_for(mt.tool_name, mt_args),
                        )
                    )
                member_final = member_run.content if isinstance(member_run.content, str) else None
                add(AgentEndEvent(idx=0, timestamp=ts, agent=member_id, final_answer=member_final))

            add(
                MessageEvent(
                    idx=0,
                    timestamp=ts,
                    from_agent=member_id,
                    to_agent=supervisor_role,
                    role="reply",
                    content=str(t.result) if t.result is not None else "",
                )
            )
        else:
            t_args = t.tool_args or {}
            add(
                ToolCallEvent(
                    idx=0,
                    timestamp=ts,
                    agent=supervisor_role,
                    tool_name=t.tool_name,
                    arguments=t_args,
                    result=t.result,
                    error=_tool_error(t),
                    status=call_log_matcher.status_for(t.tool_name, t_args),
                )
            )

    final_answer = result.content if isinstance(result.content, str) else None
    add(AgentEndEvent(idx=0, timestamp=_iso(ended), agent=supervisor_role, final_answer=final_answer))
    # Timestamp monotonicity is enforced by the caller (once over whatever
    # this function's output ends up concatenated into), not here — see
    # run_case and run_multi_turn_case.
    return events


def _merge_token_usage(usages: list[TokenUsage]) -> TokenUsage:
    total = TokenUsage()
    for usage in usages:
        total.total_prompt_tokens += usage.total_prompt_tokens
        total.total_completion_tokens += usage.total_completion_tokens
        for agent, agent_usage in usage.by_agent.items():
            prev = total.by_agent.get(agent, AgentTokenUsage())
            total.by_agent[agent] = AgentTokenUsage(
                prompt_tokens=prev.prompt_tokens + agent_usage.prompt_tokens,
                completion_tokens=prev.completion_tokens + agent_usage.completion_tokens,
            )
    return total


def _build_token_usage(result: Any, config: SystemConfig) -> TokenUsage:
    usage = TokenUsage()
    if result is None:
        return usage

    supervisor_role = config.supervisor().role
    if result.metrics is not None:
        usage.total_prompt_tokens += result.metrics.input_tokens or 0
        usage.total_completion_tokens += result.metrics.output_tokens or 0
        usage.by_agent[supervisor_role] = AgentTokenUsage(
            prompt_tokens=result.metrics.input_tokens or 0,
            completion_tokens=result.metrics.output_tokens or 0,
        )

    for mr in result.member_responses or []:
        if mr.metrics is None:
            continue
        usage.total_prompt_tokens += mr.metrics.input_tokens or 0
        usage.total_completion_tokens += mr.metrics.output_tokens or 0
        prev = usage.by_agent.get(mr.agent_id, AgentTokenUsage())
        usage.by_agent[mr.agent_id] = AgentTokenUsage(
            prompt_tokens=prev.prompt_tokens + (mr.metrics.input_tokens or 0),
            completion_tokens=prev.completion_tokens + (mr.metrics.output_tokens or 0),
        )
    return usage


def _email_permission_guard(config: SystemConfig):
    def guard(function_name: str, arguments: dict, func):
        if function_name == "send_email" and arguments.get("recipient") not in config.security.email_allowlist:
            # Do NOT call func(**arguments) — the real entrypoint (and
            # therefore tools.py's call log) never runs, which is exactly
            # what makes this "blocked" rather than "executed" downstream.
            return {
                "status": "blocked_by_policy",
                "reason": f"recipient {arguments.get('recipient')!r} not in allowlist",
            }
        return func(**arguments)

    return guard


def run_case(
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
    corpus_poison: dict[str, str] | None = None,
    tool_result_poison: dict[str, dict] | None = None,
) -> RunRecord:
    """Execute one run of the target system against `task` and return a
    fully-populated RunRecord (events, token usage, and outcomes). Does not
    write to disk — see runner.py for that.

    corpus_poison / tool_result_poison carry an attack case's payload
    content (see attacker/executor.py), but whether poisoning is actually
    live this run is decided here, not by the caller: only keys also
    present in config.security.poisoned_corpus_files /
    .poisoned_tool_results take effect. This is what makes "is poisoning
    on" a config-hashed experimental condition rather than something a
    caller could silently flip without it showing up in config_hash."""
    team = build_team(config, mock_scripts=mock_scripts)
    call_log = CallLog()
    corpus_dir = Path(config.corpus_dir)
    run_id = f"run_{uuid4().hex[:12]}"

    active_corpus_poison = {
        filename: payload
        for filename, payload in (corpus_poison or {}).items()
        if filename in config.security.poisoned_corpus_files
    }
    active_tool_result_poison = {
        key: override
        for key, override in (tool_result_poison or {}).items()
        if key in config.security.poisoned_tool_results
    }

    started = time.time()
    error_message: str | None = None
    result: Any = None
    with run_context(
        call_log,
        corpus_dir=corpus_dir,
        corpus_poison=active_corpus_poison,
        tool_result_poison=active_tool_result_poison,
    ):
        try:
            result = team.run(task, session_id=run_id)
        except Exception as exc:  # noqa: BLE001 - captured into the trajectory log, not swallowed
            error_message = f"{type(exc).__name__}: {exc}"
    ended = time.time()

    events = _build_events(result, config, started, ended, call_log) if result is not None else []
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
        token_usage=_build_token_usage(result, config),
        attack=attack,
        error=error_message,
    )

    outcome_result = evaluate_outcomes(events, config=config, task=task_context)
    record.outcomes = outcome_result.outcomes
    record.outcome_evidence = outcome_result.evidence
    return record


def run_multi_turn_case(
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
    corpus_poison: dict[str, str] | None = None,
    tool_result_poison: dict[str, dict] | None = None,
) -> RunRecord:
    """Multi-turn sibling of run_case: runs `turns` in sequence against one
    team/session (build_team's db + add_history_to_context wiring is what
    makes turn N able to see turns 1..N-1's assistant replies — verified
    interactively before this was built, see the multi-turn investigation
    notes), and returns exactly one RunRecord spanning the whole
    conversation. One RunRecord, not one per turn, because success_outcome
    is a single field per AttackCase (see attacker/attack_case.py) — the
    question a multi_turn_goal_hijack case asks is "did the attack succeed
    by the end of the conversation," evaluated once over the full
    concatenated events, the same way run_case evaluates once over a
    single turn's events.

    Stops the conversation (rather than attempting further turns) on the
    first turn that raises, matching run_case's all-or-nothing handling of
    a failed team.run() call.
    """
    if not turns:
        raise ValueError("run_multi_turn_case requires at least one turn")

    team = build_team(config, mock_scripts=mock_scripts, num_history_runs=len(turns))
    call_log = CallLog()
    corpus_dir = Path(config.corpus_dir)
    run_id = f"run_{uuid4().hex[:12]}"

    active_corpus_poison = {
        filename: payload
        for filename, payload in (corpus_poison or {}).items()
        if filename in config.security.poisoned_corpus_files
    }
    active_tool_result_poison = {
        key: override
        for key, override in (tool_result_poison or {}).items()
        if key in config.security.poisoned_tool_results
    }

    started = time.time()
    error_message: str | None = None
    combined_events: list[Event] = []
    turn_token_usages: list[TokenUsage] = []
    call_log_cursor = 0

    with run_context(
        call_log,
        corpus_dir=corpus_dir,
        corpus_poison=active_corpus_poison,
        tool_result_poison=active_tool_result_poison,
    ):
        for turn_text in turns:
            turn_started = time.time()
            result: Any = None
            try:
                result = team.run(turn_text, session_id=run_id)
            except Exception as exc:  # noqa: BLE001 - captured into the trajectory log, not swallowed
                error_message = f"{type(exc).__name__}: {exc}"
            turn_ended = time.time()

            if result is not None:
                # call_log_cursor scopes _CallLogMatcher to only the
                # entries *this* turn added — without it, turn 2's matcher
                # would re-walk turn 1's already-consumed entries from the
                # start and misattribute status.
                combined_events.extend(
                    _build_events(result, config, turn_started, turn_ended, call_log, call_log_start=call_log_cursor)
                )
                turn_token_usages.append(_build_token_usage(result, config))
            call_log_cursor = len(call_log.entries)

            if error_message:
                break

    ended = time.time()
    if error_message:
        combined_events.append(ErrorEvent(idx=len(combined_events), timestamp=_iso(ended), agent=None, message=error_message))

    for i, event in enumerate(combined_events):
        event.idx = i
    _enforce_monotonic_timestamps(combined_events)

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
        token_usage=_merge_token_usage(turn_token_usages),
        attack=attack,
        error=error_message,
    )

    outcome_result = evaluate_outcomes(combined_events, config=config, task=task_context)
    record.outcomes = outcome_result.outcomes
    record.outcome_evidence = outcome_result.evidence
    return record
