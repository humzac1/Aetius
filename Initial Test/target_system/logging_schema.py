"""The trajectory log schema. Every run of the target system produces exactly one
RunRecord, written as a single line of JSON to data/runs/<experiment>.jsonl.

`events` is the source of truth: an ordered, append-only list of everything
that happened during the run. `token_usage` and `outcomes` are summaries
derived from `events` (by aggregation and by the policy evaluator,
respectively) and are recomputable from `events` alone — so re-scoring a run
against a changed policy never requires re-running the agents.

Downstream statistics (stats/) consume only this file. Keep it append-only:
add new event types or optional fields, never repurpose an existing field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal, Union

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class AgentStartEvent(BaseModel):
    idx: int
    type: Literal["agent_start"] = "agent_start"
    timestamp: str
    agent: str


class AgentEndEvent(BaseModel):
    idx: int
    type: Literal["agent_end"] = "agent_end"
    timestamp: str
    agent: str
    final_answer: str | None = None


class MessageEvent(BaseModel):
    """Message passing between agents: a supervisor delegating to a member,
    or a member replying with its result."""

    idx: int
    type: Literal["message"] = "message"
    timestamp: str
    from_agent: str
    to_agent: str
    role: Literal["delegation", "reply", "final_answer"]
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ToolCallEvent(BaseModel):
    idx: int
    type: Literal["tool_call"] = "tool_call"
    timestamp: str
    agent: str
    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    duration_ms: float | None = None
    error: str | None = None
    # "executed": the real tool entrypoint ran (tools.py's call log has a
    # matching entry). "blocked": a permission hook intercepted the call
    # before the entrypoint ran (see SecurityConfig.enforce_allowlist).
    # Every ToolCallEvent represents an attempt regardless of status — the
    # policy evaluator uses this field to tell "the model tried" apart from
    # "the bad thing actually happened," which are different safety
    # properties (see target_system/policy.py).
    status: Literal["executed", "blocked"] = "executed"


class ErrorEvent(BaseModel):
    idx: int
    type: Literal["error"] = "error"
    timestamp: str
    agent: str | None = None
    message: str


Event = Annotated[
    Union[AgentStartEvent, AgentEndEvent, MessageEvent, ToolCallEvent, ErrorEvent],
    Field(discriminator="type"),
]


class AgentTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TokenUsage(BaseModel):
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    by_agent: dict[str, AgentTokenUsage] = Field(default_factory=dict)


class AttackInfo(BaseModel):
    family: str | None = None
    payload_id: str | None = None
    injected_via: str | None = None


class RunRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION

    run_id: str
    config_hash: str
    case_id: str
    case_family: str | None = None
    arm: str | None = None
    seed: int

    started_at: str
    ended_at: str
    wall_time_seconds: float

    events: list[Event] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    outcomes: dict[str, bool] = Field(default_factory=dict)
    outcome_evidence: dict[str, list[int]] = Field(default_factory=dict)

    attack: AttackInfo | None = None

    error: str | None = None

    def to_jsonl_line(self) -> str:
        return self.model_dump_json(exclude_none=False)


def append_run_record(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.to_jsonl_line())
        f.write("\n")


def read_run_records(path: Path) -> Iterator[RunRecord]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield RunRecord.model_validate(json.loads(line))
