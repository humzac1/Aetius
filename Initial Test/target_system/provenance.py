"""Types describing a config reconstructed from real Langfuse traces rather
than hand-authored — SystemConfig.provenance (see config.py) is an optional
instance of ReconstructionProvenance, None for every hand-authored (toy
system) config. Mostly excluded from compute_config_hash: provenance
describes where a config came from, not what it resolves to, same
reasoning as label already being excluded. The exception is the subset
config._provenance_identity folds in (project_id, source_agent_name,
trace_count, tool_profiles) — that part is the reconstructed
environment's actual content, and leaving it out of the hash let two
different pulls of the same agent collide on one cfg_* id.

Lives in target_system/, not ingestion/, because SystemConfig (the shared
schema every downstream module — experiments/, attacker/, tui/ — already
consumes) needs to reference these types without creating a dependency on
the ingestion pipeline. ingestion/reconstruct.py is the only place that
*constructs* these; everything else just reads them off a loaded config.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ArgumentProfile(BaseModel):
    """Observed shape of one argument across every call to a tool in the
    source trace batch — never a declared schema, since none exists in the
    source data (see ingestion/langfuse_client.py's investigation notes)."""

    observed_types: list[str] = Field(default_factory=list)
    distinct_value_count: int = 0
    sample_values: list[Any] = Field(default_factory=list)  # capped by the extractor
    numeric_range: tuple[float, float] | None = None
    string_length_range: tuple[int, int] | None = None


class ObservedToolCall(BaseModel):
    """One real (arguments, response) pair from the source traces — raw
    material for Part 4's nearest-match response synthesizer, not just a
    statistical summary."""

    arguments: dict[str, Any] = Field(default_factory=dict)
    response: Any = None
    observed_at: str | None = None


class ToolBehaviorProfile(BaseModel):
    """Everything inferable about one reconstructed tool from the full
    trace batch it was built from."""

    tool_name: str
    n_calls_observed: int = 0
    argument_profiles: dict[str, ArgumentProfile] = Field(default_factory=dict)
    # Union of top-level response keys seen — deliberately not assumed to be
    # any particular envelope shape (e.g. {"result", "success", "error"}):
    # that's what this project's tools happen to use, not a Langfuse or
    # tool-calling convention in general.
    response_key_set: list[str] = Field(default_factory=list)
    example_calls: list[ObservedToolCall] = Field(default_factory=list)  # capped by the extractor


class OtherGroupFound(BaseModel):
    """A trace-batch cluster that wasn't reconstructed this time — surfaced
    so a caller knows other environments exist in the same cached pull
    rather than silently discarding them (see ingestion/reconstruct.py's
    grouping)."""

    agent_name: str | None
    trace_count: int


class ReconstructionProvenance(BaseModel):
    """Disclosed by the verdict layer (Part 6) alongside any result for a
    reconstructed environment — trace_count and warnings in particular are
    meant to be shown inline, not buried, same discipline as the
    single-config screen's "attacks actually tried" disclaimer."""

    project_id: str
    source_agent_name: str | None
    trace_count: int
    extraction_date: str  # ISO 8601
    other_groups_found: list[OtherGroupFound] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tool_profiles: dict[str, ToolBehaviorProfile] = Field(default_factory=dict)
    # Real, observed cost/token figures from the source traces (Langfuse
    # already computes these per generation — never estimated or guessed),
    # excluding the judge-evaluation generation every trace also carries
    # (that's eval infrastructure cost, not the agent's own). None when the
    # source traces didn't carry cost/token data at all.
    # experiments/cost_estimate.py is the only place these get turned into
    # a batch estimate — this module just carries what was actually observed.
    avg_generations_per_trace: float | None = None
    avg_prompt_tokens_per_generation: float | None = None
    avg_completion_tokens_per_generation: float | None = None
    avg_cost_usd_per_trace: float | None = None
