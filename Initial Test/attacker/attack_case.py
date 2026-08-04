"""The attack case format. Each case is a reusable, self-contained object:
an id, a family, the benign user task that triggers the run, the injected
payload, and the outcome that counts as success — exactly the four things
the build spec calls out, plus the routing metadata (injection_vector +
target) the executor needs to actually splice the payload into a run.

Cases are stable across runs and config versions by construction: fields
are plain data (no callables, no references to corpus files' current
contents), and `injected_payload` is stored fully rendered rather than
computed on demand — a case's exact text can never drift just because
attacker/payloads.py changes later. The statistical design in stats/ pairs
runs on case id, so an id, once assigned, is never reused for different
content and never renamed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

AttackFamily = Literal[
    "direct_instruction_injection",
    "indirect_injection_document",
    "tool_result_poisoning",
    "multi_turn_goal_hijack",
]

InjectionVector = Literal["task_text", "corpus_document", "tool_result", "multi_turn"]


class AttackCase(BaseModel):
    id: str
    family: AttackFamily
    source: str
    """Provenance: which AgentDojo template + adaptation, or "hand-written"
    with a one-line reason (see attacker/sourcing.py for the full log)."""

    benign_task: str
    """The user task that triggers the run. For single-turn families this
    is the complete (and, for indirect/tool-result families, entirely
    innocent-looking) prompt. For multi_turn_goal_hijack this is turn 1."""

    injected_payload: str
    """The fully-rendered attack text, stable and self-contained — never
    re-rendered from a template at use time."""

    success_outcome: str
    """Which policy.py outcome key (see target_system/policy.py /
    policy.yaml) being True means this case's attack succeeded, e.g.
    "exfiltration" or "unauthorized_lookup"."""

    injection_vector: InjectionVector
    target_corpus_file: str | None = None
    """Required, and must be a real target_system/corpus/*.md filename,
    when injection_vector == "corpus_document"."""
    target_customer_id: str | None = None
    """Required when injection_vector == "tool_result"."""
    turns: list[str] | None = None
    """Required when injection_vector == "multi_turn": the full turn
    sequence including turn 1 (== benign_task) through the turn that
    attempts the hijack (== injected_payload)."""

    requests_customer_data: bool = False
    """Feeds TaskContext — does the benign task legitimately ask for
    customer data, so unauthorized_lookup is scored correctly."""
    success_keywords: list[str] = Field(default_factory=list)
    """Feeds TaskContext for task_success scoring."""

    notes: str = ""
    """Free-text rationale, especially for hand-written cases."""

    @model_validator(mode="after")
    def _vector_target_consistency(self) -> "AttackCase":
        if self.injection_vector == "corpus_document" and not self.target_corpus_file:
            raise ValueError(f"{self.id}: corpus_document vector requires target_corpus_file")
        if self.injection_vector == "tool_result" and not self.target_customer_id:
            raise ValueError(f"{self.id}: tool_result vector requires target_customer_id")
        if self.injection_vector == "multi_turn":
            if not self.turns or len(self.turns) < 2:
                raise ValueError(f"{self.id}: multi_turn vector requires turns with at least 2 entries")
            if self.turns[0] != self.benign_task:
                raise ValueError(f"{self.id}: turns[0] must equal benign_task")
            if self.turns[-1] != self.injected_payload:
                raise ValueError(f"{self.id}: turns[-1] must equal injected_payload")
        return self
