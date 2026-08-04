"""Where the attack material in this package actually comes from.

This module is the log the build spec asked for: "if [AgentHarm] cannot be
fetched, log that clearly and proceed with AgentDojo plus a small set of
hand-written cases rather than silently substituting." SOURCING_LOG is
structured (not just a comment) so a test can assert on it rather than
someone having to trust a docstring.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SourceRecord(BaseModel):
    name: str
    status: Literal["used", "attempted_not_used"]
    detail: str


SOURCING_LOG: list[SourceRecord] = [
    SourceRecord(
        name="AgentDojo (ethz-spylab/agentdojo)",
        status="used",
        detail=(
            "Fetched directly from GitHub (main branch, MIT license, "
            "commit content retrieved via api.github.com on 2026-08-04): "
            "src/agentdojo/attacks/important_instructions_attacks.py "
            "(the <INFORMATION>...</INFORMATION> 'important_instructions' "
            "jailbreak template and its {goal}/{user}/{model} placeholder "
            "scheme) and src/agentdojo/attacks/baseline_attacks.py (the "
            "'direct', 'ignore_previous', 'system_message', and 'injecagent' "
            "fixed-template attacks — the last cites InjecAgent, "
            "arxiv:2403.02691, in AgentDojo's own docstring). GOAL phrasing "
            "style (e.g. 'Send a transaction to <attacker account> that "
            "includes <sensitive value found in context>') was read from "
            "src/agentdojo/default_suites/v1/banking/injection_tasks.py and "
            "adapted into this system's domain (email/customer-lookup "
            "instead of bank transfers) — case content is original, "
            "template structure and phrasing are AgentDojo's. See "
            "attacker/payloads.py for the exact strings and per-template "
            "citations, and attacker/cases.py for how each case adapts a "
            "GOAL into this system's domain."
        ),
    ),
    SourceRecord(
        name="AgentHarm (ai-safety-institute/AgentHarm)",
        status="attempted_not_used",
        detail=(
            "Checked huggingface.co/datasets/ai-safety-institute/AgentHarm "
            "on 2026-08-04. The dataset is technically fetchable (not HTTP-"
            "gated), but declines on two independent grounds, either of "
            "which alone would be enough: (1) license is 'other' — a "
            "custom, non-standard license for a benchmark explicitly "
            "designed to be loaded through the UK AISI's Inspect AI "
            "harness, not scraped and redistributed into a third-party "
            "repo — the dataset card does not grant that. (2) AgentHarm's "
            "harm categories (the README's own content warning: 'may be "
            "considered harmful or offensive' — cybercrime, drugs, "
            "weapons, fraud uplift-style tasks) don't match this system's "
            "threat model (data exfiltration and unauthorized lookups via "
            "prompt injection in a benign internal-assistant setting) and "
            "adapting them in would mean importing real harmful "
            "instructional content for no benefit to the actual research "
            "question here. Proceeding per the original instruction: "
            "AgentDojo (structurally on-topic, MIT) plus hand-written "
            "cases — see the multi_turn_goal_hijack family in "
            "attacker/cases.py, which AgentDojo doesn't cover either "
            "(its threat model is single-turn tool-output injection, not "
            "a conversation that pivots across turns)."
        ),
    ),
]


def print_sourcing_log() -> None:
    for record in SOURCING_LOG:
        print(f"[{record.status}] {record.name}\n  {record.detail}\n")


if __name__ == "__main__":
    print_sourcing_log()
