"""Builds the baseline SystemConfig and named variants used by preset
experiments (Part 4). Kept separate from config.py (pure schema) and
prompts.py (pure text) so a variant is always a one-line diff against a
named constant."""

from __future__ import annotations

from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig
from target_system.prompts import OPERATOR_PROMPT, RESEARCHER_PROMPT, build_supervisor_prompt

DEFAULT_EMAIL_ALLOWLIST = [
    "billing@ourcompany.example",
    "hr@ourcompany.example",
    "it-support@ourcompany.example",
]

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


def baseline_config(
    *,
    label: str = "baseline",
    provider: str = "mock",
    model_name: str = DEFAULT_ANTHROPIC_MODEL,
    defensive_instruction: bool = True,
    cosmetic_variant: bool = False,
    email_allowlist: list[str] | None = None,
    poisoned_corpus_files: list[str] | None = None,
    poisoned_tool_results: list[str] | None = None,
    enforce_allowlist: bool = False,
    extra_agents: list[AgentSpec] | None = None,
) -> SystemConfig:
    model = ModelConfig(provider=provider, model_name=model_name, temperature=0.0)

    supervisor = AgentSpec(
        role="supervisor",
        name="Supervisor",
        system_prompt=build_supervisor_prompt(
            defensive_instruction=defensive_instruction, cosmetic_variant=cosmetic_variant
        ),
        tools=[],
    )
    researcher = AgentSpec(
        role="researcher",
        name="Researcher",
        system_prompt=RESEARCHER_PROMPT,
        tools=["search_corpus"],
    )
    operator = AgentSpec(
        role="operator",
        name="Operator",
        system_prompt=OPERATOR_PROMPT,
        tools=["send_email", "lookup_customer"],
    )

    agents = [supervisor, researcher, operator] + (extra_agents or [])

    return SystemConfig(
        label=label,
        model=model,
        agents=agents,
        security=SecurityConfig(
            email_allowlist=email_allowlist if email_allowlist is not None else list(DEFAULT_EMAIL_ALLOWLIST),
            poisoned_corpus_files=poisoned_corpus_files or [],
            poisoned_tool_results=poisoned_tool_results or [],
            enforce_allowlist=enforce_allowlist,
        ),
        defensive_instruction=defensive_instruction,
    )
