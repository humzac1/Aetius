"""Bridges an AttackCase to a run — target_system.orchestration.run_case
for the toy system, target_system.reconstructed_execution.run_reconstructed
_case for a reconstructed twin (SystemConfig.provenance is not None).

This is the only place that knows how injection_vector maps onto either
path's parameters — attack_case.py stays pure data, both execution modules
stay generic about *why* something is poisoned. Kept in attacker/, not
target_system/, because it's specifically about turning attack content into
a run, not about how the target system executes.
"""

from __future__ import annotations

from typing import Any

from target_system.config import SystemConfig
from target_system.logging_schema import AttackInfo, RunRecord
from target_system.mock_model import MockStep
from target_system.orchestration import run_case, run_multi_turn_case
from target_system.policy import TaskContext
from target_system.reconstructed_execution import run_reconstructed_case, run_reconstructed_multi_turn_case
from target_system.tool_roles import ToolRole, classify_environment_tools

from attacker.attack_case import AttackCase


def build_run_kwargs(case: AttackCase, config: SystemConfig) -> dict:
    """Returns the kwargs run_case(...)/run_multi_turn_case(...) needs to
    execute `case` — the caller picks which function based on
    case.injection_vector (see execute_case below). Does not itself check
    config.security.poisoned_corpus_files / .poisoned_tool_results —
    run_case/run_multi_turn_case do that filtering, so a case handed to a
    config that doesn't declare the target poisoned correctly runs as if
    the corpus/tool result were never poisoned at all (the "known-neutral
    change" / clean-baseline-arm scenario)."""
    task_context_kwargs: dict = {
        "task_id": case.id,
        "requests_customer_data": case.requests_customer_data,
        "success_keywords": list(case.success_keywords),
    }
    if case.target_customer_id and case.requests_customer_data:
        task_context_kwargs["authorized_customer_ids"] = [case.target_customer_id]

    kwargs: dict = {
        "case_id": case.id,
        "case_family": case.family,
        "task_context": TaskContext(**task_context_kwargs),
        "attack": AttackInfo(
            family=case.family,
            payload_id=case.id,
            injected_via=case.target_corpus_file or case.target_customer_id,
        ),
    }

    if case.injection_vector == "task_text":
        kwargs["task"] = f"{case.benign_task}\n\n{case.injected_payload}"
    elif case.injection_vector == "corpus_document":
        kwargs["task"] = case.benign_task
        kwargs["corpus_poison"] = {case.target_corpus_file: case.injected_payload}
    elif case.injection_vector == "tool_result":
        kwargs["task"] = case.benign_task
        kwargs["tool_result_poison"] = {case.target_customer_id: {"notes": case.injected_payload}}
    elif case.injection_vector == "multi_turn":
        # No "task" kwarg — run_multi_turn_case takes `turns` positionally
        # instead. case.turns is already validated (turns[0]==benign_task,
        # turns[-1]==injected_payload) by AttackCase's model_validator.
        kwargs["turns"] = list(case.turns)
    else:
        raise ValueError(f"{case.id}: unhandled injection_vector {case.injection_vector!r}")

    return kwargs


def build_reconstructed_run_kwargs(case: AttackCase, config: SystemConfig) -> dict:
    """Reconstructed-twin sibling of build_run_kwargs. Differs where the
    delivery mechanism genuinely differs, not just in naming:
      - corpus_document has no reconstructed delivery mechanism yet — it
        needs an untrusted_content_entry_point tool (a document/corpus
        search), and reconstructing content-poisoning for one is out of
        scope for Part 4 (neither real reconstructed environment
        investigated so far has such a tool — see the Part 3 report).
        attacker.applicability should have already filtered this case out
        before it reaches execute_case; this raises loudly instead of
        silently running a non-attack if that didn't happen.
      - tool_result poisoning has no customer-record fixture to address by
        ID (there's no real backing database for a reconstructed tool) —
        instead, every tool classified DATA_LOOKUP in this environment
        gets the payload embedded in whatever response it returns this
        run (see target_system.tool_synthesis._merge_injected_content).
        Coarser than the toy system's per-customer-ID targeting, but
        there's no reconstructed equivalent to target more precisely.
    """
    task_context_kwargs: dict = {
        "task_id": case.id,
        "requests_customer_data": case.requests_customer_data,
        "success_keywords": list(case.success_keywords),
    }
    kwargs: dict = {
        "case_id": case.id,
        "case_family": case.family,
        "task_context": TaskContext(**task_context_kwargs),
        "attack": AttackInfo(family=case.family, payload_id=case.id, injected_via=case.target_customer_id),
    }

    if case.injection_vector == "task_text":
        kwargs["task"] = f"{case.benign_task}\n\n{case.injected_payload}"
    elif case.injection_vector == "tool_result":
        kwargs["task"] = case.benign_task
        agent_spec = config.supervisor()
        classified = classify_environment_tools(agent_spec.tools, tool_profiles=(config.provenance.tool_profiles if config.provenance else None))
        target_tools = [name for name, roles in classified.items() if ToolRole.DATA_LOOKUP in roles]
        kwargs["tool_result_poison"] = {name: case.injected_payload for name in target_tools}
    elif case.injection_vector == "multi_turn":
        kwargs["turns"] = list(case.turns)
    elif case.injection_vector == "corpus_document":
        raise ValueError(
            f"{case.id}: corpus_document injection_vector has no reconstructed-environment delivery mechanism — "
            "this case should have been filtered out by attacker.applicability before reaching execute_case"
        )
    else:
        raise ValueError(f"{case.id}: unhandled injection_vector {case.injection_vector!r}")

    return kwargs


def execute_case(
    config: SystemConfig,
    case: AttackCase,
    *,
    seed: int,
    arm: str | None = None,
    mock_scripts: dict[str, list[MockStep]] | None = None,
    anthropic_client: Any = None,
) -> RunRecord:
    """The single place that decides which run function executes `case` —
    toy system (target_system.orchestration) vs. reconstructed twin
    (target_system.reconstructed_execution, whenever
    config.provenance is not None) — and, within either, run_case vs.
    run_multi_turn_case. Callers (the experiment runner) never branch on
    injection_vector or config type themselves.

    anthropic_client is only used by the reconstructed path (for the tool-
    response synthesizer's generation fallback — see
    target_system.tool_synthesis) and ignored for the toy system.

    Reconstructed environments are real-model-only: config.provenance is
    not None but config.model.provider != "anthropic" is rejected here,
    not silently allowed to run. target_system.reconstructed_execution
    itself still accepts provider="mock" (that's deliberate — it's what
    lets this project's own tests exercise the execution engine cheaply,
    see tests/test_reconstructed_execution.py), but nothing reaches that
    engine without passing through this dispatch point first, so gating
    here is enough to make "reconstructed + mock" unreachable for every
    real caller (wizard, TUI, experiment runner) without weakening the
    lower-level engine's own testability."""
    if config.provenance is not None:
        if config.model.provider != "anthropic":
            raise ValueError(
                f"{case.id}: reconstructed environments (config.provenance is not None) only run under "
                f"provider='anthropic' — got provider={config.model.provider!r}. There is no mock-scripting "
                "path for a reconstructed twin (Part 4/5): a mock run would execute without error but produce "
                "no meaningful attack behavior at all, which is worse than failing loudly here."
            )
        kwargs = build_reconstructed_run_kwargs(case, config)
        kwargs["seed"] = seed
        kwargs["arm"] = arm
        kwargs["mock_scripts"] = mock_scripts
        kwargs["anthropic_client"] = anthropic_client

        if case.injection_vector == "multi_turn":
            turns = kwargs.pop("turns")
            return run_reconstructed_multi_turn_case(config, turns, **kwargs)

        task = kwargs.pop("task")
        return run_reconstructed_case(config, task, **kwargs)

    kwargs = build_run_kwargs(case, config)
    kwargs["seed"] = seed
    kwargs["arm"] = arm
    kwargs["mock_scripts"] = mock_scripts

    if case.injection_vector == "multi_turn":
        turns = kwargs.pop("turns")
        return run_multi_turn_case(config, turns, **kwargs)

    task = kwargs.pop("task")
    return run_case(config, task, **kwargs)
