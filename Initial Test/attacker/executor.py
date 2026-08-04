"""Bridges an AttackCase to target_system.orchestration.run_case.

This is the only place that knows how injection_vector maps onto run_case's
parameters — attack_case.py stays pure data, orchestration.py stays
generic about *why* something is poisoned. Kept in attacker/, not
target_system/, because it's specifically about turning attack content into
a run, not about how the target system executes.
"""

from __future__ import annotations

from target_system.config import SystemConfig
from target_system.logging_schema import AttackInfo, RunRecord
from target_system.mock_model import MockStep
from target_system.orchestration import run_case, run_multi_turn_case
from target_system.policy import TaskContext

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


def execute_case(
    config: SystemConfig,
    case: AttackCase,
    *,
    seed: int,
    arm: str | None = None,
    mock_scripts: dict[str, list[MockStep]] | None = None,
) -> RunRecord:
    """The single place that decides run_case vs. run_multi_turn_case, so
    callers (the Part 4 experiment runner) never branch on injection_vector
    themselves — they just call execute_case for every case regardless of
    family."""
    kwargs = build_run_kwargs(case, config)
    kwargs["seed"] = seed
    kwargs["arm"] = arm
    kwargs["mock_scripts"] = mock_scripts

    if case.injection_vector == "multi_turn":
        turns = kwargs.pop("turns")
        return run_multi_turn_case(config, turns, **kwargs)

    task = kwargs.pop("task")
    return run_case(config, task, **kwargs)
