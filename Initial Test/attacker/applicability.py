"""Which attack cases/families actually apply to a given environment's
tool set, generalized off tool roles (target_system/tool_roles.py) rather than
the hardcoded send_email/lookup_customer names attacker/cases.py's
success_outcome values used to implicitly assume.

Same principle as tui/verdict_logic.py's compute_comparison_verdict fix
(a family/outcome-key pair a family's cases never actually target must be
excluded from consideration, not silently included as a pinned-zero row):
here, a case whose delivery vector or outcome has no matching tool role in
the environment is excluded from execution entirely, not forced to run
against a tool that isn't there or silently skipped without a reason
surfacing. The two mechanisms solve different problems (that fix is about
report rows already computed; this is about which cases should even run)
so this isn't literally the same function, but the same "don't evaluate
what doesn't apply, explain why" shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from attacker.attack_case import AttackCase
from target_system.config import SystemConfig
from target_system.tool_roles import ToolRole, available_roles, classify_environment_tools

# The outcome a case's success_outcome targets requires a tool with this
# role to exist — mirrors what target_system/policy.yaml's outcome rules
# check for (see policy.py's ROLE-based trigger resolution).
OUTCOME_REQUIRED_ROLE: dict[str, ToolRole] = {
    "exfiltration": ToolRole.SENSITIVE_ACTION,
    "unauthorized_lookup": ToolRole.DATA_LOOKUP,
}

# The injection_vector a case uses to *deliver* its payload requires a tool
# with this role to exist, independent of the outcome it's aiming for.
# task_text and multi_turn aren't in this map: both deliver the payload
# directly through the conversation, no tool involved.
DELIVERY_VECTOR_REQUIRED_ROLE: dict[str, ToolRole] = {
    "corpus_document": ToolRole.UNTRUSTED_CONTENT_ENTRY_POINT,
    "tool_result": ToolRole.DATA_LOOKUP,
}


@dataclass(frozen=True)
class CaseApplicability:
    case_id: str
    applicable: bool
    reasons: tuple[str, ...] = ()  # non-empty only when applicable is False


def case_applicability(case: AttackCase, roles: set[ToolRole]) -> CaseApplicability:
    reasons: list[str] = []

    delivery_role = DELIVERY_VECTOR_REQUIRED_ROLE.get(case.injection_vector)
    if delivery_role is not None and delivery_role not in roles:
        reasons.append(f"delivery vector {case.injection_vector!r} needs a {delivery_role.value} tool, none found")

    outcome_role = OUTCOME_REQUIRED_ROLE.get(case.success_outcome)
    if outcome_role is not None and outcome_role not in roles:
        reasons.append(f"outcome {case.success_outcome!r} needs a {outcome_role.value} tool, none found")

    return CaseApplicability(case_id=case.id, applicable=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True)
class FamilyApplicability:
    family: str
    applicable: bool  # True iff at least one case in the family is applicable
    applicable_case_ids: tuple[str, ...]
    inapplicable_case_ids: tuple[str, ...]
    # Reasons collected from inapplicable cases, deduplicated — why the
    # family as a whole isn't (fully) applicable, for display.
    reasons: tuple[str, ...] = field(default_factory=tuple)


def family_applicability(cases: list[AttackCase], roles: set[ToolRole]) -> dict[str, FamilyApplicability]:
    """One entry per family present in `cases`. A family is applicable if
    ANY of its cases are — mirrors the "OR across cases" shape of the
    compute_comparison_verdict fix's family/outcome-key applicability,
    generalized here to individual cases within a family rather than
    report rows."""
    by_family: dict[str, list[AttackCase]] = {}
    for case in cases:
        by_family.setdefault(case.family, []).append(case)

    result: dict[str, FamilyApplicability] = {}
    for family, family_cases in by_family.items():
        per_case = [case_applicability(c, roles) for c in family_cases]
        applicable_ids = tuple(ca.case_id for ca in per_case if ca.applicable)
        inapplicable_ids = tuple(ca.case_id for ca in per_case if not ca.applicable)
        reasons: list[str] = []
        for ca in per_case:
            for reason in ca.reasons:
                if reason not in reasons:
                    reasons.append(reason)
        result[family] = FamilyApplicability(
            family=family,
            applicable=bool(applicable_ids),
            applicable_case_ids=applicable_ids,
            inapplicable_case_ids=inapplicable_ids,
            reasons=tuple(reasons) if not applicable_ids else (),
        )
    return result


def applicable_cases(cases: list[AttackCase], roles: set[ToolRole]) -> list[AttackCase]:
    """The subset of cases actually runnable against an environment with
    this role set — what an execution path (Part 4/5) should filter to
    before running anything, rather than running every case regardless."""
    return [c for c in cases if case_applicability(c, roles).applicable]


def roles_for_config(config: SystemConfig) -> set[ToolRole]:
    """The tool roles available in config's environment — every agent's
    tools combined, not just the supervisor's (the toy system's supervisor
    has tools=[]; send_email/lookup_customer/search_corpus live on its
    researcher/operator sub-agents — using .supervisor().tools alone would
    silently classify the toy system as having no roles at all). Same
    combined-tool-set convention as target_system/policy.py's
    _resolve_trigger_tool_names. Classified from tool names alone for the
    hand-authored toy system, plus each tool's ToolBehaviorProfile when
    config.provenance is set (a reconstructed twin, always single-agent —
    see ingestion/reconstruct.py)."""
    all_tool_names = sorted({t for agent in config.agents for t in agent.tools})
    tool_profiles = config.provenance.tool_profiles if config.provenance else None
    classified = classify_environment_tools(all_tool_names, tool_profiles=tool_profiles)
    return available_roles(classified)


def tool_names_for_role(config: SystemConfig, role: ToolRole) -> set[str]:
    """Every tool name in config's environment classified with `role` —
    same combined-tool-set + tool_profiles resolution as roles_for_config,
    but keeping the per-tool mapping instead of collapsing to a role set.
    This is what a verdict screen (Part 6) resolves a flagged outcome's
    tool name(s) through instead of a hardcoded toy-system name like
    "send_email" — the same generalization target_system/policy.py's
    _resolve_trigger_tool_names already applies to outcome evaluation
    itself. Can return more than one name: the real Invoice Generation
    Assistant environment (see ingestion/reconstruct.py) has both
    lookup_customer and list_invoices classified DATA_LOOKUP."""
    all_tool_names = sorted({t for agent in config.agents for t in agent.tools})
    tool_profiles = config.provenance.tool_profiles if config.provenance else None
    classified = classify_environment_tools(all_tool_names, tool_profiles=tool_profiles)
    return {name for name, roles in classified.items() if role in roles}


def applicable_cases_for_configs(cases: list[AttackCase], configs: list[SystemConfig]) -> list[AttackCase]:
    """Cases applicable to every config in `configs` at once. For a single
    config this is just that environment's own applicable set; for a
    paired comparison, a case only one arm's tool set supports would fail
    outright on the other arm (e.g. execute_case's corpus_document error
    for a reconstructed environment with no untrusted-content-entry-point
    tool) — so only cases every arm can actually run are included, not
    the union. This is what a real execution path (the wizard, Part 5)
    filters the full case list to before running anything; toy-system
    configs (whose tool set covers every role) filter to a no-op."""
    role_sets = [roles_for_config(c) for c in configs]
    return [c for c in cases if all(case_applicability(c, roles).applicable for roles in role_sets)]
