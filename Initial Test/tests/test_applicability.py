from attacker.applicability import applicable_cases, applicable_cases_for_configs, case_applicability, family_applicability, roles_for_config
from attacker.attack_case import AttackCase
from attacker.cases import ATTACK_CASES
from target_system.tool_roles import ToolRole


def _case(id, family, injection_vector, success_outcome, **kwargs):
    defaults = dict(source="test", benign_task="do x", injected_payload="do y")
    if injection_vector == "corpus_document":
        defaults["target_corpus_file"] = "f.md"
    if injection_vector == "tool_result":
        defaults["target_customer_id"] = "CUST-1"
    if injection_vector == "multi_turn":
        defaults["turns"] = [defaults["benign_task"], defaults["injected_payload"]]
    defaults.update(kwargs)
    return AttackCase(id=id, family=family, injection_vector=injection_vector, success_outcome=success_outcome, **defaults)


ALL_ROLES = {ToolRole.SENSITIVE_ACTION, ToolRole.DATA_LOOKUP, ToolRole.UNTRUSTED_CONTENT_ENTRY_POINT}


# --- case_applicability -------------------------------------------------


def test_task_text_case_needs_only_outcome_role():
    case = _case("c1", "direct_instruction_injection", "task_text", "exfiltration")
    assert case_applicability(case, {ToolRole.SENSITIVE_ACTION}).applicable is True
    assert case_applicability(case, set()).applicable is False


def test_corpus_document_case_needs_entry_point_and_outcome_role():
    case = _case("c1", "indirect_injection_document", "corpus_document", "exfiltration")
    assert case_applicability(case, ALL_ROLES).applicable is True
    assert case_applicability(case, {ToolRole.UNTRUSTED_CONTENT_ENTRY_POINT}).applicable is False  # missing outcome role
    assert case_applicability(case, {ToolRole.SENSITIVE_ACTION}).applicable is False  # missing entry-point role


def test_tool_result_case_needs_data_lookup_and_outcome_role():
    case = _case("c1", "tool_result_poisoning", "tool_result", "exfiltration")
    assert case_applicability(case, {ToolRole.DATA_LOOKUP, ToolRole.SENSITIVE_ACTION}).applicable is True
    assert case_applicability(case, {ToolRole.SENSITIVE_ACTION}).applicable is False  # missing delivery role


def test_multi_turn_case_needs_only_outcome_role():
    case = _case("c1", "multi_turn_goal_hijack", "multi_turn", "unauthorized_lookup")
    assert case_applicability(case, {ToolRole.DATA_LOOKUP}).applicable is True
    assert case_applicability(case, set()).applicable is False


def test_inapplicable_case_reports_reasons():
    case = _case("c1", "indirect_injection_document", "corpus_document", "exfiltration")
    result = case_applicability(case, set())
    assert result.applicable is False
    assert len(result.reasons) == 2  # delivery vector AND outcome both unmet


def test_applicable_case_has_no_reasons():
    case = _case("c1", "direct_instruction_injection", "task_text", "exfiltration")
    result = case_applicability(case, {ToolRole.SENSITIVE_ACTION})
    assert result.reasons == ()


# --- family_applicability -------------------------------------------------


def test_family_applicable_if_any_case_applicable():
    cases = [
        _case("c1", "tool_result_poisoning", "tool_result", "exfiltration"),
        _case("c2", "tool_result_poisoning", "tool_result", "unauthorized_lookup"),
    ]
    # only unauthorized_lookup's role is available -> c1 not applicable, c2 is
    fa = family_applicability(cases, {ToolRole.DATA_LOOKUP})
    info = fa["tool_result_poisoning"]
    assert info.applicable is True
    assert info.applicable_case_ids == ("c2",)
    assert info.inapplicable_case_ids == ("c1",)


def test_family_not_applicable_when_no_case_applicable():
    cases = [_case("c1", "indirect_injection_document", "corpus_document", "exfiltration")]
    fa = family_applicability(cases, set())
    info = fa["indirect_injection_document"]
    assert info.applicable is False
    assert info.applicable_case_ids == ()
    assert len(info.reasons) > 0


def test_family_applicability_covers_every_family_present():
    fa = family_applicability(ATTACK_CASES, ALL_ROLES)
    assert set(fa.keys()) == {"direct_instruction_injection", "indirect_injection_document", "tool_result_poisoning", "multi_turn_goal_hijack"}


# --- applicable_cases --------------------------------------------------------


def test_applicable_cases_filters_correctly():
    cases = [
        _case("c1", "direct_instruction_injection", "task_text", "exfiltration"),
        _case("c2", "indirect_injection_document", "corpus_document", "exfiltration"),
    ]
    result = applicable_cases(cases, {ToolRole.SENSITIVE_ACTION})  # no entry-point role
    assert [c.id for c in result] == ["c1"]


# --- real-data regression: locks in the Part 3 report's findings -----------


def test_toy_system_all_families_applicable():
    from target_system.factory import baseline_config
    from target_system.tool_roles import available_roles, classify_environment_tools

    toy = baseline_config()
    tools = sorted({t for a in toy.agents for t in a.tools})
    roles = available_roles(classify_environment_tools(tools))
    fa = family_applicability(ATTACK_CASES, roles)
    assert all(info.applicable for info in fa.values())


def test_toolless_environment_no_families_applicable():
    fa = family_applicability(ATTACK_CASES, set())
    assert not any(info.applicable for info in fa.values())


# --- roles_for_config / applicable_cases_for_configs ------------------------


def test_roles_for_config_combines_every_agents_tools():
    # Regression: the toy system's supervisor has tools=[] -- send_email/
    # lookup_customer/search_corpus live on its researcher/operator
    # sub-agents. Looking at only one agent's tools silently classified
    # the toy system as having no roles at all.
    from target_system.factory import baseline_config

    roles = roles_for_config(baseline_config())
    assert roles == {ToolRole.SENSITIVE_ACTION, ToolRole.DATA_LOOKUP, ToolRole.UNTRUSTED_CONTENT_ENTRY_POINT}


def test_roles_for_config_reconstructed_uses_tool_profiles():
    from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig
    from target_system.provenance import ReconstructionProvenance, ToolBehaviorProfile

    config = SystemConfig(
        label="recon",
        model=ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="[unavailable]", system_prompt_source="unavailable", tools=["send_invoice"])],
        security=SecurityConfig(),
        provenance=ReconstructionProvenance(
            project_id="p", source_agent_name="A", trace_count=1, extraction_date="x",
            tool_profiles={"send_invoice": ToolBehaviorProfile(tool_name="send_invoice")},
        ),
    )
    assert roles_for_config(config) == {ToolRole.SENSITIVE_ACTION}


def test_applicable_cases_for_configs_full_toy_system_keeps_every_case():
    from target_system.factory import baseline_config

    result = applicable_cases_for_configs(list(ATTACK_CASES), [baseline_config()])
    assert len(result) == len(ATTACK_CASES)


def test_applicable_cases_for_configs_intersects_across_arms():
    from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig

    full = SystemConfig(
        label="full", model=ModelConfig(provider="mock", model_name="m"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="x", tools=["send_email", "lookup_customer", "search_corpus"])],
        security=SecurityConfig(),
    )
    no_lookup = SystemConfig(
        label="no-lookup", model=ModelConfig(provider="mock", model_name="m"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="x", tools=["send_email", "search_corpus"])],
        security=SecurityConfig(),
    )
    case = _case("c1", "tool_result_poisoning", "tool_result", "exfiltration")
    assert applicable_cases_for_configs([case], [full]) == [case]
    assert applicable_cases_for_configs([case], [full, no_lookup]) == []  # no_lookup can't deliver it
