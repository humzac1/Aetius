import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from attacker.cases import by_family
from experiments.presets import ArmSpec
from experiments.runner import OUTCOME_KEYS, run_experiment
from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig
from target_system.logging_schema import read_run_records
from target_system.provenance import ReconstructionProvenance, ToolBehaviorProfile


def _small_case_subset():
    # Keep tests fast: one case from two different families rather than
    # the full 17-case suite.
    return [by_family("direct_instruction_injection")[0], by_family("tool_result_poisoning")[0]]


def test_run_experiment_identical_arms_share_config_hash(tmp_path: Path):
    arm_a = ArmSpec(label="a", overrides={})
    arm_b = ArmSpec(label="b", overrides={})
    result = run_experiment(
        arm_a, arm_b, experiment_name="test_identical", cases=_small_case_subset(),
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )
    assert result.arm_a_hash == result.arm_b_hash
    assert set(result.family_results.keys()) == set(OUTCOME_KEYS)


def test_run_experiment_writes_expected_number_of_records(tmp_path: Path):
    cases = _small_case_subset()
    n_runs = 2
    arm_a = ArmSpec(label="a", overrides={})
    arm_b = ArmSpec(label="b", overrides={"email_allowlist": ["only-this@ourcompany.example"]})
    result = run_experiment(
        arm_a, arm_b, experiment_name="test_record_count", cases=cases,
        n_runs_per_case=n_runs, max_workers=2, runs_dir=tmp_path,
    )
    expected = len(cases) * n_runs * 2
    assert len(result.records) == expected
    assert result.n_executed == expected
    assert result.n_cached == 0

    on_disk = list(read_run_records(tmp_path / "test_record_count.jsonl"))
    assert len(on_disk) == expected


def test_run_experiment_is_resumable_and_caches(tmp_path: Path):
    cases = _small_case_subset()
    arm_a = ArmSpec(label="a", overrides={})
    arm_b = ArmSpec(label="b", overrides={})

    first = run_experiment(
        arm_a, arm_b, experiment_name="test_resume", cases=cases,
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )
    assert first.n_executed == len(cases) * 2 * 2
    assert first.n_cached == 0

    second = run_experiment(
        arm_a, arm_b, experiment_name="test_resume", cases=cases,
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )
    assert second.n_executed == 0
    assert second.n_cached == first.n_executed
    assert len(second.records) == len(first.records)


def test_run_experiment_partial_cache_only_executes_the_gap(tmp_path: Path):
    cases = _small_case_subset()
    arm_a = ArmSpec(label="a", overrides={})
    arm_b = ArmSpec(label="b", overrides={})

    run_experiment(arm_a, arm_b, experiment_name="test_partial", cases=cases, n_runs_per_case=2, max_workers=2, runs_dir=tmp_path)
    # Ask for MORE runs per case than before -- only the new ones should execute.
    result = run_experiment(arm_a, arm_b, experiment_name="test_partial", cases=cases, n_runs_per_case=4, max_workers=2, runs_dir=tmp_path)
    assert result.n_cached == len(cases) * 2 * 2
    assert result.n_executed == len(cases) * 2 * 2  # the extra 2 runs/case/arm
    assert len(result.records) == len(cases) * 4 * 2


def _reconstructed_config(label, *, tools=("send_invoice",), provider="anthropic"):
    return SystemConfig(
        label=label,
        model=ModelConfig(provider=provider, model_name="claude-haiku-4-5-20251001" if provider == "anthropic" else "mock-model"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="[unavailable]", system_prompt_source="unavailable", tools=list(tools))],
        security=SecurityConfig(),
        provenance=ReconstructionProvenance(
            project_id="proj-1", source_agent_name="A", trace_count=5, extraction_date="2026-01-01",
            tool_profiles={name: ToolBehaviorProfile(tool_name=name) for name in tools},
        ),
    )


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


def _fake_agent_run(self, task, *, session_id=None, **kwargs):
    return _FakeRunOutput(content="Done.", tools=[])  # no tool calls needed for these tests


def test_run_experiment_reconstructed_config_does_not_use_toy_mock_scripts(tmp_path: Path, monkeypatch):
    # build_mock_scripts scripts a delegate_task_to_member call the solo
    # reconstructed agent has no such function for — confirmed live this
    # produced "Function not found" noise before runner.py special-cased it.
    # This just confirms the run completes cleanly (no exception, valid
    # records written), not that any particular attack behavior fires.
    from agno.agent import Agent

    monkeypatch.setattr(Agent, "run", _fake_agent_run)
    cases = _small_case_subset()
    config_a = _reconstructed_config("a")
    config_b = _reconstructed_config("b")
    result = run_experiment(config_a, config_b, experiment_name="test_reconstructed", cases=cases, n_runs_per_case=1, max_workers=2, runs_dir=tmp_path)
    assert result.n_executed == len(cases) * 1 * 2
    assert all(r.error is None for r in result.records)


def test_run_experiment_resumable_for_reconstructed_config(tmp_path: Path, monkeypatch):
    from agno.agent import Agent

    monkeypatch.setattr(Agent, "run", _fake_agent_run)
    cases = _small_case_subset()
    config_a = _reconstructed_config("a")
    config_b = _reconstructed_config("b")
    first = run_experiment(config_a, config_b, experiment_name="test_recon_resume", cases=cases, n_runs_per_case=1, max_workers=2, runs_dir=tmp_path)
    second = run_experiment(config_a, config_b, experiment_name="test_recon_resume", cases=cases, n_runs_per_case=1, max_workers=2, runs_dir=tmp_path)
    assert second.n_executed == 0
    assert second.n_cached == first.n_executed


@dataclass
class _StubBlock:
    text: str


@dataclass
class _StubMessage:
    content: list


class _StubMessagesAPI:
    def create(self, **kwargs):
        return _StubMessage(content=[_StubBlock(text=json.dumps({"ok": True}))])


class _StubClient:
    def __init__(self):
        self.messages = _StubMessagesAPI()


def test_run_experiment_threads_anthropic_client_to_reconstructed_execution(tmp_path: Path, monkeypatch):
    # send_invoice has no example_calls at all, so any tool call must fall
    # back to generation -- only reachable if anthropic_client is actually
    # threaded through run_experiment -> execute_case -> run_reconstructed_case.
    from agno.agent import Agent

    def _fake_run_with_tool_call(self, task, *, session_id=None, **kwargs):
        return _FakeRunOutput(content="Done.", tools=[_FakeToolExecution(tool_name="send_invoice", tool_args={"invoice_id": "X"}, result="ok")])

    monkeypatch.setattr(Agent, "run", _fake_run_with_tool_call)
    cases = [by_family("direct_instruction_injection")[0]]
    config_a = _reconstructed_config("a")
    config_b = _reconstructed_config("b")
    client = _StubClient()
    result = run_experiment(
        config_a, config_b, experiment_name="test_recon_client", cases=cases, n_runs_per_case=1, max_workers=1,
        runs_dir=tmp_path, anthropic_client=client,
    )
    assert all(r.error is None for r in result.records)


# --- reconstructed environments are real-model-only -------------------------


def test_run_experiment_rejects_reconstructed_config_under_mock_provider(tmp_path: Path):
    cases = _small_case_subset()
    config_a = _reconstructed_config("a", provider="mock")
    config_b = _reconstructed_config("b", provider="mock")
    with pytest.raises(ValueError, match="only run under provider='anthropic'"):
        run_experiment(config_a, config_b, experiment_name="test_recon_mock_reject", cases=cases, n_runs_per_case=1, runs_dir=tmp_path)


def test_run_experiment_rejection_happens_before_any_job_runs(tmp_path: Path, monkeypatch):
    from agno.agent import Agent

    called = []
    monkeypatch.setattr(Agent, "run", lambda self, *a, **k: called.append(1))
    cases = _small_case_subset()
    config_a = _reconstructed_config("a", provider="mock")
    config_b = _reconstructed_config("b")  # even a well-formed second arm shouldn't matter
    with pytest.raises(ValueError):
        run_experiment(config_a, config_b, experiment_name="test_recon_mock_reject2", cases=cases, n_runs_per_case=1, runs_dir=tmp_path)
    assert called == []
    assert not (tmp_path / "test_recon_mock_reject2.jsonl").exists()  # nothing was ever written either


def test_run_experiment_concurrent_writes_produce_no_corrupted_lines(tmp_path: Path):
    cases = _small_case_subset()
    arm_a = ArmSpec(label="a", overrides={})
    arm_b = ArmSpec(label="b", overrides={})
    result = run_experiment(
        arm_a, arm_b, experiment_name="test_concurrent", cases=cases,
        n_runs_per_case=5, max_workers=8, runs_dir=tmp_path,
    )
    on_disk = list(read_run_records(tmp_path / "test_concurrent.jsonl"))  # raises on any malformed JSON line
    assert len(on_disk) == len(cases) * 5 * 2
    assert len(result.records) == len(on_disk)
