import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from agno.run.base import RunStatus

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
    status: RunStatus = RunStatus.completed

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


def _labeled_config(label, *, model_name, tools=("send_invoice",)):
    return SystemConfig(
        label=label,
        model=ModelConfig(provider="anthropic", model_name=model_name),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="[unavailable]", system_prompt_source="unavailable", tools=list(tools))],
        security=SecurityConfig(),
        provenance=ReconstructionProvenance(
            project_id="proj-1", source_agent_name="A", trace_count=5, extraction_date="2026-01-01",
            tool_profiles={name: ToolBehaviorProfile(tool_name=name) for name in tools},
        ),
    )


def test_run_experiment_separates_same_label_configs_by_config_hash(tmp_path: Path, monkeypatch):
    # Regression for the real bug found during the Braintrust E2E
    # validation: reconstruction defaults a config's label to its
    # workflow_name/agent_name, so two genuinely different reconstructions
    # of the same real workflow (e.g. re-pulled a week apart with real
    # behavior drift) end up sharing that default label. Before this fix,
    # build_paired_data/_task_success_rate grouped run records by
    # record.arm (the label) alone, so both arms' records were silently
    # merged into one indistinguishable pool at report time -- with no
    # error, just a blended/wrong result.
    #
    # config_a and config_b here share one label but differ in model_name
    # (and therefore config_hash), and are scripted so every arm-A run
    # succeeds and every arm-B run errors -- a maximally distinguishable
    # signal. If records were still being merged by label alone,
    # task_success_a and task_success_b would come out identical (the
    # same ~50/50 blend of both arms' runs) instead of cleanly 1.0/0.0.
    from agno.agent import Agent
    from agno.run.base import RunStatus
    from target_system.config import compute_config_hash

    same_label = "homepilot-ticket-analysis"
    config_a = _labeled_config(same_label, model_name="model-a")
    config_b = _labeled_config(same_label, model_name="model-b")
    assert config_a.label == config_b.label  # the exact real-world collision
    hash_a, hash_b = compute_config_hash(config_a), compute_config_hash(config_b)
    assert hash_a != hash_b

    def _fake_run(self, task, *, session_id=None, **kwargs):
        if self.model.id == "model-a":
            return _FakeRunOutput(content="Task completed.", tools=[])
        return _FakeRunOutput(content="Error code: 400 - simulated provider failure", tools=[], status=RunStatus.error)

    monkeypatch.setattr(Agent, "run", _fake_run)
    cases = _small_case_subset()
    result = run_experiment(
        config_a, config_b, experiment_name="test_same_label_diff_hash", cases=cases,
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )

    # Every record really does carry the same label -- this is the exact
    # condition that broke label-only grouping.
    assert {r.arm for r in result.records} == {same_label}
    assert {r.config_hash for r in result.records} == {hash_a, hash_b}

    assert result.task_success_a == 1.0
    assert result.task_success_b == 0.0

    records_a = [r for r in result.records if r.config_hash == hash_a]
    records_b = [r for r in result.records if r.config_hash == hash_b]
    assert len(records_a) == len(cases) * 2
    assert len(records_b) == len(cases) * 2
    assert all(r.error is None for r in records_a)
    assert all(r.error is not None for r in records_b)


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


# The real homepilot-ticket-analysis tool set, as reconstructed from live
# Braintrust traces (config cfg_4c44f09aed30, 98 traces): all read-only.
# Reproduced verbatim rather than paraphrased because the applicability
# filter's answer depends on the exact names -- every one of these
# classifies DATA_LOOKUP except search_maintenance_tickets
# (UNTRUSTED_CONTENT_ENTRY_POINT), and nothing classifies SENSITIVE_ACTION,
# which is what narrows the 17-case suite down to 2 applicable cases here.
_HOMEPILOT_TOOLS = (
    "get_occupancy_context",
    "get_open_tenant_charges",
    "get_property_level_workorders",
    "get_tenant_current_balance",
    "get_tenant_ledger",
    "list_ticket_actions",
    "search_maintenance_tickets",
    "zendesk_list_user_tickets",
)


def test_disambiguate_arm_labels_only_rewrites_the_same_hash_collision():
    from experiments.runner import disambiguate_arm_labels

    same = _labeled_config("env", model_name="m")
    # The A/A collision: one config picked twice.
    assert disambiguate_arm_labels(same, same, "cfg_x", "cfg_x") == ("env (arm A)", "env (arm B)")
    # Same label, different hash -- (hash, label) already separates these.
    assert disambiguate_arm_labels(same, same, "cfg_x", "cfg_y") == ("env", "env")
    # Ordinary comparison, untouched (so cached records still match).
    other = _labeled_config("other", model_name="m")
    assert disambiguate_arm_labels(same, other, "cfg_x", "cfg_y") == ("env", "other")


def test_run_experiment_same_config_both_arms_does_not_merge_into_one_bucket(tmp_path: Path, monkeypatch):
    # Regression for the A/A false-CLEAR bug, reproduced against the real
    # homepilot-ticket-analysis tool set. build_paired_data keys an arm on
    # (config_hash, label) and its docstring assumed the caller would give
    # a same-hash A/A check two distinct labels -- no caller ever did, so
    # picking one saved environment twice in the wizard sent both arms
    # through with identical hash AND identical label. Both arms then read
    # out of the same records bucket: n doubled, and rate_diff was pinned
    # at exactly 0.0 no matter what the runs produced -- a structural
    # false CLEAR, not a real A/A result.
    #
    # execute_case is faked so arm A and arm B produce deliberately
    # OPPOSITE outcomes. That's the part a merged bucket cannot express:
    # if the arms are still merged, both sides read the same records and
    # the diff collapses to 0.0 regardless. A real diff of 1.0 is only
    # reachable if the two arms stayed separate.
    import itertools

    from attacker.applicability import applicable_cases_for_configs
    from attacker.cases import ATTACK_CASES
    from experiments.runner import ARM_A_SUFFIX, ARM_B_SUFFIX, build_paired_data
    from target_system.config import compute_config_hash
    from target_system.logging_schema import RunRecord

    config = _labeled_config("homepilot-ticket-analysis", model_name="claude-opus-4-8", tools=_HOMEPILOT_TOOLS)

    # The real applicability filter against the real case suite -- not a
    # hand-picked subset. This environment supports exactly 2 cases, one
    # per family, which is also why its A/A run has no family data at all.
    cases = applicable_cases_for_configs(list(ATTACK_CASES), [config, config])
    assert len(cases) == 2
    assert {c.family for c in cases} == {"tool_result_poisoning", "multi_turn_goal_hijack"}

    run_ids = itertools.count()

    def _fake_execute_case(cfg, case, *, seed, arm=None, mock_scripts=None, anthropic_client=None):
        flagged = arm.endswith(ARM_B_SUFFIX)
        return RunRecord(
            run_id=f"r{next(run_ids)}", config_hash=compute_config_hash(cfg), case_id=case.id,
            case_family=case.family, arm=arm, seed=seed,
            started_at="t", ended_at="t", wall_time_seconds=0.0,
            outcomes={"unauthorized_lookup": flagged, "task_success": True},
        )

    monkeypatch.setattr("experiments.runner.execute_case", _fake_execute_case)

    n_runs = 3
    result = run_experiment(
        config, config, experiment_name="test_aa_same_config", cases=cases,
        n_runs_per_case=n_runs, max_workers=2, runs_dir=tmp_path,
    )

    # Same config really is on both arms -- the exact condition that broke.
    assert result.arm_a_hash == result.arm_b_hash
    # ...but the arms are now separable.
    assert result.arm_a_label != result.arm_b_label
    assert result.arm_a_label == f"homepilot-ticket-analysis{ARM_A_SUFFIX}"
    assert result.arm_b_label == f"homepilot-ticket-analysis{ARM_B_SUFFIX}"

    # No doubled bucket: n_runs per case per arm, not 2 * n_runs in one.
    for label in (result.arm_a_label, result.arm_b_label):
        for case in cases:
            arm_records = [r for r in result.records if r.arm == label and r.case_id == case.id]
            assert len(arm_records) == n_runs

    paired = build_paired_data(
        result.records, result.arm_a_hash, result.arm_a_label,
        result.arm_b_hash, result.arm_b_label, "unauthorized_lookup",
    )
    assert len(paired) == len(cases)
    for case_data in paired:
        assert case_data.arm_a.n == n_runs  # would be 2 * n_runs if merged
        assert case_data.arm_b.n == n_runs
        # The assertion the old code could never satisfy: a real, non-zero
        # diff survives instead of collapsing to a pinned 0.0.
        assert case_data.rate_diff == 1.0


def test_run_experiment_same_config_both_arms_executes_each_arm_once(tmp_path: Path, monkeypatch):
    # The same bug's other half: with both arms sharing (hash, label), the
    # cache key (config_hash, case_id, arm, seed) was identical for both,
    # so run_experiment queued two jobs that collided on one key -- real
    # API calls paid twice for a single arm's worth of distinguishable
    # data, and a resumed run couldn't tell which half it already had.
    from agno.agent import Agent

    monkeypatch.setattr(Agent, "run", _fake_agent_run)
    config = _labeled_config("homepilot-ticket-analysis", model_name="claude-opus-4-8", tools=_HOMEPILOT_TOOLS)
    cases = _small_case_subset()

    result = run_experiment(
        config, config, experiment_name="test_aa_cache_keys", cases=cases,
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )
    keys = {(r.config_hash, r.case_id, r.arm, r.seed) for r in result.records}
    assert len(keys) == len(result.records) == len(cases) * 2 * 2

    # Re-running is a no-op, not a second full re-execution.
    resumed = run_experiment(
        config, config, experiment_name="test_aa_cache_keys", cases=cases,
        n_runs_per_case=2, max_workers=2, runs_dir=tmp_path,
    )
    assert resumed.n_executed == 0
    assert resumed.n_cached == len(result.records)
