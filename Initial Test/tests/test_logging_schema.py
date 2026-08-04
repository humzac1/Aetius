from target_system.logging_schema import (
    AgentEndEvent,
    AgentStartEvent,
    MessageEvent,
    RunRecord,
    ToolCallEvent,
    TokenUsage,
    append_run_record,
    read_run_records,
)


def _sample_record() -> RunRecord:
    return RunRecord(
        run_id="run_test123",
        config_hash="cfg_deadbeef0000",
        case_id="benign_baseline_01",
        case_family=None,
        arm="a",
        seed=42,
        started_at="2026-08-03T14:00:00+00:00",
        ended_at="2026-08-03T14:00:07+00:00",
        wall_time_seconds=7.0,
        events=[
            AgentStartEvent(idx=0, timestamp="2026-08-03T14:00:00+00:00", agent="supervisor"),
            MessageEvent(
                idx=1, timestamp="2026-08-03T14:00:01+00:00", from_agent="supervisor",
                to_agent="researcher", role="delegation", content="Find the travel policy",
            ),
            ToolCallEvent(
                idx=2, timestamp="2026-08-03T14:00:02+00:00", agent="researcher",
                tool_name="search_corpus", arguments={"query": "travel"}, result={"files": []},
            ),
            AgentEndEvent(idx=3, timestamp="2026-08-03T14:00:07+00:00", agent="supervisor", final_answer="The per diem is $50/day."),
        ],
        token_usage=TokenUsage(total_prompt_tokens=100, total_completion_tokens=40),
        outcomes={"exfiltration": False, "unauthorized_lookup": False, "task_success": True},
    )


def test_roundtrip_preserves_event_types(tmp_path):
    record = _sample_record()
    path = tmp_path / "runs.jsonl"
    append_run_record(record, path)

    loaded = list(read_run_records(path))
    assert len(loaded) == 1
    r = loaded[0]

    assert [type(e).__name__ for e in r.events] == [
        "AgentStartEvent", "MessageEvent", "ToolCallEvent", "AgentEndEvent",
    ]
    assert r.outcomes == record.outcomes
    assert r.token_usage.total_prompt_tokens == 100


def test_multiple_records_append(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_run_record(_sample_record(), path)
    r2 = _sample_record()
    r2.run_id = "run_test456"
    append_run_record(r2, path)

    loaded = list(read_run_records(path))
    assert [r.run_id for r in loaded] == ["run_test123", "run_test456"]


def test_missing_file_yields_nothing(tmp_path):
    assert list(read_run_records(tmp_path / "does_not_exist.jsonl")) == []
