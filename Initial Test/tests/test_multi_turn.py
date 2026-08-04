"""Regression tests for run_multi_turn_case, targeting the exact gaps the
multi-turn investigation found before Part 4 was built:

  1. Without db + add_history_to_context, a second team.run() call on the
     same session sees none of the first call's conversation — confirmed
     interactively during the investigation. This test fails if that
     wiring in build_team ever regresses.
  2. Each run must get its own fresh InMemoryDb — same per-run isolation
     discipline as _build_tools' per-run Function copies (see
     tests/test_concurrency.py). A shared/leaked db between two arms of a
     paired multi-turn comparison would let one arm's conversation bleed
     into the other's, silently invalidating the comparison.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import target_system.mock_model as mock_model_module
from target_system.factory import baseline_config
from target_system.mock_model import MockStep
from target_system.orchestration import build_team, run_multi_turn_case
from target_system.policy import TaskContext


def _scripts(supervisor_steps):
    return {
        "researcher": [MockStep(content="n/a")],
        "operator": [MockStep(content="n/a")],
        "supervisor": supervisor_steps,
    }


def test_second_turn_sees_first_turns_assistant_reply_in_context():
    """Regression test for the exact gap the investigation found: spy on
    what messages the supervisor's model actually receives on turn 2, and
    confirm turn 1's assistant reply is in there — not just that the run
    completes without error."""
    captured: list[list[tuple[str, str | None]]] = []
    original_invoke = mock_model_module.MockModel.invoke

    def spy(self, *args, **kwargs):
        if self.agent_label == "supervisor":
            msgs = kwargs.get("messages") or (args[0] if args else [])
            captured.append([(m.role, m.content) for m in msgs])
        return original_invoke(self, *args, **kwargs)

    mock_model_module.MockModel.invoke = spy
    try:
        config = baseline_config(provider="mock")
        turns = ["Summarize our PTO policy.", "Great, thanks."]
        mock_scripts = _scripts([
            MockStep(content="Here is the PTO summary: fifteen days a year."),
            MockStep(content="Glad that helped."),
        ])
        run_multi_turn_case(
            config, turns, case_id="regression_history", task_context=TaskContext(task_id="regression_history"),
            seed=1, mock_scripts=mock_scripts,
        )
    finally:
        mock_model_module.MockModel.invoke = original_invoke

    assert len(captured) == 2, "expected exactly one supervisor invoke() call per turn"

    turn_1_messages = captured[0]
    turn_2_messages = captured[1]

    # Turn 1 should NOT already contain an assistant reply (nothing to
    # carry yet).
    assert not any(role == "assistant" for role, _ in turn_1_messages)

    # Turn 2 must contain turn 1's assistant reply — this is the whole
    # point of the db + add_history_to_context wiring. Without it (the
    # pre-fix state), turn 2 only sees its own new user message.
    assistant_messages = [content for role, content in turn_2_messages if role == "assistant"]
    assert any(content and "fifteen days a year" in content for content in assistant_messages), (
        f"turn 2 never saw turn 1's assistant reply; messages were: {turn_2_messages}"
    )
    # And turn 1's user message should also still be present as history.
    user_messages = [content for role, content in turn_2_messages if role == "user"]
    assert any(content and "PTO policy" in content for content in user_messages)


def test_build_team_gives_each_call_a_fresh_db_instance():
    config = baseline_config(provider="mock")
    team_a = build_team(config, mock_scripts=_scripts([MockStep(content="a")]))
    team_b = build_team(config, mock_scripts=_scripts([MockStep(content="b")]))
    assert team_a.db is not team_b.db


def test_concurrent_multi_turn_arms_do_not_leak_session_state():
    """Same isolation discipline as tests/test_concurrency.py's tool-copy
    regression test: run many pairs of multi-turn "arms" concurrently
    across real OS threads, each with a distinctive marker in its own
    turns, and confirm no run's captured message history ever contains the
    OTHER arm's marker — which is what would happen if two runs somehow
    shared (rather than each getting a fresh) InMemoryDb / session."""
    MARKER_A = "XUNIQUEMARKERAX"
    MARKER_B = "XUNIQUEMARKERBX"

    lock = threading.Lock()
    captured_by_instance: dict[int, list[str]] = {}
    original_invoke = mock_model_module.MockModel.invoke

    def spy(self, *args, **kwargs):
        if self.agent_label == "supervisor":
            msgs = kwargs.get("messages") or (args[0] if args else [])
            text = " ".join(m.content for m in msgs if isinstance(m.content, str))
            with lock:
                captured_by_instance.setdefault(id(self), []).append(text)
        return original_invoke(self, *args, **kwargs)

    mock_model_module.MockModel.invoke = spy

    def run_arm(marker: str, seed: int):
        config = baseline_config(provider="mock", label=f"arm-{marker}-{seed}")
        turns = [f"Turn 1 {marker}", f"Turn 2 {marker}"]
        mock_scripts = _scripts([
            MockStep(content=f"ack1 {marker}"),
            MockStep(content=f"ack2 {marker}"),
        ])
        return run_multi_turn_case(
            config, turns, case_id=f"concurrency_probe_{marker}_{seed}",
            task_context=TaskContext(task_id="t"), seed=seed, mock_scripts=mock_scripts,
        )

    try:
        jobs = []
        for i in range(10):
            jobs.append((MARKER_A, i))
            jobs.append((MARKER_B, i))
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_arm, marker, seed) for marker, seed in jobs]
            for future in as_completed(futures):
                future.result()  # propagate any exception
    finally:
        mock_model_module.MockModel.invoke = original_invoke

    assert len(captured_by_instance) == 20  # one MockModel instance per run, never shared

    for instance_id, turns_seen in captured_by_instance.items():
        full_history = " ".join(turns_seen)
        has_a = MARKER_A in full_history
        has_b = MARKER_B in full_history
        assert not (has_a and has_b), (
            f"session for instance {instance_id} saw both arms' markers — session state leaked "
            f"across a paired comparison's arms: {full_history!r}"
        )
        assert has_a or has_b, f"instance {instance_id} saw neither marker: {full_history!r}"
