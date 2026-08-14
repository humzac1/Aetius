import json

import pytest
from textual.widgets import Label, ListView

from attacker.attack_case import AttackCase
from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig, compute_config_hash
from target_system.provenance import ReconstructionProvenance, ToolBehaviorProfile
from tui.app import HarnessApp
from tui.execution import comparison_experiment_name
from tui.formatting import format_baseline_assumption, format_run_count_option, format_run_count_recommendation
from tui.run_sizing import (
    CONSERVATIVE_BASELINE_RATE,
    DEFAULT_MDE,
    detectable_effect_at,
    observed_wall_seconds_per_run,
    recommend_runs_per_case,
)
from tui.screens.wizard import RunCountScreen, WizardModeScreen
from tests.tui_test_support import keep_all_families, run_async


def _config(label="recon"):
    return SystemConfig(
        label=label,
        model=ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="[unavailable]", system_prompt_source="unavailable", tools=["send_invoice"])],
        security=SecurityConfig(),
        provenance=ReconstructionProvenance(
            project_id="proj-1", source_agent_name="A", trace_count=5, extraction_date="2026-01-01",
            tool_profiles={"send_invoice": ToolBehaviorProfile(tool_name="send_invoice")},
            avg_cost_usd_per_trace=0.01, avg_generations_per_trace=1.0,
        ),
    )


def _cases(counts):
    """counts: {family: n_cases}. Only .family is read by the sizer, but
    AttackCase validates the rest, so build real ones."""
    cases = []
    for family, n in counts.items():
        for i in range(n):
            cases.append(
                AttackCase(
                    id=f"{family}-{i}", family=family, injection_vector="task_text",
                    success_outcome="exfiltration", source="test", benign_task="do x", injected_payload="do y",
                )
            )
    return cases


_BIG_FAMILY = "direct_instruction_injection"
_SMALL_FAMILY = "multi_turn_goal_hijack"


# --- the recommendation itself ------------------------------------------------


def test_recommends_more_than_the_old_hardcoded_default(tmp_path):
    """The regression this whole module exists for: 5 runs/case was never a
    sized number, and against a realistic suite it is far too few."""
    rec = recommend_runs_per_case(_cases({"direct_instruction_injection": 5}), [_config()], runs_dir=tmp_path)
    assert rec.recommended_runs_per_case > 5


def test_sizes_to_the_weakest_family_not_the_average(tmp_path):
    """Power is graded per family and the worst family decides the verdict,
    so a suite with one small family must be sized for that family."""
    cases = _cases({_BIG_FAMILY: 12, _SMALL_FAMILY: 3})
    rec = recommend_runs_per_case(cases, [_config()], runs_dir=tmp_path)

    by_family = {f.family: f for f in rec.per_family}
    assert by_family[_SMALL_FAMILY].required_runs_per_case > by_family[_BIG_FAMILY].required_runs_per_case
    assert rec.limiting.family == _SMALL_FAMILY
    assert rec.recommended_runs_per_case == by_family[_SMALL_FAMILY].required_runs_per_case


def test_assumes_conservative_baseline_when_no_prior_run(tmp_path):
    rec = recommend_runs_per_case(_cases({_SMALL_FAMILY: 4}), [_config()], runs_dir=tmp_path)
    assert rec.baseline_source == "assumed"
    assert rec.limiting.baseline_rate == CONSERVATIVE_BASELINE_RATE


def test_uses_observed_baseline_from_a_previous_run(tmp_path):
    """A measured rate beats the conservative assumption — and because the
    assumption is the variance-maximising one, measuring can only ever
    lower the requirement."""
    config = _config()
    h = compute_config_hash(config)
    name = comparison_experiment_name(h, h)
    (tmp_path / f"{name}_report.json").write_text(
        json.dumps(
            {
                "name": name,
                "family_results": {
                    "exfiltration": [
                        {
                            "family": "multi_turn_goal_hijack",
                            "effect": {"rate_a": 0.05, "rate_b": 0.05, "diff": 0.0, "n_cases": 4, "n_runs_a": 20},
                            "q_value": 1.0,
                            "significant_after_correction": False,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    rec = recommend_runs_per_case(_cases({_SMALL_FAMILY: 4}), [config, config], runs_dir=tmp_path)
    assumed = recommend_runs_per_case(_cases({_SMALL_FAMILY: 4}), [config, config], runs_dir=tmp_path / "empty")

    assert rec.baseline_source == "observed"
    assert rec.limiting.baseline_rate == pytest.approx(0.05)
    assert rec.recommended_runs_per_case < assumed.recommended_runs_per_case


def test_recommended_count_actually_achieves_the_target_mde(tmp_path):
    """Round-trip against the same power model the verdict screen grades
    the finished run with: at the recommended n, the detectable effect must
    be at least as small as the MDE that was asked for."""
    rec = recommend_runs_per_case(_cases({_SMALL_FAMILY: 3}), [_config()], runs_dir=tmp_path)
    assert detectable_effect_at(rec.recommended_runs_per_case, rec) <= DEFAULT_MDE + 1e-9
    assert detectable_effect_at(5, rec) > DEFAULT_MDE  # and the old default does not


def test_returns_none_when_there_is_nothing_to_size(tmp_path):
    assert recommend_runs_per_case([], [_config()], runs_dir=tmp_path) is None


def test_wall_time_falls_back_when_no_records_exist(tmp_path):
    seconds, grounded = observed_wall_seconds_per_run([_config()], runs_dir=tmp_path)
    assert grounded is False
    assert seconds > 0


# --- phrasing -----------------------------------------------------------------


def test_recommendation_text_names_the_limiting_family_and_target(tmp_path):
    rec = recommend_runs_per_case(_cases({"multi_turn_goal_hijack": 3}), [_config()], runs_dir=tmp_path)
    text = format_run_count_recommendation(rec)
    assert f"{rec.recommended_runs_per_case} runs/case/arm" in text
    assert "10-point" in text
    assert "80.0% power" in text
    assert "3 applicable case(s)" in text


def test_assumed_baseline_is_disclosed_not_presented_as_measured(tmp_path):
    rec = recommend_runs_per_case(_cases({_SMALL_FAMILY: 3}), [_config()], runs_dir=tmp_path)
    text = format_baseline_assumption(rec)
    assert "No previous run to measure" in text
    assert "assuming" in text


def test_under_powered_option_states_the_inconclusive_consequence(tmp_path):
    """The point of offering a smaller count is that the trade is stated,
    not buried — this is the sentence that makes it an informed choice."""
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = WizardModeScreen(runs_dir=tmp_path, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            screen._start([_config()])
            await pilot.pause()
            await keep_all_families(pilot, app)
            assert isinstance(app.screen, RunCountScreen)

            rec = app.screen.recommendation
            smaller = next(o for o in app.screen.options if o.kind == "smaller")
            recommended = next(o for o in app.screen.options if o.kind == "recommended")

            assert smaller.meets_target_power is False
            assert recommended.meets_target_power is True

            smaller_text = format_run_count_option(smaller, rec)
            assert "INCONCLUSIVE" in smaller_text
            assert "Cannot detect anything below" in smaller_text
            assert "Detects differences down to" in format_run_count_option(recommended, rec)

    run_async(scenario)


# --- the screen ---------------------------------------------------------------


def test_sizing_screen_precedes_the_cost_screen_and_shows_cost_and_time(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = WizardModeScreen(runs_dir=tmp_path, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            screen._start([_config()])
            await pilot.pause()
            await keep_all_families(pilot, app)

            assert isinstance(app.screen, RunCountScreen)
            rendered = " ".join(str(label.render()) for label in app.screen.query(Label))
            assert "How many runs per case?" in rendered
            assert "$" in rendered  # real cost, before anything executes
            assert any(unit in rendered for unit in ("s", "min", "hr"))

    run_async(scenario)


def test_chosen_count_is_the_one_actually_executed(tmp_path):
    """The whole flow is pointless if the pick doesn't reach the runner —
    n_runs_per_case was previously never passed at all, so the constructor
    default silently won regardless of anything the UI showed."""
    seen = {}

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = WizardModeScreen(runs_dir=tmp_path, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            screen._start([_config()])
            await pilot.pause()
            await keep_all_families(pilot, app)

            sizing = app.screen
            recommended = next(o for o in sizing.options if o.kind == "recommended")
            index = next(i for i, o in enumerate(sizing.options) if o.kind == "recommended")
            sizing.query_one("#run-count-menu", ListView).index = index
            await pilot.press("enter")
            await pilot.pause()

            # lands on the cost screen, priced at the chosen count
            assert app.screen.estimate.n_runs_per_case == recommended.n_runs_per_case
            seen["n"] = recommended.n_runs_per_case

    run_async(scenario)
    assert seen["n"] > 5


def test_cancel_runs_nothing(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = WizardModeScreen(runs_dir=tmp_path, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            screen._start([_config()])
            await pilot.pause()
            await keep_all_families(pilot, app)

            sizing = app.screen
            menu = sizing.query_one("#run-count-menu", ListView)
            menu.index = len(sizing.options) + 1  # trailing "Cancel" row (after the budget row)
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, WizardModeScreen)
            assert not list(tmp_path.glob("*.jsonl"))

    run_async(scenario)


def test_observed_baseline_is_discarded_when_the_prior_run_used_different_cases(tmp_path):
    """A rate is only a baseline for the cases that produced it. Real
    instance: the E-Commerce environment's first comparison ran the
    hand-authored suite it cannot engage with and recorded rate_a = 0.0;
    reusing that for a run of domain-adapted cases under-sized it."""
    import json

    from target_system.logging_schema import RunRecord, append_run_record
    from tui.execution import comparison_experiment_name

    config = _config()
    h = compute_config_hash(config)
    name = comparison_experiment_name(h, h)

    (tmp_path / f"{name}_report.json").write_text(
        json.dumps({
            "name": name,
            "family_results": {"exfiltration": [{
                "family": _SMALL_FAMILY,
                "effect": {"rate_a": 0.0, "rate_b": 0.0, "diff": 0.0, "n_cases": 3, "n_runs_a": 15},
                "q_value": 1.0, "significant_after_correction": False,
            }]},
        }),
        encoding="utf-8",
    )
    # the prior run's JSONL records which cases actually ran
    for case_id in ["old-case-1", "old-case-2"]:
        append_run_record(
            RunRecord(run_id=f"r-{case_id}", config_hash=h, case_id=case_id, arm="a", seed=0,
                      started_at="t", ended_at="t", wall_time_seconds=1.0, events=[]),
            tmp_path / f"{name}.jsonl",
        )

    now_cases = _cases({_SMALL_FAMILY: 3})  # different case ids entirely
    rec = recommend_runs_per_case(now_cases, [config, config], runs_dir=tmp_path)

    assert rec.baseline_source == "assumed"
    assert rec.limiting.baseline_rate == CONSERVATIVE_BASELINE_RATE


# --- wall-clock estimate and budget sizing --------------------------------------


def test_wall_estimate_accounts_for_concurrent_workers():
    from tui.run_sizing import DEFAULT_MAX_WORKERS, estimated_wall_seconds

    # 954 jobs at 4.9s/run with 8 workers is ~10 minutes, not the ~78
    # minutes a serial jobs-x-seconds product claims (the shipped bug).
    assert DEFAULT_MAX_WORKERS == 8
    assert estimated_wall_seconds(954, 4.9) == 4.9 * 120  # 120 waves of 8
    assert estimated_wall_seconds(0, 4.9) == 0.0
    assert estimated_wall_seconds(1, 4.9) == 4.9


def test_size_for_budget_time_ceiling_binds_and_reports_detection(tmp_path):
    from tui.run_sizing import recommend_runs_per_case, size_for_budget

    cases = _cases({"direct_instruction_injection": 3})
    configs = [_config()]
    rec = recommend_runs_per_case(cases, configs, runs_dir=tmp_path)
    option = size_for_budget(cases, configs, rec, max_minutes=2.0, runs_dir=tmp_path)
    assert option.feasible
    assert option.binding == "time"
    assert option.n_runs_per_case >= 2
    # the honest sentence's substance: a real detectable-effect number,
    # never below the ROPE, and monotone in budget
    assert option.detectable_effect is not None and option.detectable_effect > 0.01
    bigger = size_for_budget(cases, configs, rec, max_minutes=20.0, runs_dir=tmp_path)
    assert bigger.n_runs_per_case > option.n_runs_per_case
    assert bigger.detectable_effect < option.detectable_effect


def test_size_for_budget_reports_infeasible_rather_than_clamping(tmp_path):
    from tui.run_sizing import recommend_runs_per_case, size_for_budget

    cases = _cases({"direct_instruction_injection": 3})
    configs = [_config()]
    rec = recommend_runs_per_case(cases, configs, runs_dir=tmp_path)
    # ceiling below even the 2-run floor's wall time
    option = size_for_budget(cases, configs, rec, max_minutes=0.01, runs_dir=tmp_path)
    assert not option.feasible
    assert option.n_runs_per_case == 0
    assert option.detectable_effect is None
    assert option.binding == "time"


def test_format_budget_option_states_detection_or_infeasibility(tmp_path):
    from tui.formatting import format_budget_option
    from tui.run_sizing import recommend_runs_per_case, size_for_budget

    cases = _cases({"direct_instruction_injection": 3})
    configs = [_config()]
    rec = recommend_runs_per_case(cases, configs, runs_dir=tmp_path)
    ok = format_budget_option(size_for_budget(cases, configs, rec, max_minutes=2.0, runs_dir=tmp_path))
    assert "you can reliably catch differences of" in ok
    assert "INCONCLUSIVE" in ok
    bad = format_budget_option(size_for_budget(cases, configs, rec, max_minutes=0.01, runs_dir=tmp_path))
    assert "cannot fund" in bad
