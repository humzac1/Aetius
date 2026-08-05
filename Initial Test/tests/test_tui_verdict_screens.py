"""Headless Textual Pilot tests for the verdict screens — checks each
tier renders without crashing, the disclaimer is always present on the
single-config screen, and 's'/'d' navigate as documented."""

import asyncio

from textual.widgets import DataTable, Label, Static

from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig, save_config
from target_system.provenance import ReconstructionProvenance, ToolBehaviorProfile
from tui.app import HarnessApp
from tui.screens.verdict import ComparisonVerdictScreen, SingleConfigVerdictScreen, StatisticsDrillDownScreen
from tui.verdict_logic import (
    AttemptedExecutedCounts,
    ComparisonVerdict,
    FamilyPower,
    FamilySingleSummary,
    SingleConfigSummary,
)


def _all_text(screen) -> str:
    return " ".join(str(label.render()) for label in screen.query(Label))


def _reconstructed_config(label, *, tools=("send_invoice",), system_prompt_source="unavailable", trace_count=11, other_groups_found=()):
    return SystemConfig(
        label=label,
        model=ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="[unavailable]", system_prompt_source=system_prompt_source, tools=list(tools))],
        security=SecurityConfig(),
        provenance=ReconstructionProvenance(
            project_id="proj-1", source_agent_name="Invoice Generation Assistant", trace_count=trace_count, extraction_date="2026-01-01T00:00:00+00:00",
            other_groups_found=list(other_groups_found),
            tool_profiles={name: ToolBehaviorProfile(tool_name=name) for name in tools},
        ),
    )


def run_async(coro_fn):
    asyncio.run(coro_fn())


def _flagged_verdict():
    effect = {
        "method": "cluster_bootstrap", "rate_a": 0.133, "rate_b": 0.733, "diff": 0.6,
        "ci_low": 0.42, "ci_high": 0.72, "alpha": 0.05, "p_value": 0.0001,
        "n_cases": 15, "n_runs_a": 75, "n_runs_b": 75, "used_fallback": False, "fallback_reason": None, "extra": {},
    }
    return ComparisonVerdict(
        tier="FLAGGED", flagged_outcome_key="exfiltration", flagged_family="direct_instruction_injection",
        flagged_effect=effect, flagged_q_value=0.001, flagged_arm_label="defensive_prompt_off", other_flagged_count=1,
    )


def _clear_verdict():
    wc = FamilyPower(outcome_key="exfiltration", family="f", n_cases=10, n_runs_per_case=8.0, baseline_rate=0.15, observed_effect=0.22, achieved_power=0.9)
    return ComparisonVerdict(tier="CLEAR", target_power=0.8, worst_case=wc, achieved_mde=0.12)


def _inconclusive_verdict():
    wc = FamilyPower(outcome_key="exfiltration", family="f", n_cases=4, n_runs_per_case=3.0, baseline_rate=0.15, observed_effect=0.02, achieved_power=0.16)
    return ComparisonVerdict(tier="INCONCLUSIVE", target_power=0.8, worst_case=wc, recommended_additional_runs=12)


def _report():
    return {
        "arm_a_label": "arm_a", "arm_b_label": "arm_b", "n_cases": 15, "n_runs_per_case": 5,
        "family_results": {
            "exfiltration": [
                {
                    "family": "direct_instruction_injection", "q_value": 0.001, "significant_after_correction": True,
                    "effect": {
                        "method": "cluster_bootstrap", "rate_a": 0.133, "rate_b": 0.733, "diff": 0.6,
                        "ci_low": 0.42, "ci_high": 0.72, "alpha": 0.05, "p_value": 0.0001,
                        "n_cases": 15, "n_runs_a": 75, "n_runs_b": 75, "used_fallback": False, "fallback_reason": None, "extra": {},
                    },
                }
            ]
        },
    }


def _records():
    return [
        {"arm": "defensive_prompt_off", "case_family": "direct_instruction_injection", "events": [{"type": "tool_call", "tool_name": "send_email", "status": "executed"}]},
        {"arm": "defensive_prompt_off", "case_family": "direct_instruction_injection", "events": [{"type": "tool_call", "tool_name": "send_email", "status": "blocked"}]},
    ]


# --- FLAGGED / CLEAR / INCONCLUSIVE render without crashing ------------------


def test_flagged_screen_renders():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(_flagged_verdict(), _report(), "toy", _records()))
            await pilot.pause()
            assert isinstance(app.screen, ComparisonVerdictScreen)

    run_async(scenario)


def test_clear_screen_renders():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(_clear_verdict(), _report(), "toy", []))
            await pilot.pause()
            assert isinstance(app.screen, ComparisonVerdictScreen)

    run_async(scenario)


def test_inconclusive_screen_renders():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(_inconclusive_verdict(), _report(), "toy", []))
            await pilot.pause()
            assert isinstance(app.screen, ComparisonVerdictScreen)

    run_async(scenario)


# --- navigation ---------------------------------------------------------------


def test_s_opens_statistics_drill_down():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(_flagged_verdict(), _report(), "toy", _records()))
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            assert isinstance(app.screen, StatisticsDrillDownScreen)

    run_async(scenario)


def test_drill_down_table_has_a_row_per_family_result():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(StatisticsDrillDownScreen(_report(), "toy"))
            await pilot.pause()
            table = app.screen.query_one("#drill-down-table", DataTable)
            assert table.row_count == 1

    run_async(scenario)


def test_back_from_drill_down_returns_to_verdict():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(_flagged_verdict(), _report(), "toy", _records()))
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, ComparisonVerdictScreen)

    run_async(scenario)


def test_open_dashboard_notifies_without_crashing_when_not_running():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(_flagged_verdict(), _report(), "toy", _records()))
            await pilot.pause()
            await pilot.press("d")
            await pilot.pause()  # should not raise even if nothing is listening on 8501

    run_async(scenario)


# --- single-config screen -----------------------------------------------------


def _single_summary():
    return SingleConfigSummary(
        config_label="baseline", config_hash="cfg_abc", total_attacks=20, succeeded=2, blocked=3, resisted=15,
        by_family=[FamilySingleSummary(family="direct_instruction_injection", total=20, succeeded=2, blocked=3)],
    )


def test_single_config_screen_renders_and_disclaimer_is_always_visible():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = SingleConfigVerdictScreen(_single_summary())
            app.push_screen(screen)
            await pilot.pause()
            disclaimer = screen.query_one("#disclaimer", Static)
            assert disclaimer.display is True
            assert disclaimer.visible is True

    run_async(scenario)


def test_single_config_screen_table_has_a_row_per_family():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = SingleConfigVerdictScreen(_single_summary())
            app.push_screen(screen)
            await pilot.pause()
            table = screen.query_one("#family-table", DataTable)
            assert table.row_count == 1

    run_async(scenario)


# --- reconstructed-environment fidelity disclosures (Part 6) -----------------


def test_single_config_screen_shows_fidelity_disclosure_for_reconstructed_config(tmp_path):
    config = _reconstructed_config("recon")
    config_hash = save_config(config, configs_dir=tmp_path)
    summary = SingleConfigSummary(
        config_label="reconstructed: Invoice Generation Assistant (11 traces)", config_hash=config_hash,
        total_attacks=5, succeeded=1, blocked=1, resisted=3, by_family=[],
    )
    records = [{"case_family": "f", "events": [{"type": "tool_call", "tool_name": "send_invoice", "response_source": "generated"}]}]

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = SingleConfigVerdictScreen(summary, records=records, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            text = _all_text(screen)
            assert "Reconstructed from 11 trace(s)" in text
            assert "Invoice Generation Assistant" in text
            assert "no system prompt at all" in text
            assert "1 model-generated" in text

    run_async(scenario)


def test_single_config_screen_no_fidelity_block_for_toy_config(tmp_path):
    from target_system.factory import baseline_config

    config_hash = save_config(baseline_config(), configs_dir=tmp_path)
    summary = SingleConfigSummary(config_label="baseline (defaults)", config_hash=config_hash, total_attacks=1, succeeded=0, blocked=0, resisted=1, by_family=[])

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = SingleConfigVerdictScreen(summary, records=[], configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            text = _all_text(screen)
            assert "Reconstructed from" not in text
            assert "no system prompt at all" not in text

    run_async(scenario)


def test_comparison_screen_shows_fidelity_block_when_an_arm_is_reconstructed(tmp_path):
    hash_a = save_config(_reconstructed_config("a"), configs_dir=tmp_path)
    hash_b = save_config(_reconstructed_config("b", other_groups_found=[]), configs_dir=tmp_path)
    report = {
        "arm_a_label": "a", "arm_b_label": "b", "arm_a_hash": hash_a, "arm_b_hash": hash_b,
        "n_cases": 1, "n_runs_per_case": 1, "family_results": {},
    }

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(_clear_verdict(), report, "test", [], configs_dir=tmp_path))
            await pilot.pause()
            text = _all_text(app.screen)
            assert "Reconstructed from 11 trace(s)" in text

    run_async(scenario)


def test_comparison_screen_no_fidelity_block_when_both_arms_toy(tmp_path):
    from target_system.factory import baseline_config

    hash_a = save_config(baseline_config(label="a"), configs_dir=tmp_path)
    hash_b = save_config(baseline_config(label="b", defensive_instruction=False), configs_dir=tmp_path)
    report = {
        "arm_a_label": "a", "arm_b_label": "b", "arm_a_hash": hash_a, "arm_b_hash": hash_b,
        "n_cases": 1, "n_runs_per_case": 1, "family_results": {},
    }

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(_clear_verdict(), report, "test", [], configs_dir=tmp_path))
            await pilot.pause()
            assert "Reconstructed from" not in _all_text(app.screen)

    run_async(scenario)


def test_comparison_screen_surfaces_other_groups_found_note(tmp_path):
    from target_system.provenance import OtherGroupFound

    hash_a = save_config(_reconstructed_config("a", other_groups_found=[OtherGroupFound(agent_name="HR Onboarding Assistant", trace_count=33)]), configs_dir=tmp_path)
    hash_b = save_config(_reconstructed_config("b"), configs_dir=tmp_path)
    report = {
        "arm_a_label": "a", "arm_b_label": "b", "arm_a_hash": hash_a, "arm_b_hash": hash_b,
        "n_cases": 1, "n_runs_per_case": 1, "family_results": {},
    }

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(_clear_verdict(), report, "test", [], configs_dir=tmp_path))
            await pilot.pause()
            assert "HR Onboarding Assistant (33)" in _all_text(app.screen)

    run_async(scenario)


def test_comparison_flagged_screen_shows_synthetic_evidence_note(tmp_path):
    hash_a = save_config(_reconstructed_config("a", system_prompt_source="observed"), configs_dir=tmp_path)
    hash_b = save_config(_reconstructed_config("b", system_prompt_source="observed"), configs_dir=tmp_path)
    effect = {
        "method": "cluster_bootstrap", "rate_a": 0.1, "rate_b": 0.6, "diff": 0.5,
        "ci_low": 0.3, "ci_high": 0.7, "alpha": 0.05, "p_value": 0.001,
        "n_cases": 5, "n_runs_a": 25, "n_runs_b": 25, "used_fallback": False, "fallback_reason": None, "extra": {},
    }
    verdict = ComparisonVerdict(
        tier="FLAGGED", flagged_outcome_key="exfiltration", flagged_family="direct_instruction_injection",
        flagged_effect=effect, flagged_q_value=0.001, flagged_arm_label="b", other_flagged_count=0,
    )
    report = {
        "arm_a_label": "a", "arm_b_label": "b", "arm_a_hash": hash_a, "arm_b_hash": hash_b,
        "n_cases": 5, "n_runs_per_case": 5, "family_results": {},
    }
    records = [
        {"arm": "b", "case_family": "direct_instruction_injection", "events": [{"type": "tool_call", "tool_name": "send_invoice", "status": "executed", "response_source": "generated"}]},
        {"arm": "b", "case_family": "direct_instruction_injection", "events": [{"type": "tool_call", "tool_name": "send_invoice", "status": "blocked", "response_source": "real"}]},
    ]

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(verdict, report, "test", records, configs_dir=tmp_path))
            await pilot.pause()
            text = _all_text(app.screen)
            assert "1 of 2 tool response(s)" in text
            assert "less weight" in text

    run_async(scenario)


def test_comparison_flagged_screen_no_synthetic_note_when_all_real(tmp_path):
    hash_a = save_config(_reconstructed_config("a", system_prompt_source="observed"), configs_dir=tmp_path)
    hash_b = save_config(_reconstructed_config("b", system_prompt_source="observed"), configs_dir=tmp_path)
    effect = {
        "method": "cluster_bootstrap", "rate_a": 0.1, "rate_b": 0.6, "diff": 0.5,
        "ci_low": 0.3, "ci_high": 0.7, "alpha": 0.05, "p_value": 0.001,
        "n_cases": 5, "n_runs_a": 25, "n_runs_b": 25, "used_fallback": False, "fallback_reason": None, "extra": {},
    }
    verdict = ComparisonVerdict(
        tier="FLAGGED", flagged_outcome_key="exfiltration", flagged_family="direct_instruction_injection",
        flagged_effect=effect, flagged_q_value=0.001, flagged_arm_label="b", other_flagged_count=0,
    )
    report = {
        "arm_a_label": "a", "arm_b_label": "b", "arm_a_hash": hash_a, "arm_b_hash": hash_b,
        "n_cases": 5, "n_runs_per_case": 5, "family_results": {},
    }
    records = [
        {"arm": "b", "case_family": "direct_instruction_injection", "events": [{"type": "tool_call", "tool_name": "send_invoice", "status": "executed", "response_source": "real"}]},
    ]

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(ComparisonVerdictScreen(verdict, report, "test", records, configs_dir=tmp_path))
            await pilot.pause()
            assert "less weight" not in _all_text(app.screen)

    run_async(scenario)


def test_drill_down_screen_populates_tool_responses_column(tmp_path):
    hash_a = save_config(_reconstructed_config("a"), configs_dir=tmp_path)
    hash_b = save_config(_reconstructed_config("b"), configs_dir=tmp_path)
    report = {
        "arm_a_hash": hash_a, "arm_b_hash": hash_b, "n_cases": 1, "n_runs_per_case": 1,
        "family_results": {
            "exfiltration": [
                {
                    "family": "direct_instruction_injection", "q_value": 0.5, "significant_after_correction": False,
                    "effect": {
                        "method": "cluster_bootstrap", "rate_a": 0.1, "rate_b": 0.2, "diff": 0.1,
                        "ci_low": 0.0, "ci_high": 0.2, "alpha": 0.05, "p_value": 0.5,
                        "n_cases": 1, "n_runs_a": 1, "n_runs_b": 1, "used_fallback": False, "fallback_reason": None, "extra": {},
                    },
                }
            ]
        },
    }
    records = [{"arm": "a", "case_family": "direct_instruction_injection", "events": [{"type": "tool_call", "tool_name": "send_invoice", "response_source": "replay"}]}]

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(StatisticsDrillDownScreen(report, "test", records=records, configs_dir=tmp_path))
            await pilot.pause()
            table = app.screen.query_one("#drill-down-table", DataTable)
            row = table.get_row_at(0)
            assert "replay" in str(row[-1])

    run_async(scenario)
