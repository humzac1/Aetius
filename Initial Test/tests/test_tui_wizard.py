from textual.widgets import Label, ListView

from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig, save_config
from target_system.factory import baseline_config
from target_system.provenance import ReconstructionProvenance, ToolBehaviorProfile
from tui.app import HarnessApp
from tui.screens.verdict import ComparisonVerdictScreen, SingleConfigVerdictScreen
from tui.screens.wizard import (
    ConfigPickerScreen,
    CostConfirmScreen,
    RunCountScreen,
    WizardModeScreen,
    WizardProgressScreen,
)
from tests.tui_test_support import keep_all_families, run_async
from tests.tui_test_support import wait_until as _wait_until


async def _choose_run_count(pilot, app, n_runs_per_case):
    """Advance past the run-count sizing screen by explicitly picking a
    count. Tests that are about the *cost* step pick the small option so
    they stay as cheap as they were before sizing existed — the sizing
    screen's own behaviour is covered in test_tui_run_sizing.py."""
    await keep_all_families(pilot, app)
    screen = app.screen
    assert isinstance(screen, RunCountScreen), f"expected RunCountScreen, got {type(screen).__name__}"
    index = next(i for i, o in enumerate(screen.options) if o.n_runs_per_case == n_runs_per_case)
    screen.query_one("#run-count-menu", ListView).index = index
    await pilot.press("enter")
    await pilot.pause()


def _reconstructed_config(label, *, provider="anthropic"):
    return SystemConfig(
        label=label,
        model=ModelConfig(provider=provider, model_name="claude-haiku-4-5-20251001"),
        agents=[AgentSpec(role="supervisor", name="A", system_prompt="[unavailable]", system_prompt_source="unavailable", tools=["send_invoice"])],
        security=SecurityConfig(),
        provenance=ReconstructionProvenance(
            project_id="proj-1", source_agent_name="A", trace_count=5, extraction_date="2026-01-01",
            tool_profiles={"send_invoice": ToolBehaviorProfile(tool_name="send_invoice")},
            avg_cost_usd_per_trace=0.01, avg_generations_per_trace=1.0,
        ),
    )


# --- mode select ---------------------------------------------------------


def test_mode_screen_lists_both_modes():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(WizardModeScreen())
            await pilot.pause()
            menu = app.screen.query_one("#wizard-mode-menu", ListView)
            assert len(menu.children) == 2

    run_async(scenario)


def test_selecting_single_mode_pushes_picker_needing_one_config():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(WizardModeScreen())
            await pilot.pause()
            await pilot.press("enter")  # first item: "single"
            await pilot.pause()
            assert isinstance(app.screen, ConfigPickerScreen)
            assert app.screen.n_needed == 1

    run_async(scenario)


def test_selecting_comparison_mode_pushes_picker_needing_two_configs():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(WizardModeScreen())
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("enter")  # second item: "comparison"
            await pilot.pause()
            assert isinstance(app.screen, ConfigPickerScreen)
            assert app.screen.n_needed == 2

    run_async(scenario)


# --- config picker --------------------------------------------------------


def test_picker_shows_empty_state_when_no_configs_saved(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = ConfigPickerScreen(n_needed=1, on_chosen=lambda configs: None, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            assert not screen.query("#config-picker-list")
            labels = " ".join(str(label.render()) for label in screen.query(Label))
            assert "No environments yet" in labels

    run_async(scenario)


def test_picker_never_offers_a_baseline_or_toy_option(tmp_path):
    save_config(baseline_config(label="one"), configs_dir=tmp_path)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = ConfigPickerScreen(n_needed=1, on_chosen=lambda configs: None, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            menu = screen.query_one("#config-picker-list", ListView)
            assert all(item.id != "__baseline__" for item in menu.children)

    run_async(scenario)


def test_picker_with_one_needed_calls_back_and_pops_on_first_pick(tmp_path):
    save_config(baseline_config(label="only-config"), configs_dir=tmp_path)
    chosen_configs = []

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            base_depth = len(app.screen_stack)
            app.push_screen(ConfigPickerScreen(n_needed=1, on_chosen=chosen_configs.append, configs_dir=tmp_path))
            await pilot.pause()
            await pilot.press("enter")  # only one row: "only-config"
            await pilot.pause()
            assert len(app.screen_stack) == base_depth  # popped back off
            assert len(chosen_configs) == 1
            assert chosen_configs[0][0].label == "only-config"

    run_async(scenario)


def test_picker_with_two_needed_waits_for_second_pick(tmp_path):
    save_config(baseline_config(label="one"), configs_dir=tmp_path)
    save_config(baseline_config(label="two", defensive_instruction=False), configs_dir=tmp_path)
    chosen_configs = []

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = ConfigPickerScreen(n_needed=2, on_chosen=chosen_configs.append, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")  # first pick
            await pilot.pause()
            assert chosen_configs == []  # not called yet — only one of two chosen
            assert isinstance(app.screen, ConfigPickerScreen)  # still on the picker
            await pilot.press("enter")  # second pick
            await pilot.pause()
            assert len(chosen_configs) == 1
            assert len(chosen_configs[0]) == 2

    run_async(scenario)


def test_picker_rows_show_description_with_hash_demoted_below(tmp_path):
    save_config(baseline_config(defensive_instruction=False), configs_dir=tmp_path)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = ConfigPickerScreen(n_needed=1, on_chosen=lambda configs: None, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            menu = screen.query_one("#config-picker-list", ListView)
            real_config_item = menu.children[0]  # no baseline row anymore — this is the only item
            text = str(real_config_item.query_one(Label).render())
            assert "baseline, but supervisor's defensive instruction removed" in text
            assert "cfg_" in text  # hash still present, just secondary

    run_async(scenario)


# --- progress + landing ----------------------------------------------------


def test_single_config_progress_lands_on_single_verdict(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            # not bit-identical to baseline, so config_label ends up as a real
            # generated description rather than the trivial "baseline (defaults)" case
            config = baseline_config(label="wizard-test-single", defensive_instruction=False)
            app.push_screen(WizardProgressScreen(mode="single", configs=[config], n_runs_per_case=1, runs_dir=tmp_path))
            await _wait_until(pilot, lambda: isinstance(app.screen, SingleConfigVerdictScreen))
            # config_label holds the auto-generated description, not the raw
            # human-chosen SystemConfig.label — see tui.data.describe_config_for_humans
            assert app.screen.summary.config_label == "baseline, but supervisor's defensive instruction removed"

    run_async(scenario)


def test_comparison_progress_lands_on_comparison_verdict_and_persists_report(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            config_a = baseline_config(label="wizard-test-a", defensive_instruction=True)
            config_b = baseline_config(label="wizard-test-b", defensive_instruction=False)
            app.push_screen(
                WizardProgressScreen(mode="comparison", configs=[config_a, config_b], n_runs_per_case=1, runs_dir=tmp_path)
            )
            await _wait_until(pilot, lambda: isinstance(app.screen, ComparisonVerdictScreen))
            assert app.screen.verdict.tier in {"FLAGGED", "CLEAR", "INCONCLUSIVE"}
            report_files = list(tmp_path.glob("adhoc_*_report.json"))
            assert len(report_files) == 1

    run_async(scenario)


# --- cost confirmation for real-model / reconstructed runs -------------------


def test_mock_config_start_skips_cost_confirm(tmp_path):
    # The mock backend is fast enough that by the time this assertion
    # runs, WizardProgressScreen may have already finished and landed on
    # the verdict screen -- what this test actually checks is that
    # CostConfirmScreen was never inserted into the flow at all, not that
    # we catch WizardProgressScreen mid-run.
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = WizardModeScreen(runs_dir=tmp_path, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            screen._start([baseline_config(label="mock-only")])
            await pilot.pause()
            await _choose_run_count(pilot, app, 5)
            assert not isinstance(app.screen, CostConfirmScreen)
            await _wait_until(pilot, lambda: isinstance(app.screen, SingleConfigVerdictScreen))
            # regression: applicable_cases_for_configs previously looked only
            # at the supervisor agent's tools (empty for the toy system),
            # silently filtering the case suite down to zero
            assert app.screen.summary.total_attacks > 0

    run_async(scenario)


def test_reconstructed_config_start_shows_cost_confirm_with_estimate(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = WizardModeScreen(runs_dir=tmp_path, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            screen._start([_reconstructed_config("recon")])
            await pilot.pause()
            await _choose_run_count(pilot, app, 5)
            assert isinstance(app.screen, CostConfirmScreen)
            text = str(app.screen.query_one(".subtitle", Label).render())
            assert "estimated cost $" in text

    run_async(scenario)


def test_reconstructed_config_forced_to_anthropic_even_if_saved_as_mock(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = WizardModeScreen(runs_dir=tmp_path, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            screen._start([_reconstructed_config("recon", provider="mock")])
            await pilot.pause()
            await _choose_run_count(pilot, app, 5)
            # forced to anthropic -> any_real_model=True -> cost confirm shown,
            # not silently routed to the (blocked) mock backend
            assert isinstance(app.screen, CostConfirmScreen)
            assert app.screen.estimate.any_real_model

    run_async(scenario)


def test_cost_confirm_cancel_returns_to_wizard_mode_without_running(tmp_path):
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = WizardModeScreen(runs_dir=tmp_path, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            screen._start([_reconstructed_config("recon")])
            await pilot.pause()
            await _choose_run_count(pilot, app, 5)
            assert isinstance(app.screen, CostConfirmScreen)
            await pilot.press("down")  # "Cancel"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WizardModeScreen)

    run_async(scenario)


def test_cost_confirm_proceed_runs_against_reconstructed_config(tmp_path, monkeypatch):
    from dataclasses import dataclass

    from agno.agent import Agent

    @dataclass
    class _FakeMetrics:
        input_tokens: int = 10
        output_tokens: int = 5

    @dataclass
    class _FakeRunOutput:
        content: str
        tools: list
        metrics: object = None

        def __post_init__(self):
            if self.metrics is None:
                self.metrics = _FakeMetrics()

    monkeypatch.setattr(Agent, "run", lambda self, task, *, session_id=None, **kwargs: _FakeRunOutput(content="Done.", tools=[]))

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = WizardModeScreen(runs_dir=tmp_path, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            screen._start([_reconstructed_config("recon")])
            await pilot.pause()
            await _choose_run_count(pilot, app, 5)
            assert isinstance(app.screen, CostConfirmScreen)
            await pilot.press("enter")  # "Proceed" (first item, already highlighted)
            await _wait_until(pilot, lambda: isinstance(app.screen, SingleConfigVerdictScreen))

    run_async(scenario)
