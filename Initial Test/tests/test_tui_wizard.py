from textual.widgets import Label, ListView

from target_system.config import AgentSpec, ModelConfig, SecurityConfig, SystemConfig, save_config
from target_system.factory import baseline_config
from target_system.provenance import ReconstructionProvenance, ToolBehaviorProfile
from tui.app import HarnessApp
from tui.screens.configs import ConfigDiffScreen
from tui.screens.verdict import ComparisonVerdictScreen, SingleConfigVerdictScreen
from tui.screens.wizard import CostConfirmScreen, ConfigPickerScreen, WizardModeScreen, WizardProgressScreen
from tests.tui_test_support import run_async
from tests.tui_test_support import wait_until as _wait_until


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


def test_picker_always_offers_baseline_first():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = ConfigPickerScreen(n_needed=1, on_chosen=lambda configs: None)
            app.push_screen(screen)
            await pilot.pause()
            menu = screen.query_one("#config-picker-list", ListView)
            assert menu.children[0].id == "__baseline__"

    run_async(scenario)


def test_picker_with_one_needed_calls_back_and_pops_on_first_pick():
    chosen_configs = []

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            base_depth = len(app.screen_stack)
            app.push_screen(ConfigPickerScreen(n_needed=1, on_chosen=chosen_configs.append))
            await pilot.pause()
            await pilot.press("enter")  # picks baseline
            await pilot.pause()
            assert len(app.screen_stack) == base_depth  # popped back off
            assert len(chosen_configs) == 1
            assert chosen_configs[0][0].label == "baseline"

    run_async(scenario)


def test_picker_with_two_needed_waits_for_second_pick():
    chosen_configs = []

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = ConfigPickerScreen(n_needed=2, on_chosen=chosen_configs.append)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")  # first pick: baseline
            await pilot.pause()
            assert chosen_configs == []  # not called yet — only one of two chosen
            assert isinstance(app.screen, ConfigPickerScreen)  # still on the picker
            await pilot.press("enter")  # second pick: baseline again
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
            real_config_item = menu.children[1]  # index 0 is "New: baseline (defaults)"
            text = str(real_config_item.query_one(Label).render())
            assert "baseline, but supervisor's defensive instruction removed" in text
            assert "cfg_" in text  # hash still present, just secondary

    run_async(scenario)


# --- view diff (v keybinding) ------------------------------------------------


def test_v_on_baseline_item_does_nothing():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = ConfigPickerScreen(n_needed=1, on_chosen=lambda configs: None)
            app.push_screen(screen)
            await pilot.pause()  # highlighted_child defaults to the first item: baseline
            await pilot.press("v")
            await pilot.pause()
            assert isinstance(app.screen, ConfigPickerScreen)

    run_async(scenario)


def test_v_on_real_config_opens_diff_against_baseline(tmp_path):
    target_hash = save_config(baseline_config(defensive_instruction=False), configs_dir=tmp_path)

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = ConfigPickerScreen(n_needed=1, on_chosen=lambda configs: None, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("down")  # move off baseline onto the real config
            await pilot.press("v")
            await pilot.pause()
            assert isinstance(app.screen, ConfigDiffScreen)
            assert app.screen.hash_b == target_hash
            # pressing v must have persisted baseline so the diff screen can actually load it
            from target_system.config import list_config_hashes

            assert app.screen.hash_a in list_config_hashes(configs_dir=tmp_path)

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
            assert isinstance(app.screen, CostConfirmScreen)
            await pilot.press("enter")  # "Proceed" (first item, already highlighted)
            await _wait_until(pilot, lambda: isinstance(app.screen, SingleConfigVerdictScreen))

    run_async(scenario)
