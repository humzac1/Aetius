from textual.widgets import Label, ListView

from tui.app import HarnessApp
from tui.run_sizing import recommend_runs_per_case
from tui.screens.wizard import FamilyScopeScreen, RunCountScreen, WizardModeScreen
from tests.tui_test_support import run_async
from tests.test_tui_run_sizing import _BIG_FAMILY, _SMALL_FAMILY, _cases, _config


def _scope_screen(cases, chosen):
    return FamilyScopeScreen(cases=cases, on_chosen=chosen.append)


def test_lists_every_applicable_family_with_its_case_count():
    cases = _cases({_BIG_FAMILY: 5, _SMALL_FAMILY: 3})
    chosen = []

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(_scope_screen(cases, chosen))
            await pilot.pause()
            menu = app.screen.query_one("#family-scope-menu", ListView)
            assert len(menu.children) == 3  # two families + Continue
            text = " ".join(str(i.query_one(Label).render()) for i in menu.children)
            assert "5 applicable case(s)" in text
            assert "3 applicable case(s)" in text

    run_async(scenario)


def test_everything_is_selected_until_the_user_narrows_it():
    """Scoping must never silently drop coverage — the default is the full
    applicable set, exactly what ran before this screen existed."""
    cases = _cases({_BIG_FAMILY: 5, _SMALL_FAMILY: 3})
    chosen = []

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = _scope_screen(cases, chosen)
            app.push_screen(screen)
            await pilot.pause()
            assert screen.selected == {_BIG_FAMILY, _SMALL_FAMILY}
            menu = screen.query_one("#family-scope-menu", ListView)
            menu.index = 2  # Continue
            await pilot.press("enter")
            await pilot.pause()

    run_async(scenario)
    assert len(chosen) == 1
    assert {c.family for c in chosen[0]} == {_BIG_FAMILY, _SMALL_FAMILY}
    assert len(chosen[0]) == 8


def test_deselecting_a_family_removes_only_its_cases():
    cases = _cases({_BIG_FAMILY: 5, _SMALL_FAMILY: 3})
    chosen = []

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = _scope_screen(cases, chosen)
            app.push_screen(screen)
            await pilot.pause()
            menu = screen.query_one("#family-scope-menu", ListView)
            drop = screen.families.index(_SMALL_FAMILY)
            menu.index = drop
            await pilot.press("enter")  # toggle it off
            await pilot.pause()
            assert screen.selected == {_BIG_FAMILY}
            assert "[ ]" in str(menu.children[drop].query_one(Label).render())
            menu.index = len(screen.families)
            await pilot.press("enter")  # Continue
            await pilot.pause()

    run_async(scenario)
    assert {c.family for c in chosen[0]} == {_BIG_FAMILY}
    assert len(chosen[0]) == 5


def test_cannot_deselect_the_last_family():
    """An empty scope would size and price a run that tests nothing —
    refused rather than allowed to proceed meaninglessly."""
    cases = _cases({_BIG_FAMILY: 5, _SMALL_FAMILY: 3})
    chosen = []

    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = _scope_screen(cases, chosen)
            app.push_screen(screen)
            await pilot.pause()
            menu = screen.query_one("#family-scope-menu", ListView)
            for i in range(len(screen.families)):
                menu.index = i
                await pilot.press("enter")
                await pilot.pause()
            assert len(screen.selected) >= 1

    run_async(scenario)


def test_narrowing_scope_lowers_the_recommended_run_count(tmp_path):
    """The payoff, and the reason this is worth a screen: the weakest
    family drives sizing, so dropping it is the biggest lever on cost —
    same math, narrower scope."""
    all_cases = _cases({_BIG_FAMILY: 5, _SMALL_FAMILY: 3})
    only_big = [c for c in all_cases if c.family == _BIG_FAMILY]

    full = recommend_runs_per_case(all_cases, [_config()], runs_dir=tmp_path)
    narrowed = recommend_runs_per_case(only_big, [_config()], runs_dir=tmp_path)

    assert full.limiting.family == _SMALL_FAMILY
    assert narrowed.limiting.family == _BIG_FAMILY
    assert narrowed.recommended_runs_per_case < full.recommended_runs_per_case


def test_scope_screen_precedes_sizing_and_its_choice_reaches_the_sizer(tmp_path):
    """Wiring check: what the user selected is what gets sized, not the
    full applicable set."""
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            screen = WizardModeScreen(runs_dir=tmp_path, configs_dir=tmp_path)
            app.push_screen(screen)
            await pilot.pause()
            screen._start([_config()])
            await pilot.pause()

            assert isinstance(app.screen, FamilyScopeScreen)
            scope = app.screen
            menu = scope.query_one("#family-scope-menu", ListView)
            kept = scope.families[0]
            for i in range(1, len(scope.families)):
                menu.index = i
                await pilot.press("enter")  # drop every family but the first
                await pilot.pause()
            menu.index = len(scope.families)
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, RunCountScreen)
            assert {c.family for c in app.screen.cases} == {kept}
            assert app.screen.recommendation.limiting.family == kept

    run_async(scenario)
