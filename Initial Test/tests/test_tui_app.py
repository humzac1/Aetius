"""Headless Textual Pilot tests for the TUI navigation shell — the
Pilot/run_test() analogue of test_dashboard_app.py's AppTest pattern.
Wrapped in asyncio.run() rather than pytest-asyncio so this needs no new
test dependency; each test function is a plain sync def pytest can collect
normally."""

import asyncio

from tui.app import HarnessApp, HomeScreen, PlaceholderScreen


def run_async(coro_fn):
    asyncio.run(coro_fn())


def test_app_starts_on_home_screen():
    async def scenario():
        app = HarnessApp()
        async with app.run_test():
            assert isinstance(app.screen, HomeScreen)

    run_async(scenario)


def test_home_screen_lists_all_six_menu_items():
    async def scenario():
        app = HarnessApp()
        async with app.run_test():
            from textual.widgets import ListView

            menu = app.screen.query_one("#menu", ListView)
            assert len(menu.children) == 6

    run_async(scenario)


def test_selecting_menu_item_pushes_a_screen():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            base_depth = len(app.screen_stack)
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.screen_stack) == base_depth + 1

    run_async(scenario)


def test_all_menu_items_now_resolve_to_real_screens():
    # every menu destination has a real screen module as of tasks #53-#55 —
    # PlaceholderScreen's ImportError fallback (tested directly below) is what
    # made incremental development possible, but nothing should hit it anymore
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            for _ in range(5):  # the first five items each push a screen; "Settings" (6th) exits instead
                await pilot.press("enter")
                await pilot.pause()
                assert not isinstance(app.screen, PlaceholderScreen)
                await pilot.press("h")
                await pilot.pause()
                await pilot.press("down")

    run_async(scenario)


def test_placeholder_screen_renders_given_title_and_supports_back():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            app.push_screen(PlaceholderScreen("Some future screen"))
            await pilot.pause()
            assert isinstance(app.screen, PlaceholderScreen)
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    run_async(scenario)


def test_back_binding_returns_to_home():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, HomeScreen)
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    run_async(scenario)


def test_home_binding_returns_to_home_from_any_depth():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, HomeScreen)
            await pilot.press("h")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    run_async(scenario)


def test_settings_menu_item_exits_app():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            from textual.widgets import ListView

            menu = app.screen.query_one("#menu", ListView)
            menu.index = 5  # "Settings / exit" is the sixth item
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.return_code == 0 or not app.is_running

    run_async(scenario)


def test_quit_binding_exits_app():
    async def scenario():
        app = HarnessApp()
        async with app.run_test() as pilot:
            await pilot.press("q")
            await pilot.pause()
            assert not app.is_running

    run_async(scenario)
