"""Shared helpers for headless Textual Pilot tests (tests/test_tui_*.py).

Not itself a test module — pytest won't collect it (no test_ prefix).
"""

from __future__ import annotations

import asyncio
from typing import Callable


def run_async(coro_fn: Callable[[], "asyncio.coroutines.Coroutine"]) -> None:
    asyncio.run(coro_fn())


async def wait_until(pilot, predicate: Callable[[], bool], *, tries: int = 200) -> None:
    """Poll `predicate` (checked after each pilot.pause()) until it's true,
    then settle once more before returning control to the caller.

    The settle matters: mounting a new screen also mounts a Header, whose
    _on_mount registers reactive watchers with init=True — each
    immediately schedules its callback via widget.call_next
    (textual/reactive.py's invoke_watcher), i.e. deferred to a LATER
    message-pump cycle, not the one that just made `predicate` true.
    Returning the instant the predicate flips risks the test (and
    app.run_test()'s teardown) racing ahead of that deferred callback; when
    it finally runs against an already-unmounted widget it raises
    NoMatches, which Header's own set_title() only catches NoScreen
    against (see textual/widgets/_header.py's _on_mount — an upstream gap
    in Header, not something catchable from here). One more pause() drains
    exactly that newly-scheduled work before we hand control back.

    Confirmed against a tight repro loop: ~30-36% failure rate over 25 runs
    without this settle step, 0/25 and 0/20 with it (see the session notes
    for the two repro scripts) — this is a targeted fix for the identified
    mechanism, not a blind retry.
    """
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            await pilot.pause()
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition never became true")


async def keep_all_families(pilot, app) -> None:
    """Advance past FamilyScopeScreen keeping every family selected — the
    no-op scope. Tests about a later step use this so they exercise the
    same case set they did before scoping existed; scoping's own behaviour
    is covered in tests/test_tui_family_scope.py.

    A no-op when the screen isn't showing: a single-family suite skips it."""
    from textual.widgets import ListView

    from tui.screens.wizard import FamilyScopeScreen

    if not isinstance(app.screen, FamilyScopeScreen):
        return
    menu = app.screen.query_one("#family-scope-menu", ListView)
    menu.index = len(app.screen.families)  # the trailing "Continue" row
    await pilot.press("enter")
    await pilot.pause()
