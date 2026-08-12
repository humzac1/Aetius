"""No live API calls here — a stub state/connection matching the one HTTP
surface this module actually uses (state.api_conn().post("btql", ...)),
shaped like the real BTQL request/response confirmed live in the
investigation behind ingestion/braintrust_client.py: cursor-based paging,
a `filter` clause distinguishing "list root spans" (is_root = true) from
"fetch every span for this batch of traces" (root_span_id in [...]) --
one bulk, cursor-paginated call for the whole batch, not one call per
trace. See ingestion/braintrust_client.py's docstring for the real,
reproduced 429 the old per-trace design caused (31 calls for a real
15-trace pull; ~201 at the default batch_size=100).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import requests

import ingestion.braintrust_client as bt_client
from ingestion.braintrust_client import (
    BraintrustRateLimitError,
    load_cached_trace,
    load_cached_trace_ids,
    load_cached_traces,
    pull_traces,
)


@dataclass
class _StubResponse:
    payload: dict[str, Any]
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.payload


@dataclass
class _StubConn:
    """roots: newest-first list of {"root_span_id": ..., "created": ...}
    dicts. spans_by_root: root_span_id -> list of span dicts (already
    ordered). bulk_fetch_calls records each bulk trace-body query's
    requested root_span_ids -- the count of THESE calls (not trace count)
    is what regression tests below assert stays small."""

    roots: list[dict[str, Any]]
    spans_by_root: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    trace_page_size: int = 1000
    root_list_calls: list[dict[str, Any]] = field(default_factory=list)
    bulk_fetch_calls: list[list[str]] = field(default_factory=list)

    def post(self, path: str, json: dict[str, Any]) -> _StubResponse:
        assert path == "btql"
        query = json["query"]
        filter_expr = query.get("filter")
        cursor = query.get("cursor")
        limit = query["limit"]

        if filter_expr is not None and filter_expr["left"]["name"] == ["is_root"]:
            self.root_list_calls.append(query)
            start = int(cursor) if cursor else 0
            page = self.roots[start : start + limit]
            next_cursor = str(start + limit) if start + limit < len(self.roots) else None
            return _StubResponse({"data": page, "cursor": next_cursor})

        if filter_expr is not None and filter_expr.get("op") == "in" and filter_expr["left"]["name"] == ["root_span_id"]:
            requested_ids = filter_expr["right"]["value"]
            self.bulk_fetch_calls.append(list(requested_ids))
            all_spans = [span for rid in requested_ids for span in self.spans_by_root.get(rid, [])]
            start = int(cursor) if cursor else 0
            page = all_spans[start : start + self.trace_page_size]
            next_cursor = str(start + self.trace_page_size) if start + self.trace_page_size < len(all_spans) else None
            return _StubResponse({"data": page, "cursor": next_cursor})

        raise AssertionError(f"unexpected BTQL query: {query}")


@dataclass
class _StubState:
    conn: _StubConn

    def api_conn(self) -> _StubConn:
        return self.conn


def _roots(n: int) -> list[dict[str, Any]]:
    # newest-first, matching what a real `sort: created desc` page returns
    return [{"root_span_id": f"root-{i}", "created": f"2026-01-01T00:00:{n - i:02d}Z"} for i in range(n)]


def _spans_for(root_id: str, n: int = 3) -> list[dict[str, Any]]:
    return [{"id": f"{root_id}-span-{i}", "root_span_id": root_id} for i in range(n)]


def test_pull_traces_caches_each_trace_to_disk(tmp_path):
    roots = _roots(5)
    conn = _StubConn(roots=roots, spans_by_root={r["root_span_id"]: _spans_for(r["root_span_id"]) for r in roots})
    client = _StubState(conn)

    ids = pull_traces(project_name="proj-1", batch_size=5, traces_dir=tmp_path, client=client)

    assert ids == [r["root_span_id"] for r in roots]
    assert set(load_cached_trace_ids("proj-1", traces_dir=tmp_path)) == set(ids)
    # all 5 traces' bodies came from ONE bulk call, not 5 individual ones
    assert len(conn.bulk_fetch_calls) == 1
    assert sorted(conn.bulk_fetch_calls[0]) == sorted(ids)


def test_pull_traces_fetches_trace_bodies_in_one_bulk_call_not_per_trace(tmp_path):
    # Regression for the real, reproduced 429: a fresh 15-trace pull
    # against the real account made 31 calls under the old per-trace
    # design (1 root-list + 15x2 trace-fetch) -- confirmed via an
    # instrumented live pull, not assumed. Call count must not scale with
    # trace count anymore.
    roots = _roots(50)
    conn = _StubConn(roots=roots, spans_by_root={r["root_span_id"]: _spans_for(r["root_span_id"]) for r in roots})
    client = _StubState(conn)

    ids = pull_traces(project_name="proj-1", batch_size=50, traces_dir=tmp_path, client=client)

    assert len(ids) == 50
    assert len(conn.bulk_fetch_calls) == 1  # not 50, not 100
    assert sorted(conn.bulk_fetch_calls[0]) == sorted(ids)


def test_pull_traces_respects_batch_size(tmp_path):
    roots = _roots(20)
    conn = _StubConn(roots=roots, spans_by_root={r["root_span_id"]: _spans_for(r["root_span_id"]) for r in roots})
    client = _StubState(conn)

    ids = pull_traces(project_name="proj-1", batch_size=7, traces_dir=tmp_path, client=client)

    assert len(ids) == 7
    assert ids == [r["root_span_id"] for r in roots[:7]]
    assert len(load_cached_trace_ids("proj-1", traces_dir=tmp_path)) == 7


def test_pull_traces_is_idempotent_and_makes_no_bulk_call_when_nothing_is_missing(tmp_path):
    roots = _roots(5)
    conn = _StubConn(roots=roots, spans_by_root={r["root_span_id"]: _spans_for(r["root_span_id"]) for r in roots})
    client = _StubState(conn)

    pull_traces(project_name="proj-1", batch_size=5, traces_dir=tmp_path, client=client)
    assert len(conn.bulk_fetch_calls) == 1
    # re-running with the same batch size should not re-fetch anything already cached --
    # not even a bulk call with an empty id list, since _fetch_traces_bulk short-circuits
    pull_traces(project_name="proj-1", batch_size=5, traces_dir=tmp_path, client=client)
    assert len(conn.bulk_fetch_calls) == 1


def test_pull_traces_extending_batch_size_only_pulls_the_gap(tmp_path):
    roots = _roots(20)
    conn = _StubConn(roots=roots, spans_by_root={r["root_span_id"]: _spans_for(r["root_span_id"]) for r in roots})
    client = _StubState(conn)

    pull_traces(project_name="proj-1", batch_size=5, traces_dir=tmp_path, client=client)
    assert len(conn.bulk_fetch_calls) == 1
    assert len(conn.bulk_fetch_calls[0]) == 5
    pull_traces(project_name="proj-1", batch_size=10, traces_dir=tmp_path, client=client)
    assert len(conn.bulk_fetch_calls) == 2  # one more bulk call, not five more individual ones
    assert len(conn.bulk_fetch_calls[1]) == 5  # only the 5 new ids, not all 10
    assert len(load_cached_trace_ids("proj-1", traces_dir=tmp_path)) == 10


def test_pull_traces_stops_when_project_has_fewer_traces_than_batch_size(tmp_path):
    roots = _roots(3)
    conn = _StubConn(roots=roots, spans_by_root={r["root_span_id"]: _spans_for(r["root_span_id"]) for r in roots})
    client = _StubState(conn)

    ids = pull_traces(project_name="proj-1", batch_size=100, traces_dir=tmp_path, client=client)
    assert len(ids) == 3


def test_pull_traces_follows_root_listing_cursor_across_pages(tmp_path):
    # more roots than one page_limit forces >1 root-listing request
    roots = _roots(25)
    conn = _StubConn(roots=roots, spans_by_root={r["root_span_id"]: _spans_for(r["root_span_id"]) for r in roots})
    client = _StubState(conn)

    ids = pull_traces(project_name="proj-1", batch_size=25, traces_dir=tmp_path, client=client, page_limit=10)

    assert ids == [r["root_span_id"] for r in roots]
    assert len(conn.root_list_calls) == 3  # 10 + 10 + 5


def test_pull_traces_follows_bulk_fetch_span_cursor_across_pages(tmp_path):
    # The bulk trace-body query still paginates via cursor (same real
    # BTQL quirk documented in the client's docstring: a page can return
    # a cursor even under its limit) -- now across the whole batch's
    # combined spans, not per trace.
    root_id = "root-0"
    spans = _spans_for(root_id, n=12)
    conn = _StubConn(roots=[{"root_span_id": root_id, "created": "2026-01-01T00:00:00Z"}], spans_by_root={root_id: spans}, trace_page_size=5)
    client = _StubState(conn)

    pull_traces(project_name="proj-1", batch_size=1, traces_dir=tmp_path, client=client)

    cached = load_cached_trace("proj-1", root_id, traces_dir=tmp_path)
    assert len(cached) == 12  # all spans recovered across 3 pages (5 + 5 + 2), not just the first page
    # 3 HTTP calls here (paginating one oversized trace's spans), but all
    # 3 are still the SAME bulk query (root_span_id in [...]) -- every
    # page requested the same single-id batch, confirming this is still
    # one logical bulk-fetch operation paginating, not a re-fetch per id.
    assert len(conn.bulk_fetch_calls) == 3
    assert all(call == [root_id] for call in conn.bulk_fetch_calls)


def test_pull_traces_bulk_fetch_spans_are_grouped_back_to_the_right_trace(tmp_path):
    # The bulk query returns a flat, interleaved span list across every
    # requested trace -- this confirms they're grouped back to the
    # correct per-root_span_id cache file, not just that call counts drop.
    roots = _roots(3)
    conn = _StubConn(roots=roots, spans_by_root={r["root_span_id"]: _spans_for(r["root_span_id"], n=2) for r in roots}, trace_page_size=2)
    client = _StubState(conn)

    pull_traces(project_name="proj-1", batch_size=3, traces_dir=tmp_path, client=client)

    for r in roots:
        cached = load_cached_trace("proj-1", r["root_span_id"], traces_dir=tmp_path)
        assert len(cached) == 2
        assert all(span["root_span_id"] == r["root_span_id"] for span in cached)


def test_load_cached_trace_ids_empty_when_nothing_pulled(tmp_path):
    assert load_cached_trace_ids("nope", traces_dir=tmp_path) == []


def test_load_cached_trace_roundtrip(tmp_path):
    root_id = "root-x"
    conn = _StubConn(roots=[{"root_span_id": root_id, "created": "2026-01-01T00:00:00Z"}], spans_by_root={root_id: _spans_for(root_id, n=2)})
    client = _StubState(conn)

    pull_traces(project_name="proj-1", batch_size=1, traces_dir=tmp_path, client=client)
    trace = load_cached_trace("proj-1", root_id, traces_dir=tmp_path)
    assert len(trace) == 2
    assert trace[0]["root_span_id"] == root_id


def test_load_cached_traces_returns_every_cached_trace(tmp_path):
    roots = _roots(4)
    conn = _StubConn(roots=roots, spans_by_root={r["root_span_id"]: _spans_for(r["root_span_id"]) for r in roots})
    client = _StubState(conn)

    pull_traces(project_name="proj-1", batch_size=4, traces_dir=tmp_path, client=client)
    traces = load_cached_traces("proj-1", traces_dir=tmp_path)
    assert len(traces) == 4
    assert {t[0]["root_span_id"] for t in traces} == {r["root_span_id"] for r in roots}


# --- _query_btql: 429/5xx retry handling -------------------------------
#
# Regression: state.api_conn() (what this module calls) returns the SDK's
# legacy HTTPConnection, which has NO working retry path for an HTTP
# status code -- confirmed by reading braintrust/api/_transport.py end to
# end (see this module's corrected docstring). Before _query_btql grew
# its own retry loop, a real 429 reached AddEnvironmentScreen as a bare
# requests.HTTPError on the very first attempt, no retry at all.


@dataclass
class _ScriptedConn:
    """Returns one canned _StubResponse per call, in order, regardless of
    query contents -- _query_btql-level tests don't need real BTQL query
    shape, just the status-code/headers/backoff behavior."""

    responses: list[_StubResponse]
    calls: int = 0

    def post(self, path: str, json: dict[str, Any]) -> _StubResponse:
        assert path == "btql"
        resp = self.responses[self.calls]
        self.calls += 1
        return resp


def _no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(bt_client.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def test_query_btql_retries_transparently_on_transient_429(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    conn = _ScriptedConn(responses=[
        _StubResponse({}, status_code=429, headers={"Retry-After": "2"}),
        _StubResponse({"data": [], "cursor": None}, status_code=200),
    ])
    state = _StubState(conn)

    result = bt_client._query_btql(state, {"select": [{"op": "star"}]})

    assert result == {"data": [], "cursor": None}
    assert conn.calls == 2
    assert sleeps == [2.0]  # honored the real Retry-After header, not a guessed backoff


def test_query_btql_retries_using_backoff_when_no_retry_after_header(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    conn = _ScriptedConn(responses=[
        _StubResponse({}, status_code=503),
        _StubResponse({"data": [], "cursor": None}, status_code=200),
    ])
    state = _StubState(conn)

    bt_client._query_btql(state, {"select": [{"op": "star"}]})

    assert len(sleeps) == 1
    assert sleeps[0] > 0  # exponential backoff, since no Retry-After was sent


def test_query_btql_raises_clear_rate_limit_error_after_exhausting_retry_budget(monkeypatch):
    _no_sleep(monkeypatch)
    conn = _ScriptedConn(responses=[_StubResponse({}, status_code=429, headers={"Retry-After": "5"})] * bt_client._MAX_ATTEMPTS)
    state = _StubState(conn)

    with pytest.raises(BraintrustRateLimitError, match="rate limited"):
        bt_client._query_btql(state, {"select": [{"op": "star"}]})

    assert conn.calls == bt_client._MAX_ATTEMPTS  # bounded, not infinite


def test_query_btql_does_not_retry_non_transient_failure(monkeypatch):
    sleeps = _no_sleep(monkeypatch)

    class _RaisingResponse(_StubResponse):
        def raise_for_status(self) -> None:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    conn = _ScriptedConn(responses=[_RaisingResponse({}, status_code=401)])
    state = _StubState(conn)

    with pytest.raises(requests.HTTPError):
        bt_client._query_btql(state, {"select": [{"op": "star"}]})

    assert conn.calls == 1  # no retry at all -- a genuine failure surfaces immediately
    assert sleeps == []
