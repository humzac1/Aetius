"""Pulls traces from a real Braintrust project and caches them to disk, so
re-running extraction (ingestion/reconstruct.py, once it grows Braintrust
support) never re-hits the API. Parallel in shape to
ingestion/langfuse_client.py -- same caching-to-disk discipline, same
idempotent gap-only re-pull behavior -- but the pull mechanics differ for
real reasons, investigated live against a real BRAINTRUST_API_KEY before
writing this (not assumed from docs, not copied from Langfuse's shape):

  - The SDK's actual expected env var is BRAINTRUST_API_KEY (no
    underscore between BRAIN and TRUST -- braintrust/env.py's
    EnvVar("BRAINTRUST_API_KEY", ...)). This repo's .env had it as
    BRAIN_TRUST_API_KEY, silently never read by the SDK; fixed as part of
    this investigation.
  - Auth is API-key-only for org resolution: braintrust.login_to_state
    auto-discovers the org from the key (confirmed against a real
    single-org account, "Bungalow") -- no separate org identifier needed
    for the common case. An account belonging to multiple orgs would need
    one (login_to_state's org_name param exists for exactly that), so
    BRAINTRUST_ORG_NAME is supported as an optional override here, not a
    required credential -- not tested against a real multi-org account
    (this key's account only has one), inferred from login_to_state's own
    source and docstring.
  - A PROJECT identifier is still required to pull anything, the same way
    Langfuse needed LANGFUSE_PROJECT_ID -- one key/org can see multiple
    projects (confirmed: this real account has 3: "homepilot",
    "homepilot-staging", "My Project"). Unlike Langfuse's opaque
    LANGFUSE_PROJECT_ID, Braintrust's BTQL project_logs() function accepts
    the human-readable project *name* directly (confirmed against the
    real account) -- so this module takes a project name, not an id.
  - No LANGFUSE_BASE_URL equivalent is required: the API host is
    auto-discovered from the org's own metadata during login.
    BRAINTRUST_API_URL exists as an optional environment override for
    self-hosted deployments (read by the SDK itself, not by this module).
  - Traces are NOT a nested trace->observations tree fetched in one call
    the way Langfuse's trace.get() was. BTQL returns a flat stream of
    spans (potentially across many traces at once), linked by
    root_span_id / span_parents / is_root. "Pull N most recent traces"
    means: list the N most recent ROOT spans (filter: is_root = true,
    sorted by created desc), then fetch every span belonging to any of
    them in one bulk, cursor-paginated query (filter: root_span_id in
    [...]) and group the flat result back into per-trace spans in memory
    -- confirmed empirically (zero overlap between pages, correct
    termination at an empty page with no cursor).
  - CORRECTION to this module's original design, found for real in a
    later investigation: the first version of this function fetched each
    trace's spans with its own separate query, one call per trace (two,
    actually -- see the next bullet). Instrumented a real fresh pull and
    counted exactly 31 calls for 15 traces (1 root-list + 15x2
    trace-fetch); at the default batch_size=100 that's ~201 calls, easily
    enough to blow through a burst rate limit -- confirmed as the real
    cause of a 429 that hit consistently around the 9th-10th trace
    (~19-20 calls in), reproduced live, not assumed. BTQL's "in" filter
    (confirmed against the installed SDK's own real usage in trace.py)
    lets the whole requested batch's spans come back in one or a few
    bulk, cursor-paginated calls instead -- root_span_id in [id1, id2,
    ...] -- so call count no longer scales with trace count at all.
  - A real BTQL pagination quirk, confirmed live and worth knowing for
    whoever paginates a BTQL query next: a page can return a non-null
    cursor even when its row count is well under the requested limit (a
    handful of rows on a limit=1000 page still got a cursor) -- the
    correct termination check is "stop once a page returns zero rows OR
    no cursor," not "stop once row count < limit." Every pagination loop
    in this module follows that rule.
  - A real (not name-substring) signal exists for judge/eval spans, for
    whoever writes Braintrust reconstruction next: span_attributes.type
    == "score" for Braintrust's own scorer executions, and separately
    this project's own judge agent tags metadata.agent_name containing
    "Judge". Confirmed against real data; excluding these is
    reconstruction's job, not this module's -- this module caches
    whatever's actually in the trace, faithfully.
  - System prompts: confirmed recoverable for this real account's data
    (present verbatim in role="system" chat messages inside an LLM span's
    input.messages) -- unlike the Langfuse project investigated
    previously, where they were not. This is a property of what the
    logging application chose to send, not a Braintrust guarantee either
    way; reconstruction should keep treating system_prompt_source as
    observed per-project, never hardcode "always available" off this one
    account's data.
  - Pagination is cursor-based, returned in the BTQL response body
    (resp["cursor"]) -- not page numbers. Confirmed empirically: passing
    the previous response's cursor back into the next request's query
    continues correctly with zero overlap; an empty page with no cursor
    signals the end.
  - No visible rate-limit headers on responses except Retry-After, sent
    on real 429s (confirmed against a real rate-limited response during
    this investigation).
  - CORRECTION to an earlier (wrong) assumption in this docstring: the
    SDK does define a policy-aware retry engine with real 429 backoff
    (braintrust/api/policies.py's RetryPolicy, braintrust/api/
    _transport.py's Transport class) -- but state.api_conn() (what this
    module actually calls) returns the OLDER HTTPConnection class, which
    Transport's own module docstring says explicitly: "Existing
    HTTPConnection call sites deliberately do not use this class yet."
    HTTPConnection's default adapter is built with Retry(total=0) --
    zero retries -- and even calling its make_long_lived() (which swaps
    in RetryRequestExceptionsAdapter) doesn't help: that adapter only
    retries on network-level exceptions (connection errors, timeouts),
    never on an HTTP status code -- a 429 comes back as an ordinary,
    non-raising Response, so raise_for_status() turning it into an
    exception happens entirely outside that adapter's retry loop.
    Confirmed by reading both classes end to end, not assumed. So a real
    429 previously reached this module's caller immediately, as a bare
    requests.HTTPError, on the first attempt -- the exponential-backoff
    retry below is hand-rolled in _query_btql specifically because the
    SDK has no working retry path for this call at all, not because a
    working one just needed a bigger budget.

Never logs or prints BRAINTRUST_API_KEY.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from braintrust.logger import BraintrustState, login_to_state

from config import paths
from config.credentials import ensure_env_loaded

DEFAULT_TRACES_DIR = paths.TRACES_BRAINTRUST_DIR
DEFAULT_BATCH_SIZE = 100
DEFAULT_PAGE_LIMIT = 100  # root-span listing pages
DEFAULT_TRACE_FETCH_LIMIT = 1000  # per-trace span fetch -- matches the SDK's own DEFAULT_FETCH_BATCH_SIZE


def build_client(*, api_key: str | None = None, org_name: str | None = None) -> BraintrustState:
    """Reads BRAINTRUST_API_KEY (and optionally BRAINTRUST_ORG_NAME) from
    the environment unless explicitly overridden. login_to_state returns
    an isolated BraintrustState (not the SDK's global singleton), so this
    never has the side effect of logging in the whole process -- callers
    (e.g. config/credentials.py's validate_braintrust) can validate
    candidate values before they're written anywhere. Raises with a clear
    message naming which var is missing, never touching its value; the
    SDK's own login_to_state already masks the key in ITS error messages
    (mask_api_key), a real, confirmed behavior -- but this still never
    passes str(exc) through, same discipline as build_client's caller."""
    ensure_env_loaded()
    api_key = api_key or os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        raise RuntimeError("missing required credentials: BRAINTRUST_API_KEY")
    org_name = org_name or os.environ.get("BRAINTRUST_ORG_NAME") or None
    return login_to_state(api_key=api_key, org_name=org_name)


def default_project_name() -> str:
    ensure_env_loaded()
    project_name = os.environ.get("BRAINTRUST_PROJECT_NAME")
    if not project_name:
        raise RuntimeError("BRAINTRUST_PROJECT_NAME not set")
    return project_name


def _ident(*name: str) -> dict[str, Any]:
    return {"op": "ident", "name": list(name)}


def _literal(value: Any) -> dict[str, Any]:
    return {"op": "literal", "value": value}


def _eq_filter(field: str, value: Any) -> dict[str, Any]:
    return {"op": "eq", "left": _ident(field), "right": _literal(value)}


def _in_filter(field: str, values: list[Any]) -> dict[str, Any]:
    return {"op": "in", "left": _ident(field), "right": _literal(values)}


_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 5  # 1 initial + 4 retries
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 20.0


class BraintrustRateLimitError(RuntimeError):
    """A retryable status (408/429/5xx) persisted across every attempt in
    the retry budget -- distinct from a plain requests.HTTPError so a
    caller (AddEnvironmentScreen) can tell "still rate limited/unavailable
    after retrying" apart from a genuine, non-transient failure (bad
    credentials, missing project, ...), which still raises immediately
    via resp.raise_for_status() below and is never retried or reworded."""


def _retry_after_seconds(resp: Any) -> float | None:
    value = resp.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # an HTTP-date value instead of delta-seconds -- falls back to exponential backoff below


def _query_btql(state: BraintrustState, query: dict[str, Any]) -> dict[str, Any]:
    """Retries 408/429/500/502/503/504 with backoff (honoring a real
    Retry-After header when the server sends one, exponential backoff
    otherwise) before giving up -- see this module's docstring for why
    the SDK's own transport doesn't already do this for state.api_conn().
    Any other status (401/403/404/400, ...) is not retryable and raises
    immediately on the first attempt, unchanged from before -- a
    persistent, non-transient failure is never hidden behind retrying."""
    resp = None
    for attempt in range(_MAX_ATTEMPTS):
        resp = state.api_conn().post("btql", json={"query": query})
        if resp.status_code not in _RETRYABLE_STATUSES:
            resp.raise_for_status()
            return resp.json()
        if attempt == _MAX_ATTEMPTS - 1:
            break
        wait = _retry_after_seconds(resp)
        if wait is None:
            wait = min(_BASE_BACKOFF_SECONDS * (2**attempt), _MAX_BACKOFF_SECONDS)
        time.sleep(wait)

    kind = "rate limited" if resp.status_code in (429, 503) else "temporarily unavailable"
    retry_hint = _retry_after_seconds(resp)
    hint_text = f"retry in {retry_hint:.0f}s" if retry_hint is not None else "try again shortly"
    raise BraintrustRateLimitError(
        f"Braintrust API is {kind} (HTTP {resp.status_code}) — still failing after {_MAX_ATTEMPTS} attempts, {hint_text}."
    )


def _project_logs_page(
    state: BraintrustState,
    project_name: str,
    *,
    filter_expr: dict[str, Any] | None,
    sort_desc_by: str | None,
    cursor: str | None,
    limit: int,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "select": [{"op": "star"}],
        "from": {"op": "function", "name": _ident("project_logs"), "args": [_literal(project_name)]},
        "cursor": cursor,
        "limit": limit,
    }
    if filter_expr is not None:
        query["filter"] = filter_expr
    if sort_desc_by is not None:
        query["sort"] = [{"expr": _ident(sort_desc_by), "dir": "desc"}]
    return _query_btql(state, query)


def _list_recent_root_span_ids(state: BraintrustState, project_name: str, *, batch_size: int, page_limit: int) -> list[str]:
    """Newest-first root_span_ids (one per trace), up to batch_size --
    the "which traces exist" step, analogous to Langfuse's trace.list()."""
    root_ids: list[str] = []
    cursor: str | None = None
    while len(root_ids) < batch_size:
        page = _project_logs_page(
            state,
            project_name,
            filter_expr=_eq_filter("is_root", True),
            sort_desc_by="created",
            cursor=cursor,
            limit=min(page_limit, batch_size - len(root_ids)),
        )
        rows = page.get("data") or []
        if not rows:
            break
        root_ids.extend(row["root_span_id"] for row in rows)
        cursor = page.get("cursor")
        if not cursor:
            break
    return root_ids[:batch_size]


def _fetch_traces_bulk(
    state: BraintrustState, project_name: str, root_span_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Every span belonging to any of root_span_ids, in one bulk,
    cursor-paginated query (filter: root_span_id in [...]) -- not one
    query per trace. See this module's docstring for why: the original
    per-trace design made ~2 calls per trace (confirmed live), which is
    what actually caused a real, reproduced 429 at ordinary batch sizes.
    Groups the flat result back into per-root_span_id lists in memory."""
    if not root_span_ids:
        return {}
    spans_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    filter_expr = _in_filter("root_span_id", root_span_ids)
    cursor: str | None = None
    while True:
        page = _project_logs_page(
            state,
            project_name,
            filter_expr=filter_expr,
            sort_desc_by=None,
            cursor=cursor,
            limit=DEFAULT_TRACE_FETCH_LIMIT,
        )
        rows = page.get("data") or []
        for row in rows:
            spans_by_root[row["root_span_id"]].append(row)
        cursor = page.get("cursor")
        if not rows or not cursor:
            break
    return dict(spans_by_root)


def _trace_cache_path(project_name: str, root_span_id: str, *, traces_dir: Path) -> Path:
    return traces_dir / project_name / f"{root_span_id}.json"


def pull_traces(
    *,
    project_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    traces_dir: Path = DEFAULT_TRACES_DIR,
    client: BraintrustState | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
) -> list[str]:
    """Pulls up to batch_size most-recent traces for project_name
    (defaults to BRAINTRUST_PROJECT_NAME), caching each trace's full body
    (every span sharing its root_span_id) to
    <traces_dir>/<project_name>/<root_span_id>.json.

    Idempotent per trace, same discipline as
    ingestion/langfuse_client.py's pull_traces: a trace already cached on
    disk is not re-fetched -- re-running this with a larger batch_size
    only pulls the gap. Returns every root_span_id now in the cache for
    this project (not just newly pulled ones), up to batch_size, newest
    first.

    The gap (root_span_ids not already cached) is fetched in one bulk
    call to _fetch_traces_bulk, not one call per trace -- see this
    module's docstring for why that distinction is the whole point.
    """
    project_name = project_name or default_project_name()
    state = client or build_client()
    cache_dir = traces_dir / project_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    root_span_ids = _list_recent_root_span_ids(state, project_name, batch_size=batch_size, page_limit=page_limit)
    missing_ids = [rid for rid in root_span_ids if not _trace_cache_path(project_name, rid, traces_dir=traces_dir).exists()]
    spans_by_root = _fetch_traces_bulk(state, project_name, missing_ids)
    for root_span_id in missing_ids:
        path = _trace_cache_path(project_name, root_span_id, traces_dir=traces_dir)
        path.write_text(json.dumps(spans_by_root.get(root_span_id, []), indent=2), encoding="utf-8")
    return root_span_ids


def load_cached_trace_ids(project_name: str, *, traces_dir: Path = DEFAULT_TRACES_DIR) -> list[str]:
    cache_dir = traces_dir / project_name
    if not cache_dir.exists():
        return []
    return sorted(p.stem for p in cache_dir.glob("*.json"))


def load_cached_trace(project_name: str, root_span_id: str, *, traces_dir: Path = DEFAULT_TRACES_DIR) -> list[dict[str, Any]]:
    path = _trace_cache_path(project_name, root_span_id, traces_dir=traces_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def load_cached_traces(project_name: str, *, traces_dir: Path = DEFAULT_TRACES_DIR) -> list[list[dict[str, Any]]]:
    return [load_cached_trace(project_name, rid, traces_dir=traces_dir) for rid in load_cached_trace_ids(project_name, traces_dir=traces_dir)]
