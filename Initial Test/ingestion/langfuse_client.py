"""Pulls traces from a real Langfuse project and caches them to disk, so
re-running extraction (ingestion/reconstruct.py) never re-hits the API.

Investigated live against real .env credentials before writing this (not
assumed from docs):
  - Reading historical traces goes through Langfuse(...).api — a generated
    REST client (lf.api.trace, lf.api.observations, ...) — a completely
    different surface from the @observe/OTel tracing side of the SDK,
    which is for *producing* traces, not reading them back.
  - lf.api.trace.get(trace_id) returns that trace's full observation
    bodies in one call (not just observation IDs) — the right primitive
    for a per-trace pull: N+1 API calls for N traces, not N + N*M.
  - The generated client already retries 429/408/409 with backoff+jitter
    and honors X-RateLimit-Reset when present (langfuse/api/core/
    http_client.py) — nothing to hand-roll here.
  - The default client timeout was too short for this connection (a plain
    trace.list() call timed out); pass an explicit, generous one.
  - trace.list(order_by="timestamp.desc") is what "a fixed recent batch"
    actually means here — unordered-by-default would silently drift.
  - Cached trace JSON keeps the API's native camelCase field names
    (model_dump(mode="json") serializes via alias by default for this
    generated client) rather than being renamed to snake_case — it's a
    faithful cache of what the API returned, not a second schema to keep
    in sync; ingestion/reconstruct.py reads those keys directly.

Never logs or prints LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langfuse import Langfuse

from config.credentials import ensure_env_loaded

DEFAULT_TRACES_DIR = Path(__file__).parent.parent / "data" / "traces"
DEFAULT_BATCH_SIZE = 100
DEFAULT_TIMEOUT = 30.0
DEFAULT_PAGE_LIMIT = 100  # traces per trace.list() page — Langfuse's own page-size ceiling in practice


def build_client(
    *,
    timeout: float = DEFAULT_TIMEOUT,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
) -> Langfuse:
    """Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
    from the environment (real env vars first, config-file-backed ones
    next -- see config/credentials.py's ensure_env_loaded) unless
    explicitly overridden by a caller. The override params exist so
    config/credentials.py's validate_langfuse can auth-check candidate
    values before they're written anywhere, without this function ever
    touching os.environ as a side effect of validation. Raises with a
    clear message (naming which vars are missing, never their values)
    rather than letting the SDK fail opaquely."""
    ensure_env_loaded()
    public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    host = host or os.environ.get("LANGFUSE_BASE_URL")
    missing = [
        name
        for name, value in (("LANGFUSE_PUBLIC_KEY", public_key), ("LANGFUSE_SECRET_KEY", secret_key), ("LANGFUSE_BASE_URL", host))
        if not value
    ]
    if missing:
        raise RuntimeError(f"missing required credentials: {', '.join(missing)}")
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host, timeout=timeout)


def default_project_id() -> str:
    ensure_env_loaded()
    project_id = os.environ.get("LANGFUSE_PROJECT_ID")
    if not project_id:
        raise RuntimeError("LANGFUSE_PROJECT_ID not set")
    return project_id


def _trace_cache_path(project_id: str, trace_id: str, *, traces_dir: Path) -> Path:
    return traces_dir / project_id / f"{trace_id}.json"


def pull_traces(
    *,
    project_id: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    traces_dir: Path = DEFAULT_TRACES_DIR,
    client: Langfuse | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
) -> list[str]:
    """Pulls up to batch_size most-recent traces for project_id (defaults
    to LANGFUSE_PROJECT_ID from .env), caching each trace's full body
    (all its observations, via trace.get()) to
    <traces_dir>/<project_id>/<trace_id>.json.

    Idempotent per trace: a trace already cached on disk is not re-fetched
    — re-running this with a larger batch_size only pulls the gap, same
    resumability discipline as experiments/runner.py's CacheIndex. Returns
    every trace ID now in the cache for this project (not just newly
    pulled ones), up to batch_size, newest first.
    """
    project_id = project_id or default_project_id()
    lf = client or build_client()
    cache_dir = traces_dir / project_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    trace_ids: list[str] = []
    page = 1
    while len(trace_ids) < batch_size:
        remaining = batch_size - len(trace_ids)
        limit = min(page_limit, remaining)
        response = lf.api.trace.list(page=page, limit=limit, order_by="timestamp.desc")
        if not response.data:
            break
        for summary in response.data:
            path = _trace_cache_path(project_id, summary.id, traces_dir=traces_dir)
            if not path.exists():
                full = lf.api.trace.get(summary.id)
                path.write_text(json.dumps(full.model_dump(mode="json"), indent=2), encoding="utf-8")
            trace_ids.append(summary.id)
            if len(trace_ids) >= batch_size:
                break
        if page >= response.meta.total_pages:
            break
        page += 1
    return trace_ids


def load_cached_trace_ids(project_id: str, *, traces_dir: Path = DEFAULT_TRACES_DIR) -> list[str]:
    cache_dir = traces_dir / project_id
    if not cache_dir.exists():
        return []
    return sorted(p.stem for p in cache_dir.glob("*.json"))


def load_cached_trace(project_id: str, trace_id: str, *, traces_dir: Path = DEFAULT_TRACES_DIR) -> dict:
    path = _trace_cache_path(project_id, trace_id, traces_dir=traces_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def load_cached_traces(project_id: str, *, traces_dir: Path = DEFAULT_TRACES_DIR) -> list[dict]:
    return [load_cached_trace(project_id, tid, traces_dir=traces_dir) for tid in load_cached_trace_ids(project_id, traces_dir=traces_dir)]
