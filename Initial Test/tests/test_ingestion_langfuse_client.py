"""No live API calls here — a stub client matching the two lf.api.trace
methods this module actually uses (list, get), shaped like the real
responses confirmed in the Part 1 investigation."""

from dataclasses import dataclass, field

from ingestion.langfuse_client import load_cached_trace, load_cached_trace_ids, load_cached_traces, pull_traces


@dataclass
class _StubSummary:
    id: str


@dataclass
class _StubMeta:
    total_pages: int


@dataclass
class _StubListResponse:
    data: list
    meta: _StubMeta


@dataclass
class _StubFullTrace:
    id: str
    payload: dict = field(default_factory=dict)

    def model_dump(self, mode="json"):
        return {"id": self.id, **self.payload}


class _StubTraceAPI:
    def __init__(self, trace_ids: list[str], page_limit: int = 100):
        self._trace_ids = trace_ids
        self._page_limit = page_limit
        self.get_calls: list[str] = []

    def list(self, *, page, limit, order_by=None):
        start = (page - 1) * limit
        chunk = self._trace_ids[start : start + limit]
        total_pages = max(1, -(-len(self._trace_ids) // limit))
        return _StubListResponse(data=[_StubSummary(id=tid) for tid in chunk], meta=_StubMeta(total_pages=total_pages))

    def get(self, trace_id):
        self.get_calls.append(trace_id)
        return _StubFullTrace(id=trace_id, payload={"metadata": {"agent_name": "Stub Assistant"}, "observations": []})


class _StubApi:
    def __init__(self, trace_ids: list[str]):
        self.trace = _StubTraceAPI(trace_ids)


class _StubClient:
    def __init__(self, trace_ids: list[str]):
        self.api = _StubApi(trace_ids)


def test_pull_traces_caches_each_trace_to_disk(tmp_path):
    client = _StubClient([f"trace-{i}" for i in range(5)])
    ids = pull_traces(project_id="proj-1", batch_size=5, traces_dir=tmp_path, client=client)
    assert ids == [f"trace-{i}" for i in range(5)]
    assert set(load_cached_trace_ids("proj-1", traces_dir=tmp_path)) == set(ids)
    assert client.api.trace.get_calls == ids


def test_pull_traces_respects_batch_size(tmp_path):
    client = _StubClient([f"trace-{i}" for i in range(20)])
    ids = pull_traces(project_id="proj-1", batch_size=7, traces_dir=tmp_path, client=client)
    assert len(ids) == 7
    assert len(load_cached_trace_ids("proj-1", traces_dir=tmp_path)) == 7


def test_pull_traces_is_idempotent_per_trace(tmp_path):
    client = _StubClient([f"trace-{i}" for i in range(5)])
    pull_traces(project_id="proj-1", batch_size=5, traces_dir=tmp_path, client=client)
    first_get_calls = list(client.api.trace.get_calls)
    # re-running with the same batch size should not re-fetch anything already cached
    pull_traces(project_id="proj-1", batch_size=5, traces_dir=tmp_path, client=client)
    assert client.api.trace.get_calls == first_get_calls


def test_pull_traces_extending_batch_size_only_pulls_the_gap(tmp_path):
    client = _StubClient([f"trace-{i}" for i in range(20)])
    pull_traces(project_id="proj-1", batch_size=5, traces_dir=tmp_path, client=client)
    assert len(client.api.trace.get_calls) == 5
    pull_traces(project_id="proj-1", batch_size=10, traces_dir=tmp_path, client=client)
    assert len(client.api.trace.get_calls) == 10  # 5 old + 5 new, not 15
    assert len(load_cached_trace_ids("proj-1", traces_dir=tmp_path)) == 10


def test_pull_traces_stops_when_project_has_fewer_traces_than_batch_size(tmp_path):
    client = _StubClient([f"trace-{i}" for i in range(3)])
    ids = pull_traces(project_id="proj-1", batch_size=100, traces_dir=tmp_path, client=client)
    assert len(ids) == 3


def test_load_cached_trace_ids_empty_when_nothing_pulled(tmp_path):
    assert load_cached_trace_ids("nope", traces_dir=tmp_path) == []


def test_load_cached_trace_roundtrip(tmp_path):
    client = _StubClient(["trace-x"])
    pull_traces(project_id="proj-1", batch_size=1, traces_dir=tmp_path, client=client)
    trace = load_cached_trace("proj-1", "trace-x", traces_dir=tmp_path)
    assert trace["id"] == "trace-x"
    assert trace["metadata"]["agent_name"] == "Stub Assistant"


def test_load_cached_traces_returns_every_cached_trace(tmp_path):
    client = _StubClient([f"trace-{i}" for i in range(4)])
    pull_traces(project_id="proj-1", batch_size=4, traces_dir=tmp_path, client=client)
    traces = load_cached_traces("proj-1", traces_dir=tmp_path)
    assert len(traces) == 4
    assert {t["id"] for t in traces} == {f"trace-{i}" for i in range(4)}
