"""The Render API calls report failure instead of raising, so the panel always renders."""

from typing import Any

import pytest

from src.dashboard import render_ops

_CONFIGURED = {
    "RENDER_API_KEY": "rnd_key",
    "RENDER_PRECOMPUTE_CRON_ID": "crn-abc",
    "RENDER_WEB_SERVICE_ID": "srv-xyz",
}


class _Response:
    def __init__(self, status_code: int, text: str = "", payload: Any = None):
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


class _Requests:
    """Stub for the requests module, recording the single call it expects."""

    def __init__(self, response: Any = None, error: Exception | None = None):
        self._response = response or _Response(200)
        self._error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append((url, kwargs))
        if self._error is not None:
            raise self._error
        return self._response

    def get(self, url: str, **kwargs: Any) -> Any:
        return self.post(url, **kwargs)


def _patch_requests(monkeypatch: pytest.MonkeyPatch, stub: _Requests) -> _Requests:
    monkeypatch.setattr(render_ops, "_requests_module", lambda: stub)
    return stub


def test_nothing_configured_is_reported_not_raised():
    assert render_ops.render_ops_configured({}) is False
    assert render_ops.missing_settings({}) == [
        "RENDER_API_KEY",
        "RENDER_PRECOMPUTE_CRON_ID",
        "RENDER_WEB_SERVICE_ID",
    ]


def test_a_missing_variable_makes_no_http_call(monkeypatch: pytest.MonkeyPatch):
    stub = _patch_requests(monkeypatch, _Requests())

    succeeded, message = render_ops.trigger_precompute_run({"RENDER_API_KEY": "rnd_key"})

    assert succeeded is False
    assert "RENDER_PRECOMPUTE_CRON_ID" in message
    assert stub.calls == []


def test_triggering_the_cron_posts_to_its_runs_endpoint(monkeypatch: pytest.MonkeyPatch):
    stub = _patch_requests(monkeypatch, _Requests(_Response(200)))

    succeeded, _ = render_ops.trigger_precompute_run(_CONFIGURED)

    assert succeeded is True
    url, kwargs = stub.calls[0]
    assert url == "https://api.render.com/v1/cron-jobs/crn-abc/runs"
    assert kwargs["headers"]["Authorization"] == "Bearer rnd_key"
    assert kwargs["timeout"] == 15


def test_restarting_posts_to_the_service_restart_endpoint(monkeypatch: pytest.MonkeyPatch):
    stub = _patch_requests(monkeypatch, _Requests(_Response(200)))

    succeeded, _ = render_ops.restart_web_service(_CONFIGURED)

    assert succeeded is True
    assert stub.calls[0][0] == "https://api.render.com/v1/services/srv-xyz/restart"


def test_the_cron_id_is_never_used_as_a_service_id(monkeypatch: pytest.MonkeyPatch):
    stub = _patch_requests(monkeypatch, _Requests(_Response(200)))

    render_ops.restart_web_service(_CONFIGURED)

    assert "crn-abc" not in stub.calls[0][0]


@pytest.mark.parametrize("status_code", [401, 404, 500])
def test_a_rejected_call_is_reported_with_its_status(
    monkeypatch: pytest.MonkeyPatch, status_code: int
):
    _patch_requests(monkeypatch, _Requests(_Response(status_code, "no")))

    succeeded, message = render_ops.trigger_precompute_run(_CONFIGURED)

    assert succeeded is False
    assert str(status_code) in message


def test_an_unreachable_api_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch):
    _patch_requests(monkeypatch, _Requests(error=OSError("connection reset")))

    succeeded, message = render_ops.restart_web_service(_CONFIGURED)

    assert succeeded is False
    assert "connection reset" in message


def test_cron_events_are_read_from_the_service_event_feed(monkeypatch: pytest.MonkeyPatch):
    payload = [
        {
            "event": {
                "timestamp": "2026-09-02T19:42:07.123Z",
                "type": "cron_job_run_ended",
                "details": {"status": "successful", "reason": {"oomKilled": False}},
            }
        }
    ]
    stub = _patch_requests(monkeypatch, _Requests(_Response(200, payload=payload)))

    rows, error = render_ops.precompute_run_events(_CONFIGURED)

    assert error == ""
    assert rows == [
        {
            "timestamp": "2026-09-02 19:42",
            "type": "cron job run ended",
            "outcome": "successful",
        }
    ]
    url, kwargs = stub.calls[0]
    assert url == "https://api.render.com/v1/services/crn-abc/events"
    assert kwargs["params"]["type"] == ["cron_job_run_started", "cron_job_run_ended"]


def test_a_failed_run_reports_why_it_died(monkeypatch: pytest.MonkeyPatch):
    payload = [
        {
            "event": {
                "timestamp": "2026-09-02T19:42:07Z",
                "type": "cron_job_run_ended",
                "details": {
                    "status": "unsuccessful",
                    "reason": {"oomKilled": True, "nonZeroExit": False},
                },
            }
        }
    ]
    _patch_requests(monkeypatch, _Requests(_Response(200, payload=payload)))

    rows, _ = render_ops.precompute_run_events(_CONFIGURED)

    assert rows[0]["outcome"] == "unsuccessful, oomKilled"


def test_web_service_events_use_the_web_service_id(monkeypatch: pytest.MonkeyPatch):
    stub = _patch_requests(monkeypatch, _Requests(_Response(200, payload=[])))

    rows, error = render_ops.web_service_events(_CONFIGURED)

    assert (rows, error) == ([], "")
    assert stub.calls[0][0] == "https://api.render.com/v1/services/srv-xyz/events"


def test_an_unreadable_event_body_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch):
    _patch_requests(monkeypatch, _Requests(_Response(200)))

    rows, error = render_ops.precompute_run_events(_CONFIGURED)

    assert rows == []
    assert "could not read" in error


def test_events_need_the_same_configuration_as_the_buttons(monkeypatch: pytest.MonkeyPatch):
    stub = _patch_requests(monkeypatch, _Requests(_Response(200, payload=[])))

    rows, error = render_ops.precompute_run_events({})

    assert rows == []
    assert "RENDER_API_KEY" in error
    assert stub.calls == []


class _Lock:
    """Stand-in runtime store: the first caller per key gets the lock."""

    def __init__(self):
        self.keys: list[str] = []

    def acquire_lock(self, lock_key: str, owner_id: str, ttl_seconds: int = 900) -> bool:
        del owner_id, ttl_seconds
        acquired = lock_key not in self.keys
        self.keys.append(lock_key)
        return acquired


def test_auto_trigger_runs_once_per_hash_on_render(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(render_ops, "precompute_run_events", lambda env=None, limit=1: ([], ""))
    stub = _patch_requests(monkeypatch, _Requests(_Response(200)))
    store = _Lock()
    env = {**_CONFIGURED, "RENDER": "true"}

    first = render_ops.trigger_precompute_once("hash_a", env, store=store)
    again = render_ops.trigger_precompute_once("hash_a", env, store=store)
    other = render_ops.trigger_precompute_once("hash_b", env, store=store)

    assert (first, again, other) == (True, False, True)
    assert [url for url, _ in stub.calls] == [
        "https://api.render.com/v1/cron-jobs/crn-abc/runs",
        "https://api.render.com/v1/cron-jobs/crn-abc/runs",
    ]


def test_auto_trigger_never_runs_off_render(monkeypatch: pytest.MonkeyPatch):
    """A local dashboard pointed at production must not start production runs."""
    stub = _patch_requests(monkeypatch, _Requests(_Response(200)))
    store = _Lock()

    assert render_ops.trigger_precompute_once("hash_a", _CONFIGURED, store=store) is False
    assert stub.calls == []
    assert store.keys == []


def test_auto_trigger_survives_a_lock_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(render_ops, "precompute_run_events", lambda env=None, limit=1: ([], ""))
    stub = _patch_requests(monkeypatch, _Requests(_Response(200)))

    class _Broken:
        def acquire_lock(self, *args: Any, **kwargs: Any) -> bool:
            raise RuntimeError("db down")

    env = {**_CONFIGURED, "RENDER": "true"}
    assert render_ops.trigger_precompute_once("hash_a", env, store=_Broken()) is False
    assert stub.calls == []


def test_auto_trigger_skips_while_a_run_is_in_flight(monkeypatch: pytest.MonkeyPatch):
    """Render cancels an in-flight run on a new trigger; that run builds this hash anyway."""
    from datetime import UTC, datetime, timedelta

    stub = _patch_requests(monkeypatch, _Requests(_Response(200)))
    store = _Lock()
    env = {**_CONFIGURED, "RENDER": "true"}

    def _events_started(minutes_ago: int):
        stamp = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M")
        return lambda env=None, limit=1: (
            [{"timestamp": stamp, "type": "cron job run started", "outcome": ""}],
            "",
        )

    monkeypatch.setattr(render_ops, "precompute_run_events", _events_started(3))
    assert render_ops.trigger_precompute_once("hash_a", env, store=store) is False
    assert stub.calls == []
    # Not locked, so the hash can still trigger once the run ends.
    assert store.keys == []

    # A start with no end for hours is treated as lost.
    monkeypatch.setattr(render_ops, "precompute_run_events", _events_started(300))
    assert render_ops.trigger_precompute_once("hash_a", env, store=store) is True
