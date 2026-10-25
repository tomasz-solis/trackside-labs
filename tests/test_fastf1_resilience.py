from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import src.utils.fastf1_resilience as resilience


@pytest.fixture(autouse=True)
def _reset_circuit_state():
    resilience._circuit_state.clear()


def test_call_with_resilience_succeeds_without_retry():
    value = resilience.call_with_resilience("unit_op", lambda: 42)
    assert value == 42


def test_call_with_resilience_retries_and_recovers(patcher):
    calls = {"count": 0}

    def _flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("temporary FastF1 error")
        return "ok"

    patcher.setattr(resilience.time, "sleep", lambda _seconds: None)
    value = resilience.call_with_resilience(
        "retry_op",
        _flaky,
        policy=resilience.FastF1ResiliencePolicy(max_attempts=3, timeout_budget_seconds=5.0),
    )

    assert value == "ok"
    assert calls["count"] == 2


def test_call_with_resilience_opens_circuit_after_threshold(patcher):
    patcher.setattr(resilience.time, "sleep", lambda _seconds: None)
    policy = resilience.FastF1ResiliencePolicy(
        max_attempts=1,
        timeout_budget_seconds=5.0,
        circuit_breaker_failure_threshold=1,
        circuit_breaker_cooldown_seconds=60.0,
    )

    with pytest.raises(resilience.FastF1RetryExhaustedError):
        resilience.call_with_resilience(
            "circuit_op", lambda: (_ for _ in ()).throw(RuntimeError("boom")), policy=policy
        )

    with pytest.raises(resilience.FastF1CircuitOpenError):
        resilience.call_with_resilience("circuit_op", lambda: "should not run", policy=policy)


_PATH = "/static/2026/2026-10-11_Singapore_Grand_Prix/2026-10-10_Qualifying/"
_F1 = "https://livetiming.formula1.com" + _PATH
_MIRROR = "https://livetiming-mirror.fastf1.dev" + _PATH


@pytest.fixture
def archive(monkeypatch):
    """Install the archive guard over a fake network that answers from ``statuses``."""
    from fastf1.req import Cache

    statuses: dict[str, int] = {}
    sent: list[str] = []

    def fake_get(cls, url, **_kwargs):
        sent.append(url)
        return SimpleNamespace(status_code=statuses.get(url, 200), content=b"")

    monkeypatch.setattr(Cache, "requests_get", classmethod(fake_get))
    monkeypatch.setattr(resilience, "_archive_guard_installed", False)
    monkeypatch.setattr(resilience, "_archive_refusals", {})
    resilience.install_archive_guard()
    return Cache, statuses, sent


def test_archive_guard_answers_a_refused_season_without_a_call(archive):
    cache, statuses, sent = archive
    statuses[_F1 + "SessionInfo.jsonStream"] = 401
    race = _F1.replace("2026-10-10_Qualifying", "2026-10-11_Race")
    season_2025 = "https://livetiming.formula1.com/static/2025/x/y/SessionInfo.jsonStream"
    jolpica = "https://api.jolpi.ca/ergast/f1/2026/17/results.json"

    assert cache.requests_get(_F1 + "SessionInfo.jsonStream").status_code == 401
    assert cache.requests_get(race + "TimingData.jsonStream").status_code == 401
    assert cache.requests_get(season_2025).status_code == 200
    assert cache.requests_get(jolpica).status_code == 200
    assert sent == [_F1 + "SessionInfo.jsonStream", season_2025, jolpica]


def test_archive_guard_skips_only_the_session_the_mirror_lacks(archive):
    cache, statuses, sent = archive
    race = _MIRROR.replace("2026-10-10_Qualifying", "2026-10-11_Race")
    statuses[_MIRROR + "SessionInfo.jsonStream"] = 404
    statuses[race + "LapCount.jsonStream"] = 404

    assert cache.requests_get(_MIRROR + "SessionInfo.jsonStream").status_code == 404
    assert cache.requests_get(_MIRROR + "TimingData.jsonStream").status_code == 404
    # A missing page other than SessionInfo says nothing about the rest of the session.
    assert cache.requests_get(race + "LapCount.jsonStream").status_code == 404
    assert cache.requests_get(race + "TimingData.jsonStream").status_code == 200
    assert sent == [
        _MIRROR + "SessionInfo.jsonStream",
        race + "LapCount.jsonStream",
        race + "TimingData.jsonStream",
    ]


def test_archive_guard_skips_only_an_unpublished_session_when_signed_in(archive):
    # With an F1 TV token the F1 host answers 403 for a session it has not published.
    cache, statuses, sent = archive
    race = _F1.replace("2026-10-10_Qualifying", "2026-10-11_Race")
    statuses[race + "SessionInfo.jsonStream"] = 403

    assert cache.requests_get(race + "SessionInfo.jsonStream").status_code == 403
    assert cache.requests_get(race + "TimingData.jsonStream").status_code == 403
    assert cache.requests_get(_F1 + "SessionInfo.jsonStream").status_code == 200
    assert sent == [race + "SessionInfo.jsonStream", _F1 + "SessionInfo.jsonStream"]


def test_archive_guard_asks_again_after_ten_minutes(archive, monkeypatch):
    cache, statuses, sent = archive
    clock = {"now": 1000.0}
    monkeypatch.setattr(resilience.time, "monotonic", lambda: clock["now"])
    statuses[_F1 + "SessionInfo.jsonStream"] = 401

    cache.requests_get(_F1 + "SessionInfo.jsonStream")
    clock["now"] += 599
    cache.requests_get(_F1 + "SessionInfo.jsonStream")
    assert len(sent) == 1
    clock["now"] += 2
    statuses.clear()
    assert cache.requests_get(_F1 + "SessionInfo.jsonStream").status_code == 200
    assert len(sent) == 2


def test_fastf1_fetch_page_fails_as_before_and_asks_once(archive):
    from fastf1 import _api

    _cache, statuses, sent = archive
    statuses[_F1 + "SessionInfo.jsonStream"] = 401
    statuses[_MIRROR + "SessionInfo.jsonStream"] = 404

    assert _api.fetch_page(_PATH, "session_info") is None
    assert _api.fetch_page(_PATH, "timing_data") is None
    assert sent == [_F1 + "SessionInfo.jsonStream", _MIRROR + "SessionInfo.jsonStream"]


def test_archive_guard_takes_url_pairs_and_stops_at_the_first_answer(archive):
    # FastF1 3.9 passes the F1 host and the mirror together.
    cache, statuses, sent = archive
    pair = (_F1 + "SessionInfo.jsonStream", _MIRROR + "SessionInfo.jsonStream")

    assert cache.requests_get(pair).status_code == 200
    assert sent == [_F1 + "SessionInfo.jsonStream"]


def test_install_archive_guard_wraps_once(archive):
    cache, _statuses, sent = archive
    guard = cache.__dict__["requests_get"]
    resilience.install_archive_guard()
    assert cache.__dict__["requests_get"] is guard
    cache.requests_get(_F1 + "SessionInfo.jsonStream")
    assert len(sent) == 1


@pytest.fixture
def f1tv(monkeypatch, tmp_path):
    """Point FastF1 at a temporary token file and Supabase at an in-memory store."""
    import fastf1.internals.f1account as f1account
    from fastf1.internals.f1account.tokens import TokenManager

    import src.persistence.runtime_state_store as runtime_state_store

    rows: dict[tuple[str, str], dict] = {}

    class FakeStore:
        def get_record(self, namespace, state_key):
            return rows.get((namespace, state_key))

        def upsert_record(self, namespace, state_key, payload):
            rows[(namespace, state_key)] = payload

    manager = TokenManager(auth_func=lambda: None, path=tmp_path / "f1auth_token.json")
    monkeypatch.setattr(f1account, "token_manager", manager)
    monkeypatch.setattr(runtime_state_store, "RuntimeStateStore", FakeStore)
    monkeypatch.setenv("F1_USERNAME", "fan@example.com")
    return manager, rows


def test_saved_f1tv_token_is_reused_and_a_new_one_saved(f1tv):
    from fastf1.internals.f1account import SubscriptionToken

    manager, rows = f1tv
    saved = {"value": "saved", "expires_at": time.time() + 3600}
    rows[resilience._F1TV_TOKEN_STATE] = saved

    restored = resilience.restore_f1tv_token()
    assert restored == "saved"
    assert manager.get_stored().value == "saved"
    resilience.save_f1tv_token(restored)
    assert rows[resilience._F1TV_TOKEN_STATE] is saved  # unchanged token, no write

    # A rejected token makes FastF1 sign in again; the new one is saved.
    manager._token = SubscriptionToken(value="new", expires_at=time.time() + 3600)
    resilience.save_f1tv_token(restored)
    assert rows[resilience._F1TV_TOKEN_STATE]["value"] == "new"


def test_expired_f1tv_token_is_not_restored(f1tv):
    manager, rows = f1tv
    rows[resilience._F1TV_TOKEN_STATE] = {"value": "old", "expires_at": time.time() + 60}

    assert resilience.restore_f1tv_token() is None
    assert manager.get_stored() is None


def test_f1tv_token_is_left_alone_without_a_login(f1tv, monkeypatch):
    manager, rows = f1tv
    monkeypatch.delenv("F1_USERNAME")
    rows[resilience._F1TV_TOKEN_STATE] = {"value": "saved", "expires_at": time.time() + 3600}

    assert resilience.restore_f1tv_token() is None
    assert manager.get_stored() is None
