"""Tests for the OpenF1 fallback used while FastF1's archive is down."""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.data import openf1
from src.data.actual_results_fetcher import (
    fetch_actual_session_results,
    fetch_actual_starting_grid,
)
from src.utils.session_detector import SessionDetector

_MEETINGS = [
    {
        "meeting_key": 1,
        "meeting_name": "Bahrain Grand Prix",
        "date_start": "2026-04-10",
        "is_cancelled": True,
    },
    {
        "meeting_key": 2,
        "meeting_name": "Bahrain Grand Prix",
        "date_start": "2026-10-02",
        "is_cancelled": False,
    },
    {
        "meeting_key": 3,
        "meeting_name": "São Paulo Grand Prix",
        "date_start": "2026-11-06",
        "is_cancelled": False,
    },
]
_SESSIONS = {
    2: [
        {"session_key": 20, "session_name": "Practice 1", "date_end": "2026-10-02T12:30:00+00:00"},
        {"session_key": 21, "session_name": "Qualifying", "date_end": "2026-10-03T17:00:00+00:00"},
        {"session_key": 22, "session_name": "Race", "date_end": "2026-10-04T17:00:00+00:00"},
    ],
    3: [{"session_key": 30, "session_name": "Race", "date_end": "2026-11-08T19:00:00+00:00"}],
}
_TEAMS = ["McLaren", "Ferrari", "Red Bull Racing", "Mercedes", "Williams", "Haas F1 Team"]
_DRIVERS = [
    {"driver_number": n, "name_acronym": f"D{n:02d}", "team_name": _TEAMS[n % len(_TEAMS)]}
    for n in range(1, 23)
]


def _race_results() -> list[dict]:
    """20 classified finishers and two retirements, the later retirement first."""
    rows = [
        {
            "driver_number": n,
            "position": n,
            "number_of_laps": 57,
            "dnf": False,
            "dns": False,
            "dsq": False,
        }
        for n in range(1, 21)
    ]
    # Retired late but classified: OpenF1 says dnf, the scoring rule says finisher.
    rows[-1]["dnf"] = True
    rows.append({"driver_number": 21, "position": None, "number_of_laps": 3, "dnf": True})
    rows.append({"driver_number": 22, "position": None, "number_of_laps": 40, "dnf": True})
    return rows


def _fake_get(results_by_session: dict[int, list[dict]], grid: list[dict] | None = None):
    def fake(endpoint, **params):
        if endpoint == "meetings":
            return _MEETINGS
        if endpoint == "sessions":
            return _SESSIONS.get(params["meeting_key"], [])
        if endpoint == "drivers":
            return _DRIVERS
        if endpoint == "session_result":
            return results_by_session.get(params["session_key"], [])
        if endpoint == "starting_grid":
            return grid or []
        raise AssertionError(endpoint)

    return fake


def test_disabled_env_blocks_network():
    with pytest.raises(openf1.OpenF1Unavailable):
        openf1._get("meetings", year=2026)


def test_find_session_skips_cancelled_meeting_and_ignores_accents():
    with patch.object(openf1, "_get", _fake_get({})):
        assert openf1.find_session(2026, "Bahrain Grand Prix", "R")["session_key"] == 22
        assert openf1.find_session(2026, "Sao Paulo Grand Prix", "R")["session_key"] == 30
        assert openf1.find_session(2026, "Bahrain Grand Prix", "Sprint") is None


def test_race_results_order_retirements_by_laps_and_flag_dnf():
    with patch.object(openf1, "_get", _fake_get({22: _race_results()})):
        rows = openf1.fetch_session_results(2026, "Bahrain Grand Prix", "R")
    assert [row["position"] for row in rows] == list(range(1, 23))
    assert [row["driver"] for row in rows[-2:]] == ["D22", "D21"]
    assert [row["dnf"] for row in rows[-3:]] == [False, True, True]
    assert {row["source"] for row in rows} == {"openf1"}
    assert rows[0]["team"] == "Ferrari"


def test_qualifying_drivers_without_a_time_go_last_by_car_number():
    quali = [{"driver_number": n, "position": n, "number_of_laps": 12} for n in range(1, 21)]
    with patch.object(openf1, "_get", _fake_get({21: quali})):
        rows = openf1.fetch_session_results(2026, "Bahrain Grand Prix", "Q")
    assert [(row["driver"], row["position"]) for row in rows[-2:]] == [("D21", 21), ("D22", 22)]


def test_rate_limited_request_is_retried(monkeypatch):
    import urllib.error

    monkeypatch.delenv(openf1.DISABLE_ENV)
    monkeypatch.setattr(openf1, "_RATE_LIMIT_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(openf1, "_MIN_INTERVAL_SECONDS", 0.0)
    calls = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'[{"meeting_key": 1}]'

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", None, None)
        return _Response()

    monkeypatch.setattr(openf1.urllib.request, "urlopen", fake_urlopen)
    assert openf1._get("meetings", year=2026) == [{"meeting_key": 1}]
    assert len(calls) == 2


class _Body:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def test_live_session_401_closes_openf1_until_the_lock_expires(monkeypatch):
    import io
    import urllib.error

    monkeypatch.delenv(openf1.DISABLE_ENV)
    monkeypatch.setattr(openf1, "_MIN_INTERVAL_SECONDS", 0.0)
    calls = []
    body = b'{"detail":"Live F1 session in progress. Global API access is restricted"}'

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", None, io.BytesIO(body))

    monkeypatch.setattr(openf1.urllib.request, "urlopen", fake_urlopen)
    for _ in range(2):
        with pytest.raises(openf1.OpenF1Unavailable, match="live session"):
            openf1._get("meetings", year=2026)
    assert len(calls) == 1

    monkeypatch.setattr(openf1, "_locked_until", 0.0)
    monkeypatch.setattr(openf1.urllib.request, "urlopen", lambda request, timeout: _Body(b"[]"))
    assert openf1._get("meetings", year=2026) == []


def test_cached_responses_expire(monkeypatch):
    monkeypatch.delenv(openf1.DISABLE_ENV)
    monkeypatch.setattr(openf1, "_MIN_INTERVAL_SECONDS", 0.0)
    bodies = iter([b"[]", b'[{"position": 1}]'])
    monkeypatch.setattr(
        openf1.urllib.request, "urlopen", lambda request, timeout: _Body(next(bodies))
    )
    assert openf1._get("session_result", session_key=1) == []
    assert openf1._get("session_result", session_key=1) == []
    monkeypatch.setattr(openf1, "_CACHE_TTL_SECONDS", 0.0)
    assert openf1._get("session_result", session_key=1) == [{"position": 1}]


def test_qualifying_results_have_no_dnf_flag():
    quali = [{"driver_number": n, "position": n, "number_of_laps": 12} for n in range(1, 22)]
    quali.append({"driver_number": 22, "position": "RT", "number_of_laps": 2})
    with patch.object(openf1, "_get", _fake_get({21: quali})):
        rows = openf1.fetch_session_results(2026, "Bahrain Grand Prix", "Q")
    assert rows[-1] == {"driver": "D22", "team": "Williams", "position": 22, "source": "openf1"}


@pytest.mark.parametrize(
    ("now", "results", "expected"),
    [
        (datetime(2026, 10, 4, 16, 0, tzinfo=UTC), [], "incomplete"),
        (datetime(2026, 10, 4, 18, 0, tzinfo=UTC), _race_results(), "completed"),
        (datetime(2026, 10, 4, 18, 0, tzinfo=UTC), [], "incomplete"),
        (datetime(2026, 10, 5, 12, 0, tzinfo=UTC), [], "unknown"),
    ],
)
def test_completion_state_follows_published_results(now, results, expected):
    with patch.object(openf1, "_get", _fake_get({22: results})):
        assert openf1.session_completion_state(2026, "Bahrain Grand Prix", "R", now=now) == expected


def test_practice_is_complete_after_its_scheduled_end():
    now = datetime(2026, 10, 2, 13, 0, tzinfo=UTC)
    with patch.object(openf1, "_get", _fake_get({})):
        assert (
            openf1.session_completion_state(2026, "Bahrain Grand Prix", "FP1", now=now)
            == "completed"
        )


def test_fetcher_falls_back_to_openf1_when_fastf1_fails():
    with (
        patch(
            "src.data.actual_results_fetcher.fastf1.get_session", side_effect=RuntimeError("401")
        ),
        patch.object(openf1, "_get", _fake_get({22: _race_results()})),
    ):
        results = fetch_actual_session_results(2026, "Bahrain Grand Prix", "R")
    assert results is not None and len(results) == 22
    assert results[-1]["dnf"] is True and results[0]["source"] == "openf1"


def test_fetcher_returns_none_when_both_sources_fail():
    with patch(
        "src.data.actual_results_fetcher.fastf1.get_session", side_effect=RuntimeError("401")
    ):
        assert fetch_actual_session_results(2026, "Bahrain Grand Prix", "R") is None


def test_starting_grid_falls_back_to_openf1():
    grid = [{"driver_number": n, "position": 23 - n} for n in range(1, 23)]
    with (
        patch(
            "src.data.actual_results_fetcher.fastf1.get_session", side_effect=RuntimeError("401")
        ),
        patch.object(openf1, "_get", _fake_get({}, grid=grid)),
    ):
        rows = fetch_actual_starting_grid(2026, "Bahrain Grand Prix")
    assert rows is not None and rows[0]["driver"] == "D22"
    assert {row["start_type"] for row in rows} == {"grid"}


def test_detector_uses_openf1_when_fastf1_cannot_decide():
    detector = SessionDetector()
    with (
        patch.object(
            SessionDetector, "_compute_session_completion_state", side_effect=RuntimeError("401")
        ),
        patch("src.data.openf1.session_completion_state", return_value="completed") as fallback,
    ):
        assert detector.get_session_completion_state(2026, "Bahrain Grand Prix", "R") == "completed"
    fallback.assert_called_once_with(2026, "Bahrain Grand Prix", "R")


def test_detector_stays_unknown_when_openf1_is_off():
    detector = SessionDetector()
    with patch.object(
        SessionDetector, "_compute_session_completion_state", side_effect=RuntimeError("401")
    ):
        assert detector.get_session_completion_state(2026, "Bahrain Grand Prix", "R") == "unknown"
