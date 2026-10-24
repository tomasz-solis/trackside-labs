"""Tests for filling a FastF1 session from OpenF1 laps, stints, weather and car data."""

from types import SimpleNamespace
from unittest.mock import patch

import fastf1
import pandas as pd
import pytest
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError

from src.data import openf1, openf1_session
from src.systems import testing_updater_sessions
from src.utils import fp_blending

_T0 = "2026-10-09T08:30:00+00:00"
_ROW = {"session_key": 9, "date_start": _T0, "date_end": "2026-10-09T09:30:00+00:00"}


def _session() -> fastf1.core.Session:
    """A FastF1 practice session built offline."""
    event = {
        "RoundNumber": 18,
        "Country": "Singapore",
        "Location": "Marina Bay",
        "EventName": "Singapore Grand Prix",
        "EventDate": pd.Timestamp("2026-10-11"),
        "EventFormat": "conventional",
        "F1ApiSupport": True,
    }
    for n in range(1, 6):
        event[f"Session{n}"] = "Practice 1" if n == 1 else ""
        event[f"Session{n}Date"] = pd.Timestamp("2026-10-09 16:30", tz="Asia/Singapore")
        event[f"Session{n}DateUtc"] = pd.Timestamp("2026-10-09 08:30")
    return fastf1.core.Session(
        fastf1.events.Event(event, year=2026), "Practice 1", f1_api_support=True
    )


def _lap(n: int, start_s: float, duration: float | None, *, pit_out: bool = False) -> dict:
    start = pd.Timestamp(_T0) + pd.to_timedelta(start_s, unit="s")
    return {
        "driver_number": 1,
        "lap_number": n,
        "date_start": start.isoformat(),
        "lap_duration": duration,
        "duration_sector_1": 30.0 if duration else None,
        "i1_speed": 250,
        "i2_speed": 260,
        "st_speed": 300,
        "is_pit_out_lap": pit_out,
    }


# Out-lap, two timed laps, in-lap, out-lap on the refitted set, a timed lap, last lap.
_LAPS = [
    _lap(1, 0, 120.0, pit_out=True),
    _lap(2, 120, 95.0),
    _lap(3, 215, 94.0),
    _lap(4, 309, 140.0),
    _lap(5, 900, 200.0, pit_out=True),
    _lap(6, 1100, 93.5),
    _lap(7, 1193.5, 130.0),
]
_STINTS = [
    {
        "driver_number": 1,
        "stint_number": 1,
        "lap_start": 1,
        "lap_end": 4,
        "compound": "SOFT",
        "tyre_age_at_start": 0,
    },
    {
        "driver_number": 1,
        "stint_number": 2,
        "lap_start": 5,
        "lap_end": 7,
        "compound": "SOFT",
        "tyre_age_at_start": 0,
    },
]
_DRIVERS = {1: ("NOR", "McLaren")}


def _frame(**kwargs) -> pd.DataFrame:
    return openf1_session.build_laps_frame(
        _LAPS, _STINTS, _DRIVERS, pd.Timestamp("2026-10-09 08:30"), **kwargs
    )


def test_out_laps_have_no_time_and_the_lap_before_is_an_in_lap():
    laps = _frame().set_index("LapNumber")
    assert laps.loc[1, "LapTime"] is pd.NaT and laps.loc[1, "PitOutTime"] == pd.Timedelta(0)
    assert laps.loc[4, "PitInTime"] == pd.to_timedelta(449, unit="s")
    assert laps.loc[[2, 3, 6], "IsAccurate"].all()
    assert not laps.loc[[1, 4, 5], "IsAccurate"].any()


def test_last_lap_is_an_in_lap_only_when_asked():
    assert _frame().set_index("LapNumber").loc[7, "IsAccurate"]
    practice = _frame(last_lap_is_in_lap=True).set_index("LapNumber")
    assert pd.notna(practice.loc[7, "PitInTime"]) and not practice.loc[7, "IsAccurate"]


def test_refitted_set_of_the_same_compound_keeps_its_age():
    laps = _frame().set_index("LapNumber")
    assert laps.loc[[1, 4, 5, 7], "TyreLife"].tolist() == [1.0, 4.0, 5.0, 7.0]
    assert laps.loc[5, "Stint"] == 2.0 and laps.loc[5, "Compound"] == "SOFT"


def test_a_new_compound_or_a_reported_age_is_kept():
    stints = [
        {
            "stint_number": 1,
            "lap_start": 1,
            "lap_end": 4,
            "compound": "SOFT",
            "tyre_age_at_start": 0,
        },
        {
            "stint_number": 2,
            "lap_start": 5,
            "lap_end": 9,
            "compound": "MEDIUM",
            "tyre_age_at_start": 0,
        },
        {
            "stint_number": 3,
            "lap_start": 10,
            "lap_end": 12,
            "compound": "MEDIUM",
            "tyre_age_at_start": 2,
        },
    ]
    assert [s["tyre_age_at_start"] for s in openf1_session._carry_tyre_age(stints)] == [0, 0, 2]


def test_personal_best_runs_over_clean_laps():
    laps = _frame().set_index("LapNumber")
    assert laps.loc[laps["IsPersonalBest"]].index.tolist() == [2, 3, 6]


def _iso(seconds: float) -> str:
    return (pd.Timestamp(_T0) + pd.to_timedelta(seconds, unit="s")).isoformat()


def _fake_get(endpoint, *, cache=True, **params):
    rows = {
        "laps": _LAPS,
        "session_result": [{"driver_number": 1, "number_of_laps": 7}],
        "stints": _STINTS,
        "weather": [{"date": _T0, "rainfall": 1, "air_temperature": 30.0}],
        "race_control": [
            {
                "date": _iso(100),
                "category": "Flag",
                "flag": "YELLOW",
                "scope": "Sector",
                "sector": 4,
            },
            {
                "date": _iso(130),
                "category": "Flag",
                "flag": "CLEAR",
                "scope": "Sector",
                "sector": 4,
            },
        ],
        "car_data": [
            {"date": _iso(s), "brake": b, "speed": 200, "throttle": 100, "rpm": 11000, "n_gear": 7}
            for s, b in [(125, 0), (150, 100), (200, 0), (300, 100)]
        ],
    }
    return rows[endpoint]


def test_load_session_fills_laps_weather_and_car_data():
    session = _session()
    with (
        patch.object(openf1, "find_session", return_value=_ROW),
        patch.object(openf1, "_driver_table", return_value=_DRIVERS),
        patch.object(openf1, "_get", _fake_get),
    ):
        openf1_session.load_session(session, 2026, "Singapore Grand Prix", "FP1", car_data=True)

    assert len(session.laps) == len(_LAPS) and set(session.laps["Source"]) == {"openf1"}
    assert session.weather_data["Rainfall"].tolist() == [True]
    # Lap 1 runs 0 to 120 s, through the sector 4 yellow from 100 s.
    assert session.laps["TrackStatus"].tolist()[:3] == ["12", "21", "1"]
    lap_2 = session.laps.pick_laps(2).iloc[0]
    car = lap_2.get_car_data()
    assert car["Brake"].tolist() == [False, True, False]


def test_load_session_raises_when_openf1_has_no_laps():
    with (
        patch.object(openf1, "find_session", return_value=_ROW),
        patch.object(openf1, "_driver_table", return_value=_DRIVERS),
        patch.object(openf1, "_get", return_value=[]),
        pytest.raises(openf1.OpenF1Unavailable),
    ):
        openf1_session.load_session(_session(), 2026, "Singapore Grand Prix", "FP1")


class _RefusedSession:
    """A FastF1 session whose archive request was refused: laps never load."""

    def load(self, **_kwargs):
        return None

    @property
    def laps(self):
        raise DataNotLoadedError("archive returned 401")


def test_fp_blending_loader_falls_back_to_openf1(monkeypatch, tmp_path):
    filled = SimpleNamespace(laps=pd.DataFrame({"LapTime": [pd.Timedelta(seconds=93)]}))
    monkeypatch.setattr(fp_blending, "_iter_fastf1_cache_dirs", lambda: (tmp_path,))
    monkeypatch.setattr(fp_blending, "_enable_fastf1_cache", lambda _dir: None)
    monkeypatch.setattr(fp_blending.ff1, "get_session", lambda *a: _RefusedSession())
    monkeypatch.setattr(fp_blending, "_load_session_from_openf1", lambda *a: filled)

    session = fp_blending._load_session_with_cache_fallback(
        year=2026, race_name="Singapore Grand Prix", session_type="FP1", weather=True
    )
    assert session is filled


def test_fp_blending_loader_reraises_when_openf1_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(fp_blending, "_iter_fastf1_cache_dirs", lambda: (tmp_path,))
    monkeypatch.setattr(fp_blending, "_enable_fastf1_cache", lambda _dir: None)
    monkeypatch.setattr(fp_blending.ff1, "get_session", lambda *a: _RefusedSession())
    monkeypatch.setattr(fp_blending.time, "sleep", lambda _s: None)

    with pytest.raises(DataNotLoadedError):
        fp_blending._load_session_with_cache_fallback(
            year=2026, race_name="Singapore Grand Prix", session_type="FP1", weather=True
        )


def test_practice_capture_loader_takes_laps_and_car_data_from_openf1():
    refused = _RefusedSession()
    calls = []
    loaded = testing_updater_sessions.load_sessions_for_event(
        2026,
        "Singapore Grand Prix",
        ["FP1"],
        fastf1_get_session=lambda *a: refused,
        load_openf1_session_fn=lambda session, *a, **kw: calls.append((session, a, kw)),
        load_car_data_only_fn=lambda _s: pytest.fail("FastF1 car data must not load"),
    )
    assert loaded == [("FP1", refused)]
    assert calls == [(refused, (2026, "Singapore Grand Prix", "FP1"), {"car_data": True})]


class _RateLimitedSession:
    """A FastF1 session whose load ran into FastF1's hourly call limit."""

    def load(self, **_kwargs):
        raise RateLimitExceededError("any API: 500 calls/h")


def test_practice_capture_loader_takes_openf1_when_fastf1_hits_its_call_limit():
    limited = _RateLimitedSession()
    calls = []
    loaded = testing_updater_sessions.load_sessions_for_event(
        2026,
        "Barcelona Grand Prix",
        ["FP1"],
        fastf1_get_session=lambda *a: limited,
        load_openf1_session_fn=lambda session, *a, **kw: calls.append((session, a, kw)),
    )
    assert loaded == [("FP1", limited)]
    assert calls == [(limited, (2026, "Barcelona Grand Prix", "FP1"), {"car_data": True})]


def test_get_session_falls_back_when_fastf1_hits_its_call_limit():
    limited = _RateLimitedSession()
    with (
        patch.object(fastf1, "get_session", return_value=limited),
        patch.object(openf1_session, "load_session") as fill,
    ):
        assert openf1_session.get_session(2026, "Barcelona Grand Prix", "R") is limited
        fill.assert_called_once_with(limited, 2026, "Barcelona Grand Prix", "R", car_data=False)


def test_practice_capture_loader_skips_session_when_openf1_fails_too():
    def _unavailable(*_a, **_kw):
        raise openf1.OpenF1Unavailable("off")

    errors: list[str] = []
    loaded = testing_updater_sessions.load_sessions_for_event(
        2026,
        "Singapore Grand Prix",
        ["FP1"],
        error_messages=errors,
        fastf1_get_session=lambda *a: _RefusedSession(),
        load_openf1_session_fn=_unavailable,
    )
    assert loaded == [] and "OpenF1Unavailable" in errors[0]


def _rc(seconds: float, message: str, **fields) -> dict:
    return {"date": _iso(seconds), "message": message, **fields}


def test_track_status_follows_race_control_like_fastf1():
    race_control = [
        _rc(
            10, "YELLOW IN TRACK SECTOR 3", category="Flag", flag="YELLOW", scope="Sector", sector=3
        ),
        _rc(20, "SAFETY CAR DEPLOYED", category="SafetyCar"),
        _rc(30, "SAFETY CAR IN THIS LAP", category="SafetyCar"),
        _rc(40, "TRACK CLEAR", category="Flag", flag="CLEAR", scope="Track"),
        _rc(50, "VSC DEPLOYED", category="SafetyCar"),
        _rc(60, "VSC ENDING", category="SafetyCar"),
        _rc(65, "TRACK CLEAR", category="Flag", flag="CLEAR", scope="Track"),
        _rc(70, "RED FLAG - RACE SUSPENDED", category="Other"),
    ]
    t0 = pd.Timestamp("2026-10-09 08:30")
    laps = pd.DataFrame(
        {
            "LapStartDate": [t0 + pd.Timedelta(seconds=s) for s in (0, 15, 45, 66)],
            "Time": [pd.Timedelta(seconds=s) for s in (5, 35, 62, 80)],
        }
    )
    status = openf1_session.lap_track_status(laps, race_control, t0)
    assert status.tolist() == ["1", "24", "167", "15"]


def test_timed_deletions_are_matched_on_lap_time_and_can_be_reinstated():
    race_control = [
        _rc(1, "CAR 1 (NOR) TIME 1:35.000 DELETED - TRACK LIMITS AT TURN 3 LAP 3 16:02:18"),
        _rc(2, "CAR 1 (NOR) LAP DELETED - TRACK LIMITS AT TURN 9 LAP 5 16:03:35 (PIT)"),
        _rc(3, "CAR 4 (PIA) TIME 1:31.451 DELETED - TRACK LIMITS AT TURN 9 LAP 9 16:18:19"),
        _rc(4, "CAR 4 (PIA) LAP 9 WILL BE REINSTATED"),
    ]
    deleted = openf1_session.deleted_laps(race_control)
    assert deleted == frozenset({(1, 95.0)})
    laps = _frame(deleted=deleted).set_index("LapNumber")
    assert laps.loc[laps["Deleted"]].index.tolist() == [2]
    assert laps.loc[laps["IsPersonalBest"]].index.tolist() == [3, 6]


def test_formation_lap_after_a_red_flag_has_no_time():
    t0 = pd.Timestamp("2026-10-09 08:30")
    laps = pd.DataFrame(
        {
            "LapStartDate": [t0 + pd.Timedelta(seconds=s) for s in (0, 100)],
            "Time": [pd.Timedelta(seconds=s) for s in (100, 190)],
        }
    )
    race_control = [_rc(120, "EXTRA FORMATION LAP"), _rc(10, "RACE START")]
    assert openf1_session.untimed_restart_laps(laps, race_control, t0).tolist() == [False, True]


def test_get_session_keeps_fastf1_laps_and_falls_back_when_refused():
    class _Loaded:
        laps = pd.DataFrame({"LapTime": [pd.Timedelta(seconds=90)]})

        def load(self, **_kwargs):
            return None

    loaded = _Loaded()
    with (
        patch.object(fastf1, "get_session", return_value=loaded),
        patch.object(openf1_session, "load_session") as fill,
    ):
        assert openf1_session.get_session(2026, "Singapore Grand Prix", "Q") is loaded
        fill.assert_not_called()

    refused = _RefusedSession()
    with (
        patch.object(fastf1, "get_session", return_value=refused),
        patch.object(openf1_session, "load_session") as fill,
    ):
        assert (
            openf1_session.get_session(2026, "Singapore Grand Prix", "Q", telemetry=True) is refused
        )
        fill.assert_called_once_with(refused, 2026, "Singapore Grand Prix", "Q", car_data=True)


def test_load_session_refuses_a_session_that_has_not_finished():
    running = {**_ROW, "date_end": "2999-01-01T00:00:00+00:00"}
    with (
        patch.object(openf1, "find_session", return_value=running),
        patch.object(openf1, "_get", _fake_get),
        pytest.raises(openf1.OpenF1Unavailable, match="not finished"),
    ):
        openf1_session.load_session(_session(), 2026, "Singapore Grand Prix", "FP1")


def test_restart_behind_the_safety_car_after_a_red_flag():
    race_control = [
        _rc(10, "RED FLAG - RACE SUSPENDED", category="Other"),
        _rc(20, "TRACK CLEAR", category="Flag", flag="CLEAR", scope="Track"),
        _rc(30, "SAFETY CAR LIGHTS ON", category="Other"),
        _rc(40, "SESSION STARTED", category="SessionStatus"),
        _rc(60, "STANDING START", category="Other"),
    ]
    t0 = pd.Timestamp("2026-10-09 08:30")
    laps = pd.DataFrame(
        {
            "LapStartDate": [t0 + pd.Timedelta(seconds=s) for s in (41, 61)],
            "Time": [pd.Timedelta(seconds=s) for s in (59, 80)],
        }
    )
    assert openf1_session.lap_track_status(laps, race_control, t0).tolist() == ["4", "1"]


def test_laps_still_arriving_are_refused():
    results = [{"driver_number": 1, "number_of_laps": 7}, {"driver_number": 4, "number_of_laps": 3}]
    assert openf1_session.drivers_short_of_results(_LAPS, results) == [4]
    assert openf1_session.drivers_short_of_results(_LAPS, results[:1]) == []
    assert openf1_session.drivers_short_of_results(_LAPS, []) == ["no session_result"]
