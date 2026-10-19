"""Tests for filling a FastF1 session from OpenF1 laps, stints, weather and car data."""

from types import SimpleNamespace
from unittest.mock import patch

import fastf1
import pandas as pd
import pytest
from fastf1.exceptions import DataNotLoadedError

from src.data import openf1, openf1_session
from src.systems import testing_updater_sessions
from src.utils import fp_blending

_T0 = "2026-10-09T08:30:00+00:00"


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


def _fake_get(endpoint, *, cache=True, **params):
    rows = {
        "laps": _LAPS,
        "stints": _STINTS,
        "weather": [{"date": _T0, "rainfall": 1, "air_temperature": 30.0}],
        "car_data": [
            {"date": (pd.Timestamp(_T0) + pd.to_timedelta(s, unit="s")).isoformat(), "brake": b}
            for s, b in [(125, 0), (150, 100), (200, 0), (300, 100)]
        ],
    }
    return rows[endpoint]


def test_load_session_fills_laps_weather_and_car_data():
    session = _session()
    with (
        patch.object(openf1, "find_session", return_value={"session_key": 9, "date_start": _T0}),
        patch.object(openf1, "_driver_table", return_value=_DRIVERS),
        patch.object(openf1, "_get", _fake_get),
    ):
        openf1_session.load_session(session, 2026, "Singapore Grand Prix", "FP1", car_data=True)

    assert len(session.laps) == len(_LAPS) and set(session.laps["Source"]) == {"openf1"}
    assert session.weather_data["Rainfall"].tolist() == [True]
    lap_2 = session.laps.pick_laps(2).iloc[0]
    car = lap_2.get_car_data()
    assert car["Brake"].tolist() == [False, True, False]


def test_load_session_raises_when_openf1_has_no_laps():
    with (
        patch.object(openf1, "find_session", return_value={"session_key": 9, "date_start": _T0}),
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
