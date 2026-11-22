"""Tests for the 2027 protocol's tyre weather rule (plan row 45)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import openf1, openf1_session
from src.data.session_conditions import UNKNOWN, conditions_from_laps, session_conditions


def _laps(rows: list[tuple[str, object, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driver": [driver for driver, _, _ in rows],
            "Compound": [compound for _, compound, _ in rows],
            "LapTime": [pd.to_timedelta(t, unit="s") if t else pd.NaT for _, _, t in rows],
        }
    )


def test_a_session_on_slicks_is_dry():
    conditions = conditions_from_laps(_laps([("VER", "SOFT", 90), ("NOR", "MEDIUM", 91)]))

    assert (conditions.scenario, conditions.known, conditions.wet_laps) == ("dry", True, 0)
    assert conditions.timed_laps == 2


def test_one_timed_lap_on_intermediates_makes_it_wet():
    laps = _laps([("VER", "SOFT", 90), ("BOT", "INTERMEDIATE", 98), ("BOT", "intermediate", 97)])
    conditions = conditions_from_laps(laps)

    assert (conditions.scenario, conditions.wet_laps, conditions.wet_drivers) == (
        "rain",
        2,
        ("BOT",),
    )


def test_full_wets_make_it_wet():
    assert conditions_from_laps(_laps([("HAM", "WET", 110)])).scenario == "rain"


def test_an_untimed_lap_on_intermediates_does_not_count():
    conditions = conditions_from_laps(_laps([("VER", "SOFT", 90), ("SAI", "INTERMEDIATE", None)]))

    assert (conditions.scenario, conditions.wet_laps) == ("dry", 0)


@pytest.mark.parametrize(
    "laps",
    [
        _laps([("VER", None, 90), ("NOR", "UNKNOWN", 91)]),
        _laps([]),
        pd.DataFrame({"Driver": ["VER"], "LapTime": [pd.to_timedelta(90, unit="s")]}),
        None,
    ],
)
def test_no_readable_compound_is_scored_dry_and_flagged(laps):
    assert conditions_from_laps(laps) == UNKNOWN
    assert (UNKNOWN.scenario, UNKNOWN.known) == ("dry", False)


def test_a_session_with_no_laps_anywhere_is_flagged(monkeypatch):
    def no_laps(*_args, **_kwargs):
        raise openf1.OpenF1Unavailable("no OpenF1 session")

    monkeypatch.setattr(openf1_session, "get_session", no_laps)
    assert session_conditions(2027, "Bahrain Grand Prix", "Q") == UNKNOWN


def test_an_unreachable_openf1_is_raised_for_a_retry(monkeypatch):
    def unreachable(*_args, **_kwargs):
        raise openf1.OpenF1TransportError("failed after retries")

    monkeypatch.setattr(openf1_session, "get_session", unreachable)
    with pytest.raises(openf1.OpenF1TransportError):
        session_conditions(2027, "Bahrain Grand Prix", "Q")
