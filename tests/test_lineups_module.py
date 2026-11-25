from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.utils import driver_substitutions, lineups


def test_get_lineups_from_session_extracts_participants(patcher):
    results = pd.DataFrame(
        {
            "TeamName": ["McLaren", "McLaren", "Ferrari"],
            "Abbreviation": ["NOR", "PIA", "LEC"],
        }
    )
    session = SimpleNamespace(results=results)
    session.load = lambda laps, telemetry, weather: None

    patcher.setattr(lineups.ff1, "get_session", lambda year, race, session_type: session)

    extracted = lineups.get_lineups_from_session(2026, "Bahrain Grand Prix", "Q")

    assert extracted == {"McLaren": ["NOR", "PIA"], "Ferrari": ["LEC"]}


def test_get_lineups_from_session_returns_none_on_exception(patcher):
    patcher.setattr(
        lineups.ff1,
        "get_session",
        lambda year, race, session_type: (_ for _ in ()).throw(TypeError("boom")),
    )

    assert lineups.get_lineups_from_session(2026, "Bahrain Grand Prix", "Q") is None


def test_load_current_lineups_handles_missing_and_existing_files(tmp_path):
    missing_path = tmp_path / "missing.json"
    assert lineups.load_current_lineups(str(missing_path)) is None

    config_path = tmp_path / "current_lineups.json"
    payload = {"current_lineups": {"McLaren": ["NOR", "PIA"]}}
    config_path.write_text(json.dumps(payload))

    loaded = lineups.load_current_lineups(str(config_path))
    assert loaded == {"McLaren": ["NOR", "PIA"]}


def test_get_lineups_historical_prefers_session_data(patcher):
    patcher.setattr(
        lineups,
        "get_lineups_from_session",
        lambda year, race, session_type: {"Ferrari": ["LEC", "HAM"]},
    )

    result = lineups.get_lineups(2025, "Monaco Grand Prix")
    assert result == {"Ferrari": ["LEC", "HAM"]}


def test_get_lineups_falls_back_to_config(patcher):
    patcher.setattr(lineups, "get_lineups_from_session", lambda year, race, session_type: None)
    patcher.setattr(
        lineups, "load_current_lineups", lambda config_path: {"McLaren": ["NOR", "PIA"]}
    )

    result = lineups.get_lineups(2025, "Monaco Grand Prix")
    assert result == {"McLaren": ["NOR", "PIA"]}


def test_get_lineups_raises_without_any_data(patcher):
    patcher.setattr(lineups, "get_lineups_from_session", lambda year, race, session_type: None)
    patcher.setattr(lineups, "load_current_lineups", lambda config_path: None)

    with pytest.raises(ValueError, match="No lineup data available"):
        lineups.get_lineups(2026, "Bahrain Grand Prix")


def test_get_lineups_applies_a_race_scoped_substitution(patcher, tmp_path):
    """The injured driver's seat is filled before any caller sees the lineup."""
    config_path = tmp_path / "current_lineups.json"
    config_path.write_text(
        json.dumps({"current_lineups": {"Red Bull Racing": ["VER", "HAD"], "RB": ["LAW", "LIN"]}})
    )
    patcher.setattr(
        driver_substitutions,
        "load_configured_substitutions",
        lambda **_: {"HAD": "LAW", "LAW": "TSU"},
    )

    resolved = lineups.get_lineups(2026, "Dutch Grand Prix", str(config_path))

    assert resolved["Red Bull Racing"] == ["VER", "LAW"]
    assert resolved["RB"] == ["TSU", "LIN"]


_CONFIGURED = {
    "Haas F1 Team": ["OCO", "BEA"],
    "Red Bull Racing": ["VER", "HAD"],
    "RB": ["LAW", "LIN"],
}


def _entry_lists(by_session: dict[str, list[tuple[str, str]]], asked: list[str]):
    def fake(year, race_name, session_name):
        asked.append(session_name)
        return by_session.get(session_name, [])

    return fake


def test_a_substitute_in_the_entry_list_takes_the_seat(patcher):
    """Las Vegas 2026: CAM drove for OCO; raw team names are mapped to the configured ones."""
    asked: list[str] = []
    fp2 = [("CAM", "Haas F1 Team"), ("BEA", "Haas F1 Team"), ("LAW", "Racing Bulls")]
    patcher.setattr(lineups, "_session_entry_list", _entry_lists({"FP2": fp2}, asked))

    seated = lineups.seat_entered_drivers(
        _CONFIGURED, year=2026, race_name="Las Vegas Grand Prix", checkpoint_session="FP2"
    )

    assert seated["Haas F1 Team"] == ["CAM", "BEA"]
    assert seated["RB"] == ["LAW", "LIN"]
    assert asked == ["FP2"]


@pytest.mark.parametrize("checkpoint", ["PRE", "FP1", "fp1", ""])
def test_pre_and_fp1_keep_the_configured_line_up(patcher, checkpoint):
    """FP1 lists rookies who drive only that session, so it never sets the line-up."""
    asked: list[str] = []
    rookie = [("ROO", "Haas F1 Team"), ("BEA", "Haas F1 Team")]
    patcher.setattr(lineups, "_session_entry_list", _entry_lists({"FP1": rookie}, asked))

    seated = lineups.seat_entered_drivers(
        _CONFIGURED, year=2026, race_name="Mexico City Grand Prix", checkpoint_session=checkpoint
    )

    assert seated == _CONFIGURED and asked == []


def test_the_latest_session_with_an_entry_list_answers(patcher):
    asked: list[str] = []
    fp3 = [("HIR", "Haas F1 Team"), ("BEA", "Haas F1 Team")]
    patcher.setattr(lineups, "_session_entry_list", _entry_lists({"FP3": fp3}, asked))

    seated = lineups.seat_entered_drivers(
        _CONFIGURED, year=2026, race_name="Mexico City Grand Prix", checkpoint_session="Q"
    )

    assert seated["Haas F1 Team"] == ["HIR", "BEA"]
    assert asked == ["Q", "Sprint", "SQ", "FP3"]


def test_entry_list_swaps_are_one_for_one_and_handle_a_chain():
    entered = {
        "Red Bull Racing": ["VER", "LAW"],
        "RB": ["TSU", "LIN"],
        "Haas F1 Team": ["OCO", "BEA", "ROO"],
    }
    swaps = driver_substitutions.entry_list_swaps(_CONFIGURED, entered)

    assert swaps == {"HAD": "LAW", "LAW": "TSU"}
    assert driver_substitutions.entry_list_swaps(_CONFIGURED, {"Haas F1 Team": ["BEA"]}) == {}


def test_an_admin_substitution_overrides_the_entry_list(patcher, tmp_path):
    config_path = tmp_path / "current_lineups.json"
    config_path.write_text(json.dumps({"current_lineups": _CONFIGURED}))
    fp2 = [("CAM", "Haas F1 Team"), ("BEA", "Haas F1 Team")]
    patcher.setattr(lineups, "_session_entry_list", _entry_lists({"FP2": fp2}, []))
    patcher.setattr(
        driver_substitutions,
        "load_configured_substitutions",
        lambda **_: {"CAM": "OCO"},
    )

    resolved = lineups.get_lineups(
        2026, "Las Vegas Grand Prix", str(config_path), checkpoint_session="FP2"
    )

    assert resolved["Haas F1 Team"] == ["OCO", "BEA"]


def test_an_entry_list_comes_from_openf1_when_fastf1_fails_and_is_cached(patcher):
    calls: list[str] = []

    def fastf1_fails(year, race, session_name):
        calls.append("fastf1")
        raise ConnectionError("FastF1 down")

    def openf1_answers(year, race, session_name):
        calls.append("openf1")
        return [("BEA", "Haas F1 Team"), ("CAM", "Haas F1 Team")]

    patcher.setattr(lineups.ff1, "get_session", fastf1_fails)
    patcher.setattr("src.data.openf1.fetch_entry_list", openf1_answers)
    patcher.setattr(lineups, "_ENTRY_LISTS", {})

    first = lineups._session_entry_list(2026, "Las Vegas Grand Prix", "FP2")
    second = lineups._session_entry_list(2026, "Las Vegas Grand Prix", "FP2")

    assert first == second == [("BEA", "Haas F1 Team"), ("CAM", "Haas F1 Team")]
    assert calls == ["fastf1", "openf1"]


def test_an_entry_list_with_a_missing_code_goes_to_openf1_and_is_not_cached(patcher):
    """One São Paulo 2026 Q load named Câmara "nan"; that must never take a seat."""
    import math

    class _Session:
        results = {
            "Abbreviation": ["BEA", math.nan],
            "TeamName": ["Haas F1 Team", "Haas F1 Team"],
        }

        def load(self, **kwargs):
            return None

    answers = {"openf1": [("BEA", "Haas F1 Team"), ("CAM", "Haas F1 Team")]}
    patcher.setattr(lineups.ff1, "get_session", lambda *args: _Session())
    patcher.setattr("src.data.openf1.fetch_entry_list", lambda *args: answers["openf1"])
    patcher.setattr(lineups, "_ENTRY_LISTS", {})

    assert lineups._session_entry_list(2026, "Qatar Grand Prix", "FP2") == answers["openf1"]

    answers["openf1"] = [("BEA", "Haas F1 Team"), ("51", "Haas F1 Team")]
    assert lineups._session_entry_list(2026, "Qatar Grand Prix", "FP3") == []
    assert (2026, "Qatar Grand Prix", "FP3") not in lineups._ENTRY_LISTS


def test_an_admin_swap_keyed_to_the_configured_seat_beats_a_conflicting_entry_list(
    patcher, tmp_path
):
    """Codex 2026-11-23: entry list seats HIR for OCO, admin saved OCO > CAM; CAM must race.

    The swap is saved through the admin's own validation, not patched in.
    """
    config_path = tmp_path / "current_lineups.json"
    config_path.write_text(json.dumps({"current_lineups": _CONFIGURED}))

    class _Store:
        saved: dict = {}

        def save_artifact(self, **kwargs):
            self.saved = kwargs["data"]
            return kwargs

        def load_artifact(self, artifact_type, artifact_key):
            return self.saved

    store = _Store()
    driver_substitutions.save_substitutions(
        race_name="Qatar Grand Prix",
        year=2026,
        substitutions={"OCO": "CAM"},
        lineups=_CONFIGURED,
        store=store,
    )
    patcher.setattr(driver_substitutions, "_default_store", lambda: store)
    fp2 = [("HIR", "Haas F1 Team"), ("BEA", "Haas F1 Team"), ("LAW", "Racing Bulls")]
    patcher.setattr(lineups, "_session_entry_list", _entry_lists({"FP2": fp2}, []))

    resolved = lineups.get_lineups(
        2026, "Qatar Grand Prix", str(config_path), checkpoint_session="FP2"
    )
    assert resolved["Haas F1 Team"] == ["CAM", "BEA"]

    # The entry list may not seat the admin's driver in another seat either.
    fp2 = [("CAM", "Haas F1 Team"), ("HIR", "Haas F1 Team")]
    patcher.setattr(lineups, "_session_entry_list", _entry_lists({"FP2": fp2}, []))
    resolved = lineups.get_lineups(
        2026, "Qatar Grand Prix", str(config_path), checkpoint_session="FP2"
    )
    assert resolved["Haas F1 Team"] == ["CAM", "HIR"]
