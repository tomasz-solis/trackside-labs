"""Tests for track parameter helpers."""

from unittest.mock import MagicMock, patch

from src.utils.track_data_loader import (
    get_available_compounds,
    resolve_race_distance_laps,
)


def setup_function():
    resolve_race_distance_laps.cache_clear()


def test_get_available_compounds_is_weather_aware():
    assert get_available_compounds("Bahrain Grand Prix", weather="dry") == [
        "SOFT",
        "MEDIUM",
        "HARD",
    ]
    assert get_available_compounds("Bahrain Grand Prix", weather="rain") == ["INTERMEDIATE", "WET"]
    assert get_available_compounds("Bahrain Grand Prix", weather="mixed") == [
        "SOFT",
        "MEDIUM",
        "HARD",
        "INTERMEDIATE",
    ]


def test_resolve_race_distance_uses_known_track_mapping():
    assert resolve_race_distance_laps(2026, "Monaco Grand Prix", is_sprint=False) == 78
    assert resolve_race_distance_laps(2026, "British Grand Prix", is_sprint=True) == 17


def test_resolve_race_distance_uses_fastf1_metadata_for_unknown_tracks():
    mock_session = MagicMock()
    mock_session.total_laps = 63

    with patch("src.utils.track_data_loader.fastf1.get_session", return_value=mock_session):
        laps = resolve_race_distance_laps(2026, "Imaginary Grand Prix", is_sprint=False)

    assert laps == 63
    mock_session.load.assert_not_called()


def test_resolve_race_distance_falls_back_when_fastf1_fails():
    with patch("src.utils.track_data_loader.fastf1.get_session", side_effect=RuntimeError("boom")):
        assert resolve_race_distance_laps(2026, "Unknown Grand Prix", is_sprint=False) == 60
        assert resolve_race_distance_laps(2026, "Unknown Grand Prix", is_sprint=True) == 20
