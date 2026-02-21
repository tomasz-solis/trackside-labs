"""Tests for FP blending team-name canonicalization."""

import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.utils.fp_blending import FPDataError, _circuit_breaker, get_fp_team_performance


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Reset circuit breaker before each test to avoid cross-test contamination."""
    _circuit_breaker.reset()
    yield
    _circuit_breaker.reset()


def test_get_fp_team_performance_maps_fastf1_team_names():
    """FastF1 sponsor-form team names should map to characteristics team names."""
    # Need at least 10 laps to pass red flag detection
    lap_data = []
    for lap_num in range(3):  # 3 laps per driver × 4 drivers = 12 total laps
        lap_data.extend(
            [
                {
                    "Driver": "VER",
                    "Team": "Oracle Red Bull Racing",
                    "LapTime": pd.Timedelta(seconds=90.1 + lap_num * 0.1),
                    "Compound": "SOFT",
                },
                {
                    "Driver": "PER",
                    "Team": "Oracle Red Bull Racing",
                    "LapTime": pd.Timedelta(seconds=90.3 + lap_num * 0.1),
                    "Compound": "SOFT",
                },
                {
                    "Driver": "LEC",
                    "Team": "Scuderia Ferrari",
                    "LapTime": pd.Timedelta(seconds=90.0 + lap_num * 0.1),
                    "Compound": "SOFT",
                },
                {
                    "Driver": "HAM",
                    "Team": "Scuderia Ferrari",
                    "LapTime": pd.Timedelta(seconds=90.4 + lap_num * 0.1),
                    "Compound": "SOFT",
                },
            ]
        )
    laps = pd.DataFrame(lap_data)

    mock_session = MagicMock()
    mock_session.laps = laps
    mock_session.date = datetime.now(tz=UTC)

    with patch("src.utils.fp_blending.ff1.get_session", return_value=mock_session):
        perf, session_laps, error = get_fp_team_performance(2026, "Bahrain Grand Prix", "FP1")

    assert perf is not None
    assert session_laps is not None
    assert "Red Bull Racing" in perf
    assert "Ferrari" in perf


def test_get_fp_team_performance_load_none_does_not_emit_warning(caplog):
    """load(None) should quietly return API failure (warning is emitted upstream once)."""
    mock_session = MagicMock()
    mock_session.load.return_value = None

    with patch("src.utils.fp_blending.ff1.get_session", return_value=mock_session):
        with caplog.at_level(logging.WARNING):
            perf, session_laps, error = get_fp_team_performance(
                2026, "Australian Grand Prix", "FP1"
            )

    assert perf is None
    assert session_laps is None
    assert error == FPDataError.API_FAILURE
    assert "session.load() returned None" not in caplog.text
