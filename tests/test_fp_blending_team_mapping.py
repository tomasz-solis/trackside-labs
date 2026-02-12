"""Tests for FP blending team-name canonicalization."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.utils.fp_blending import get_fp_team_performance


def test_get_fp_team_performance_maps_fastf1_team_names():
    """FastF1 sponsor-form team names should map to characteristics team names."""
    laps = pd.DataFrame(
        [
            {
                "Driver": "VER",
                "Team": "Oracle Red Bull Racing",
                "LapTime": pd.Timedelta(seconds=90.1),
                "Compound": "SOFT",
            },
            {
                "Driver": "PER",
                "Team": "Oracle Red Bull Racing",
                "LapTime": pd.Timedelta(seconds=90.3),
                "Compound": "SOFT",
            },
            {
                "Driver": "LEC",
                "Team": "Scuderia Ferrari",
                "LapTime": pd.Timedelta(seconds=90.0),
                "Compound": "SOFT",
            },
            {
                "Driver": "HAM",
                "Team": "Scuderia Ferrari",
                "LapTime": pd.Timedelta(seconds=90.4),
                "Compound": "SOFT",
            },
        ]
    )

    mock_session = MagicMock()
    mock_session.laps = laps

    with patch("src.utils.fp_blending.ff1.get_session", return_value=mock_session):
        perf = get_fp_team_performance(2026, "Bahrain Grand Prix", "FP1")

    assert perf is not None
    assert "Red Bull Racing" in perf
    assert "Ferrari" in perf
