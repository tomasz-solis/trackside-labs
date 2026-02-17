"""Tests for qualifying-stage-aware FP blending session selection."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.fp_blending import _circuit_breaker, get_best_fp_performance


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Reset circuit breaker before each test to avoid cross-test contamination."""
    _circuit_breaker.reset()
    yield
    _circuit_breaker.reset()


def test_sprint_qualifying_stage_uses_fp1_only():
    """Sprint-qualifying context should only probe FP1 on sprint weekends."""
    calls = []

    def _mock_get_fp_team_performance(year, race_name, session_type):
        calls.append(session_type)
        if session_type == "FP1":
            return {"McLaren": 0.9}, pd.DataFrame({"Driver": ["NOR"]}), None
        return None, None, None

    with patch(
        "src.utils.fp_blending.get_fp_team_performance",
        side_effect=_mock_get_fp_team_performance,
    ):
        session_label, perf, laps = get_best_fp_performance(
            year=2026,
            race_name="Chinese Grand Prix",
            is_sprint=True,
            qualifying_stage="sprint",
        )

    assert calls == ["FP1"]
    assert session_label == "FP1 times"
    assert perf == {"McLaren": 0.9}
    assert laps is not None


def test_main_qualifying_stage_prefers_sq_then_sprint_then_fp1():
    """Main-qualifying context should prioritize SQ, then Sprint, then FP1."""
    calls = []

    def _mock_get_fp_team_performance(year, race_name, session_type):
        calls.append(session_type)
        if session_type == "Sprint Qualifying":
            return {"Mercedes": 0.8}, pd.DataFrame({"Driver": ["RUS"]}), None
        return None, None, None

    with patch(
        "src.utils.fp_blending.get_fp_team_performance",
        side_effect=_mock_get_fp_team_performance,
    ):
        session_label, perf, laps = get_best_fp_performance(
            year=2026,
            race_name="Chinese Grand Prix",
            is_sprint=True,
            qualifying_stage="main",
        )

    assert calls == ["Sprint Qualifying"]
    assert session_label == "Sprint Qualifying times"
    assert perf == {"Mercedes": 0.8}
    assert laps is not None


def test_invalid_qualifying_stage_raises_value_error():
    with pytest.raises(ValueError, match="qualifying_stage"):
        get_best_fp_performance(
            year=2026,
            race_name="Chinese Grand Prix",
            is_sprint=True,
            qualifying_stage="invalid",
        )
