from __future__ import annotations

import pytest

from src.systems.weight_schedule import (
    calculate_blended_performance,
    format_schedule_summary,
    get_recommended_schedule,
    get_schedule_weights,
)


def test_get_schedule_weights_validates_inputs():
    with pytest.raises(ValueError, match="Unknown schedule"):
        get_schedule_weights(1, schedule="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="race_number must be >= 1"):
        get_schedule_weights(0, schedule="extreme")


def test_get_schedule_weights_interpolates_between_checkpoints():
    weights = get_schedule_weights(race_number=7, schedule="conservative")
    assert weights == pytest.approx({"baseline": 0.2625, "testing": 0.0875, "current": 0.65})


def test_get_schedule_weights_clamps_after_last_checkpoint():
    weights = get_schedule_weights(race_number=99, schedule="insane")
    assert weights == {"baseline": 0.0, "testing": 0.0, "current": 1.0}


def test_calculate_blended_performance_uses_schedule_weights():
    blended = calculate_blended_performance(
        baseline_score=0.8,
        testing_modifier=0.1,
        current_score=0.6,
        race_number=1,
        schedule="extreme",
    )
    assert blended == pytest.approx(0.56)


def test_get_recommended_schedule_switches_with_regulation_flag():
    assert get_recommended_schedule(is_regulation_change=True) == "extreme"
    assert get_recommended_schedule(is_regulation_change=False) == "moderate"


def test_format_schedule_summary_contains_schedule_lines():
    summary = format_schedule_summary("extreme")
    assert "Weight Schedule: EXTREME" in summary
    assert "Race  1+" in summary
