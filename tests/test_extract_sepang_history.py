"""Safety car inference from lap times: gaps close under a safety car, not in rain."""

import pandas as pd
from scripts.extract_sepang_history import had_safety_car


def _race(slow_lap_spread: float) -> list[pd.DataFrame]:
    """Five cars over 10 laps. Laps 4-5 run at 1.5x pace; on them each place behind
    the leader gains or loses ``slow_lap_spread`` seconds (negative closes the field)."""
    rows = [
        {
            "number": lap,
            "driverId": f"d{position}",
            "time": pd.Timedelta(
                seconds=150.0 + position * slow_lap_spread if lap in (4, 5) else 100.0 + position
            ),
        }
        for lap in range(1, 11)
        for position in range(5)
    ]
    return [pd.DataFrame(rows)]


def test_a_slow_run_that_closes_the_gaps_is_a_safety_car():
    assert had_safety_car(_race(slow_lap_spread=-1.0))


def test_a_slow_run_that_keeps_the_gaps_is_not_a_safety_car():
    """Rain slows everyone but the field stays spread out."""
    assert not had_safety_car(_race(slow_lap_spread=1.0))
