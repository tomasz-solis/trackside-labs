"""Tests for the track profile's top speed and energy clipping measures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.extractors import car_track_traits


def _telemetry(
    speeds: list[float], *, throttle: float = 100.0, brake: bool = False
) -> pd.DataFrame:
    """Car data sampled every 0.25 s."""
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta(np.arange(len(speeds)) * 0.25, unit="s"),
            "Speed": speeds,
            "Throttle": throttle,
            "Brake": brake,
        }
    )


def test_clipping_share_counts_full_throttle_slowing_above_200():
    rising = list(np.linspace(250, 300, 20))
    clipping = list(np.linspace(300, 280, 21))[1:]  # 4 km/h per second for 5 s
    tel = _telemetry(rising + clipping)

    assert car_track_traits.clipping_share(tel) == pytest.approx(0.5, abs=0.03)


def test_clipping_share_ignores_braking_low_speed_and_part_throttle():
    falling = list(np.linspace(300, 250, 20))

    assert car_track_traits.clipping_share(_telemetry(falling, brake=True)) == 0.0
    assert car_track_traits.clipping_share(_telemetry(falling, throttle=60.0)) == 0.0
    assert car_track_traits.clipping_share(_telemetry(list(np.linspace(190, 150, 20)))) == 0.0


def test_energy_profile_takes_medians_and_skips_drivers_without_car_data(monkeypatch):
    class _Lap:
        def __init__(self, tel):
            self._tel = tel

        def get_car_data(self):
            if self._tel is None:
                raise KeyError("12")
            return self._tel

    flat = _telemetry([300.0] * 20)
    laps = {
        "A": _Lap(flat),
        "B": _Lap(_telemetry(list(np.linspace(300, 320, 20)))),
        "C": _Lap(_telemetry(list(np.linspace(330, 310, 20)))),
        "D": _Lap(None),
    }
    monkeypatch.setattr(car_track_traits, "_fastest_laps", lambda session: laps)

    profile = car_track_traits.energy_profile(object())

    assert profile["top_speed_kph"] == 320.0  # median of 300, 320, 330
    assert profile["clipping"] == 0.0  # median of 0, 0, about 1


def test_energy_profile_is_empty_without_any_car_data(monkeypatch):
    monkeypatch.setattr(car_track_traits, "_fastest_laps", lambda session: {})

    assert car_track_traits.energy_profile(object()) == {}
