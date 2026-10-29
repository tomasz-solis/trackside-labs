"""Tests for the tyre deg severity script's race regression and season combination."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_script_module():
    """Load the tyre deg severity script as a module."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "extract_tyre_deg_severity.py"
    spec = importlib.util.spec_from_file_location("extract_tyre_deg_severity_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _race_laps(deg: float, trend: float, stops: dict[str, list[int]]) -> pd.DataFrame:
    """Synthetic 50-lap race: each driver pits on the given laps and gets fresh MEDIUMs."""
    rng = np.random.default_rng(0)
    rows = []
    for offset, (driver, pit_laps) in enumerate(stops.items()):
        age = 0
        for lap in range(1, 51):
            age += 1
            pit_in = lap in pit_laps
            rows.append(
                {
                    "Driver": driver,
                    "LapNumber": lap,
                    "TyreLife": age,
                    "Compound": "MEDIUM",
                    "TrackStatus": "1",
                    "LapTime": pd.Timedelta(
                        seconds=90 + 0.2 * offset + deg * age + trend * lap + rng.normal(0, 0.05)
                    ),
                    "PitInTime": pd.Timedelta(seconds=1) if pit_in else pd.NaT,
                    "PitOutTime": pd.Timedelta(seconds=1) if age == 1 and lap > 1 else pd.NaT,
                }
            )
            if pit_in:
                age = 0
    return pd.DataFrame(rows)


def test_race_tyre_deg_separates_deg_from_fuel_trend():
    module = _load_script_module()
    stops = {f"D{i}": [12 + 2 * i, 30 + i] for i in range(10)}
    laps = _race_laps(deg=0.06, trend=-0.05, stops=stops)

    measured = module.race_tyre_deg(laps)

    assert measured["deg"] == pytest.approx(0.06, abs=0.005)
    assert measured["trend"] == pytest.approx(-0.05, abs=0.005)
    assert measured["wet_share"] == 0.0


def test_race_tyre_deg_returns_none_when_too_few_clean_laps():
    module = _load_script_module()
    laps = _race_laps(deg=0.06, trend=-0.05, stops={"D0": [20]})

    assert module.race_tyre_deg(laps) is None


def test_race_tyre_deg_returns_none_when_the_whole_field_stops_on_one_lap():
    module = _load_script_module()
    laps = _race_laps(deg=0.06, trend=-0.05, stops={f"D{i}": [3] for i in range(10)})

    assert module.race_tyre_deg(laps) is None


def test_dry_relative_drops_wet_races_and_divides_by_season_median():
    module = _load_script_module()
    races = {
        "A": {"deg": 0.02, "wet_share": 0.0, "laps_used": 900},
        "B": {"deg": 0.04, "wet_share": 0.0, "laps_used": 900},
        "C": {"deg": 0.08, "wet_share": 0.0, "laps_used": 900},
        "Wet": {"deg": 0.5, "wet_share": 0.4, "laps_used": 900},
    }

    assert module.dry_relative(races) == {"A": 0.5, "B": 1.0, "C": 2.0}


def test_severities_take_median_over_seasons_and_shrink_toward_one():
    module = _load_script_module()
    season = lambda a, b: {  # noqa: E731
        "A": {"deg": a, "wet_share": 0.0, "laps_used": 900},
        "B": {"deg": b, "wet_share": 0.0, "laps_used": 900},
        "M": {"deg": 0.04, "wet_share": 0.0, "laps_used": 900},
    }

    result = module.severities([season(0.08, 0.02), season(0.08, 0.04), season(0.16, 0.02)])

    # A is 2.0x the median in two seasons and 4.0x in one: median 2.0, shrunk (3 x 2 + 2) / 5.
    assert result["A"] == (1.6, 3)
    assert result["B"] == (0.7, 3)  # 0.5, 1.0, 0.5: median 0.5, (1.5 + 2) / 5
