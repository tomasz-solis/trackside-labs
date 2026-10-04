"""Carry-over tyre deg: field slope per compound plus a Kalman team estimate."""

import pytest

from src.models import tyre_deg as td
from src.predictors.baseline.race.preparation_mixin import apply_carry_over_tyre_deg


def _race(compound_deg, deg_by_team):
    return {"compound_deg": compound_deg, "traits": {t: {"deg": v} for t, v in deg_by_team.items()}}


def test_field_slope_is_the_median_raw_slope_over_earlier_races():
    races = {
        "R1": _race({"SOFT": 0.01, "MEDIUM": 0.0}, {}),
        "R2": _race({"SOFT": 0.03}, {}),
        "R3": _race({"SOFT": -0.30}, {}),  # an outlier like Monaco does not drag the median
    }

    slopes = td.field_raw_slopes(races, ["R1", "R2", "R3"])

    assert slopes["SOFT"] == pytest.approx(0.01)
    assert slopes["MEDIUM"] == pytest.approx(0.0)
    assert slopes["HARD"] == pytest.approx(td.FALLBACK_RAW_SLOPE["HARD"])  # no data yet


def test_team_estimate_grows_with_evidence_and_stays_shrunk():
    one = td.team_deg_deltas({"R1": _race({}, {"A": 0.03})}, ["R1"])
    many = td.team_deg_deltas(
        {f"R{i}": _race({}, {"A": 0.03}) for i in range(15)}, [f"R{i}" for i in range(15)]
    )

    assert 0.0 < one["A"] < many["A"] < 0.03  # visible, growing, never the raw value


def test_apply_replaces_practice_slopes(monkeypatch):
    monkeypatch.setattr(
        td,
        "tyre_deg_slopes",
        lambda year, race, gain: ({"SOFT": 0.07, "MEDIUM": 0.05, "HARD": 0.05}, {"A": 0.01}),
    )
    info = {
        "X": {"team": "A", "tire_deg_by_compound": {"SOFT": 0.43, "MEDIUM": 0.04, "HARD": 0.35}},
        "Y": {"team": "B", "tire_deg_by_compound": {"SOFT": 0.28, "MEDIUM": 0.20, "HARD": 0.21}},
    }

    apply_carry_over_tyre_deg(info, year=2026, race_name="R", cfg={})

    assert info["X"]["tire_deg_by_compound"] == pytest.approx(
        {"SOFT": 0.08, "MEDIUM": 0.06, "HARD": 0.06}
    )
    assert info["Y"]["tire_deg_by_compound"] == pytest.approx(
        {"SOFT": 0.07, "MEDIUM": 0.05, "HARD": 0.05}
    )


def test_default_is_practice():
    from src.utils import config_loader

    assert config_loader.get("baseline_predictor.race.tyre_deg_model") == "practice"
