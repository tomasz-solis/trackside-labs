"""Model 3.1 candidate switches: off by default, and each does what it says when on."""

from __future__ import annotations

import numpy as np

from src.predictors.baseline.race.result_processing import build_finish_order
from src.utils import config_loader
from src.utils.lap_by_lap_simulator import _calculate_safety_car_lap_probability


class _Cfg:
    """Real config with a few keys overridden."""

    def __init__(self, **overrides):
        self._overrides = overrides

    def get(self, key, default=None):
        if key in self._overrides:
            return self._overrides[key]
        return config_loader.get(key, default)


def _finish_order(cfg) -> list[str]:
    # A wins 3 times in 5 but retires otherwise; B and C are steady P2 and P3.
    # Mean rank: A 1.8, B 1.6. Median rank: A 1, B 2.
    info = {
        "team": "T",
        "grid_pos": 1,
        "overtaking_skill": 0.5,
        "race_advantage": 0.0,
        "skill": 0.5,
    }
    rows = build_finish_order(
        aggregated={
            "median_positions": {"A": 1, "B": 2, "C": 3},
            "position_distributions": {
                "A": [1, 1, 1, 22, 22] * 100,
                "B": [2, 2, 2, 2, 2] * 100,
                "C": [3, 3, 3, 3, 3] * 100,
            },
            "dnf_rates": {"A": 0.4, "B": 0.0, "C": 0.0},
        },
        driver_info_map={
            "A": {**info, "team": "TA"},
            "B": {**info, "team": "TB", "grid_pos": 2},
            "C": {**info, "team": "TC", "grid_pos": 3},
        },
        grid_position_samples_by_driver={},
        field_size=22,
        weather="dry",
        is_sprint=True,
        input_confidence=1.0,
        cfg=cfg,
        race_params={},
        weather_feature_modifiers={},
        get_learned_position_adjustment=lambda **_: 0.0,
        learned_interval_radius=0.0,
        enforce_non_increasing=lambda values: values,
        base_seed=1,
    )
    return [row["driver"] for row in sorted(rows, key=lambda row: row["position"])]


def test_shipped_31_switches():
    assert config_loader.get("model.version") == "3.4"
    assert config_loader.get("baseline_predictor.race.finish_order_sort") == "median_rank"
    assert config_loader.get("baseline_predictor.race.overtake_gap_from_lap_start") is True
    assert config_loader.get("baseline_predictor.race.dnf_per_lap_hazard") is False


def test_median_rank_sort_stops_a_dnf_tail_burying_the_usual_winner():
    mean_order = _finish_order(_Cfg(**{"baseline_predictor.race.finish_order_sort": "mean_rank"}))
    median_order = _finish_order(
        _Cfg(**{"baseline_predictor.race.finish_order_sort": "median_rank"})
    )

    assert mean_order[:2] == ["B", "A"]
    assert median_order[:2] == ["A", "B"]


def test_per_lap_hazard_delivers_the_race_level_probability():
    p, laps = 0.20, 56
    hazard = _calculate_safety_car_lap_probability(p, laps)

    assert np.isclose(1 - (1 - hazard) ** laps, p)
    # The old p / N form under-delivers by ~9% at p 0.2.
    assert 1 - (1 - p / laps) ** laps < 0.92 * p


def test_learned_race_radius_widens_bands_and_never_moves_the_order():
    """Singapore item D: the widening is a band floor. Positions stay where they were."""
    from src.predictors.baseline.race.result_processing import apply_learned_interval_radius

    rows = [
        {"driver": "A", "position": 1, "median_position": 1, "p5": 1, "p95": 2},
        {"driver": "B", "position": 2, "median_position": 2, "p5": 1, "p95": 3},
        {"driver": "C", "position": 22, "median_position": 22, "p5": 21, "p95": 22},
    ]
    before = [(row["driver"], row["position"]) for row in rows]

    apply_learned_interval_radius(finish_order=rows, learned_interval_radius=5.0, field_size=22)

    assert [(row["driver"], row["position"]) for row in rows] == before
    assert (rows[0]["p5"], rows[0]["p95"]) == (1, 6)
    assert (rows[2]["p5"], rows[2]["p95"]) == (17, 22)
    assert all(row["p5"] <= row["position"] <= row["p95"] for row in rows)


def test_race_interval_widening_is_off_by_default():
    assert config_loader.get("learning.race_interval_widening") is False
    assert config_loader.get("learning.apply_adaptive_calibration") is True


def test_race_interval_widening_switch_gates_the_live_predictor():
    """The predictor class resolves one radius getter; the switch must gate that one."""
    from types import SimpleNamespace

    from src.predictors.baseline_2026 import Baseline2026Predictor

    def radius(switch: bool, session: str) -> float:
        stub = SimpleNamespace(
            calibration_system=SimpleNamespace(get_interval_radius=lambda **_: 5.0),
            config={"learning.race_interval_widening": switch},
        )
        return Baseline2026Predictor._get_learned_interval_radius(stub, session=session)

    assert radius(False, "race") == 0.0
    assert radius(False, "sprint") == 0.0
    assert radius(True, "race") == 5.0
    assert radius(True, "sprint") == 5.0
    assert radius(False, "qualifying") == 5.0
