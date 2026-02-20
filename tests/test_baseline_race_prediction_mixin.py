from __future__ import annotations

import src.predictors.baseline.race.prediction_mixin as prediction_module
from src.predictors.baseline.race.prediction_mixin import BaselineRacePredictionMixin


class DummyRacePredictor(BaselineRacePredictionMixin):
    seed = 7

    def _load_race_params(self) -> dict:
        return {}

    def _prepare_driver_info_with_compounds(
        self, qualifying_grid: list[dict], race_name: str | None
    ) -> tuple[dict, int]:
        return (
            {
                "DRV": {
                    "driver": "DRV",
                    "team": "Team",
                    "grid_pos": 1,
                    "team_strength": 0.5,
                    "team_strength_by_compound": {"SOFT": 0.5, "MEDIUM": 0.5, "HARD": 0.5},
                    "tire_deg_by_compound": {"SOFT": 0.1, "MEDIUM": 0.1, "HARD": 0.1},
                    "skill": 0.5,
                    "race_advantage": 0.0,
                    "overtaking_skill": 0.5,
                    "defensive_skill": 0.5,
                    "dnf_probability": 0.0,
                }
            },
            0,
        )


def _stub_prediction_dependencies(monkeypatch):
    monkeypatch.setattr(prediction_module, "load_track_specific_params", lambda _race_name: {})
    monkeypatch.setattr(prediction_module, "get_tire_stress_score", lambda _race_name: 3.0)
    monkeypatch.setattr(
        prediction_module,
        "get_available_compounds",
        lambda _race_name, weather="dry": ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"],
    )
    monkeypatch.setattr(
        prediction_module,
        "resolve_race_distance_laps",
        lambda year, race_name, is_sprint: 60,
    )
    monkeypatch.setattr(
        prediction_module,
        "simulate_race_lap_by_lap",
        lambda **kwargs: {
            "finish_order": ["DRV"],
            "dnf_drivers": [],
            "strategies_used": kwargs["strategies"],
        },
    )
    monkeypatch.setattr(
        prediction_module,
        "aggregate_simulation_results",
        lambda _simulation_results: {
            "median_positions": {"DRV": 1},
            "position_distributions": {"DRV": [1]},
            "dnf_rates": {"DRV": 0.0},
            "compound_strategy_distribution": {"SOFT→MEDIUM": 1.0},
            "pit_lap_distribution": {"lap_30-35": 1.0},
        },
    )


def test_predict_race_enforces_two_compounds_for_mixed_weather(monkeypatch):
    predictor = DummyRacePredictor()
    _stub_prediction_dependencies(monkeypatch)

    enforce_flags: list[bool] = []

    def _fake_generate_pit_strategy(**kwargs):
        enforce_flags.append(bool(kwargs["enforce_two_compound_rule"]))
        return {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["SOFT", "MEDIUM"],
            "stint_lengths": [30, 30],
        }

    monkeypatch.setattr(prediction_module, "generate_pit_strategy", _fake_generate_pit_strategy)

    predictor.predict_race(
        qualifying_grid=[{"driver": "DRV", "team": "Team", "position": 1}],
        weather="mixed",
        race_name="Bahrain Grand Prix",
        n_simulations=1,
    )

    assert enforce_flags == [True]


def test_predict_race_allows_single_compound_rule_override_for_rain(monkeypatch):
    predictor = DummyRacePredictor()
    _stub_prediction_dependencies(monkeypatch)

    enforce_flags: list[bool] = []

    def _fake_generate_pit_strategy(**kwargs):
        enforce_flags.append(bool(kwargs["enforce_two_compound_rule"]))
        return {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["INTERMEDIATE", "INTERMEDIATE"],
            "stint_lengths": [30, 30],
        }

    monkeypatch.setattr(prediction_module, "generate_pit_strategy", _fake_generate_pit_strategy)

    predictor.predict_race(
        qualifying_grid=[{"driver": "DRV", "team": "Team", "position": 1}],
        weather="rain",
        race_name="Bahrain Grand Prix",
        n_simulations=1,
    )

    assert enforce_flags == [False]


def test_predict_race_caps_extreme_backmarker_recovery(monkeypatch):
    class ExtremeRecoveryPredictor(BaselineRacePredictionMixin):
        seed = 11

        def _load_race_params(self) -> dict:
            return {}

        def _prepare_driver_info_with_compounds(
            self, qualifying_grid: list[dict], race_name: str | None
        ) -> tuple[dict, int]:
            info_map = {}
            for entry in qualifying_grid:
                driver = entry["driver"]
                is_extreme_recovery = driver == "D22"
                info_map[driver] = {
                    "driver": driver,
                    "team": entry["team"],
                    "grid_pos": entry["position"],
                    "team_strength": 0.5,
                    "team_strength_by_compound": {"SOFT": 0.5, "MEDIUM": 0.5, "HARD": 0.5},
                    "tire_deg_by_compound": {"SOFT": 0.1, "MEDIUM": 0.1, "HARD": 0.1},
                    "skill": 1.0 if is_extreme_recovery else 0.5,
                    "race_advantage": 0.35 if is_extreme_recovery else 0.0,
                    "overtaking_skill": 1.0 if is_extreme_recovery else 0.5,
                    "defensive_skill": 0.5,
                    "dnf_probability": 0.0,
                }
            return info_map, 0

    predictor = ExtremeRecoveryPredictor()

    monkeypatch.setattr(
        prediction_module,
        "load_track_specific_params",
        lambda _race_name: {"track_overtaking": 0.05},
    )
    monkeypatch.setattr(prediction_module, "get_tire_stress_score", lambda _race_name: 3.0)
    monkeypatch.setattr(
        prediction_module,
        "get_available_compounds",
        lambda _race_name, weather="dry": ["SOFT", "MEDIUM", "HARD"],
    )
    monkeypatch.setattr(
        prediction_module,
        "resolve_race_distance_laps",
        lambda year, race_name, is_sprint: 60,
    )
    monkeypatch.setattr(
        prediction_module,
        "generate_pit_strategy",
        lambda **kwargs: {
            "num_stops": 0,
            "pit_laps": [],
            "compound_sequence": ["MEDIUM"],
            "stint_lengths": [60],
        },
    )
    monkeypatch.setattr(
        prediction_module,
        "simulate_race_lap_by_lap",
        lambda **kwargs: {
            "finish_order": [entry["driver"] for entry in kwargs["driver_info_map"].values()],
            "dnf_drivers": [],
            "strategies_used": kwargs["strategies"],
        },
    )

    def _fake_aggregate(_simulation_results):
        median_positions = {}
        position_distributions = {}
        for idx in range(1, 23):
            driver = f"D{idx:02d}"
            median = 1 if driver == "D22" else idx
            median_positions[driver] = median
            position_distributions[driver] = [median, median, median]
        return {
            "median_positions": median_positions,
            "position_distributions": position_distributions,
            "dnf_rates": {f"D{idx:02d}": 0.0 for idx in range(1, 23)},
            "compound_strategy_distribution": {"MEDIUM": 1.0},
            "pit_lap_distribution": {},
        }

    monkeypatch.setattr(prediction_module, "aggregate_simulation_results", _fake_aggregate)

    original_config_get = prediction_module.config_loader.get
    overrides = {
        "baseline_predictor.race.grid_anchor.base": 0.0,
        "baseline_predictor.race.grid_anchor.track_scale": 0.0,
        "baseline_predictor.race.grid_anchor.min": 0.0,
        "baseline_predictor.race.grid_anchor.sprint_min": 0.0,
        "baseline_predictor.race.final_blend.overtaking_skill_scale": 6.0,
        "baseline_predictor.race.final_blend.race_advantage_scale": 6.0,
        "baseline_predictor.race.final_blend.driver_skill_scale": 6.0,
        "baseline_predictor.race.final_blend.max_driver_adjustment_positions": 20.0,
    }

    def _config_get(key, default=None):
        if key in overrides:
            return overrides[key]
        return original_config_get(key, default)

    monkeypatch.setattr(prediction_module.config_loader, "get", _config_get)

    qualifying_grid = [
        {"driver": f"D{idx:02d}", "team": f"Team{idx:02d}", "position": idx} for idx in range(1, 23)
    ]

    result = predictor.predict_race(
        qualifying_grid=qualifying_grid,
        weather="dry",
        race_name="Bahrain Grand Prix",
        n_simulations=1,
    )

    positions = {entry["driver"]: entry["position"] for entry in result["finish_order"]}
    assert positions["D22"] > 1
    assert positions["D22"] >= 10
