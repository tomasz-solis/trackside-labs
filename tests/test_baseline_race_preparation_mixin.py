from __future__ import annotations

import json

import pytest

import src.predictors.baseline.race.preparation_mixin as prep_module
from src.predictors.baseline.race.preparation_mixin import BaselineRacePreparationMixin


class DummyPreparation(BaselineRacePreparationMixin):
    def __init__(self):
        self.teams = {}
        self.drivers = {}
        self.compound_strength = 0.9
        self.blended_strength = 0.7
        self.profile_modifier = (0.0, False)

    def get_compound_adjusted_team_strength(
        self, team: str, race_name: str, race_compound: str
    ) -> float:
        return self.compound_strength

    def get_blended_team_strength(self, team: str, race_name: str) -> float:
        return self.blended_strength

    def _compute_testing_profile_modifier(
        self, team: str, profile: str, metric_weights: dict, scale: float
    ):
        return self.profile_modifier


def test_load_track_overtaking_difficulty_from_file_and_fallbacks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    track_dir = tmp_path / "data" / "processed" / "track_characteristics"
    track_dir.mkdir(parents=True)
    track_file = track_dir / "2026_track_characteristics.json"
    track_file.write_text(
        json.dumps({"tracks": {"Bahrain Grand Prix": {"overtaking_difficulty": 0.82}}})
    )

    prep = DummyPreparation()
    monkeypatch.setattr(prep_module, "validate_track_characteristics", lambda payload: None)

    assert prep._load_track_overtaking_difficulty(None) == 0.5
    assert prep._load_track_overtaking_difficulty("Bahrain Grand Prix") == 0.82
    assert prep._load_track_overtaking_difficulty("Unknown Race") == 0.5

    track_file.write_text("{bad json")
    assert prep._load_track_overtaking_difficulty("Bahrain Grand Prix") == 0.5


def test_load_track_overtaking_difficulty_handles_schema_validation_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    track_dir = tmp_path / "data" / "processed" / "track_characteristics"
    track_dir.mkdir(parents=True)
    (track_dir / "2026_track_characteristics.json").write_text(json.dumps({"tracks": {}}))

    prep = DummyPreparation()
    monkeypatch.setattr(
        prep_module,
        "validate_track_characteristics",
        lambda payload: (_ for _ in ()).throw(ValueError("schema error")),
    )

    assert prep._load_track_overtaking_difficulty("Bahrain Grand Prix") == 0.5


def test_prepare_driver_info_applies_caps_and_profile_modifiers(monkeypatch):
    prep = DummyPreparation()
    prep.compound_strength = 0.9
    prep.profile_modifier = (0.25, True)
    prep.teams = {"McLaren": {"overall_performance": 0.55, "uncertainty": 0.50}}
    prep.drivers = {
        "NOR": {
            "pace": {"quali_pace": 0.6, "race_pace": 0.7},
            "racecraft": {"skill_score": 0.8, "overtaking_skill": 0.7},
            "dnf_risk": {"dnf_rate": 0.6},
            "experience": {"tier": "rookie"},
        }
    }

    monkeypatch.setattr(
        prep_module.config_loader,
        "get",
        lambda key, default: {
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.33,
            "baseline_predictor.race.testing_long_run_modifier_scale": 0.05,
        }.get(key, default),
    )

    info_map, long_profile_count = prep._prepare_driver_info(
        qualifying_grid=[{"driver": "NOR", "team": "McLaren", "position": 1}],
        race_name="Bahrain Grand Prix",
        race_compound="SOFT",
    )

    info = info_map["NOR"]
    assert long_profile_count == 1
    assert info["team_strength"] == 1.0
    assert info["race_advantage"] == pytest.approx(0.1)
    assert info["defensive_skill"] == pytest.approx(0.735)
    assert info["dnf_probability"] == pytest.approx(0.33)


def test_prepare_driver_info_with_compounds_builds_per_compound_strengths(monkeypatch):
    prep = DummyPreparation()
    prep.blended_strength = 0.7
    prep.profile_modifier = (0.1, True)
    prep.teams = {
        "McLaren": {
            "uncertainty": 0.2,
            "compound_characteristics": {
                "SOFT": {
                    "tire_deg_slope": 0.20,
                }
            },
        }
    }
    prep.drivers = {
        "NOR": {
            "pace": {"quali_pace": 0.6, "race_pace": 0.65},
            "racecraft": {"skill_score": 0.7, "overtaking_skill": 0.6},
            "dnf_risk": {"dnf_rate": 0.12},
            "experience": {"tier": "established"},
        }
    }

    monkeypatch.setattr(
        prep_module.config_loader,
        "get",
        lambda key, default: {
            "baseline_predictor.race.tire_physics.default_deg_slope": 0.12,
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.35,
            "baseline_predictor.race.testing_long_run_modifier_scale": 0.05,
        }.get(key, default),
    )
    monkeypatch.setattr(
        "src.utils.compound_performance.get_compound_performance_modifier",
        lambda team_compound_chars, compound: 0.05 if compound == "SOFT" else 0.0,
    )

    info_map, long_profile_count = prep._prepare_driver_info_with_compounds(
        qualifying_grid=[{"driver": "NOR", "team": "McLaren", "position": 2}],
        race_name="Bahrain Grand Prix",
    )

    info = info_map["NOR"]
    assert long_profile_count == 1
    assert info["team_strength"] == pytest.approx(0.8)
    assert info["team_strength_by_compound"]["SOFT"] == pytest.approx(0.85)
    assert info["team_strength_by_compound"]["MEDIUM"] == pytest.approx(0.8)
    assert info["team_strength_by_compound"]["HARD"] == pytest.approx(0.8)
    assert info["tire_deg_by_compound"]["SOFT"] == pytest.approx(0.2)
    assert info["tire_deg_by_compound"]["MEDIUM"] == pytest.approx(0.12)


def test_get_driver_data_or_fallback_uses_teammate_profile_for_missing_lineup_driver(monkeypatch):
    prep = DummyPreparation()
    prep.drivers = {
        "LAW": {
            "pace": {"quali_pace": 0.70, "race_pace": 0.65},
            "racecraft": {"skill_score": 0.62, "overtaking_skill": 0.60},
            "dnf_risk": {"dnf_rate": 0.03},
            "experience": {"tier": "developing"},
        }
    }

    monkeypatch.setattr(
        "src.utils.lineups.load_current_lineups",
        lambda: {"RB": ["LAW", "LIN"]},
    )
    monkeypatch.setattr(
        prep_module.config_loader,
        "get",
        lambda key, default: {
            "baseline_predictor.qualifying.default_skill": 0.5,
            "baseline_predictor.race.missing_driver_teammate_weight": 0.80,
            "baseline_predictor.race.missing_driver_default_dnf_rate": 0.10,
            "baseline_predictor.race.missing_driver_rookie_dnf_penalty": 0.02,
            "baseline_predictor.race.missing_driver_rookie_quali_penalty": 0.08,
            "baseline_predictor.race.missing_driver_rookie_race_penalty": 0.07,
            "baseline_predictor.race.missing_driver_rookie_skill_penalty": 0.08,
            "baseline_predictor.race.missing_driver_rookie_overtaking_penalty": 0.06,
        }.get(key, default),
    )

    fallback = prep._get_driver_data_or_fallback("LIN", "RB")

    assert fallback["pace"]["quali_pace"] == pytest.approx(0.58)
    assert fallback["pace"]["race_pace"] == pytest.approx(0.55)
    assert fallback["racecraft"]["skill_score"] == pytest.approx(0.516)
    assert fallback["racecraft"]["overtaking_skill"] == pytest.approx(0.52)
    assert fallback["dnf_risk"]["dnf_rate"] == pytest.approx(0.05)
    assert fallback["experience"]["tier"] == "rookie"
    assert prep.drivers["LIN"] == fallback


def test_infer_missing_driver_experience_tier_uses_debuts_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "driver_debuts.csv").write_text(
        "Driver,First F1 season,Notes\n"
        "Lewis Hamilton,2007,McLaren debut\n"
        "Arvid Lindblad,2026,Racing Bulls debut\n"
    )

    prep = DummyPreparation()
    prep.year = 2026

    assert prep._infer_missing_driver_experience_tier("HAM") == "veteran"
    assert prep._infer_missing_driver_experience_tier("LIN") == "rookie"
    assert prep._infer_missing_driver_experience_tier("ZZZ") == "rookie"


def test_infer_missing_driver_experience_tier_prefers_artifact_store():
    class StubStore:
        def load_artifact(self, artifact_type: str, artifact_key: str):
            assert artifact_type == "driver_debuts"
            assert artifact_key == "driver_debuts"
            return {"driver_debuts": {"HAM": 2007, "LIN": 2026}}

    prep = DummyPreparation()
    prep.year = 2026
    prep.artifact_store = StubStore()

    assert prep._infer_missing_driver_experience_tier("HAM") == "veteran"
    assert prep._infer_missing_driver_experience_tier("LIN") == "rookie"
    assert prep._infer_missing_driver_experience_tier("ZZZ") == "rookie"
