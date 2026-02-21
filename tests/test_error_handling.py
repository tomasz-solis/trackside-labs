"""Negative-path tests for validation and error handling."""

import pytest

from src.predictors.baseline.race.preparation_mixin import BaselineRacePreparationMixin


class _DummyPreparation(BaselineRacePreparationMixin):
    def __init__(self):
        self.teams = {"McLaren": {"uncertainty": 0.2, "compound_characteristics": {}}}
        self.drivers = {}

    def get_blended_team_strength(self, team: str, race_name: str) -> float:
        return 0.5

    def _compute_testing_profile_modifier(
        self,
        team: str,
        profile: str,
        metric_weights: dict[str, float],
        scale: float,
    ) -> tuple[float, bool]:
        return 0.0, False


def test_prepare_driver_info_with_unknown_driver_raises():
    prep = _DummyPreparation()

    with pytest.raises(ValueError, match="Driver .* not found"):
        prep._prepare_driver_info_with_compounds(
            qualifying_grid=[{"driver": "UNKNOWN", "team": "McLaren", "position": 1}],
            race_name="Bahrain Grand Prix",
        )
