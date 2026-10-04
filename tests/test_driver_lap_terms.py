"""Driver lap-time terms can be centred within each team (C3, arm L)."""

import pytest

from src.utils import config_loader
from src.utils.lap_by_lap_simulator import _center_by_team


def test_centering_keeps_the_teammate_gap_and_drops_the_team_average():
    info = {
        "VER": {"team": "Red Bull Racing"},
        "HAD": {"team": "Red Bull Racing"},
        "RUS": {"team": "Mercedes"},
        "ANT": {"team": "Mercedes"},
    }
    bonus = {"VER": 2.6, "HAD": 1.4, "RUS": 0.8, "ANT": 1.0}

    centered = _center_by_team(bonus, info)

    assert centered["VER"] - centered["HAD"] == pytest.approx(1.2)
    assert centered["ANT"] - centered["RUS"] == pytest.approx(0.2)
    assert centered["VER"] + centered["HAD"] == pytest.approx(0.0)
    assert centered["RUS"] + centered["ANT"] == pytest.approx(0.0)


def test_centering_is_off_by_default():
    assert config_loader.get("baseline_predictor.race.center_driver_lap_terms_by_team") is False
