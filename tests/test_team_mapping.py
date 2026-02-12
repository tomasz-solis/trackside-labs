"""Tests for team mapping helpers."""

from src.utils.team_mapping import map_team_to_characteristics


def test_map_team_to_characteristics_known_alias():
    """Known aliases should map to characteristics naming."""
    assert map_team_to_characteristics("Oracle Red Bull Racing") == "Red Bull Racing"
    assert map_team_to_characteristics("Scuderia Ferrari") == "Ferrari"


def test_map_team_to_characteristics_respects_known_teams():
    """When known teams are provided, mapping should resolve into that set."""
    known = {"Red Bull", "McLaren"}
    assert map_team_to_characteristics("Oracle Red Bull Racing", known_teams=known) == "Red Bull"
    assert map_team_to_characteristics("Unknown Team", known_teams=known) is None
