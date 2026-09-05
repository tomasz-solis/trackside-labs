"""Tests for saved-actual team scoring helpers."""

import pytest

from src.predictors.baseline.data_support import score_teams_from_actual_rows


def test_score_teams_from_actual_rows_uses_rank_spacing():
    actual_rows = [
        {"team": "McLaren", "position": 1},
        {"team": "McLaren", "position": 2},
        {"team": "Ferrari", "position": 3},
        {"team": "Ferrari", "position": 4},
        {"team": "Mercedes", "position": 5},
        {"team": "Mercedes", "position": 6},
        {"team": "Aston Martin", "position": 7},
        {"team": "Aston Martin", "position": 8},
        {"team": "Haas F1 Team", "position": 9},
        {"team": "Haas F1 Team", "position": 10},
    ]

    scores = score_teams_from_actual_rows(
        actual_rows,
        known_teams={"McLaren", "Ferrari", "Mercedes", "Aston Martin", "Haas F1 Team"},
    )

    assert scores == {
        "McLaren": 1.0,
        "Ferrari": 0.75,
        "Mercedes": 0.5,
        "Aston Martin": 0.25,
        "Haas F1 Team": 0.0,
    }


def test_score_teams_from_actual_rows_returns_empty_for_single_resolvable_team():
    """A single team carries no relative ranking information, so it must not
    fabricate a neutral 0.5 score that then enters the season-form series."""
    scores = score_teams_from_actual_rows(
        [{"team": "Ferrari", "position": 2}],
        known_teams={"Ferrari"},
    )

    assert scores == {}


def test_score_teams_from_actual_rows_excludes_unknown_team_without_shifting_known_teams():
    """An unresolved team name must not take a rank slot that shifts real teams' scores."""
    actual_rows = [
        {"team": "McLaren", "position": 1},
        {"team": "Ferrari", "position": 2},
        {"team": "Williams", "position": 3},
        {"team": "Some New Entrant", "position": 4},
    ]
    known_teams = {"McLaren", "Ferrari", "Williams"}

    scores = score_teams_from_actual_rows(actual_rows, known_teams=known_teams)

    assert scores == {"McLaren": 1.0, "Ferrari": 0.5, "Williams": 0.0}
    assert "Some New Entrant" not in scores


def test_score_teams_from_actual_rows_returns_empty_when_every_team_unresolved():
    scores = score_teams_from_actual_rows(
        [{"team": "Unknown One", "position": 1}, {"team": "Unknown Two", "position": 2}],
        known_teams={"McLaren", "Ferrari"},
    )

    assert scores == {}


def test_score_teams_from_actual_rows_counts_invalid_positions():
    actual_rows = [
        {"team": "McLaren", "position": 1},
        {"team": "McLaren", "position": None},
        {"team": "Ferrari", "position": 0},
        {"team": "Ferrari", "position": 2},
    ]
    known_teams = {"McLaren", "Ferrari"}

    scores = score_teams_from_actual_rows(actual_rows, known_teams=known_teams)

    assert scores == {"McLaren": 1.0, "Ferrari": 0.0}


def test_score_teams_from_actual_rows_without_dnf_key_is_unaffected():
    """Rows carrying no ``dnf`` key (every artifact predating the flag) must score
    identically to before the DNF-exclusion branch was added."""
    actual_rows = [
        {"team": "McLaren", "position": 1},
        {"team": "McLaren", "position": 2},
        {"team": "Ferrari", "position": 3},
        {"team": "Ferrari", "position": 4},
    ]
    known_teams = {"McLaren", "Ferrari"}

    assert score_teams_from_actual_rows(actual_rows, known_teams=known_teams) == {
        "McLaren": 1.0,
        "Ferrari": 0.0,
    }


@pytest.mark.parametrize(
    "dnf_signal",
    [
        {"dnf": True},
        {"status": "Retired"},
        {"classified": False},
    ],
    ids=["dnf_flag", "status_string", "classified_flag"],
)
def test_score_teams_from_actual_rows_excludes_retirements_from_team_mean(dnf_signal):
    """A retirement classified far down the order must not drag its team's mean,
    regardless of which of the three shapes `row_is_dnf` recognizes it flags the DNF."""
    actual_rows = [
        {"team": "McLaren", "position": 1},
        {"team": "McLaren", "position": 20, **dnf_signal},
        {"team": "Ferrari", "position": 2},
        {"team": "Ferrari", "position": 3},
    ]
    known_teams = {"McLaren", "Ferrari"}

    scores = score_teams_from_actual_rows(actual_rows, known_teams=known_teams)

    # McLaren's classified mean (position 1) beats Ferrari's (mean 2.5), even though
    # the raw (unfiltered) McLaren mean of 10.5 would have lost.
    assert scores == {"McLaren": 1.0, "Ferrari": 0.0}


@pytest.mark.parametrize(
    "dnf_signal",
    [
        {"dnf": True},
        {"status": "Retired"},
        {"classified": False},
    ],
    ids=["dnf_flag", "status_string", "classified_flag"],
)
def test_score_teams_from_actual_rows_keeps_all_retired_team_rows(dnf_signal):
    """A team with no classified finishers keeps its raw rows rather than dropping out."""
    actual_rows = [
        {"team": "McLaren", "position": 1, **dnf_signal},
        {"team": "McLaren", "position": 2, **dnf_signal},
        {"team": "Ferrari", "position": 3},
        {"team": "Ferrari", "position": 4},
    ]
    known_teams = {"McLaren", "Ferrari"}

    scores = score_teams_from_actual_rows(actual_rows, known_teams=known_teams)

    assert scores == {"McLaren": 1.0, "Ferrari": 0.0}
