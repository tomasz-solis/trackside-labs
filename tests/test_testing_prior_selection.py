"""A team whose car changed over the winter drops last season's testing prior; others keep it."""

import json

import pandas as pd

from src.systems.testing_prior_selection import (
    apply_testing_prior_choice,
    choose_resets,
    track_session,
)

_TEAMS = ["Alpine", "Aston Martin", "Audi", "Cadillac", "Ferrari", "Haas"]
_PRIOR = {"Alpine": 0.30, "Aston Martin": 0.40, "Audi": 0.20, "Cadillac": 0.10, "Ferrari": 0.80}
_PRIOR["Haas"] = 0.50
# Small session-to-session noise around each team's true 2027 pace.
_NOISE = [0.02, -0.03, 0.01, -0.01, 0.03]


def _run_sessions(true_pace: dict[str, float], sessions: int) -> dict[str, dict]:
    """Blend sessions like the stored profile does, and track both options."""
    teams: dict[str, dict] = {}
    for team in _TEAMS:
        team_data = teams.setdefault(team, {})
        stored = {"overall_pace": _PRIOR[team]}
        for index in range(sessions):
            observed = {"overall_pace": true_pace[team] + _NOISE[index]}
            track_session(team_data, "short_run", observed, stored, session_count=1)
            weight = 0.25 / (index + 1) ** 0.5
            stored = {
                "overall_pace": (1 - weight) * stored["overall_pace"]
                + weight * observed["overall_pace"]
            }
    return teams


def test_a_team_that_changed_a_lot_drops_the_prior_and_the_others_keep_it():
    rocketship = {**_PRIOR, "Audi": 0.85}

    assert choose_resets(_run_sessions(rocketship, sessions=5)) == {"short_run": ["Audi"]}


def test_no_team_drops_the_prior_when_no_car_changed():
    assert choose_resets(_run_sessions(_PRIOR, sessions=5)) == {}


def test_a_car_that_got_slower_drops_the_prior_too():
    assert choose_resets(_run_sessions({**_PRIOR, "Ferrari": 0.2}, sessions=5)) == {
        "short_run": ["Ferrari"]
    }


def test_two_sessions_are_not_enough_to_drop_the_prior():
    assert choose_resets(_run_sessions({**_PRIOR, "Audi": 0.85}, sessions=2)) == {}


def test_overlay_swaps_in_this_seasons_own_profile_for_the_chosen_team_only():
    teams = _run_sessions({**_PRIOR, "Audi": 0.85}, sessions=5)
    for team in _TEAMS:
        teams[team]["overall_performance"] = 0.5
        teams[team]["testing_characteristics_profiles"] = {
            "short_run": {"overall_pace": _PRIOR[team], "sessions_blended": 5}
        }
    payload = {"year": 2027, "seeded_from": 2026, "teams": teams}

    result = apply_testing_prior_choice(payload)

    audi = result["teams"]["Audi"]["testing_characteristics_profiles"]["short_run"]
    assert (
        audi["overall_pace"]
        == teams["Audi"]["testing_reset"]["short_run"]["metrics"]["overall_pace"]
    )
    assert audi["overall_pace"] > 0.8
    assert audi["sessions_blended"] == 5
    haas = result["teams"]["Haas"]["testing_characteristics_profiles"]["short_run"]
    assert haas["overall_pace"] == _PRIOR["Haas"]
    # The stored payload is not changed, and a season that was not seeded is left alone.
    assert (
        payload["teams"]["Audi"]["testing_characteristics_profiles"]["short_run"]["overall_pace"]
        == 0.20
    )
    unseeded = {"year": 2027, "teams": teams}
    assert apply_testing_prior_choice(unseeded) is unseeded


def test_session_updates_track_both_options_only_in_a_seeded_season(tmp_path, patcher):
    from src.systems import testing_updater

    data_dir = tmp_path / "data" / "processed" / "car_characteristics"
    data_dir.mkdir(parents=True)
    characteristics_file = data_dir / "2027_car_characteristics.json"

    class DummySession:
        def __init__(self, event_name: str):
            self.event_name = event_name
            self.laps = pd.DataFrame(
                {
                    "Team": ["Audi"] * 6,
                    "Driver": ["HUL"] * 6,
                    "LapTime": [pd.to_timedelta(f"0:01:{30 + i:02d}") for i in range(6)],
                }
            )

    patcher.setattr(
        testing_updater,
        "_load_sessions_for_event",
        lambda **kwargs: [("FP1", DummySession(kwargs["event_name"]))],
    )
    patcher.setattr(
        testing_updater,
        "_collect_session_metrics",
        lambda **kwargs: ({"Audi": {"overall_pace": 0.9}}, {}),
    )
    patcher.setattr(
        testing_updater,
        "_count_team_selected_laps",
        lambda session, known_teams, run_profile: {"Audi": 8.0},
    )

    def run(seeded: bool) -> dict:
        payload = {
            "year": 2027,
            "teams": {
                "Audi": {
                    "overall_performance": 0.3,
                    "testing_characteristics": {"overall_pace": 0.2},
                    "testing_characteristics_profiles": {"short_run": {"overall_pace": 0.2}},
                }
            },
        }
        if seeded:
            payload["seeded_from"] = 2026
        characteristics_file.write_text(json.dumps(payload))
        for event in ("Bahrain Grand Prix", "Australian Grand Prix"):
            testing_updater.update_from_testing_sessions(
                year=2027,
                events=[event],
                data_dir=str(tmp_path / "data" / "processed"),
                new_weight=0.25,
                dry_run=False,
            )
        return json.loads(characteristics_file.read_text())["teams"]["Audi"]

    audi = run(seeded=True)
    short_run = audi["testing_reset"]["short_run"]
    assert short_run["metrics"]["overall_pace"] == 0.9
    assert short_run["sessions"] == 2
    # The prior is last season's value, kept from before the first session blended in.
    assert short_run["prior"] == 0.2
    assert short_run["observed"] == [0.9, 0.9]
    assert "testing_reset" not in run(seeded=False)
