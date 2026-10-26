"""Car traits x track composition adjust measured team race pace."""

import numpy as np
import pytest

from src.extractors import car_track_traits as ctt
from src.models import track_traits as tt
from src.utils import lap_by_lap_simulator as sim

TEAMS = ["A", "B", "C", "D", "E", "F"]
TOP_SPEED = {"A": 8.0, "B": 4.0, "C": 0.0, "D": -2.0, "E": -4.0, "F": -6.0}


def _season(n_races: int, seed: int = 0):
    """Synthetic season: teams with more top speed gain on full-throttle tracks."""
    rng = np.random.default_rng(seed)
    traits, pace = {}, {}
    for i in range(n_races):
        race = f"R{i}"
        share = 0.35 + 0.35 * rng.random()
        profile = {k: 0.1 for k in ("slow", "medium", "fast", "braking", "deg_severity")}
        profile["full_throttle"] = share
        traits[race] = {
            "traits": {
                t: {
                    "top_speed": TOP_SPEED[t],
                    "slow": 0.0,
                    "medium": 0.0,
                    "fast": 0.0,
                    "braking": 0.0,
                    "deg": 0.0,
                }
                for t in TEAMS
            },
            "profile": profile,
        }
        # gap: smaller is faster; top-speed teams gain 0.1 s/lap per km/h per unit share.
        pace[race] = {
            t: 1.0 - 0.1 * TOP_SPEED[t] * (share - 0.5) + rng.normal(0, 0.01) for t in TEAMS
        }
    return traits, pace


def test_model_recovers_a_planted_top_speed_effect():
    traits, pace = _season(10)
    fast_track = {**traits["R0"]["profile"], "full_throttle": 0.75}

    adj = tt.fit_and_predict(traits, pace, [f"R{i}" for i in range(10)], fast_track)

    assert adj["A"] > adj["C"] > adj["F"]  # most top speed gains most at a fast track
    assert sum(adj.values()) == pytest.approx(0.0, abs=1e-9)


def test_no_adjustment_with_too_few_races_or_no_profile():
    traits, pace = _season(10)
    profile = traits["R0"]["profile"]

    assert tt.fit_and_predict(traits, pace, ["R0", "R1", "R2"], profile) == {}
    assert tt.fit_and_predict(traits, pace, [f"R{i}" for i in range(10)], None) == {}


def test_adjustment_is_added_to_measured_pace(monkeypatch):
    monkeypatch.setattr(
        sim, "_load_track_trait_adjustments", lambda year, race: {"A": 0.2, "B": -0.2}
    )

    adjusted = sim._with_track_trait_adjustment({"A": 0.5, "B": -0.5, "C": 0.0}, 2026, "X")

    assert adjusted == pytest.approx({"A": 0.7, "B": -0.7, "C": 0.0})
    assert sim._with_track_trait_adjustment(None, 2026, "X") is None  # round 1: nothing to adjust


def test_refresh_measures_only_missing_races_and_the_upcoming_track(monkeypatch):
    class _Store:
        def __init__(self):
            self.payload = {"races": {"R0": {"traits": {}, "profile": {"deg_severity": 0.04}}}}
            self.saved = 0

        def load_artifact(self, artifact_type, artifact_key):
            return self.payload

        def save_artifact(self, artifact_type, artifact_key, data):
            self.saved += 1
            self.payload = data

    measured = []
    monkeypatch.setattr(
        ctt,
        "measure_completed_race",
        lambda y, r: measured.append(r) or {"traits": {}, "profile": {"deg_severity": 0.02}},
    )
    monkeypatch.setattr(
        ctt,
        "measure_weekend_profile",
        lambda y, r, newer_than=None, sessions=None: (
            None if newer_than == "FP1" else {"full_throttle": 0.5, "session": "FP1"}
        ),
    )
    monkeypatch.setattr(ctt, "measure_practice_gaps", lambda y, r, have, sessions=None: {})
    store = _Store()

    added = ctt.refresh_car_track_traits(2026, ["R0", "R1"], upcoming_race="R2", store=store)

    assert added == {"races": ["R1"], "profiles": ["R2"], "fp_gaps": []}
    assert measured == ["R1"]
    assert store.payload["profiles"]["R2"]["deg_severity"] == pytest.approx(
        0.03
    )  # season median stands in
    assert ctt.refresh_car_track_traits(2026, ["R0", "R1"], upcoming_race="R2", store=store) == {
        "races": [],
        "profiles": [],
        "fp_gaps": [],
    }
    assert store.saved == 1


def test_weekend_profile_moves_to_the_latest_dry_session(monkeypatch):
    from types import SimpleNamespace

    import pandas as pd

    from src.data import openf1_session

    rain = {"FP2": [False], "FP3": [True]}  # FP3 wet, FP2 dry, Q not run yet
    loads: list[tuple[str, bool]] = []

    def _get_session(year, race_name, code, telemetry=False, weather=False):
        if code not in rain:
            raise RuntimeError("not run yet")
        loads.append((code, telemetry))
        return SimpleNamespace(
            laps=pd.DataFrame({"LapTime": [90.0]}),
            weather_data=pd.DataFrame({"Rainfall": rain[code]}),
        )

    monkeypatch.setattr(openf1_session, "get_session", _get_session)
    monkeypatch.setattr(ctt, "track_profile", lambda session: {"full_throttle": 0.6})

    assert ctt.measure_weekend_profile(2026, "X") == {"full_throttle": 0.6, "session": "FP2"}
    assert ("FP3", True) not in loads  # the wet session never loads car data
    loads.clear()
    assert ctt.measure_weekend_profile(2026, "X", newer_than="FP2") is None
    assert all(not telemetry for _, telemetry in loads)  # nothing newer: no car data


def test_replay_target_uses_earlier_races_deg_not_its_own(monkeypatch):
    from src.utils import weekend

    traits, pace = _season(8)
    for i, race in enumerate(traits):
        traits[race]["profile"]["deg_severity"] = 0.01 * i
        for team in TEAMS:
            traits[race]["traits"][team]["deg"] = float(TEAMS.index(team))
    traits["R7"]["profile"]["deg_severity"] = 5.0  # only the race itself could know this
    # A profile stored live before the race; the replay keeps the measured geometry, as
    # the baselines it is compared with did.
    live = {key: value + 1.0 for key, value in traits["R7"]["profile"].items()}
    data = {"races": traits, "profiles": {"R7": live}, "fp_gaps": {}}
    monkeypatch.setattr(ctt, "load_car_track_traits", lambda year: data)
    monkeypatch.setattr(weekend, "get_schedule_rows", lambda year: [(r, "x") for r in traits])
    monkeypatch.setattr(
        "src.extractors.team_race_pace.load_team_race_pace", lambda year: {"races": pace}
    )
    tt.track_trait_adjustments.cache_clear()

    adjusted = tt.track_trait_adjustments(2026, "R7", "race")
    prior = [f"R{i}" for i in range(7)]
    pre_weekend = {**traits["R7"]["profile"], "deg_severity": 0.03}  # median of 0.00 to 0.06

    assert adjusted == pytest.approx(tt.fit_and_predict(traits, pace, prior, pre_weekend))
    tt.track_trait_adjustments.cache_clear()


def test_switch_is_off_by_default():
    from src.utils import config_loader

    assert config_loader.get("baseline_predictor.race.track_trait_adjustment") is False


def test_team_quali_gaps_uses_each_teams_best_lap():
    from types import SimpleNamespace

    import pandas as pd

    laps = pd.DataFrame(
        {
            "Driver": ["X1", "X2", "Y1", "Y2"],
            "LapTime": pd.to_timedelta([80.5, 80.2, 81.0, 80.9], unit="s"),
        }
    )
    team_of = {"X1": "X", "X2": "X", "Y1": "Y", "Y2": "Y"}

    gaps = ctt.team_quali_gaps(SimpleNamespace(laps=laps), team_of)

    assert gaps == {"X": 0.0, "Y": pytest.approx(0.7)}


def test_qualifying_adjustment_moves_team_seconds(monkeypatch):
    from src.predictors.baseline import qualifying_preparation as qp

    monkeypatch.setattr(
        "src.models.track_traits.track_trait_adjustments",
        lambda year, race, kind, features=None: {"A": 0.1} if kind == "qualifying" else {},
    )
    records = [
        {"team": "A", "team_strength_seconds_delta": 0.3},
        {"team": "B", "team_strength_seconds_delta": -0.2},
        {"team": "C"},  # no seconds mapping: untouched
    ]

    qp.apply_track_trait_adjustment(records, 2026, "Italian Grand Prix")

    assert records[0]["team_strength_seconds_delta"] == pytest.approx(0.4)
    assert records[1]["team_strength_seconds_delta"] == pytest.approx(-0.2)
    assert "team_strength_seconds_delta" not in records[2]


def test_qualifying_switch_is_on_since_model_32():
    from src.utils import config_loader

    assert config_loader.get("baseline_predictor.qualifying.track_trait_adjustment") is True


def test_unmeasurable_traits_are_saved_as_null_without_a_warning(monkeypatch, caplog):
    import logging

    class _Store:
        def __init__(self):
            self.payload = {"races": {}}
            self.saved = None

        def load_artifact(self, artifact_type, artifact_key):
            return self.payload

        def save_artifact(self, artifact_type, artifact_key, data):
            self.saved = data

    monkeypatch.setattr(
        ctt,
        "measure_completed_race",
        lambda y, r: {"traits": {"A": {"fast": float("nan"), "top_speed": 3.0}}, "profile": {}},
    )
    monkeypatch.setattr(ctt, "measure_practice_gaps", lambda y, r, have, sessions=None: {})
    store = _Store()

    with caplog.at_level(logging.INFO):
        ctt.refresh_car_track_traits(2026, ["R1"], store=store)

    assert store.saved["races"]["R1"]["traits"]["A"]["fast"] is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "1 car trait value(s) not measurable" in caplog.text


def test_feature_list_none_equals_all_six_and_dropping_one_changes_the_fit():
    traits, pace = _season(10)
    for race in traits.values():  # give braking its own varying signal
        for i, team in enumerate(TEAMS):
            race["traits"][team]["braking"] = float(i % 3)
        race["profile"]["braking"] = race["profile"]["full_throttle"] / 2
    prior = [f"R{k}" for k in range(10)]
    target = {**traits["R0"]["profile"], "full_throttle": 0.75, "braking": 0.5}

    default = tt.fit_and_predict(traits, pace, prior, target)
    explicit = tt.fit_and_predict(traits, pace, prior, target, tuple(tt.TRAIT_TO_SHARE))
    without = tt.fit_and_predict(
        traits, pace, prior, target, ("top_speed", "slow", "medium", "fast", "deg")
    )

    assert default == pytest.approx(explicit)
    assert without != pytest.approx(default)


def test_live_config_drops_braking_since_model_33():
    from src.utils import config_loader

    features = set(config_loader.get("baseline_predictor.qualifying.track_trait_features"))
    assert features == set(tt.TRAIT_TO_SHARE) - {"braking"}


def test_weekend_profile_and_practice_gaps_skip_sessions_not_run_at_the_checkpoint(monkeypatch):
    from types import SimpleNamespace

    import pandas as pd

    from src.data import openf1_session

    loads: list[str] = []

    def _get_session(year, race_name, code, telemetry=False, weather=False):
        loads.append(code)  # a finished season: every session loads
        return SimpleNamespace(
            laps=pd.DataFrame({"LapTime": [90.0]}),
            weather_data=pd.DataFrame({"Rainfall": [False]}),
        )

    monkeypatch.setattr(openf1_session, "get_session", _get_session)
    monkeypatch.setattr(ctt, "track_profile", lambda session: {"full_throttle": 0.6})
    monkeypatch.setattr(ctt, "team_quali_gaps", lambda session, team_of: {"A": 0.0})
    monkeypatch.setattr(ctt, "_team_of", lambda session: {})

    profile = ctt.measure_weekend_profile(2026, "X", sessions={"FP1", "FP2"})
    gaps = ctt.measure_practice_gaps(2026, "X", set(), sessions={"FP1", "FP2"})

    assert profile == {"full_throttle": 0.6, "session": "FP2"}
    assert set(gaps) == {"FP1", "FP2"}
    assert not {"Q", "SQ", "FP3"} & set(loads)


def test_refresh_limits_only_the_upcoming_race_to_the_sessions_done(monkeypatch):
    class _Store:
        payload = {"races": {"R0": {"traits": {}, "profile": {"deg_severity": 0.0}}}}

        def load_artifact(self, artifact_type, artifact_key):
            return self.payload

        def save_artifact(self, artifact_type, artifact_key, data):
            self.payload = data

    practice_calls = []
    monkeypatch.setattr(
        ctt,
        "measure_practice_gaps",
        lambda y, r, have, sessions=None: practice_calls.append((r, sessions)) or {},
    )
    profile_calls = []
    monkeypatch.setattr(
        ctt,
        "measure_weekend_profile",
        lambda y, r, newer_than=None, sessions=None: profile_calls.append((r, sessions)),
    )

    ctt.refresh_car_track_traits(
        2026, ["R0"], upcoming_race="R1", store=_Store(), sessions_done=["FP1"]
    )

    assert practice_calls == [("R0", None), ("R1", ["FP1"])]
    assert profile_calls == [("R1", ["FP1"])]


def test_replay_root_never_falls_back_to_the_committed_file(monkeypatch, tmp_path):
    from src.utils.historical_replay import _reset_replay_track_traits

    monkeypatch.setenv("USE_DB_STORAGE", "file_only")
    monkeypatch.setenv(ctt.CAR_TRACK_TRAITS_ROOT_ENV, str(tmp_path))
    assert ctt.load_car_track_traits(2026)["races"] == {}  # no file at all

    _reset_replay_track_traits(tmp_path / "processed", 2026)

    assert ctt.load_car_track_traits(2026) == {"races": {}, "profiles": {}, "fp_gaps": {}}


def test_saving_traits_clears_the_fits_cached_from_them(monkeypatch):
    from src.utils import weekend

    class _Store:
        def load_artifact(self, artifact_type, artifact_key):
            return {"races": {}}

        def save_artifact(self, artifact_type, artifact_key, data):
            pass

    monkeypatch.setattr(weekend, "get_schedule_rows", lambda year: [])
    monkeypatch.setattr(ctt, "measure_completed_race", lambda y, r: {"traits": {}, "profile": {}})
    monkeypatch.setattr(ctt, "measure_practice_gaps", lambda y, r, have, sessions=None: {})
    tt.track_trait_adjustments.cache_clear()
    tt.track_trait_adjustments(2026, "R1", "qualifying")
    assert tt.track_trait_adjustments.cache_info().currsize == 1

    ctt.refresh_car_track_traits(2026, ["R0"], store=_Store())

    assert tt.track_trait_adjustments.cache_info().currsize == 0
    for module_name, function_name in ctt._DERIVED_CACHES:  # a rename must not slip by
        module = __import__(module_name, fromlist=[function_name])
        assert callable(getattr(module, function_name).cache_clear)
