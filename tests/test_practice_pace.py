"""This weekend's practice adjusts race pace, using only sessions the checkpoint may see."""

import numpy as np
import pytest

from src.models import practice_pace as pp
from src.utils import lap_by_lap_simulator as sim
from src.utils.prediction_context import (
    PredictionContext,
    activate_prediction_runtime,
    build_historical_prediction_context,
)

TEAMS = ["A", "B", "C", "D", "E"]


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("PRE", ()),
        ("FP1", ("FP1",)),
        ("FP2", ("FP1", "FP2")),
        ("FP3", ("FP1", "FP2", "FP3")),
        ("Q", ("FP1", "FP2", "FP3")),
    ],
)
def test_checkpoint_sees_only_completed_practice(checkpoint, expected):
    assert pp.allowed_sessions(checkpoint, ["FP1", "FP2", "FP3"]) == expected


def test_live_uses_whatever_is_stored():
    assert pp.allowed_sessions(None, ["FP1", "FP2"]) == ("FP1", "FP2")


def _season(n: int, seed: int = 0):
    """Synthetic season: practice gap vs usual pace predicts the race deviation."""
    rng = np.random.default_rng(seed)
    usual = {t: float(i) * 0.3 for i, t in enumerate(TEAMS)}
    pace, fp = {}, {}
    for k in range(n):
        form = {t: rng.normal(0, 0.2) for t in TEAMS}  # this weekend's real form
        pace[f"R{k}"] = {t: usual[t] - form[t] + rng.normal(0, 0.02) for t in TEAMS}
        fp[f"R{k}"] = {"FP1": {t: usual[t] - form[t] + rng.normal(0, 0.05) for t in TEAMS}}
    return pace, fp


def test_model_recovers_a_planted_practice_signal():
    pace, fp = _season(10)
    target_fp = {"FP1": {"A": -0.5, "B": 0.3, "C": 0.6, "D": 0.9, "E": 1.2}}  # A flying in practice

    adj = pp.fit_and_predict(fp, pace, [f"R{k}" for k in range(10)], target_fp, ("FP1",))

    assert adj["A"] == max(adj.values())
    assert adj["A"] > 0.2


def test_no_adjustment_without_sessions_or_history():
    pace, fp = _season(10)
    assert pp.fit_and_predict(fp, pace, ["R0", "R1", "R2"], fp["R3"], ("FP1",)) == {}
    assert pp.fit_and_predict(fp, pace, [f"R{k}" for k in range(10)], fp["R3"], ()) == {}


def test_replay_fp1_checkpoint_never_reads_later_practice(monkeypatch):
    """Leak guard: FP2 and FP3 are stored, but an FP1 forecast must not use them."""
    seen = []
    monkeypatch.setattr(
        "src.extractors.car_track_traits.load_car_track_traits",
        lambda year: {
            "races": {},
            "profiles": {},
            "fp_gaps": {"X GP": {"FP1": {}, "FP2": {}, "FP3": {}}},
        },
    )
    monkeypatch.setattr(
        pp, "practice_adjustments", lambda year, race, sessions: seen.append(sessions) or {"A": 0.1}
    )

    for checkpoint, expected in (("PRE", ()), ("FP1", ("FP1",)), ("FP3", ("FP1", "FP2", "FP3"))):
        seen.clear()
        context = PredictionContext(
            mode="historical", season_year=2026, checkpoint_session=checkpoint
        )
        with activate_prediction_runtime(config=None, prediction_context=context):
            sim._with_practice_adjustment({"A": 0.5}, 2026, "X GP")
        assert seen == [expected]


def test_replay_context_carries_the_checkpoint(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr("fastf1.get_event", boom)
    context = build_historical_prediction_context(
        year=2026, race_name="X GP", target_session_name="R", checkpoint_session="FP2"
    )

    assert context.checkpoint_session == "FP2"
    assert context.normalized().checkpoint_session == "FP2"


def test_switch_is_off_by_default():
    from src.utils import config_loader

    assert config_loader.get("baseline_predictor.race.practice_pace_adjustment") is False
