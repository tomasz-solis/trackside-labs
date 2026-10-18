"""Tests for the challenger slot: registry, isolation, storage and scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models import challengers
from src.models.challengers import (
    ChallengerSpec,
    challenger_prediction_view,
    enabled_challengers,
    run_challengers,
)
from src.utils.prediction_metrics import PredictionMetrics

ORDER_MAP = {"australian grand prix": 0, "chinese grand prix": 1, "japanese grand prix": 2}


@pytest.fixture(autouse=True)
def fixed_calendar(monkeypatch):
    """Calendar order without touching FastF1."""
    monkeypatch.setattr(challengers, "_calendar_order_map", lambda year: ORDER_MAP)


def _rows(*drivers: str) -> list[dict]:
    return [
        {"driver": driver, "team": "McLaren", "position": position}
        for position, driver in enumerate(drivers, start=1)
    ]


def _prediction(race: str, predicted: list[dict], actual: list[dict] | None = None) -> dict:
    return {
        "metadata": {
            "year": 2026,
            "race_name": race,
            "session_name": "FP3",
            "weekend_format": "normal",
        },
        "targets": {
            "main_qualifying": {
                "target_session": "Q",
                "predicted_order": predicted,
                "result_mode": "PREDICTED",
                "eligible_at_save": True,
            }
        },
        "actuals": {"targets": {"main_qualifying": actual}},
    }


def test_previous_event_order_repeats_last_result_and_appends_new_drivers():
    history = [
        _prediction("Australian Grand Prix", _rows("NOR", "VER"), _rows("VER", "NOR", "LEC"))
    ]
    current = _prediction("Chinese Grand Prix", _rows("NOR", "PIA", "VER", "LEC"))

    result = run_challengers(
        current, historical_predictions=history, names=["previous_event_order"]
    )

    payload = result["previous_event_order"]
    assert payload["status"] == "active"
    rows = payload["targets"]["main_qualifying"]["predicted_order"]
    # Last event's order first, then the driver it did not have, in champion order.
    assert [row["driver"] for row in rows] == ["VER", "NOR", "LEC", "PIA"]
    assert [row["position"] for row in rows] == [1, 2, 3, 4]
    assert rows[0]["champion_position"] == 3


def test_later_events_never_feed_the_naive_order():
    history = [_prediction("Japanese Grand Prix", _rows("NOR"), _rows("LEC", "NOR"))]
    current = _prediction("Chinese Grand Prix", _rows("NOR", "LEC"))

    result = run_challengers(
        current, historical_predictions=history, names=["previous_event_order"]
    )

    assert result["previous_event_order"]["status"] == "insufficient_history"
    assert result["previous_event_order"]["targets"] == {}


def test_a_failing_challenger_is_recorded_and_the_others_still_run(monkeypatch):
    def broken(ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        challengers.CHALLENGERS, "broken", ChallengerSpec("broken", "0", broken, "test")
    )
    history = [_prediction("Australian Grand Prix", _rows("NOR"), _rows("VER", "NOR"))]
    current = _prediction("Chinese Grand Prix", _rows("NOR", "VER"))

    result = run_challengers(
        current, historical_predictions=history, names=["broken", "previous_event_order"]
    )

    assert result["broken"]["status"] == "error" and "boom" in result["broken"]["error"]
    assert result["previous_event_order"]["status"] == "active"


def test_actual_sections_are_not_forecast():
    current = _prediction("Chinese Grand Prix", _rows("NOR", "VER"))
    current["targets"]["main_qualifying"]["result_mode"] = "ACTUAL"
    history = [_prediction("Australian Grand Prix", _rows("NOR"), _rows("VER", "NOR"))]

    result = run_challengers(
        current, historical_predictions=history, names=["previous_event_order"]
    )

    assert result["previous_event_order"]["targets"] == {}


def test_view_scores_the_challenger_with_the_champion_metrics():
    prediction = _prediction(
        "Chinese Grand Prix", _rows("NOR", "VER", "LEC"), _rows("VER", "NOR", "LEC")
    )
    prediction["challengers"] = {
        "previous_event_order": {
            "status": "active",
            "targets": {"main_qualifying": {"predicted_order": _rows("VER", "NOR", "LEC")}},
        }
    }

    view = challenger_prediction_view(prediction, "previous_event_order")
    champion = PredictionMetrics.calculate_prediction_target_metrics(prediction, is_sprint=False)
    challenger = PredictionMetrics.calculate_prediction_target_metrics(view, is_sprint=False)

    assert champion["main_qualifying"]["overall_mae"] > 0
    assert challenger["main_qualifying"]["overall_mae"] == 0
    assert challenger_prediction_view(prediction, "missing") is None


def test_logger_saves_challengers_beside_the_champion(tmp_path):
    from src.utils.prediction_logger import PredictionLogger

    logger = PredictionLogger(predictions_dir=str(tmp_path / "predictions"))
    rows = _rows("NOR", "VER", "LEC")
    logger.save_prediction(
        year=2026,
        race_name="Australian Grand Prix",
        session_name="FP3",
        qualifying_prediction=rows,
        race_prediction=rows,
        weather="dry",
        target_predictions={"main_qualifying": {"predicted_order": rows, "target_session": "Q"}},
    )

    saved = logger.load_prediction(2026, "Australian Grand Prix", "FP3")

    # First event of the season: the naive challenger has nothing to repeat yet.
    assert saved["challengers"]["previous_event_order"]["status"] == "insufficient_history"
    assert saved["targets"]["main_qualifying"]["predicted_order"][0]["driver"] == "NOR"


def test_replay_writes_challenger_snapshots_where_compare_replay_arms_reads(tmp_path):
    from src.utils.historical_replay import _save_challenger_snapshots

    prediction = _prediction(
        "Chinese Grand Prix", _rows("NOR", "VER", "LEC"), _rows("VER", "NOR", "LEC")
    )
    prediction["challengers"] = {
        "previous_event_order": {
            "status": "active",
            "targets": {"main_qualifying": {"predicted_order": _rows("VER", "NOR", "LEC")}},
        }
    }

    _save_challenger_snapshots(
        prediction,
        output_root=tmp_path,
        is_sprint=False,
        metrics_calculator=PredictionMetrics(),
    )

    files = list(
        (tmp_path / "challengers" / "previous_event_order" / "accuracy_snapshot" / "2026").glob(
            "*/*/main_qualifying.json"
        )
    )
    assert len(files) == 1
    payload = json.loads(Path(files[0]).read_text())
    assert payload["metrics"]["overall_mae"] == 0
    assert payload["metadata"]["race_name"] == "Chinese Grand Prix"


def test_config_enables_the_naive_challenger_and_skips_unknown_names():
    assert enabled_challengers() == [
        "previous_event_order",
        "qualifying_ridge",
        "qualifying_laptime",
    ]

    class _Cfg:
        def get(self, key, default=None):
            return ["previous_event_order", "not_registered"]

    assert enabled_challengers(_Cfg()) == ["previous_event_order"]


# --- qualifying_ridge ------------------------------------------------------------------

RIDGE_RACES = [
    "Australian Grand Prix",
    "Chinese Grand Prix",
    "Japanese Grand Prix",
    "Miami Grand Prix",
    "Canadian Grand Prix",
    "Monaco Grand Prix",
]
DRIVERS = ["VER", "NOR", "LEC", "PIA", "HAM", "RUS"]


def _ridge_prediction(race: str, session: str, predicted, actual=None) -> dict:
    prediction = _prediction(race, predicted, actual)
    prediction["metadata"]["session_name"] = session
    return prediction


class _Store:
    def __init__(self):
        self.loaded: list[str] = []

    def load_artifact(self, artifact_type, key):
        self.loaded.append(key)
        teams = {
            team: {
                "profiles": {"short_run": {"overall_pace_seconds": 90.0 + i * 0.2}},
                "driver_deltas_seconds": {"short_run": {}},
            }
            for i, team in enumerate(["A", "B", "C", "D", "E", "F"])
        }
        return {"teams": teams}


def _ridge_ctx(monkeypatch, current_race: str, checkpoint: str, events: int):
    from types import SimpleNamespace

    monkeypatch.setattr(
        challengers,
        "_calendar_order_map",
        lambda year: {challengers._normalize_label(r): i for i, r in enumerate(RIDGE_RACES)},
    )
    history = [
        _ridge_prediction(race, "FP3", _rows(*DRIVERS), _rows(*DRIVERS))
        for race in RIDGE_RACES[:events]
    ]
    current = _ridge_prediction(current_race, checkpoint, _rows(*DRIVERS))
    store = _Store()
    predictor = SimpleNamespace(artifact_store=store, data_dir="", config=None)
    return current, history, predictor, store


def test_ridge_needs_four_earlier_events(monkeypatch):
    current, history, predictor, _ = _ridge_ctx(monkeypatch, "Canadian Grand Prix", "FP3", 3)
    history = history[:3]

    result = run_challengers(
        current, historical_predictions=history, predictor=predictor, names=["qualifying_ridge"]
    )

    assert result["qualifying_ridge"]["status"] == "insufficient_history"


def test_ridge_forecasts_a_permutation_from_earlier_events_only(monkeypatch):
    current, history, predictor, store = _ridge_ctx(monkeypatch, "Monaco Grand Prix", "FP1", 5)
    # A later event in the history must never be read.
    history.append(_ridge_prediction("Monaco Grand Prix", "FP3", _rows(*DRIVERS), _rows(*DRIVERS)))

    result = run_challengers(
        current, historical_predictions=history, predictor=predictor, names=["qualifying_ridge"]
    )

    rows = result["qualifying_ridge"]["targets"]["main_qualifying"]["predicted_order"]
    assert sorted(row["position"] for row in rows) == list(range(1, 7))
    assert {row["driver"] for row in rows} == set(DRIVERS)
    monaco_reads = [key for key in store.loaded if "Monaco" in key]
    # At the FP1 checkpoint only FP1 of the current weekend is read.
    assert monaco_reads == ["2026::Monaco Grand Prix::FP1"]


def test_ridge_reads_no_practice_at_pre(monkeypatch):
    current, history, predictor, store = _ridge_ctx(monkeypatch, "Monaco Grand Prix", "PRE", 5)

    run_challengers(
        current, historical_predictions=history, predictor=predictor, names=["qualifying_ridge"]
    )

    assert not [key for key in store.loaded if "Monaco" in key]


# --- qualifying_laptime ----------------------------------------------------------------


def _laptime_setup(monkeypatch, checkpoint: str, events: int):
    import src.models.challenger_laptime as laptime

    current, history, predictor, store = _ridge_ctx(
        monkeypatch, "Monaco Grand Prix", checkpoint, events
    )
    teams = {driver: team for driver, team in zip(DRIVERS, "AABBCC", strict=True)}
    for prediction in [*history, current]:
        for row in prediction["targets"]["main_qualifying"]["predicted_order"]:
            row["team"] = teams[row["driver"]]
        for row in prediction["actuals"]["targets"].get("main_qualifying") or []:
            row["team"] = teams[row["driver"]]
    gaps = {race: {"A": 0.0, "B": 0.4, "C": 0.9} for race in RIDGE_RACES}
    monkeypatch.setattr(
        "src.extractors.car_track_traits.load_car_track_traits",
        lambda year, store=None: {"races": {r: {"quali_gaps": g} for r, g in gaps.items()}},
    )
    monkeypatch.setattr(laptime, "_track_fit", lambda year, race: {})
    laptime._best_lap_cache.clear()
    return current, history, predictor, store


def test_laptime_needs_four_earlier_events(monkeypatch):
    current, history, predictor, _ = _laptime_setup(monkeypatch, "FP3", 3)

    result = run_challengers(
        current, historical_predictions=history, predictor=predictor, names=["qualifying_laptime"]
    )

    assert result["qualifying_laptime"]["status"] == "insufficient_history"


def test_laptime_orders_by_car_pace_without_openf1(monkeypatch):
    current, history, predictor, store = _laptime_setup(monkeypatch, "PRE", 5)

    result = run_challengers(
        current, historical_predictions=history, predictor=predictor, names=["qualifying_laptime"]
    )

    rows = result["qualifying_laptime"]["targets"]["main_qualifying"]["predicted_order"]
    assert sorted(row["position"] for row in rows) == list(range(1, 7))
    # Team A is fastest every earlier race, team C slowest.
    assert [row["team"] for row in rows] == ["A", "A", "B", "B", "C", "C"]
    assert rows[0]["predicted_gap_s"] == 0.0
    assert not [key for key in store.loaded if "Monaco" in key]
