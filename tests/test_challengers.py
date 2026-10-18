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
    assert enabled_challengers() == ["previous_event_order"]

    class _Cfg:
        def get(self, key, default=None):
            return ["previous_event_order", "not_registered"]

    assert enabled_challengers(_Cfg()) == ["previous_event_order"]
