"""Parity and identity checks for saved qualifying forecast evaluation."""

from __future__ import annotations

import copy
import json
import logging
import math

import pytest
from scripts import generate_evaluation_report as report_module

from src.analysis import frozen_forecast_scoring as scoring_module
from src.analysis.frozen_forecast_scoring import (
    EVALUATION_SPEC_VERSION,
    score_frozen_forecast,
)
from src.dashboard.prediction_checkpointing import prediction_targets_for_checkpoint
from src.dashboard.prediction_serving import load_served_prediction_bundle


def _rows() -> list[dict]:
    """Return ranked saved rows with asymmetric odds and narrow bands."""
    return [
        {"driver": "AAA", "team": "A", "position": 1, "pole_probability": 0.23, "p5": 1, "p95": 1},
        {"driver": "BBB", "team": "B", "position": 2, "pole_probability": 0.61, "p5": 2, "p95": 2},
    ]


def test_frozen_scorer_keeps_rows_hash_and_outcome_revision_separate():
    forecast = {"event_id": "2026:Test", "target": "main_qualifying", "predicted_rows": _rows()}
    outcome = {
        "event_id": "2026:Test",
        "target": "main_qualifying",
        "revision_id": "r1",
        "actual_rows": _rows(),
    }
    untouched = copy.deepcopy(forecast)
    first = score_frozen_forecast(forecast, outcome, EVALUATION_SPEC_VERSION)
    repeat = score_frozen_forecast(forecast, outcome, EVALUATION_SPEC_VERSION)
    revised = score_frozen_forecast(
        forecast,
        {
            **outcome,
            "revision_id": "r2",
            "actual_rows": list(
                reversed([{**row, "position": 3 - row["position"]} for row in _rows()])
            ),
        },
        EVALUATION_SPEC_VERSION,
    )

    assert first == repeat
    assert forecast == untouched
    assert first["released_rows"] == _rows()
    assert first["forecast_core_hash"] == revised["forecast_core_hash"]
    assert first["rank_metrics"]["mae"] != revised["rank_metrics"]["mae"]
    assert revised["outcome_revision_id"] == "r2"
    first["released_rows"][0]["position"] = 99
    assert forecast == untouched


def test_frozen_scorer_builds_calibration_components_once(monkeypatch):
    """One saved forecast computes interval samples once for both uses."""
    original = scoring_module.build_calibration_components
    calls = 0

    def count_build(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(scoring_module, "build_calibration_components", count_build)
    result = score_frozen_forecast(
        {"event_id": "2026:Test", "target": "main_qualifying", "predicted_rows": _rows()},
        {"event_id": "2026:Test", "target": "main_qualifying", "actual_rows": _rows()},
        EVALUATION_SPEC_VERSION,
    )

    assert calls == 1
    assert result["calibration_metrics"]["interval_count"] == 2.0


def test_frozen_scorer_rejects_mismatch_and_handles_empty_rows():
    forecast = {"event_id": "2026:Test", "target": "sprint_qualifying", "predicted_rows": []}
    outcome = {"event_id": "2026:Test", "target": "sprint_qualifying", "actual_rows": _rows()}
    assert (
        score_frozen_forecast(forecast, outcome, EVALUATION_SPEC_VERSION)["status"] == "unscorable"
    )
    with pytest.raises(ValueError, match="target differ"):
        score_frozen_forecast(
            forecast, {**outcome, "target": "main_qualifying"}, EVALUATION_SPEC_VERSION
        )
    with pytest.raises(ValueError, match="event_id differ"):
        score_frozen_forecast(
            forecast, {**outcome, "event_id": "2026:Else"}, EVALUATION_SPEC_VERSION
        )
    with pytest.raises(ValueError, match="Unsupported evaluation spec"):
        score_frozen_forecast(forecast, outcome, "future")


@pytest.mark.parametrize(
    "bad_rows",
    [
        [{"driver": "AAA", "team": "A", "position": "1"}],
        [{"driver": "AAA", "team": "A", "position": 1.0}],
        [{"driver": "AAA", "team": "A", "position": True}],
        [{"driver": "AAA", "position": 1}],
        [{"team": "A", "position": 1}],
        ["AAA"],
        [
            {"driver": "AAA", "team": "A", "position": 1},
            {"driver": "AAA", "team": "A", "position": 2},
        ],
        [
            {"driver": "AAA", "team": "A", "position": 1},
            {"driver": "BBB", "team": "B", "position": 1},
        ],
    ],
)
def test_pure_scorer_rejects_noncanonical_raw_forecast_rows(bad_rows):
    """Direct callers cannot bypass the report's saved-row boundary."""
    forecast = {"event_id": "2026:Test", "target": "main_qualifying", "predicted_rows": bad_rows}
    outcome = {
        "event_id": "2026:Test",
        "target": "main_qualifying",
        "actual_rows": _rows(),
    }
    original = copy.deepcopy(forecast)

    with pytest.raises(ValueError, match="predicted_rows"):
        score_frozen_forecast(forecast, outcome, EVALUATION_SPEC_VERSION)
    assert forecast == original


@pytest.mark.parametrize(
    "bad_fields",
    [
        {"p5": "1", "p95": 2},
        {"p5": True, "p95": 2},
        {"p5": math.nan, "p95": 2},
        {"p5": 1, "p95": None},
        {"p5": 2, "p95": 1},
        {"p5": -1, "p95": 2},
        {"pole_probability": "0.23"},
        {"pole_probability": False},
        {"pole_probability": math.inf},
        {"dnf_probability": "0.09"},
        {"podium_probability": math.nan},
        {"pole_probability": -1},
        {"pole_probability": 101},
        {"dnf_probability": -0.01},
        {"podium_probability": 100.01},
    ],
)
def test_pure_scorer_rejects_invalid_saved_bands_and_odds(bad_fields):
    """Saved numeric evidence is scored only when its original type is valid."""
    rows = _rows()
    rows[0].update(bad_fields)
    forecast = {"event_id": "2026:Test", "target": "main_qualifying", "predicted_rows": rows}
    outcome = {
        "event_id": "2026:Test",
        "target": "main_qualifying",
        "actual_rows": _rows(),
    }
    original = copy.deepcopy(forecast)

    with pytest.raises(ValueError, match="predicted_rows"):
        score_frozen_forecast(forecast, outcome, EVALUATION_SPEC_VERSION)
    assert forecast == original


def test_pure_scorer_keeps_percentage_probability_and_missing_bands():
    """Saved percent units and absent intervals pass through unchanged."""
    rows = _rows()
    rows[0]["podium_probability"] = 78.8
    rows[0]["dnf_probability"] = 0.09
    rows[0].pop("p5")
    rows[0].pop("p95")
    scored = score_frozen_forecast(
        {"event_id": "2026:Test", "target": "main_qualifying", "predicted_rows": rows},
        {"event_id": "2026:Test", "target": "main_qualifying", "actual_rows": _rows()},
        EVALUATION_SPEC_VERSION,
    )

    assert scored["released_rows"] == rows
    assert scored["released_rows"][0]["podium_probability"] == 78.8
    assert scored["released_rows"][0]["dnf_probability"] == 0.09


@pytest.mark.parametrize(
    ("is_sprint", "checkpoint", "target"),
    [
        (False, "FP2", "main_qualifying"),
        (True, "FP1", "sprint_qualifying"),
        (True, "SQ", "main_qualifying"),
    ],
)
def test_actual_serving_adapter_and_report_score_same_target_rows(
    monkeypatch, tmp_path, is_sprint, checkpoint, target
):
    """Serve canonical sections, project targets for save, then score exact saved rows."""
    released = _rows()
    section = {
        "main_qualifying": "main_quali" if is_sprint else "qualifying",
        "sprint_qualifying": "sprint_quali",
    }[target]
    served_payload = {section: {"grid": released, "result_mode": "PREDICTED"}}
    persisted_serving_file = tmp_path / "persisted_serving.json"
    persisted_serving_file.write_text(json.dumps(served_payload), encoding="utf-8")
    bundle = load_served_prediction_bundle(
        race_name="Test Grand Prix",
        weather="dry",
        year=2026,
        is_sprint=is_sprint,
        boundary_refresh={"boundary_signature": "sig", "latest_elapsed_session": checkpoint},
        session_detector=object(),
        precompute_settings={"horizon_races": 1},
        get_artifact_versions_fn=lambda **_: {},
        compute_artifact_hash_fn=lambda _: "hash",
        resolve_race_boundary_context_fn=lambda **_: ("sig", checkpoint),
        resolve_precompute_targets_fn=lambda **_: ["Test Grand Prix"],
        prediction_cache_key_fn=lambda **_: "key",
        get_cached_prediction_fn=lambda _: None,
        resolve_persisted_boundary_fallback_fn=lambda **_: None,
        load_precomputed_prediction_fn=lambda **_: json.loads(
            persisted_serving_file.read_text(encoding="utf-8")
        ),
        cached_prediction_matches_persisted_fn=lambda **_: False,
        store_cached_prediction_fn=lambda *_: None,
        load_warmed_boundary_fallback_prediction_fn=lambda **_: None,
        load_precompute_horizon_index_fn=lambda **_: None,
        served_prediction_boundary_session_name_fn=lambda **_: checkpoint,
        prediction_unavailable_error_type=RuntimeError,
        notify_fn=lambda _: None,
        logger=logging.getLogger("test"),
    )
    served = bundle["prediction_results"]
    targets = prediction_targets_for_checkpoint(
        prediction_results=served, is_sprint=is_sprint, session_name=checkpoint
    )
    assert targets[target]["predicted_order"] == released
    artifact = {
        "metadata": {
            "year": 2026,
            "race_name": "Test Grand Prix",
            "session_name": checkpoint,
            "weekend_format": "sprint" if is_sprint else "normal",
            "predicted_at": "2026-04-01T10:00:00+00:00",
            "forecast_id": "forecast-fixture",
            "run_id": "run-fixture",
            "model_version": "fixture-v1",
            "top_level_qualifying_target": target,
        },
        "qualifying": {
            "predicted_grid": [{**row, "position": 3 - row["position"]} for row in released]
        },
        "targets": targets,
        "actuals": {
            "qualifying": [{**row, "position": 3 - row["position"]} for row in released],
            "targets": {target: copy.deepcopy(released)},
        },
    }
    predictions_dir = tmp_path / "predictions"
    saved_path = predictions_dir / "2026" / "test_grand_prix" / f"{checkpoint.lower()}.json"
    saved_path.parent.mkdir(parents=True)
    saved_path.write_text(json.dumps(artifact), encoding="utf-8")
    saved_bytes = saved_path.read_bytes()
    monkeypatch.setattr(
        report_module,
        "_load_predictions",
        lambda year, directory: (
            report_module._load_predictions_from_files(year, directory),
            "persisted_file_fixture",
        ),
    )
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})
    report = report_module.build_report(2026, predictions_dir)
    scored = report["scoring_records"][0]

    assert bundle["boundary_session_name"] == checkpoint
    assert scored["target"] == target
    assert scored["released_rows"] == json.loads(saved_bytes)["targets"][target]["predicted_order"]
    assert scored["released_rows"] == released
    assert scored["forecast_id"] == "forecast-fixture"
    assert scored["run_id"] == "run-fixture"
    assert scored["model_version"] == "fixture-v1"
    manual_score = score_frozen_forecast(
        {
            "event_id": "2026:Test Grand Prix",
            "target": target,
            "checkpoint": checkpoint,
            "information_cutoff_at": None,
            "predicted_at": "2026-04-01T10:00:00+00:00",
            "forecast_id": "forecast-fixture",
            "run_id": "run-fixture",
            "model_version": "fixture-v1",
            "predicted_rows": released,
        },
        {
            "event_id": "2026:Test Grand Prix",
            "target": target,
            "revision_id": None,
            "actual_rows": released,
        },
        EVALUATION_SPEC_VERSION,
    )
    assert scored["forecast_core_hash"] == manual_score["forecast_core_hash"]
    assert scored["released_rows"][0]["pole_probability"] == 0.23
    assert report["qualifying_accuracy"]["mae"] == 0.0
    assert report["calibration"]["empirical_coverage"] == 1.0


def test_counterfactual_is_opt_in_and_cannot_change_released_headline(monkeypatch, tmp_path):
    """The research adjustment never supplies the report's headline or gate."""
    first = {
        "metadata": {"year": 2026, "race_name": "A", "session_name": "FP1"},
        "qualifying": {"predicted_grid": _rows()},
        "actuals": {"qualifying": _rows()},
    }
    second = copy.deepcopy(first)
    second["metadata"]["race_name"] = "B"
    second["qualifying"]["predicted_grid"] = [
        {**row, "position": index} for index, row in enumerate(reversed(_rows()), start=1)
    ]
    monkeypatch.setattr(report_module, "_load_predictions", lambda *_: ([first, second], "fixture"))
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})
    saved = report_module.build_report(2026, tmp_path)
    research = report_module.build_report(2026, tmp_path, include_counterfactual=True)

    assert saved["counterfactual_metrics"] is None
    assert research["counterfactual_metrics"]["run_kind"] == "counterfactual"
    assert research["qualifying_accuracy"] == saved["qualifying_accuracy"]
    assert research["calibration"] == saved["calibration"]
    assert research["segment_breakdown"] == saved["segment_breakdown"]
    assert research["error_analysis"] == saved["error_analysis"]
    assert research["qualifying_bias"] == saved["qualifying_bias"]
    assert [record["status"] for record in research["scoring_records"]] == ["scored", "scored"]
    assert research["scoring_records"] == saved["scoring_records"]
    assert research["qualifying_accuracy"]["events_evaluated"] == 2
    assert research["qualifying_accuracy"]["mae"] > 0
    assert research["counterfactual_metrics"]["cohort_size"] == 2
    assert (
        research["production_gate"]["metrics"]["qualifying_mae_improvement_vs_naive"]
        == (saved["production_gate"]["metrics"]["qualifying_mae_improvement_vs_naive"])
    )
    assert (
        research["counterfactual_metrics"]["qualifying_accuracy"]["mae"]
        < research["qualifying_accuracy"]["mae"]
    )
    assert "Counterfactual Qualifying Research" not in report_module.render_markdown(saved)
    assert "Counterfactual Qualifying Research" in report_module.render_markdown(research)


def test_counterfactual_excludes_unscorable_event_and_binds_scored_cohort(monkeypatch, tmp_path):
    """A selected event without matching drivers cannot enter research metrics."""
    scored = {
        "metadata": {"year": 2026, "race_name": "A", "session_name": "FP1"},
        "qualifying": {"predicted_grid": _rows()},
        "actuals": {"qualifying": _rows()},
    }
    unmatched = {
        "metadata": {"year": 2026, "race_name": "B", "session_name": "FP1"},
        "qualifying": {"predicted_grid": _rows()},
        "actuals": {
            "qualifying": [
                {"driver": "CCC", "team": "C", "position": 1},
                {"driver": "DDD", "team": "D", "position": 2},
            ]
        },
    }
    monkeypatch.setattr(
        report_module, "_load_predictions", lambda *_: ([scored, unmatched], "fixture")
    )
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})
    released = report_module.build_report(2026, tmp_path)
    research = report_module.build_report(2026, tmp_path, include_counterfactual=True)

    assert [row["status"] for row in research["scoring_records"]] == [
        "scored",
        "unscorable",
    ]
    assert research["evaluation_scope"]["selected_prediction_counts"]["qualifying"] == 2
    assert research["evaluation_scope"]["scored_qualifying_count"] == 1
    assert research["qualifying_pairs"] == 1
    assert research["counterfactual_metrics"]["cohort_size"] == 1
    assert (
        research["counterfactual_metrics"]["cohort_hash"]
        == research["evaluation_scope"]["cohort_hash"]
    )
    assert (
        research["released_model_metrics"]["cohort_hash"]
        == research["evaluation_scope"]["cohort_hash"]
    )
    assert research["counterfactual_metrics"]["adjustments"]["event_margins"][0]["race_name"] == "A"
    assert research["counterfactual_metrics"]["qualifying_accuracy"]["events_evaluated"] == 1
    assert math.isfinite(research["counterfactual_metrics"]["qualifying_accuracy"]["mae"])
    assert research["qualifying_accuracy"] == released["qualifying_accuracy"]
    assert research["calibration"] == released["calibration"]
    assert research["production_gate"]["checks"] == released["production_gate"]["checks"]


def test_raw_model_metrics_are_machine_readable_unavailable(monkeypatch, tmp_path):
    """Legacy raw field names cannot carry saved released metrics."""
    artifact = {
        "metadata": {"year": 2026, "race_name": "A", "session_name": "FP1"},
        "qualifying": {"predicted_grid": _rows()},
        "actuals": {"qualifying": _rows()},
    }
    monkeypatch.setattr(report_module, "_load_predictions", lambda *_: ([artifact], "fixture"))
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})
    report = report_module.build_report(2026, tmp_path)

    assert report["raw_model_metrics"]["status"] == "unavailable"
    assert report["raw_qualifying_accuracy"] is None
    assert report["raw_qualifying_calibration"] is None
    assert report["raw_qualifying_vs_baseline"] is None
    assert report["released_model_metrics"]["qualifying_accuracy"] == report["qualifying_accuracy"]
