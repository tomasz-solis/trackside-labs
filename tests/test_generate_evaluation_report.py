"""Tests for evaluation report generation and output schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import generate_evaluation_report as report_module
from scripts.generate_evaluation_report import (
    _build_dnf_calibration_section,
    _resolve_session_pair,
    _sanitize_non_finite,
    _sort_selected_predictions,
    build_report,
    evaluate_production_gate,
    render_error_analysis_markdown,
    render_markdown,
)

from src.analysis.qualifying_counterfactual import apply_qualifying_counterfactual


def _write_prediction(
    root: Path,
    *,
    race_slug: str,
    weekend_format: str,
    weather: str = "dry",
    session_name: str = "FP1",
    predicted_at: str | None = None,
    top_level_qualifying_target: str = "main_qualifying",
    top_level_race_target: str = "grand_prix_race",
    qualifying_rows: list[dict],
    race_rows: list[dict],
    actual_quali: list[dict],
    actual_race: list[dict],
    target_qualifying_rows: list[dict] | None = None,
    target_race_rows: list[dict] | None = None,
    target_actual_quali: list[dict] | None = None,
    target_actual_race: list[dict] | None = None,
) -> None:
    """Write one synthetic prediction artifact for report testing."""
    race_dir = root / "2026" / race_slug
    race_dir.mkdir(parents=True, exist_ok=True)
    stored_target_qualifying_rows = target_qualifying_rows or qualifying_rows
    stored_target_race_rows = target_race_rows or race_rows
    stored_target_actual_quali = target_actual_quali or actual_quali
    stored_target_actual_race = target_actual_race or actual_race
    payload = {
        "metadata": {
            "year": 2026,
            "race_name": race_slug.replace("_", " ").title(),
            "session_name": session_name,
            "weekend_format": weekend_format,
            "weather": weather,
            "predicted_at": predicted_at or f"2026-01-01T00:00:0{len(race_rows)}+00:00",
            "information_cutoff_at": predicted_at or f"2026-01-01T00:00:0{len(race_rows)}+00:00",
            "top_level_qualifying_target": top_level_qualifying_target,
            "top_level_race_target": top_level_race_target,
        },
        "qualifying": {"predicted_grid": qualifying_rows},
        "race": {"predicted_results": race_rows},
        "targets": {
            top_level_qualifying_target: {
                "target_session": "Q",
                "predicted_order": stored_target_qualifying_rows,
                "result_mode": "PREDICTED",
                "grid_source": "PREDICTED",
                "mean_confidence": 0.6,
                "eligible_at_save": True,
            },
            top_level_race_target: {
                "target_session": "R",
                "predicted_order": stored_target_race_rows,
                "result_mode": "PREDICTED",
                "grid_source": "PREDICTED",
                "mean_confidence": 0.6,
                "eligible_at_save": True,
            },
        },
        "actuals": {
            "qualifying": actual_quali,
            "race": actual_race,
            "targets": {
                top_level_qualifying_target: stored_target_actual_quali,
                top_level_race_target: stored_target_actual_race,
            },
        },
    }
    (race_dir / f"{race_slug}_{session_name.lower()}.json").write_text(json.dumps(payload))


def _legacy_prediction(
    race_name: str,
    predicted_rows: list[dict],
    actual_rows: list[dict],
) -> dict:
    """Return a minimal legacy-shaped prediction payload for helper tests."""
    return {
        "metadata": {
            "year": 2026,
            "race_name": race_name,
            "session_name": "FP3",
            "predicted_at": "2026-01-01T00:00:00+00:00",
        },
        "qualifying": {"predicted_grid": predicted_rows},
        "actuals": {"qualifying": actual_rows},
    }


def test_report_scores_saved_qualifying_rows_without_reordering_or_widening(monkeypatch, tmp_path):
    """The authoritative report must score the rows saved for serving."""
    first = _legacy_prediction(
        "Australian Grand Prix",
        [
            {"driver": "AAA", "team": "A", "position": 1, "p5": 1, "p95": 1},
            {"driver": "BBB", "team": "B", "position": 2, "p5": 2, "p95": 2},
        ],
        [
            {"driver": "AAA", "team": "A", "position": 1},
            {"driver": "BBB", "team": "B", "position": 2},
        ],
    )
    second = _legacy_prediction(
        "Chinese Grand Prix",
        [
            {"driver": "BBB", "team": "B", "position": 1, "p5": 1, "p95": 1},
            {"driver": "AAA", "team": "A", "position": 2, "p5": 2, "p95": 2},
        ],
        [
            {"driver": "AAA", "team": "A", "position": 1},
            {"driver": "BBB", "team": "B", "position": 2},
        ],
    )
    monkeypatch.setattr(report_module, "_load_predictions", lambda *_: ([first, second], "fixture"))
    monkeypatch.setattr(
        report_module,
        "_calendar_order_map",
        lambda *_: {"australian grand prix": 0, "chinese grand prix": 1},
    )

    report = build_report(2026, tmp_path)

    assert report["qualifying_accuracy"]["mae"] == 0.5
    assert report["calibration"]["empirical_coverage"] == 0.5
    assert report["evaluation_scope"]["production_adjustments_applied"]["qualifying"] is False


def test_report_uses_prepared_scorer_metrics_without_re_resolving(monkeypatch, tmp_path):
    """Headline, segments, errors, and released group share one scoring pass."""
    actual = [
        {"driver": "AAA", "team": "A", "position": 1},
        {"driver": "BBB", "team": "B", "position": 2},
    ]
    first = _legacy_prediction("First Grand Prix", actual, actual)
    second = _legacy_prediction(
        "Second Grand Prix",
        [
            {"driver": "BBB", "team": "B", "position": 1},
            {"driver": "AAA", "team": "A", "position": 2},
        ],
        actual,
    )
    monkeypatch.setattr(report_module, "_load_predictions", lambda *_: ([first, second], "fixture"))
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})
    original_scorer = report_module.score_frozen_forecast
    original_resolver = report_module._resolve_session_pair
    original_rank = report_module.compute_prediction_accuracy
    original_summary = report_module.summarize_calibration_components
    scoring_started = False
    report_rank_calls = 0
    pooled_summary_calls = 0

    def score_once(*args):
        nonlocal scoring_started
        result = original_scorer(*args)
        scoring_started = True
        return result

    def reject_late_resolution(*args, **kwargs):
        if scoring_started:
            raise AssertionError("report re-resolved a pair after pure scoring")
        return original_resolver(*args, **kwargs)

    def count_rank(*args, **kwargs):
        nonlocal report_rank_calls
        report_rank_calls += 1
        return original_rank(*args, **kwargs)

    def count_summary(*args):
        nonlocal pooled_summary_calls
        pooled_summary_calls += 1
        return original_summary(*args)

    monkeypatch.setattr(report_module, "score_frozen_forecast", score_once)
    monkeypatch.setattr(report_module, "_resolve_session_pair", reject_late_resolution)
    monkeypatch.setattr(report_module, "compute_prediction_accuracy", count_rank)
    monkeypatch.setattr(report_module, "summarize_calibration_components", count_summary)

    report = build_report(2026, tmp_path)

    assert report_rank_calls == 1  # The preceding-actual baseline only.
    assert pooled_summary_calls == 1
    assert report["qualifying_accuracy"]["events_evaluated"] == 2
    assert report["qualifying_accuracy"]["mae"] == 0.5
    assert report["released_model_metrics"]["qualifying_accuracy"] is report["qualifying_accuracy"]
    assert report["released_model_metrics"]["qualifying_calibration"] is report["calibration"]
    assert report["segment_breakdown"]["qualifying"]["weekend_format"]["unknown"]["mae"] == 0.5
    assert report["error_analysis"]["qualifying"]["worst_events"][0]["race_name"] == (
        "Second Grand Prix"
    )
    expected_baseline = report_module._build_baseline_comparison_section(
        [first["qualifying"]["predicted_grid"], second["qualifying"]["predicted_grid"]],
        [actual, actual],
        "qualifying",
    )
    assert report["qualifying_vs_baseline"] == expected_baseline


def test_pooled_calibration_keeps_driver_and_miss_weights(monkeypatch, tmp_path):
    """Unequal event sizes and missing bands retain pooled interval semantics."""
    hit = _legacy_prediction(
        "A Grand Prix",
        [{"driver": "AAA", "team": "A", "position": 1, "p5": 1, "p95": 1}],
        [{"driver": "AAA", "team": "A", "position": 1}],
    )
    misses = _legacy_prediction(
        "B Grand Prix",
        [
            {"driver": "BBB", "team": "B", "position": 1, "p5": 1, "p95": 1},
            {"driver": "CCC", "team": "C", "position": 2, "p5": 2, "p95": 2},
            {"driver": "DDD", "team": "D", "position": 3, "p5": 3, "p95": 3},
        ],
        [
            {"driver": "BBB", "team": "B", "position": 2},
            {"driver": "CCC", "team": "C", "position": 1},
            {"driver": "DDD", "team": "D", "position": 6},
        ],
    )
    no_bands = _legacy_prediction(
        "C Grand Prix",
        [{"driver": "EEE", "team": "E", "position": 1}],
        [{"driver": "EEE", "team": "E", "position": 1}],
    )
    unmatched = _legacy_prediction(
        "D Grand Prix",
        [{"driver": "FFF", "team": "F", "position": 1, "p5": 1, "p95": 1}],
        [{"driver": "GGG", "team": "G", "position": 1}],
    )
    monkeypatch.setattr(
        report_module,
        "_load_predictions",
        lambda *_: ([hit, misses, no_bands, unmatched], "fixture"),
    )
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})

    report = build_report(2026, tmp_path)

    assert [record["status"] for record in report["scoring_records"]] == [
        "scored",
        "scored",
        "scored",
        "unscorable",
    ]
    assert report["qualifying_accuracy"]["events_evaluated"] == 3
    assert report["calibration"]["interval_count"] == 4.0
    assert report["calibration"]["empirical_coverage"] == 0.25
    assert report["calibration"]["average_miss_distance"] == pytest.approx(5 / 3)
    assert report["calibration"]["races_with_band_data"] == 2.0
    assert report["calibration"]["total_races_evaluated"] == 3.0
    assert report["released_model_metrics"]["qualifying_calibration"] is report["calibration"]
    assert report["error_analysis"]["qualifying"]["worst_events"][0]["race_name"] == (
        "B Grand Prix"
    )


@pytest.mark.parametrize(
    ("session_kind", "target", "other_target", "payload_key"),
    [
        ("qualifying", "main_qualifying", "sprint_qualifying", "predicted_grid"),
        ("race", "grand_prix_race", "sprint_race", "predicted_results"),
    ],
)
@pytest.mark.parametrize(
    "target_actuals", ["other_target", "missing", "malformed", "missing_prediction"]
)
def test_explicit_target_never_scores_conflicting_top_level_alias(
    monkeypatch, tmp_path, session_kind, target, other_target, payload_key, target_actuals
):
    """A missing explicit outcome cannot turn another session into the selected target."""
    main_rows = [{"driver": "MAIN", "team": "A", "position": 1}]
    alias_rows = [{"driver": "SPRINT", "team": "B", "position": 1}]
    actual_targets = {
        "other_target": {other_target: alias_rows},
        "missing": {},
        "malformed": {target: "invalid"},
        "missing_prediction": {target: main_rows},
    }[target_actuals]
    target_predictions = (
        {other_target: {"predicted_order": alias_rows}}
        if target_actuals == "missing_prediction"
        else {target: {"predicted_order": main_rows}}
    )
    artifact = {
        "metadata": {
            "year": 2026,
            "race_name": "Conflict Grand Prix",
            "session_name": "FP1",
            f"top_level_{session_kind}_target": target,
        },
        session_kind: {payload_key: alias_rows},
        "targets": target_predictions,
        "actuals": {session_kind: alias_rows, "targets": actual_targets},
    }
    monkeypatch.setattr(report_module, "_load_predictions", lambda *_: ([artifact], "fixture"))
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})

    assert _resolve_session_pair(artifact, session_kind=session_kind) == ([], [])
    report = build_report(2026, tmp_path, include_counterfactual=True)
    assert report[f"{session_kind}_accuracy"]["events_evaluated"] == 0
    assert report["evaluation_scope"]["selected_prediction_counts"][session_kind] == 0
    if session_kind == "qualifying":
        assert report["scoring_records"] == []
        assert report["counterfactual_metrics"]["cohort_size"] == 0


def test_unidentified_target_payload_cannot_use_legacy_alias():
    """Target maps without an explicit selected key have no safe top-level join."""
    rows = [{"driver": "AAA", "team": "A", "position": 1}]
    artifact = _legacy_prediction("Conflict Grand Prix", rows, rows)
    artifact["targets"] = {"main_qualifying": {"predicted_order": rows}}
    artifact["actuals"]["targets"] = {"sprint_qualifying": rows}

    assert _resolve_session_pair(artifact, session_kind="qualifying") == ([], [])

    artifact.pop("targets")
    artifact["actuals"].pop("targets")
    assert _resolve_session_pair(artifact, session_kind="qualifying") == (rows, rows)


@pytest.mark.parametrize("session_kind", ["qualifying", "race"])
@pytest.mark.parametrize(
    "explicit_target", ["legacy_qualifying", "legacy_grand_prix_race", "legacy_custom", False, 0]
)
def test_explicit_unknown_target_never_uses_legacy_fallback(
    monkeypatch, tmp_path, session_kind, explicit_target
):
    """Only an absent target field can use top-level legacy rows."""
    rows = [{"driver": "AAA", "team": "A", "position": 1}]
    payload_key = "predicted_grid" if session_kind == "qualifying" else "predicted_results"
    artifact = {
        "metadata": {
            "year": 2026,
            "race_name": "Explicit Legacy Grand Prix",
            "session_name": "FP1",
            f"top_level_{session_kind}_target": explicit_target,
        },
        session_kind: {payload_key: rows},
        "actuals": {session_kind: rows},
    }
    monkeypatch.setattr(report_module, "_load_predictions", lambda *_: ([artifact], "fixture"))
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})

    assert _resolve_session_pair(artifact, session_kind=session_kind) == ([], [])
    report = build_report(2026, tmp_path, include_counterfactual=True)
    assert report[f"{session_kind}_accuracy"]["events_evaluated"] == 0
    assert report["evaluation_scope"]["selected_prediction_counts"][session_kind] == 0
    assert report["counterfactual_metrics"]["cohort_size"] == 0


@pytest.mark.parametrize("session_kind", ["qualifying", "race"])
@pytest.mark.parametrize("missing_target", ["absent", "null", "empty"])
def test_genuine_legacy_without_target_field_still_scores(session_kind, missing_target):
    """Old files with no declared target retain the legacy pair path."""
    rows = [{"driver": "AAA", "team": "A", "position": 1}]
    payload_key = "predicted_grid" if session_kind == "qualifying" else "predicted_results"
    metadata = {"year": 2026, "race_name": "Legacy Grand Prix", "session_name": "FP1"}
    if missing_target == "null":
        metadata[f"top_level_{session_kind}_target"] = None
    elif missing_target == "empty":
        metadata[f"top_level_{session_kind}_target"] = ""
    artifact = {
        "metadata": metadata,
        session_kind: {payload_key: rows},
        "actuals": {session_kind: rows},
    }

    assert _resolve_session_pair(artifact, session_kind=session_kind) == (rows, rows)


@pytest.mark.parametrize(
    ("session_kind", "wrong_target", "target_session"),
    [
        ("qualifying", "grand_prix_race", "R"),
        ("race", "main_qualifying", "Q"),
    ],
)
def test_explicit_target_from_other_session_family_cannot_score(
    monkeypatch, tmp_path, session_kind, wrong_target, target_session
):
    """A race target cannot be scored as qualifying, or the reverse."""
    rows = [{"driver": "AAA", "team": "A", "position": 1}]
    artifact = {
        "metadata": {
            "year": 2026,
            "race_name": "Wrong Family Grand Prix",
            "session_name": "FP1",
            f"top_level_{session_kind}_target": wrong_target,
        },
        "targets": {
            wrong_target: {
                "target_session": target_session,
                "result_mode": "PREDICTED",
                "eligible_at_save": True,
                "predicted_order": rows,
            }
        },
        "actuals": {"targets": {wrong_target: rows}},
    }
    monkeypatch.setattr(report_module, "_load_predictions", lambda *_: ([artifact], "fixture"))
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})

    assert _resolve_session_pair(artifact, session_kind=session_kind) == ([], [])
    report = build_report(2026, tmp_path, include_counterfactual=True)
    assert report[f"{session_kind}_accuracy"]["events_evaluated"] == 0
    assert report["evaluation_scope"]["selected_prediction_counts"][session_kind] == 0
    assert report["counterfactual_metrics"]["cohort_size"] == 0


@pytest.mark.parametrize("invalid_probability", [-1, 101])
def test_report_excludes_impossible_saved_probability_without_repair(
    monkeypatch, tmp_path, invalid_probability
):
    """Invalid odds keep the paired artifact visible but outside headline metrics."""
    row = {
        "driver": "AAA",
        "team": "A",
        "position": 1,
        "pole_probability": invalid_probability,
    }
    artifact = {
        "metadata": {
            "year": 2026,
            "race_name": "Odds Grand Prix",
            "session_name": "FP1",
            "top_level_qualifying_target": "main_qualifying",
        },
        "targets": {
            "main_qualifying": {
                "target_session": "Q",
                "result_mode": "PREDICTED",
                "eligible_at_save": True,
                "predicted_order": [row],
            }
        },
        "actuals": {
            "targets": {"main_qualifying": [{"driver": "AAA", "team": "A", "position": 1}]}
        },
    }
    original = json.dumps(artifact, sort_keys=True)
    monkeypatch.setattr(report_module, "_load_predictions", lambda *_: ([artifact], "fixture"))
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})

    report = build_report(2026, tmp_path, include_counterfactual=True)

    assert report["evaluation_scope"]["selected_prediction_counts"]["qualifying"] == 1
    assert report["scoring_records"][0]["status"] == "unscorable"
    assert "pole_probability" in report["scoring_records"][0]["reason"]
    assert report["qualifying_accuracy"]["events_evaluated"] == 0
    assert report["counterfactual_metrics"]["cohort_size"] == 0
    assert json.dumps(artifact, sort_keys=True) == original


@pytest.mark.parametrize(
    ("session_kind", "target", "expected_session", "wrong_session", "payload_key", "closed"),
    [
        ("qualifying", "main_qualifying", "Q", "SQ", "predicted_grid", "Q"),
        ("race", "grand_prix_race", "R", "SPRINT", "predicted_results", "R"),
    ],
)
@pytest.mark.parametrize(
    "invalid_state",
    [
        "actual_mode",
        "ineligible",
        "metadata_actual_mode",
        "metadata_ineligible",
        "session_actual_mode",
        "session_ineligible",
        "closed",
        "wrong_session",
    ],
)
def test_target_state_and_declared_session_must_be_forecast_eligible(
    monkeypatch,
    tmp_path,
    session_kind,
    target,
    expected_session,
    wrong_session,
    payload_key,
    closed,
    invalid_state,
):
    """Post-outcome or wrongly declared target rows cannot enter the report."""
    rows = [{"driver": "MAIN", "team": "A", "position": 1}]
    alias_rows = [{"driver": "ALIAS", "team": "B", "position": 1}]
    payload = {
        "target_session": wrong_session if invalid_state == "wrong_session" else expected_session,
        "result_mode": "ACTUAL" if invalid_state == "actual_mode" else "PREDICTED",
        "eligible_at_save": invalid_state != "ineligible",
        "predicted_order": rows,
    }
    artifact = {
        "metadata": {
            "year": 2026,
            "race_name": "Conflict Grand Prix",
            "session_name": closed if invalid_state == "closed" else "FP1",
            f"top_level_{session_kind}_target": target,
        },
        session_kind: {payload_key: alias_rows},
        "targets": {target: payload},
        "actuals": {session_kind: alias_rows, "targets": {target: rows}},
    }
    if invalid_state == "metadata_actual_mode":
        artifact["metadata"][f"top_level_{session_kind}_result_mode"] = "ACTUAL"
    elif invalid_state == "metadata_ineligible":
        artifact["metadata"][f"top_level_{session_kind}_eligible_at_save"] = False
    elif invalid_state == "session_actual_mode":
        artifact[session_kind]["result_mode"] = "ACTUAL"
    elif invalid_state == "session_ineligible":
        artifact[session_kind]["eligible_at_save"] = False
    monkeypatch.setattr(report_module, "_load_predictions", lambda *_: ([artifact], "fixture"))
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})

    assert _resolve_session_pair(artifact, session_kind=session_kind) == ([], [])
    report = build_report(2026, tmp_path, include_counterfactual=True)
    assert report[f"{session_kind}_accuracy"]["events_evaluated"] == 0
    if session_kind == "qualifying":
        assert not any(record["status"] == "scored" for record in report["scoring_records"])
        assert report["counterfactual_metrics"]["cohort_size"] == 0


def test_malformed_latest_paired_forecast_does_not_resurrect_older_checkpoint(
    monkeypatch, tmp_path
):
    """A bad latest saved forecast remains selected and visibly unscorable."""
    actual = [{"driver": "AAA", "team": "A", "position": 1}]

    def artifact(checkpoint, time, position):
        return {
            "metadata": {
                "year": 2026,
                "race_name": "Checkpoint Grand Prix",
                "session_name": checkpoint,
                "predicted_at": time,
                "forecast_id": f"forecast-{checkpoint}",
                "run_id": f"run-{checkpoint}",
                "top_level_qualifying_target": "main_qualifying",
            },
            "targets": {
                "main_qualifying": {
                    "target_session": "Q",
                    "result_mode": "PREDICTED",
                    "eligible_at_save": True,
                    "predicted_order": [{"driver": "AAA", "team": "A", "position": position}],
                }
            },
            "actuals": {"targets": {"main_qualifying": actual}},
        }

    old = artifact("PRE", "2026-01-01T00:00:00+00:00", 1)
    latest = artifact("FP1", "2026-01-02T00:00:00+00:00", "1")
    original = json.dumps([old, latest], sort_keys=True)
    monkeypatch.setattr(report_module, "_load_predictions", lambda *_: ([old, latest], "fixture"))
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})
    monkeypatch.setattr("src.utils.accuracy_targets._scheduled_session_start", lambda **_: None)

    report = build_report(2026, tmp_path, include_counterfactual=True)

    assert report["evaluation_scope"]["selected_prediction_counts"]["qualifying"] == 1
    assert report["evaluation_scope"]["selected_checkpoint_breakdown"]["qualifying"] == {"FP1": 1}
    assert report["qualifying_accuracy"]["events_evaluated"] == 0
    assert report["counterfactual_metrics"]["cohort_size"] == 0
    assert report["scoring_records"][0]["status"] == "unscorable"
    assert "position" in report["scoring_records"][0]["reason"]
    assert report["scoring_records"][0]["forecast_id"] == "forecast-FP1"
    assert report["scoring_records"][0]["run_id"] == "run-FP1"
    assert json.dumps([old, latest], sort_keys=True) == original


@pytest.mark.parametrize(
    "invalid_state",
    ["actual_mode", "ineligible", "payload_actual_mode", "payload_ineligible", "rank_string"],
)
def test_legacy_saved_forecast_rejects_invalid_state_or_rows(invalid_state):
    """Legacy aliases retain unknown metadata but reject explicit invalid facts."""
    rows = [{"driver": "AAA", "team": "A", "position": 1}]
    artifact = _legacy_prediction("Legacy Grand Prix", rows, rows)
    metadata = artifact["metadata"]
    if invalid_state == "actual_mode":
        metadata["top_level_qualifying_result_mode"] = "ACTUAL"
    elif invalid_state == "ineligible":
        metadata["top_level_qualifying_eligible_at_save"] = False
    elif invalid_state == "payload_actual_mode":
        artifact["qualifying"]["result_mode"] = "ACTUAL"
    elif invalid_state == "payload_ineligible":
        artifact["qualifying"]["eligible_at_save"] = False
    else:
        artifact["qualifying"]["predicted_grid"][0]["position"] = "1"

    assert _resolve_session_pair(artifact, session_kind="qualifying") == ([], [])


@pytest.mark.parametrize(
    "bad_row",
    [
        {"driver": "AAA", "team": "A", "position": "1"},
        {"driver": "AAA", "team": "A", "position": 1.0},
        {"driver": "AAA", "team": "A", "position": True},
        {"driver": "AAA", "position": 1},
        {"team": "A", "position": 1},
        "AAA",
    ],
)
def test_report_rejects_noncanonical_saved_rows_without_repair(bad_row):
    """The evaluator must not coerce rank or silently drop a served row."""
    valid = {"driver": "BBB", "team": "B", "position": 2}
    rows = [bad_row, valid]
    artifact = {
        "metadata": {
            "year": 2026,
            "race_name": "Conflict Grand Prix",
            "session_name": "FP1",
            "top_level_qualifying_target": "main_qualifying",
        },
        "targets": {
            "main_qualifying": {
                "target_session": "Q",
                "result_mode": "PREDICTED",
                "eligible_at_save": True,
                "predicted_order": rows,
            }
        },
        "actuals": {
            "targets": {
                "main_qualifying": [
                    {"driver": "AAA", "team": "A", "position": 1},
                    valid,
                ]
            }
        },
    }
    original = json.dumps(artifact, sort_keys=True)

    assert _resolve_session_pair(artifact, session_kind="qualifying") == ([], [])
    assert json.dumps(artifact, sort_keys=True) == original


def test_dnf_calibration_aggregate_skill_survives_zero_dnf_event():
    # Regression: a zero-DNF event has an undefined (NaN) per-event skill score;
    # the aggregate must be derived from pooled brier/baseline components instead
    # of averaging per-event skills, or the NaN poisons the whole season number.
    with_dnf_predicted = [
        {"driver": "A", "position": 1, "dnf_probability": 0.05},
        {"driver": "B", "position": 2, "dnf_probability": 0.80},
    ]
    with_dnf_actual = [
        {"driver": "A", "position": 1, "status": "Finished"},
        {"driver": "B", "position": 20, "status": "Retired"},
    ]
    zero_dnf_predicted = [
        {"driver": "A", "position": 1, "dnf_probability": 0.05},
        {"driver": "B", "position": 2, "dnf_probability": 0.10},
    ]
    zero_dnf_actual = [
        {"driver": "A", "position": 1, "status": "Finished"},
        {"driver": "B", "position": 2, "status": "Finished"},
    ]

    section = _build_dnf_calibration_section(
        [with_dnf_predicted, zero_dnf_predicted],
        [with_dnf_actual, zero_dnf_actual],
    )

    assert section["events_scored"] == 2
    assert section["scored_drivers"] == 4
    assert section["actual_dnf_count"] == 1
    # Pooled components: brier = mean over both events weighted by drivers,
    # baseline pools the per-event base-rate scores (0.25 and 0.0).
    expected_brier = ((0.0025 + 0.04) / 2 + (0.0025 + 0.01) / 2) / 2
    assert section["brier_score"] == pytest.approx(expected_brier)
    assert section["baseline_brier"] == pytest.approx(0.125)
    assert section["brier_skill_score"] == pytest.approx(1.0 - expected_brier / 0.125)


def test_dnf_calibration_aggregate_skill_is_none_when_no_dnfs_at_all():
    predicted = [
        {"driver": "A", "position": 1, "dnf_probability": 0.05},
        {"driver": "B", "position": 2, "dnf_probability": 0.10},
    ]
    actual = [
        {"driver": "A", "position": 1, "status": "Finished"},
        {"driver": "B", "position": 2, "status": "Finished"},
    ]

    section = _build_dnf_calibration_section([predicted], [actual])

    assert section["events_scored"] == 1
    assert section["baseline_brier"] == 0.0
    assert section["brier_skill_score"] is None


def test_sanitize_non_finite_makes_report_strict_json_safe():
    report = {
        "ok": 1.5,
        "bad": float("nan"),
        "nested": {"inf": float("inf"), "list": [1, float("-inf"), "text"]},
    }

    sanitized = _sanitize_non_finite(report)

    assert sanitized == {
        "ok": 1.5,
        "bad": None,
        "nested": {"inf": None, "list": [1, None, "text"]},
    }
    # Must not raise: the report file is written with allow_nan=False.
    json.dumps(sanitized, allow_nan=False)


def test_selected_predictions_are_sorted_by_calendar_order():
    predictions = [
        _legacy_prediction("Canadian Grand Prix", [], []),
        _legacy_prediction("Australian Grand Prix", [], []),
        _legacy_prediction("Chinese Grand Prix", [], []),
    ]

    ordered = _sort_selected_predictions(
        predictions,
        year=2026,
        calendar_order={
            "australian grand prix": 0,
            "chinese grand prix": 1,
            "canadian grand prix": 2,
        },
    )

    assert [row["metadata"]["race_name"] for row in ordered] == [
        "Australian Grand Prix",
        "Chinese Grand Prix",
        "Canadian Grand Prix",
    ]


def test_qualifying_research_counterfactual_uses_previous_selected_actuals_only():
    first_actual = [
        {"driver": "AAA", "team": "A", "position": 1},
        {"driver": "BBB", "team": "B", "position": 2},
    ]
    second_actual = [
        {"driver": "AAA", "team": "A", "position": 1},
        {"driver": "BBB", "team": "B", "position": 2},
    ]
    first_prediction = _legacy_prediction(
        "Australian Grand Prix",
        [
            {"driver": "AAA", "team": "A", "position": 1, "p5": 1, "p95": 1},
            {"driver": "BBB", "team": "B", "position": 2, "p5": 2, "p95": 2},
        ],
        first_actual,
    )
    second_prediction = _legacy_prediction(
        "Chinese Grand Prix",
        [
            {"driver": "BBB", "team": "B", "position": 1, "p5": 1, "p95": 1},
            {"driver": "AAA", "team": "A", "position": 2, "p5": 2, "p95": 2},
        ],
        second_actual,
    )

    adjusted, metadata = apply_qualifying_counterfactual(
        [
            {
                "predicted_rows": prediction["qualifying"]["predicted_grid"],
                "actual_rows": prediction["actuals"]["qualifying"],
                "race_name": prediction["metadata"]["race_name"],
                "target": "legacy_qualifying",
            }
            for prediction in (first_prediction, second_prediction)
        ]
    )

    second_rows = adjusted[1]
    assert [row["driver"] for row in second_rows] == ["AAA", "BBB"]
    assert second_rows[0]["released_position"] == 2
    assert second_rows[0]["previous_selected_event_position"] == 1
    assert second_rows[0]["conformal_interval_margin"] == 1.0
    assert metadata["qualifying_rank_stabilizer"]["uses_current_event_actuals"] is False
    assert metadata["run_kind"] == "counterfactual"
    assert (
        metadata["qualifying_interval_stabilizer"]["uses_current_event_actuals_for_current_margin"]
        is False
    )


def test_production_gate_passes_when_model_is_fresh_calibrated_and_beats_naive():
    report = {
        "generated_at": "2026-06-15T12:00:00+00:00",
        "qualifying_accuracy": {"events_evaluated": 5},
        "race_accuracy": {"events_evaluated": 5},
        "qualifying_vs_baseline": {"improvement": {"mae_improvement": 0.12}},
        "race_vs_baseline": {"improvement": {"mae_improvement": 0.08}},
        "calibration": {"empirical_coverage": 0.90},
        "qualifying_bias": {
            "most_overestimated_drivers": [{"entity": "AAA", "mean_abs_error": 2.0}]
        },
        "race_bias": {"most_underestimated_drivers": [{"entity": "BBB", "mean_abs_error": 3.0}]},
    }

    gate = evaluate_production_gate(
        report,
        latest_completed_race_at="2026-06-14T18:00:00+00:00",
    )

    assert gate["status"] == "pass"
    assert gate["score_estimate"] == 95
    assert gate["reasons"] == []


def test_production_gate_fails_stale_underperforming_report():
    report = {
        "generated_at": "2026-04-20T12:00:00+00:00",
        "qualifying_accuracy": {"events_evaluated": 3},
        "race_accuracy": {"events_evaluated": 3},
        "qualifying_vs_baseline": {"improvement": {"mae_improvement": -0.23}},
        "race_vs_baseline": {"improvement": {"mae_improvement": -0.18}},
        "calibration": {"empirical_coverage": 0.803},
        "qualifying_bias": {
            "most_overestimated_drivers": [{"entity": "VER", "mean_abs_error": 11.0}]
        },
        "race_bias": {},
    }

    gate = evaluate_production_gate(
        report,
        latest_completed_race_at="2026-06-01T18:00:00+00:00",
    )

    assert gate["status"] == "fail"
    assert gate["score_estimate"] < 95
    assert "older than the latest completed race" in " ".join(gate["reasons"])
    assert "qualifying MAE does not beat" in " ".join(gate["reasons"])
    assert "race MAE does not beat" in " ".join(gate["reasons"])


def test_build_report_includes_accuracy_and_format_breakdown(tmp_path):
    """Report payload should include standardized accuracy and coverage sections."""
    predictions_dir = tmp_path / "predictions"
    rows_a = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 3},
        {"position": 2, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 4},
        {"position": 3, "driver": "LEC", "team": "Ferrari", "p5": 2, "p95": 5},
    ]
    rows_b = [
        {"position": 1, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 3},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 4},
        {"position": 3, "driver": "LEC", "team": "Ferrari", "p5": 2, "p95": 5},
    ]
    actual_a = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 2, "driver": "NOR", "team": "McLaren"},
        {"position": 3, "driver": "LEC", "team": "Ferrari"},
    ]
    actual_b = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 2, "driver": "LEC", "team": "Ferrari"},
        {"position": 3, "driver": "NOR", "team": "McLaren"},
    ]

    _write_prediction(
        predictions_dir,
        race_slug="bahrain_grand_prix",
        weekend_format="normal",
        weather="dry",
        qualifying_rows=rows_a,
        race_rows=rows_a,
        actual_quali=actual_a,
        actual_race=actual_a,
    )
    _write_prediction(
        predictions_dir,
        race_slug="monaco_grand_prix",
        weekend_format="sprint",
        weather="mixed",
        qualifying_rows=rows_b,
        race_rows=rows_b,
        actual_quali=actual_b,
        actual_race=actual_b,
    )

    report = build_report(2026, predictions_dir)

    assert report["predictions_analyzed"] == 2
    assert report["qualifying_accuracy"]["events_evaluated"] == 2
    assert report["race_accuracy"]["events_evaluated"] == 2
    assert report["evaluation_scope"]["selection_policy"] == "latest_checkpoint_per_race_and_target"
    assert report["format_breakdown"]["normal"]["prediction_files"] == 1
    assert report["format_breakdown"]["sprint"]["prediction_files"] == 1
    assert report["segment_breakdown"]["race"]["weather"]["mixed"]["events"] == 1
    assert report["segment_breakdown"]["race"]["track_type"]["street"]["events"] == 1
    assert report["error_analysis"]["race"]["worst_events"][0]["race_name"] == "Monaco Grand Prix"
    assert report["production_gate"]["status"] == "fail"
    assert "mae" in report["qualifying_accuracy"]
    assert "mae" in report["race_accuracy"]


def test_render_markdown_mentions_accuracy_overview_and_format_breakdown(tmp_path):
    """Rendered markdown should surface the new review-oriented sections."""
    predictions_dir = tmp_path / "predictions"
    rows = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 3},
        {"position": 2, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 4},
    ]
    actual = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 2, "driver": "NOR", "team": "McLaren"},
    ]
    _write_prediction(
        predictions_dir,
        race_slug="bahrain_grand_prix",
        weekend_format="normal",
        weather="dry",
        qualifying_rows=rows,
        race_rows=rows,
        actual_quali=actual,
        actual_race=actual,
    )

    markdown = render_markdown(build_report(2026, predictions_dir))

    assert "## 0. Accuracy Overview" in markdown
    assert "## Production Readiness Gate" in markdown
    assert "### Weekend Format Coverage" in markdown
    assert "## Selection Policy" in markdown
    assert "## 1. Segmented Performance" in markdown
    assert "## 3. Error Analysis" in markdown


def test_build_report_uses_latest_checkpoint_per_race_and_target(monkeypatch, tmp_path):
    """Canonical evaluation should dedupe repeated race checkpoints down to the latest artifact."""
    # This test isolates selection from the live race schedule.
    monkeypatch.setattr(report_module, "_calendar_order_map", lambda *_: {})
    monkeypatch.setattr("src.utils.accuracy_targets._scheduled_session_start", lambda **_: None)
    predictions_dir = tmp_path / "predictions"
    earlier_rows = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 3},
        {"position": 2, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 4},
    ]
    later_rows = [
        {"position": 1, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 3},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 4},
    ]
    actual_rows = [
        {"position": 1, "driver": "NOR", "team": "McLaren"},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
    ]
    _write_prediction(
        predictions_dir,
        race_slug="australian_grand_prix",
        weekend_format="normal",
        session_name="PRE",
        predicted_at="2026-03-10T10:00:00+00:00",
        qualifying_rows=earlier_rows,
        race_rows=earlier_rows,
        actual_quali=actual_rows,
        actual_race=actual_rows,
    )
    _write_prediction(
        predictions_dir,
        race_slug="australian_grand_prix",
        weekend_format="normal",
        session_name="FP3",
        predicted_at="2026-03-12T10:00:00+00:00",
        qualifying_rows=later_rows,
        race_rows=later_rows,
        actual_quali=actual_rows,
        actual_race=actual_rows,
    )

    report = build_report(2026, predictions_dir)

    assert report["predictions_analyzed"] == 2
    assert report["qualifying_pairs"] == 1
    assert report["race_pairs"] == 1
    assert report["qualifying_accuracy"]["mae"] == 0.0
    assert report["race_accuracy"]["mae"] == 0.0
    assert report["evaluation_scope"]["selected_prediction_counts"]["qualifying"] == 1
    assert report["evaluation_scope"]["selected_checkpoint_breakdown"]["qualifying"] == {"FP3": 1}
    assert report["evaluation_scope"]["ignored_intermediate_checkpoints"]["qualifying"] == 1


def test_render_error_analysis_markdown_surfaces_worst_events(tmp_path):
    """Standalone error-analysis markdown should focus on worst weekends and repeat misses."""
    predictions_dir = tmp_path / "predictions"
    predicted_rows = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 3},
        {"position": 2, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 4},
        {"position": 3, "driver": "LEC", "team": "Ferrari", "p5": 2, "p95": 5},
    ]
    actual_rows = [
        {"position": 1, "driver": "NOR", "team": "McLaren"},
        {"position": 2, "driver": "LEC", "team": "Ferrari"},
        {"position": 3, "driver": "VER", "team": "Red Bull Racing"},
    ]
    _write_prediction(
        predictions_dir,
        race_slug="monaco_grand_prix",
        weekend_format="normal",
        weather="mixed",
        qualifying_rows=predicted_rows,
        race_rows=predicted_rows,
        actual_quali=actual_rows,
        actual_race=actual_rows,
    )

    markdown = render_error_analysis_markdown(build_report(2026, predictions_dir))

    assert "# Model error analysis" in markdown
    assert "Worst weekends:" in markdown
    assert "Drivers that show up repeatedly among the largest misses:" in markdown
    assert "Monaco Grand Prix" in markdown


def test_build_report_reads_target_aware_intervals_when_legacy_rows_lack_bands(tmp_path):
    """Calibration and accuracy should use canonical target payloads before legacy rows."""
    predictions_dir = tmp_path / "predictions"
    legacy_rows = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 2, "driver": "NOR", "team": "McLaren"},
    ]
    target_rows = [
        {"position": 1, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 2},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 2},
    ]
    actual_rows = [
        {"position": 1, "driver": "NOR", "team": "McLaren"},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
    ]
    _write_prediction(
        predictions_dir,
        race_slug="australian_grand_prix",
        weekend_format="normal",
        qualifying_rows=legacy_rows,
        race_rows=legacy_rows,
        actual_quali=actual_rows,
        actual_race=actual_rows,
        target_qualifying_rows=target_rows,
        target_race_rows=target_rows,
    )

    report = build_report(2026, predictions_dir)

    assert report["qualifying_accuracy"]["mae"] == 0.0
    assert report["race_accuracy"]["mae"] == 0.0
    assert report["calibration"]["races_with_band_data"] == 1.0
    assert report["calibration"]["total_races_evaluated"] == 1.0


def test_track_type_is_read_by_circuit_for_a_moved_race():
    """The 2026 Bahrain GP ran at Sepang: its segment reads Sepang's entry, not Sakhir's."""
    track_types = {"Bahrain Grand Prix": "street", "Malaysian Grand Prix": "permanent"}

    from scripts.generate_evaluation_report import _track_type_for_race

    assert _track_type_for_race(track_types, "Bahrain Grand Prix", 2026) == "permanent"
    assert _track_type_for_race(track_types, "Bahrain Grand Prix", 2025) == "street"
