"""Tests for session selector weekend-type compatibility."""

import pytest

from src.utils.session_selector import (
    calculate_overtaking_difficulty,
    get_prediction_context,
    get_prediction_workflow,
    map_session_name_to_key,
    select_best_session,
)


def test_get_prediction_context_accepts_conventional_alias():
    """`conventional` should be accepted as alias for legacy `normal`."""
    context = get_prediction_context("post_fp2", weekend_type="conventional")
    assert context["next_prediction"] == "qualifying"


def test_get_prediction_workflow_accepts_conventional_alias():
    """Workflow helper should accept `conventional` weekend type."""
    workflow = get_prediction_workflow("conventional")
    assert len(workflow) > 0


def test_get_prediction_workflow_supports_sprint_weekends():
    workflow = get_prediction_workflow("sprint")
    assert workflow[-1]["prediction"] == "Main Race prediction"


def test_get_prediction_context_unknown_session_defaults_to_pre_fp1():
    context = get_prediction_context("post_unknown", weekend_type="normal")
    assert context["next_prediction"] == "qualifying"
    assert context["available"] == []


def test_get_prediction_context_rejects_invalid_weekend_type():
    with pytest.raises(ValueError, match="Unsupported weekend_type"):
        get_prediction_context("post_fp1", weekend_type="weird")


def test_map_session_name_to_key_supports_exact_and_suffix_matching():
    sessions = {
        "fp1": {"score": 0.3},
        "2026_bahrain_fp2": {"score": 0.7},
    }
    assert map_session_name_to_key("fp1", sessions) == "fp1"
    assert map_session_name_to_key("fp2", sessions) == "2026_bahrain_fp2"
    assert map_session_name_to_key("fp3", sessions) is None


def test_select_best_session_returns_baseline_when_no_data_available():
    best_session, confidence, reasoning = select_best_session(
        team_sessions={},
        track_chars={},
        current_session="pre_fp1",
        weekend_type="conventional",
        prediction_target="qualifying",
    )

    assert best_session is None
    assert confidence == pytest.approx(0.15)
    assert "baseline" in reasoning.lower()


def test_select_best_session_qualifying_prefers_fp3():
    team_sessions = {
        "fp1": {"session": "fp1"},
        "fp2": {"session": "fp2"},
        "fp3": {"session": "fp3"},
    }
    best_session, confidence, reasoning = select_best_session(
        team_sessions=team_sessions,
        track_chars={"corner_density_z": 0.1},
        current_session="post_fp3",
        weekend_type="normal",
        prediction_target="qualifying",
    )

    assert best_session == {"session": "fp3"}
    assert confidence == pytest.approx(0.85)
    assert "fp3" in reasoning


def test_select_best_session_race_prefers_main_quali_on_tight_track():
    best_session, confidence, reasoning = select_best_session(
        team_sessions={
            "main_quali": {"session": "main_quali"},
            "fp3": {"session": "fp3"},
        },
        track_chars={
            "is_street_circuit_z": 1.0,
            "corner_density_z": 1.0,
            "full_throttle_pct_z": -1.0,
        },
        current_session="post_quali",
        weekend_type="sprint",
        prediction_target="race",
    )

    assert best_session == {"session": "main_quali"}
    assert confidence == pytest.approx(0.8)
    assert "main_quali" in reasoning


def test_select_best_session_race_prefers_sprint_data_on_easy_track():
    best_session, confidence, reasoning = select_best_session(
        team_sessions={
            "sprint": {"session": "sprint"},
            "main_quali": {"session": "main_quali"},
        },
        track_chars={
            "is_street_circuit_z": 0.0,
            "corner_density_z": 0.0,
            "full_throttle_pct_z": 0.5,
        },
        current_session="post_quali",
        weekend_type="sprint",
        prediction_target="race",
    )

    assert best_session == {"session": "sprint"}
    assert confidence == pytest.approx(0.8)
    assert "sprint" in reasoning


def test_calculate_overtaking_difficulty_clips_to_unit_interval():
    hard = calculate_overtaking_difficulty(
        {"is_street_circuit_z": 10.0, "corner_density_z": 10.0, "full_throttle_pct_z": -10.0}
    )
    easy = calculate_overtaking_difficulty(
        {"is_street_circuit_z": -10.0, "corner_density_z": -10.0, "full_throttle_pct_z": 10.0}
    )

    assert hard == 1.0
    assert easy == 0.0
