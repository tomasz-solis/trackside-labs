"""Tests for adaptive FP blend weight calculation."""

import pytest

from src.utils.adaptive_fp_blending import calculate_adaptive_blend_weight


def test_baseline_weight_with_perfect_conditions():
    """Perfect conditions return base weight of 0.70."""
    session_data = {
        "session_type": "FP2",
        "weather": "dry",
        "total_laps": 50,
        "track_limits_violations": 10,
    }

    weight = calculate_adaptive_blend_weight(
        session_data=session_data,
        predicted_race_weather="dry",
        track_name="Bahrain Grand Prix",
    )

    assert weight == 0.70


def test_weather_mismatch_reduces_weight():
    """Weather mismatch (FP dry, race wet) reduces weight by 0.15."""
    session_data = {
        "session_type": "FP2",
        "weather": "dry",
        "total_laps": 50,
        "track_limits_violations": 10,
    }

    weight = calculate_adaptive_blend_weight(
        session_data=session_data,
        predicted_race_weather="rain",  # Mismatch
        track_name="Bahrain Grand Prix",
    )

    assert weight == pytest.approx(0.55, abs=0.01)  # 0.70 - 0.15


def test_truncated_session_reduces_weight():
    """Truncated session (<30 laps) reduces weight by 0.10."""
    session_data = {
        "session_type": "FP2",
        "weather": "dry",
        "total_laps": 25,  # <30 laps
        "track_limits_violations": 10,
    }

    weight = calculate_adaptive_blend_weight(
        session_data=session_data,
        predicted_race_weather="dry",
        track_name="Bahrain Grand Prix",
    )

    assert weight == pytest.approx(0.60, abs=0.01)  # 0.70 - 0.10


def test_fp1_receives_evolution_penalty():
    """FP1 (Friday) receives 0.08 penalty for track evolution."""
    session_data = {
        "session_type": "FP1",
        "weather": "dry",
        "total_laps": 50,
        "track_limits_violations": 10,
    }

    weight = calculate_adaptive_blend_weight(
        session_data=session_data,
        predicted_race_weather="dry",
        track_name="Bahrain Grand Prix",
    )

    assert weight == pytest.approx(0.62, abs=0.01)  # 0.70 - 0.08


def test_fp3_receives_evolution_bonus():
    """FP3 (Saturday) receives 0.05 bonus for track evolution."""
    session_data = {
        "session_type": "FP3",
        "weather": "dry",
        "total_laps": 50,
        "track_limits_violations": 10,
    }

    weight = calculate_adaptive_blend_weight(
        session_data=session_data,
        predicted_race_weather="dry",
        track_name="Bahrain Grand Prix",
    )

    assert weight == pytest.approx(0.75, abs=0.01)  # 0.70 + 0.05


def test_high_track_limits_violations_reduce_weight():
    """Excessive track limits violations (>50) reduce weight by 0.05."""
    session_data = {
        "session_type": "FP2",
        "weather": "dry",
        "total_laps": 50,
        "track_limits_violations": 75,  # >50
    }

    weight = calculate_adaptive_blend_weight(
        session_data=session_data,
        predicted_race_weather="dry",
        track_name="Bahrain Grand Prix",
    )

    assert weight == pytest.approx(0.65, abs=0.01)  # 0.70 - 0.05


def test_street_circuit_receives_penalty():
    """Street circuits receive 0.05 penalty due to track evolution."""
    session_data = {
        "session_type": "FP2",
        "weather": "dry",
        "total_laps": 50,
        "track_limits_violations": 10,
    }

    weight = calculate_adaptive_blend_weight(
        session_data=session_data,
        predicted_race_weather="dry",
        track_name="Monaco Grand Prix",  # Street circuit
    )

    assert weight == pytest.approx(0.65, abs=0.01)  # 0.70 - 0.05


def test_combined_penalties_monaco_fp1_red_flag():
    """Worst case: Monaco FP1 with red flag and weather mismatch."""
    session_data = {
        "session_type": "FP1",
        "weather": "rain",
        "total_laps": 15,  # Red flagged
        "track_limits_violations": 5,
    }

    weight = calculate_adaptive_blend_weight(
        session_data=session_data,
        predicted_race_weather="dry",  # Weather mismatch
        track_name="Monaco Grand Prix",  # Street circuit
    )

    # Penalties: -0.15 (weather) -0.10 (truncated) -0.08 (FP1) -0.05 (street)
    # = 0.70 - 0.38 = 0.32, but clamped to 0.50 minimum
    assert weight == pytest.approx(0.50, abs=0.01)


def test_best_case_fp3_dry():
    """Best case: FP3 with perfect conditions."""
    session_data = {
        "session_type": "FP3",
        "weather": "dry",
        "total_laps": 60,
        "track_limits_violations": 5,
    }

    weight = calculate_adaptive_blend_weight(
        session_data=session_data,
        predicted_race_weather="dry",
        track_name="Silverstone Circuit",  # Permanent circuit
    )

    # Bonus: +0.05 (FP3)
    # = 0.70 + 0.05 = 0.75
    assert weight == pytest.approx(0.75, abs=0.01)


def test_weight_clamped_to_valid_range():
    """Weight is always clamped to 0.50-0.85."""
    # Test lower bound
    session_data_bad = {
        "session_type": "FP1",
        "weather": "rain",
        "total_laps": 10,
        "track_limits_violations": 100,
    }

    weight_low = calculate_adaptive_blend_weight(
        session_data=session_data_bad,
        predicted_race_weather="dry",
        track_name="Monaco Grand Prix",
    )

    assert 0.50 <= weight_low <= 0.85

    # Test upper bound (can't exceed 0.85 even with all bonuses)
    session_data_good = {
        "session_type": "FP3",
        "weather": "dry",
        "total_laps": 100,
        "track_limits_violations": 0,
    }

    weight_high = calculate_adaptive_blend_weight(
        session_data=session_data_good,
        predicted_race_weather="dry",
        track_name="Bahrain Grand Prix",
    )

    assert 0.50 <= weight_high <= 0.85
