"""Tests for the ML challenger dataset helpers and walk-forward fitting."""

import json

import numpy as np
import pandas as pd
import pytest

import src.models.ml_challenger as ml


def _frame(rounds: int = 8, drivers: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for rnd in range(1, rounds + 1):
        x = rng.normal(size=drivers)
        for d in range(drivers):
            rows.append(
                {
                    "race": f"R{rnd}",
                    "round": rnd,
                    "x": x[d],
                    "y": float(np.argsort(np.argsort(x))[d] + 1),
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_never_sees_the_predicted_race_or_later():
    data = _frame()
    before, _ = ml.walk_forward(data, "y", ["x"], ml.ridge_model)
    poisoned = data.copy()
    poisoned.loc[poisoned["round"] >= 6, "y"] = 1000.0  # later targets must not leak backwards
    after, _ = ml.walk_forward(poisoned, "y", ["x"], ml.ridge_model)
    early = data["round"] <= 6
    assert before[early].equals(after[early])
    assert before[data["round"] <= 4].isna().all()  # fewer than 4 training races: no forecast


def test_score_is_perfect_for_the_actual_order():
    data = _frame()
    result = ml.score(data, "y", data["y"])
    assert result["corr"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["top3"] == pytest.approx(1.0)


def test_practice_features_are_gaps_to_the_fastest_team(tmp_path):
    teams = {
        f"T{i}": {
            "profiles": {
                "short_run": {"overall_pace_seconds": 80.0 + i, "top_speed_kph": 300.0 + i}
            }
        }
        for i in range(6)
    }
    (tmp_path / "fp1.json").write_text(json.dumps({"teams": teams}))
    frame = ml.practice_features(tmp_path, ["fp1", "fp2"])  # missing fp2 is skipped
    assert frame.at["T0", "fp_short_gap_pct"] == pytest.approx(0.0)
    assert frame.at["T4", "fp_short_gap_pct"] == pytest.approx(5.0)
    assert frame.at["T0", "fp_top_speed_kph"] == pytest.approx(-2.5)
