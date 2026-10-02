"""Tests for the race likely-range quantile fitter."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.fit_race_band_quantiles import (
    Residual,
    bucket_of,
    fit_quantiles,
    leave_one_race_out_coverage,
    load_residuals,
)


def test_bucket_of_assigns_the_configured_edges():
    assert bucket_of(1) == "1-5"
    assert bucket_of(5) == "1-5"
    assert bucket_of(6) == "6-10"
    assert bucket_of(10) == "6-10"
    assert bucket_of(11) == "11-16"
    assert bucket_of(16) == "11-16"
    assert bucket_of(17) == "17-22"
    assert bucket_of(22) == "17-22"
    assert bucket_of(23) == "23+"


def test_fit_quantiles_computes_n_and_p25_p75_per_bucket():
    residuals = [
        Residual(race="R1", bucket="1-5", residual=value) for value in (-4.0, -2.0, 0.0, 2.0, 4.0)
    ] + [Residual(race="R1", bucket="6-10", residual=1.0)]

    fitted = fit_quantiles(residuals)

    assert fitted["1-5"]["n"] == 5
    assert fitted["1-5"]["q25"] == -2.0
    assert fitted["1-5"]["q75"] == 2.0
    assert fitted["6-10"] == {"n": 1, "q25": 1.0, "q75": 1.0}


def test_leave_one_race_out_coverage_pools_across_held_out_races():
    # Three races, each contributing one bucket-"1-5" residual. Two of the three
    # residuals are close (0.0), one is a wide outlier (10.0); holding out the
    # outlier's race should still cover it because the band is fit without it,
    # while the two close races cover each other more often than not.
    residuals = [
        Residual(race="R1", bucket="1-5", residual=0.0),
        Residual(race="R2", bucket="1-5", residual=0.5),
        Residual(race="R3", bucket="1-5", residual=10.0),
    ]

    coverage = leave_one_race_out_coverage(residuals)

    assert set(coverage) == {"1-5"}
    assert 0.0 <= coverage["1-5"] <= 100.0


def _write_checkpoint(
    race_dir: Path,
    filename: str,
    *,
    target: str,
    predicted_order: list[dict[str, object]],
    actuals: list[dict[str, object]],
) -> None:
    race_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "targets": {target: {"predicted_order": predicted_order}},
        "actuals": {"targets": {target: actuals}},
    }
    (race_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_load_residuals_pairs_predictions_with_actuals_and_drops_dnfs(tmp_path):
    replay_root = tmp_path / "replay"
    race_dir = replay_root / "predictions" / "2026" / "test_grand_prix"
    _write_checkpoint(
        race_dir,
        "test_grand_prix_pre.json",
        target="grand_prix_race",
        predicted_order=[
            {"driver": "VER", "position": 1, "position_blend_score": 1.5},
            {"driver": "NOR", "position": 2, "position_blend_score": 2.5},
        ],
        actuals=[
            {"driver": "VER", "position": 1, "dnf": False},
            {"driver": "NOR", "position": 15, "dnf": True},
        ],
    )

    residuals = load_residuals(replay_root, 2026, "grand_prix_race")

    assert len(residuals) == 1
    assert residuals[0].bucket == "1-5"
    assert residuals[0].residual == 0.0  # actual P1 minus shown P1; blend score ignored
