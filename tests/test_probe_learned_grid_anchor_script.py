"""Tests for the learned grid anchor sizing probe."""

from __future__ import annotations

import numpy as np
from scripts.probe_learned_grid_anchor import Checkpoint, fit_weight, rebuilt_positions


def _checkpoint(sim, grid, actual, weight=0.5, max_gain=30.0) -> Checkpoint:
    n = len(sim)
    weights = np.full(n, weight) if np.isscalar(weight) else np.array(weight, dtype=float)
    return Checkpoint(
        race="R",
        session="FP1",
        predicted_at="2026-01-01",
        drivers=[f"D{i}" for i in range(n)],
        sim=np.array(sim, dtype=float),
        grid=np.array(grid, dtype=float),
        weight=weights,
        racecraft=np.zeros(n),
        max_gain=np.full(n, max_gain),
        saved=np.arange(1, n + 1, dtype=float),
        actual=np.array(actual, dtype=float),
    )


def test_fit_recovers_the_weight_that_made_the_result():
    sim = [1.0, 5.0, 2.0, 8.0, 3.0]
    grid = [4.0, 1.0, 6.0, 2.0, 9.0]
    actual = [0.3 * s + 0.7 * g for s, g in zip(sim, grid, strict=True)]

    assert np.isclose(fit_weight([_checkpoint(sim, grid, actual)]), 0.7)


def test_fit_is_clipped_to_the_unit_interval():
    sim, grid = [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]

    assert fit_weight([_checkpoint(sim, grid, [5.0, 2.0, -1.0])]) == 1.0
    assert fit_weight([_checkpoint(sim, grid, [-1.0, 2.0, 5.0])]) == 0.0


def test_rebuild_ranks_the_blend_and_leaves_a_penalised_driver_unanchored():
    cp = _checkpoint(
        sim=[1.0, 2.0, 3.0], grid=[3.0, 2.0, 1.5], actual=[1, 2, 3], weight=[0.0, 0.9, 0.9]
    )

    # Weight 1 would put the grid order first, but D0 is penalised and keeps the sim P1.
    assert rebuilt_positions(cp, 1.0).tolist() == [1.0, 3.0, 2.0]
    assert rebuilt_positions(cp, 0.0).tolist() == [1.0, 2.0, 3.0]
