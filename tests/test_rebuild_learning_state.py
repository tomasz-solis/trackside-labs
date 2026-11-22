"""Tests for ``scripts/rebuild_learning_state.py --sprint-only`` (plan row 25)."""

from __future__ import annotations

import copy

from scripts.rebuild_learning_state import with_rebuilt_sprint_history


def _state(qualifying: list, race: list, sprint: list | None) -> dict:
    history = {"qualifying": qualifying, "race": race}
    if sprint is not None:
        history["sprint"] = sprint
    return {
        "season": 2026,
        "adaptive_calibration": {
            "driver_position_error": {"qualifying": {"VER": 1.0}, "race": {"VER": 2.0}},
            "interval_residual_history": history,
        },
    }


def test_sprint_only_replaces_the_sprint_residuals_and_nothing_else():
    stored = _state([1.0, 2.0], [3.0], None)
    rebuilt = _state([9.0], [9.0], [0.5, 1.5])
    before = copy.deepcopy(stored)

    merged = with_rebuilt_sprint_history(stored, rebuilt)

    history = merged["adaptive_calibration"]["interval_residual_history"]
    assert history["sprint"] == [0.5, 1.5]
    del history["sprint"]
    assert merged == before
    assert stored == before
