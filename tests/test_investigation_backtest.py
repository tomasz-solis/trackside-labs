"""Tests for the row 33 investigation backtest, on synthetic weekends only."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TEAMS = [f"T{i}" for i in range(1, 9)]


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "investigation_backtest.py"
    spec = importlib.util.spec_from_file_location("investigation_backtest_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def _rows(order: list[str]) -> list[dict]:
    """Two drivers per team, the team's drivers side by side in ``order``."""
    rows = []
    for index, team in enumerate(order):
        rows += [
            {"driver": f"{team}a", "team": team, "position": 2 * index + 1},
            {"driver": f"{team}b", "team": team, "position": 2 * index + 2},
        ]
    return rows


def _snapshot(order: list[str], laps: int = 20) -> dict:
    teams = {
        team: {"profiles": {"short_run": {"overall_pace_seconds": 90.0 + 0.1 * index}}}
        for index, team in enumerate(order)
    }
    return {"teams": teams, "team_clean_lap_counts": {team: laps for team in order}}


def _units(module, *, practice, actual, wet=(), laps=20, drop_ends=False):
    return module.weekend_units(
        year=2026,
        race="Test GP",
        forecast=_rows(_TEAMS),
        actual=_rows(actual),
        snapshots={"FP1": _snapshot(practice, laps)},
        checkpoints=("FP1",),
        is_wet=lambda year, race, session: session in wet,
        min_laps=10,
        drop_ends=drop_ends,
    )


def test_a_practice_gap_that_qualifying_follows_is_supported():
    module = _load()
    # T6 is expected sixth, runs second in practice and qualifies third.
    practice = ["T1", "T6", "T2", "T3", "T4", "T5", "T7", "T8"]
    actual = ["T1", "T2", "T6", "T3", "T4", "T5", "T7", "T8"]
    t6 = next(u for u in _units(module, practice=practice, actual=actual) if u.team == "T6")
    assert (t6.expected_rank, t6.practice_rank, t6.gap, t6.outcome) == (6.0, 2.0, 4.0, "supported")

    flat = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    t6 = next(u for u in _units(module, practice=practice, actual=flat) if u.team == "T6")
    assert t6.outcome == "weakened"


def test_wet_qualifying_leaves_every_card_unresolved_and_wet_practice_is_no_evidence():
    module = _load()
    practice = ["T1", "T6", "T2", "T3", "T4", "T5", "T7", "T8"]
    units = _units(module, practice=practice, actual=_TEAMS, wet={"Q"})
    assert {u.outcome for u in units} == {"unresolved"}
    assert _units(module, practice=practice, actual=_TEAMS, wet={"FP1"}) == []


def test_too_few_laps_and_the_end_teams_are_left_out():
    module = _load()
    assert _units(module, practice=_TEAMS, actual=_TEAMS, laps=5) == []
    kept = {u.team for u in _units(module, practice=_TEAMS, actual=_TEAMS, drop_ends=True)}
    assert kept == set(_TEAMS) - {"T1", "T8"}


def test_the_summary_counts_cards_and_its_lifts_are_reproducible():
    module = _load()
    practice = ["T1", "T6", "T2", "T3", "T4", "T5", "T7", "T8"]
    supported = ["T1", "T2", "T6", "T3", "T4", "T5", "T7", "T8"]
    weekends = {
        "A GP": _units(module, practice=practice, actual=supported),
        "B GP": _units(module, practice=_TEAMS, actual=_TEAMS),
    }
    first = module.summarise(weekends, threshold=3, bands=4, shuffles=20, bootstrap=50)
    again = module.summarise(weekends, threshold=3, bands=4, shuffles=20, bootstrap=50)

    assert first == again
    assert (first["weekends"], first["weekends_without_card"], first["cards"]) == (2, 1, 1)
    assert (first["supported"], first["weakened"], first["unresolved"]) == (1, 0, 0)
    assert module.summarise({}, threshold=3, bands=4) == {"weekends": 0}
