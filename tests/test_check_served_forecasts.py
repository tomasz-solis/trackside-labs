"""Tests for the served forecast rollback check."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_served_forecasts.py"
    spec = importlib.util.spec_from_file_location("check_served_forecasts_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload() -> dict:
    drivers = ["VER", "NOR", "LEC", "HAM"]
    win = [50.0, 30.0, 15.0, 5.0]
    podium = [90.0, 80.0, 75.0, 55.0]
    finish = [
        {"driver": d, "position": i, "win_probability": w, "podium_probability": p}
        for i, (d, w, p) in enumerate(zip(drivers, win, podium, strict=True), start=1)
    ]
    grid = [
        {"driver": d, "position": i, "champion_position": i} for i, d in enumerate(drivers, start=1)
    ]
    return {
        "metadata": {"model_version": "3.9", "boundary_session_name": "PRE"},
        "weather": "dry",
        "prediction_results": {
            "race": {"finish_order": finish, "starting_grid": copy.deepcopy(grid)},
            "qualifying": {"grid": grid, "qualifying_order_source": "qualifying_ridge"},
        },
    }


def test_a_sound_payload_passes():
    assert _load_module().payload_problems(_payload(), "3.9") == []


def test_each_rollback_condition_fails():
    module = _load_module()
    cases = {
        "model version 3.7": lambda s: s["metadata"].update(model_version="3.7"),
        "positions are not 1 to 4": lambda s: s["prediction_results"]["race"]["finish_order"][
            3
        ].update(position=5),
        "bad or repeated driver codes ['nan']": lambda s: [
            row.update(driver="nan")
            for section in (
                s["prediction_results"]["race"]["finish_order"],
                s["prediction_results"]["qualifying"]["grid"],
            )
            for row in section
            if row["driver"] == "HAM"
        ],
        "below the same driver's win chance": lambda s: s["prediction_results"]["race"][
            "finish_order"
        ][0].update(podium_probability=12.0),
        "win chances sum to 105": lambda s: s["prediction_results"]["race"]["finish_order"][
            3
        ].update(win_probability=10.0),
        "not served by the ridge": lambda s: s["prediction_results"]["qualifying"].pop(
            "qualifying_order_source"
        ),
        "lack champion_position": lambda s: s["prediction_results"]["qualifying"]["grid"][0].pop(
            "champion_position"
        ),
        "drivers differ": lambda s: s["prediction_results"]["qualifying"]["grid"][3].update(
            driver="ALO"
        ),
    }
    for expected, break_it in cases.items():
        state = copy.deepcopy(_payload())
        break_it(state)
        problems = module.payload_problems(state, "3.9")
        assert any(expected in p for p in problems), (expected, problems)


def test_actual_qualifying_needs_no_ridge():
    state = _payload()
    qualifying = state["prediction_results"]["qualifying"]
    qualifying.pop("qualifying_order_source")
    qualifying["result_mode"] = "actual"
    assert _load_module().payload_problems(state, "3.9") == []


def test_accuracy_fails_only_when_every_checkpoint_is_below_zero():
    module = _load_module()
    key = "2026::Qatar Grand Prix::{}::grand_prix_race"
    one_bad = {
        key.format("PRE"): {"metrics": {"correlation": -0.1}},
        key.format("Q"): {"metrics": {"correlation": 0.4}},
    }
    all_bad = {
        key.format("PRE"): {"metrics": {"correlation": -0.1}},
        key.format("Q"): {"metrics": {"correlation": -0.2}},
    }
    assert module.accuracy_problems(one_bad) == []
    assert module.accuracy_problems(all_bad) == [
        "grand_prix_race: correlation below 0 at all 2 checkpoints"
    ]


def test_missing_grids_and_missing_coverage_fail():
    module = _load_module()
    no_grids = _payload()
    del no_grids["prediction_results"]["qualifying"]["grid"]
    assert "no qualifying grid" in module.payload_problems(no_grids, "3.9")
    no_start = _payload()
    del no_start["prediction_results"]["race"]["starting_grid"]
    assert "no starting grid" in module.payload_problems(no_start, "3.9")

    def row(checkpoint, weather):
        return {"state": {"metadata": {"boundary_session_name": checkpoint}, "weather": weather}}

    full = [row("PRE", "dry"), row("PRE", "rain"), row("FP1", "dry"), row("FP1", "rain")]
    assert module.coverage_problems(full, ["dry", "rain"]) == []
    assert module.coverage_problems(full[:3], ["dry", "rain"]) == ["FP1: no rain scenario"]
    assert module.coverage_problems(full[2:], ["dry", "rain"]) == ["no PRE payload"]
    # Qatar 2026 still stores a 3.7 "mixed" PRE from before the scenario was dropped.
    assert module.served_rows([*full, row("PRE", "mixed")], ["dry", "rain"]) == full
