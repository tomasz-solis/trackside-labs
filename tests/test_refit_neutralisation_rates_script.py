"""Tests for the safety car and VSC rate refit script's counting and shift logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script_module():
    """Load the neutralisation-rate refit script as a module."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "refit_neutralisation_rates.py"
    spec = importlib.util.spec_from_file_location("refit_neutralisation_rates_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_neutralisations_in_reads_both_phrasings_and_ignores_other_messages():
    module = _load_script_module()
    vsc_2026 = [("SafetyCar", "VSC DEPLOYED"), ("SafetyCar", "VSC ENDING")]
    both_2023 = [
        ("SafetyCar", "VIRTUAL SAFETY CAR DEPLOYED"),
        ("SafetyCar", "SAFETY CAR DEPLOYED"),
        ("SafetyCar", "SAFETY CAR IN THIS LAP"),
    ]
    neither = [("Other", "SAFETY CAR LIGHTS ON"), ("Flag", "YELLOW IN TRACK SECTOR 4")]

    assert module.neutralisations_in(vsc_2026) == {"sc": False, "vsc": True}
    assert module.neutralisations_in(both_2023) == {"sc": True, "vsc": True}
    assert module.neutralisations_in(neither) == {"sc": False, "vsc": False}


def test_season_rate_weighs_history_as_one_extra_season():
    module = _load_script_module()
    # 60 history races at 50% weigh as 20 races: (10 + 8) / (20 + 10)
    assert module.season_rate(30, 60, 8, 10) == pytest.approx(0.6)


def test_logit_shift_hits_target_mean_and_keeps_track_order():
    module = _load_script_module()
    probs = {"street": 0.6, "mixed": 0.3, "open": 0.15}

    shifted = module.logit_shift(probs, 0.5)

    assert sum(shifted.values()) / len(shifted) == pytest.approx(0.5, abs=0.002)
    assert shifted["street"] > shifted["mixed"] > shifted["open"]
    assert all(shifted[key] > probs[key] for key in probs)


def test_refit_only_touches_calendar_tracks_with_the_field():
    module = _load_script_module()
    data = {
        "tracks": {
            "on calendar": {"safety_car_prob": 0.3, "vsc_prob": 0.2, "pit_stop_loss": 22.0},
            "no vsc field": {"safety_car_prob": 0.4},
            "off calendar": {"safety_car_prob": 0.5, "vsc_prob": 0.5},
        }
    }

    module.refit(data, {"on calendar", "no vsc field"}, {"sc": 0.5, "vsc": 0.4})

    tracks = data["tracks"]
    assert tracks["on calendar"]["vsc_prob"] == pytest.approx(0.4, abs=0.002)
    assert tracks["on calendar"]["pit_stop_loss"] == 22.0
    assert "vsc_prob" not in tracks["no vsc field"]
    assert tracks["off calendar"] == {"safety_car_prob": 0.5, "vsc_prob": 0.5}
