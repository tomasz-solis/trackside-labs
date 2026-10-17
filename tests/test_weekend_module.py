"""Focused tests for weekend-type utilities and fallback schedule behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils import weekend


def test_get_schedule_rows_uses_track_fallback_when_fastf1_empty(patcher, tmp_path):
    patcher.chdir(tmp_path)
    weekend.refresh_schedule_cache()

    fallback_file = Path("data/processed/track_characteristics/2027_track_characteristics.json")
    fallback_file.parent.mkdir(parents=True, exist_ok=True)
    fallback_file.write_text(
        json.dumps(
            {
                "tracks": {
                    "Chinese Grand Prix": {"has_sprint": True},
                    "Bahrain Grand Prix": {"has_sprint": False},
                }
            }
        )
    )

    patcher.setattr(
        weekend.fastf1,
        "get_event_schedule",
        lambda year: pd.DataFrame(columns=["EventName", "EventFormat"]),
    )

    rows = weekend._get_schedule_rows(2027)

    assert ("Chinese Grand Prix", "sprint") in rows
    assert ("Bahrain Grand Prix", "conventional") in rows


def test_get_schedule_rows_supplements_missing_fastf1_races_from_fallback(patcher, tmp_path):
    patcher.chdir(tmp_path)
    weekend.refresh_schedule_cache()

    fallback_file = Path("data/processed/track_characteristics/2027_track_characteristics.json")
    fallback_file.parent.mkdir(parents=True, exist_ok=True)
    fallback_file.write_text(
        json.dumps(
            {
                "tracks": {
                    "Chinese Grand Prix": {"has_sprint": True},
                    "Bahrain Grand Prix": {"has_sprint": False},
                }
            }
        )
    )
    patcher.setattr(
        weekend.fastf1,
        "get_event_schedule",
        lambda year: pd.DataFrame(
            {
                "EventName": ["Chinese Grand Prix"],
                "EventFormat": ["sprint"],
            }
        ),
    )

    rows = weekend._get_schedule_rows(2027)

    assert rows.count(("Chinese Grand Prix", "sprint")) == 1
    assert ("Bahrain Grand Prix", "conventional") in rows


def test_refresh_schedule_cache_forces_new_fastf1_fetch(patcher):
    weekend.refresh_schedule_cache()

    schedules = [
        pd.DataFrame(
            {"EventName": ["Chinese Grand Prix"], "EventFormat": ["sprint"]},
        ),
        pd.DataFrame(
            {"EventName": ["Chinese Grand Prix"], "EventFormat": ["conventional"]},
        ),
    ]

    call_count = {"n": 0}

    def _get_event_schedule(year: int):
        current = schedules[min(call_count["n"], len(schedules) - 1)]
        call_count["n"] += 1
        return current

    patcher.setattr(weekend.fastf1, "get_event_schedule", _get_event_schedule)

    first = weekend.is_sprint_weekend(2026, "Chinese Grand Prix")
    weekend.refresh_schedule_cache()
    second = weekend.is_sprint_weekend(2026, "Chinese Grand Prix")

    assert first is True
    assert second is False
    assert call_count["n"] >= 2


def test_get_all_conventional_races(patcher, tmp_path):
    patcher.chdir(tmp_path)
    weekend.refresh_schedule_cache()
    patcher.setattr(
        weekend.fastf1,
        "get_event_schedule",
        lambda year: pd.DataFrame(
            {
                "EventName": ["Chinese Grand Prix", "Australian Grand Prix"],
                "EventFormat": ["sprint_shootout", "conventional"],
            }
        ),
    )

    conventional_races = weekend.get_all_conventional_races(2026)
    assert "Australian Grand Prix" in conventional_races
    assert "Chinese Grand Prix" not in conventional_races


def test_get_schedule_rows_trusts_fastf1_over_the_cancelled_list(patcher, caplog):
    """FastF1 is the source of truth: a listed race it still carries is kept, with a warning.

    The cancelled list once hid the 2026 Bahrain GP after it was reinstated at Sepang.
    """
    weekend.refresh_schedule_cache()
    patcher.setattr(
        weekend.fastf1,
        "get_event_schedule",
        lambda year: pd.DataFrame(
            {
                "EventName": [
                    "Pre-Season Testing",
                    "Australian Grand Prix",
                    "Bahrain Grand Prix",
                    "Saudi Arabian Grand Prix",
                    "Chinese Grand Prix",
                ],
                "EventFormat": [
                    "testing",
                    "conventional",
                    "conventional",
                    "conventional",
                    "sprint",
                ],
            }
        ),
    )

    rows = weekend._get_schedule_rows(2026)

    assert ("Australian Grand Prix", "conventional") in rows
    assert ("Chinese Grand Prix", "sprint") in rows
    assert all(event_name != "Pre-Season Testing" for event_name, _event_format in rows)
    assert ("Bahrain Grand Prix", "conventional") in rows
    assert ("Saudi Arabian Grand Prix", "conventional") in rows
    assert "cancelled list" in caplog.text


def test_sepang_track_key_is_not_a_2026_event():
    """The Malaysian GP key holds Sepang's track data; it must never appear as a race."""
    assert weekend.should_skip_schedule_event(2026, "Malaysian Grand Prix")


def test_is_sprint_weekend_raises_when_lookup_fails(patcher):
    patcher.setattr(
        weekend,
        "get_weekend_type",
        lambda year, race_name: (_ for _ in ()).throw(ValueError("missing race")),
    )

    with pytest.raises(ValueError, match="missing race"):
        weekend.is_sprint_weekend(2026, "Missing Race")


def test_cancelled_list_still_filters_the_local_fallback_schedule():
    """Track-file keys that are not 2026 races (Saudi Arabia, Sepang's key) stay out."""
    assert weekend.should_skip_schedule_event(2026, "Saudi Arabian Grand Prix")
    assert not weekend.should_skip_schedule_event(2026, "Bahrain Grand Prix")


def test_schedule_falls_back_to_the_saved_fastf1_calendar(patcher):
    """The track file has no calendar order; when FastF1 is down use its last calendar.

    Rounds from the track file put Azerbaijan after Singapore, so the Mexico City and
    Sao Paulo PRE forecasts picked an Azerbaijan snapshot during the 2026-10 outage.
    """
    saved: dict[int, tuple[tuple[str, str], ...]] = {}
    calendar = pd.DataFrame(
        {
            "EventName": ["Azerbaijan Grand Prix", "Singapore Grand Prix"],
            "EventFormat": ["conventional", "sprint_qualifying"],
        }
    )
    patcher.setattr(
        weekend, "_save_schedule_snapshot", lambda year, rows: saved.update({year: rows})
    )
    patcher.setattr(weekend, "_load_saved_schedule_rows", lambda year: saved.get(year, ()))

    patcher.setattr(weekend.fastf1, "get_event_schedule", lambda year: calendar)
    weekend.refresh_schedule_cache()
    live_rows = weekend._get_schedule_rows(2026)

    def _down(year):
        raise RuntimeError("RateLimitExceededError: any API: 500 calls/h")

    patcher.setattr(weekend.fastf1, "get_event_schedule", _down)
    weekend.refresh_schedule_cache()
    outage_rows = weekend._get_schedule_rows(2026)
    weekend.refresh_schedule_cache()

    assert saved[2026] == (
        ("Azerbaijan Grand Prix", "conventional"),
        ("Singapore Grand Prix", "sprint_qualifying"),
    )
    assert outage_rows == saved[2026]
    assert live_rows[: len(outage_rows)] == outage_rows


def test_schedule_snapshot_is_a_no_op_in_file_mode():
    """Tests and local file runs must never write a calendar artifact."""
    from src.persistence.config import should_write_to_db

    assert should_write_to_db() is False
    weekend._save_schedule_snapshot(2026, (("A Grand Prix", "conventional"),))
    assert weekend._load_saved_schedule_rows(2026) == ()
    assert not Path("data/season_schedule").exists() or not any(
        Path("data/season_schedule").rglob("*.json")
    )
