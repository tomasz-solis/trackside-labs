"""Tests for the calibrated 50% likely-range attached to the race finish order."""

from __future__ import annotations

from src.predictors.baseline.race.result_processing import assign_likely_range


class _Cfg:
    """Minimal cfg stub exposing one likely_range table per (race, sprint)."""

    def __init__(self, race: dict | None = None, sprint: dict | None = None):
        self._tables = {"race": race or {}, "sprint": sprint or {}}

    def get(self, key: str, default=None):
        prefix = "baseline_predictor.race.likely_range."
        if key.startswith(prefix):
            return self._tables.get(key[len(prefix) :], default)
        return default


_RACE_TABLE = {
    "1-5": {"q25": -2.0, "q75": 1.0},
    "6-10": {"q25": -3.0, "q75": 1.0},
    "11-16": {"q25": -4.0, "q75": 1.0},
    "17-22": {"q25": -5.0, "q75": -1.0},
}
_SPRINT_TABLE = {
    "1-5": {"q25": -1.0, "q75": 1.0},
    "6-10": {"q25": -1.0, "q75": 1.0},
    "11-16": {"q25": -1.0, "q75": 1.0},
    "17-22": {"q25": -1.0, "q75": 1.0},
}


def test_assign_likely_range_uses_the_bucket_for_the_final_position():
    finish_order = [
        {"driver": "A", "position": 1, "position_blend_score": 1.2},
        {"driver": "B", "position": 8, "position_blend_score": 8.1},
    ]

    assign_likely_range(
        finish_order=finish_order, is_sprint=False, field_size=22, cfg=_Cfg(race=_RACE_TABLE)
    )

    assert finish_order[0]["likely_lo"] == 1  # round(1.2 - 2.0) clipped to 1
    assert finish_order[0]["likely_hi"] == 2  # round(1.2 + 1.0)
    assert finish_order[1]["likely_lo"] == 5  # round(8.1 - 3.0)
    assert finish_order[1]["likely_hi"] == 9  # round(8.1 + 1.0)


def test_assign_likely_range_clips_to_field_and_keeps_lo_le_hi():
    # A wide offset table that would push lo below 1 and hi above the field size.
    wide_table = {"17-22": {"q25": -30.0, "q75": 5.0}}
    finish_order = [{"driver": "Z", "position": 22, "position_blend_score": 22.0}]

    assign_likely_range(
        finish_order=finish_order, is_sprint=False, field_size=22, cfg=_Cfg(race=wide_table)
    )

    row = finish_order[0]
    assert row["likely_lo"] == 1
    assert row["likely_hi"] == 22
    assert row["likely_lo"] <= row["likely_hi"]


def test_assign_likely_range_picks_the_sprint_table_when_is_sprint():
    finish_order = [{"driver": "A", "position": 3, "position_blend_score": 3.0}]

    assign_likely_range(
        finish_order=finish_order,
        is_sprint=True,
        field_size=22,
        cfg=_Cfg(race=_RACE_TABLE, sprint=_SPRINT_TABLE),
    )

    row = finish_order[0]
    assert row["likely_lo"] == 2  # sprint table q25 -1.0
    assert row["likely_hi"] == 4  # sprint table q75 +1.0


def test_assign_likely_range_is_a_no_op_without_a_config_table():
    finish_order = [{"driver": "A", "position": 1, "position_blend_score": 1.0}]

    assign_likely_range(finish_order=finish_order, is_sprint=False, field_size=22, cfg=_Cfg())

    assert "likely_lo" not in finish_order[0]
    assert "likely_hi" not in finish_order[0]
