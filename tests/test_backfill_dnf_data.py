"""Tests for the DNF backfill script's merge logic (non-destructive actuals update)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _restore_event_writes(monkeypatch):
    """backfill_store suspends event writes for the process; undo it after each test."""
    from src.utils import operational_observability as obs

    monkeypatch.setattr(obs, "_db_writes_disabled_until", obs._db_writes_disabled_until)


def _load_backfill_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_dnf_data.py"
    spec = importlib.util.spec_from_file_location("backfill_dnf_data_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_dnf_preserves_order_and_only_adds_flag():
    module = _load_backfill_module()
    existing = [
        {"position": 1, "driver": "ANT", "team": "Mercedes"},
        {"position": 2, "driver": "RUS", "team": "Mercedes"},
        {"position": 20, "driver": "VER", "team": "Red Bull Racing"},
    ]
    # Fresh fetch may differ in order; only the per-driver DNF flag should be used.
    fetched_dnf = {"ANT": False, "RUS": False, "VER": True}
    labelled = module._merge_dnf_into_block(existing, fetched_dnf)

    assert labelled == 1
    # Positions/drivers/teams untouched; dnf attached.
    assert [r["driver"] for r in existing] == ["ANT", "RUS", "VER"]
    assert existing[0] == {"position": 1, "driver": "ANT", "team": "Mercedes", "dnf": False}
    assert existing[2]["dnf"] is True


def test_match_session_for_block_picks_best_order_match():
    module = _load_backfill_module()
    block = [
        {"position": 1, "driver": "RUS"},
        {"position": 2, "driver": "PIA"},
        {"position": 3, "driver": "NOR"},
    ]
    fetched = {
        "R": [{"driver": "ANT"}, {"driver": "RUS"}, {"driver": "HAM"}],  # GP order
        "Sprint": [{"driver": "RUS"}, {"driver": "PIA"}, {"driver": "NOR"}],  # matches block
    }
    assert module._match_session_for_block(block, fetched) == "Sprint"


def test_match_session_for_block_rejects_zero_match():
    # GP fetch failed; the Sprint rows share no position with the block, so its
    # DNF flags must not be merged onto it.
    module = _load_backfill_module()
    block = [{"driver": "ANT"}, {"driver": "NOR"}]
    fetched = {"R": None, "Sprint": [{"driver": "RUS"}, {"driver": "VER"}]}
    assert module._match_session_for_block(block, fetched) is None


def test_match_session_for_block_rejects_tie():
    module = _load_backfill_module()
    block = [{"driver": "RUS"}, {"driver": "ANT"}]
    fetched = {
        "R": [{"driver": "RUS"}, {"driver": "NOR"}],
        "Sprint": [{"driver": "RUS"}, {"driver": "VER"}],
    }
    assert module._match_session_for_block(block, fetched) is None


def test_backfill_actuals_labels_targets_and_legacy_block():
    module = _load_backfill_module()
    prediction = {
        "actuals": {
            "race": [  # sprint weekend: legacy race mirrors the Sprint
                {"position": 1, "driver": "RUS", "team": "Mercedes"},
                {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
            ],
            "targets": {
                "grand_prix_race": [
                    {"position": 1, "driver": "ANT", "team": "Mercedes"},
                    {"position": 2, "driver": "NOR", "team": "McLaren"},
                ],
                "sprint_race": [
                    {"position": 1, "driver": "RUS", "team": "Mercedes"},
                    {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
                ],
            },
        }
    }
    fetched = {
        "R": [
            {"position": 1, "driver": "ANT", "team": "Mercedes", "dnf": False},
            {"position": 2, "driver": "NOR", "team": "McLaren", "dnf": True},
        ],
        "Sprint": [
            {"position": 1, "driver": "RUS", "team": "Mercedes", "dnf": False},
            {"position": 2, "driver": "VER", "team": "Red Bull Racing", "dnf": True},
        ],
    }
    labelled = module._backfill_actuals(prediction, fetched=fetched)

    # 1 DNF in the GP target + 1 in the sprint target.
    assert labelled == 2
    gpr = prediction["actuals"]["targets"]["grand_prix_race"]
    assert gpr[1]["driver"] == "NOR" and gpr[1]["dnf"] is True
    assert gpr[0]["dnf"] is False
    # Legacy race block (the sprint) gets VER's DNF from the matched Sprint session.
    legacy = prediction["actuals"]["race"]
    assert legacy[1]["driver"] == "VER" and legacy[1]["dnf"] is True


def test_merge_is_noop_without_fetched_data():
    module = _load_backfill_module()
    rows = [{"position": 1, "driver": "VER", "team": "RBR"}]
    assert module._merge_dnf_into_block(rows, {}) == 0
    assert "dnf" not in rows[0]


class _FakeStore:
    storage_mode = "db_only"

    def __init__(self, rows):
        self.rows = rows
        self.saved = []

    def list_artifacts(self, artifact_type, key_prefix, limit):
        return self.rows

    def save_artifact(self, **kwargs):
        self.saved.append(kwargs)


class _FakeLogger:
    def __init__(self, store):
        self.artifact_store = store

    def _prediction_payload_from_listing_row(self, row):
        return row["data"]

    def _deduplicate_predictions(self, predictions):
        return predictions


def _stored(race, session, dnf_flag=None):
    row = {"position": 1, "driver": "NOR", "team": "McLaren"}
    if dnf_flag is not None:
        row["dnf"] = dnf_flag
    return {
        "data": {
            "metadata": {"year": 2026, "race_name": race, "session_name": session, "run_id": "r1"},
            "actuals": {"targets": {"grand_prix_race": [row]}},
        }
    }


def test_backfill_store_exports_before_saving_and_skips_labelled(tmp_path):
    """Store mode labels only unlabelled records, after writing the originals out."""
    import json

    import pytest

    module = _load_backfill_module()
    store = _FakeStore(
        [_stored("Monaco Grand Prix", "FP1"), _stored("Dutch Grand Prix", "PRE", False)]
    )

    def fetch(year, race):
        # NOR retired at Monaco; the Dutch record already holds the right flag.
        return {"R": [{"driver": "NOR", "dnf": race == "Monaco Grand Prix"}], "Sprint": None}

    def run(**kwargs):
        return module.backfill_store(2026, logger_inst=_FakeLogger(store), fetch_fn=fetch, **kwargs)

    assert run(export_path=None, dry_run=True) == 1
    assert store.saved == []
    assert "dnf" not in store.rows[0]["data"]["actuals"]["targets"]["grand_prix_race"][0]
    with pytest.raises(SystemExit):
        run(export_path=None, dry_run=False)
    assert store.saved == []

    export = tmp_path / "originals.json"
    assert run(export_path=export, dry_run=False) == 1
    originals = json.loads(export.read_text(encoding="utf-8"))
    assert [o["metadata"]["race_name"] for o in originals] == ["Monaco Grand Prix"]
    assert "dnf" not in originals[0]["actuals"]["targets"]["grand_prix_race"][0]
    (saved,) = store.saved
    assert saved["artifact_key"] == "2026::Monaco Grand Prix::FP1"
    assert saved["data"]["actuals"]["targets"]["grand_prix_race"][0]["dnf"] is True


def test_backfill_store_skips_races_without_results_and_writes_no_events(monkeypatch):
    """A race not run yet is never fetched, and no fetch can write an operational event."""
    from src.utils import operational_observability as obs

    monkeypatch.setattr(obs, "_db_writes_disabled_until", 0.0)
    monkeypatch.setattr(obs, "should_write_to_db", lambda: True)
    inserts = []

    class _Client:
        def table(self, name):
            inserts.append(name)
            raise AssertionError("operational event written")

    monkeypatch.setattr(obs, "get_supabase_client", _Client)
    module = _load_backfill_module()
    future = {
        "data": {"metadata": {"year": 2026, "race_name": "Qatar Grand Prix", "session_name": "PRE"}}
    }
    store = _FakeStore([future, _stored("Monaco Grand Prix", "FP1")])
    fetched = []

    def fetch(year, race):
        fetched.append(race)
        obs.record_counter("fastf1_results_empty_total", labels={"race_name": race})
        return {"R": [{"driver": "NOR", "dnf": True}], "Sprint": None}

    module.backfill_store(
        2026, logger_inst=_FakeLogger(store), fetch_fn=fetch, export_path=None, dry_run=True
    )
    assert fetched == ["Monaco Grand Prix"]
    assert inserts == []


def test_backfill_store_reports_and_refuses_a_flip(tmp_path, caplog):
    """A stored flag the fetch disagrees with is a flip: counted, and the write stops."""
    import pytest

    module = _load_backfill_module()
    store = _FakeStore([_stored("Monaco Grand Prix", "FP1", False)])

    def fetch(year, race):
        return {"R": [{"driver": "NOR", "dnf": True}], "Sprint": None}

    def run(**kwargs):
        return module.backfill_store(2026, logger_inst=_FakeLogger(store), fetch_fn=fetch, **kwargs)

    with caplog.at_level("INFO", logger="backfill_dnf"):
        assert run(export_path=None, dry_run=True) == 1
    assert "0 Grand Prix, 0 sprint, 0 legacy race; 1 flips, nothing written" in caplog.text
    export = tmp_path / "originals.json"
    with pytest.raises(SystemExit, match="1 stored flag"):
        run(export_path=export, dry_run=False)
    assert store.saved == [] and not export.exists()


def test_flag_changes_counts_added_flags_per_block():
    module = _load_backfill_module()
    original = {
        "actuals": {
            "race": [{"driver": "NOR"}, {"driver": "VER"}],
            "targets": {"grand_prix_race": [{"driver": "NOR", "dnf": True}]},
        }
    }
    updated = {
        "actuals": {
            "race": [{"driver": "NOR", "dnf": True}, {"driver": "VER", "dnf": False}],
            "targets": {"grand_prix_race": [{"driver": "NOR", "dnf": True}]},
        }
    }
    added, flips = module._flag_changes(original, updated)
    assert added == {"race": 2, "grand_prix_race": 0, "sprint_race": 0}
    assert flips == 0


def test_backfill_store_keeps_the_stored_run_id(tmp_path):
    """An actuals update stamps its own run_id column; the label write must not reset it."""
    module = _load_backfill_module()
    row = {**_stored("Monaco Grand Prix", "FP1"), "artifact_key": "2026::Monaco Grand Prix::FP1"}
    row["run_id"] = "11111111-1111-1111-1111-111111111111"
    store = _FakeStore([row])

    def fetch(year, race):
        return {"R": [{"driver": "NOR", "dnf": True}], "Sprint": None}

    module.backfill_store(
        2026,
        logger_inst=_FakeLogger(store),
        fetch_fn=fetch,
        export_path=tmp_path / "originals.json",
        dry_run=False,
    )
    (saved,) = store.saved
    assert saved["run_id"] == "11111111-1111-1111-1111-111111111111"
    assert saved["data"]["metadata"]["run_id"] == "r1"


def test_backfill_store_refuses_file_mode():
    module = _load_backfill_module()
    store = _FakeStore([])
    store.storage_mode = "file_only"
    import pytest

    with pytest.raises(SystemExit):
        module.backfill_store(2026, logger_inst=_FakeLogger(store), export_path=None, dry_run=True)
