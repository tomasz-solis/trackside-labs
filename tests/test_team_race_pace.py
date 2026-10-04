"""Measured team race pace refreshes itself after each race."""

import pytest

from src.dashboard import warmup
from src.extractors import team_race_pace as trp


class _Store:
    """In-memory ArtifactStore stand-in."""

    def __init__(self, payload=None):
        self.payload = payload
        self.saved = []

    def load_artifact(self, artifact_type, artifact_key):
        return self.payload

    def save_artifact(self, artifact_type, artifact_key, data):
        self.saved.append((artifact_type, artifact_key, data))
        self.payload = data


def test_refresh_measures_only_missing_races_and_saves(monkeypatch):
    store = _Store({"races": {"Australian Grand Prix": {"A": 0.0, "B": 1.0}}})
    measured = []

    def fake_measure(year, race_name):
        measured.append(race_name)
        return {"A": 0.5, "B": 0.0}

    monkeypatch.setattr(trp, "measure_race", fake_measure)

    added = trp.refresh_team_race_pace(
        2026, ["Australian Grand Prix", "Chinese Grand Prix"], store=store
    )

    assert added == ["Chinese Grand Prix"]
    assert measured == ["Chinese Grand Prix"]  # Australia is never re-measured
    _type, key, data = store.saved[0]
    assert key == "2026::team_race_pace"
    assert set(data["races"]) == {"Australian Grand Prix", "Chinese Grand Prix"}
    assert data["teams"]["A"] == {"gap_s": 0.25, "races": 2}


def test_refresh_saves_nothing_when_nothing_new_or_laps_not_ready(monkeypatch):
    store = _Store({"races": {"Australian Grand Prix": {"A": 0.0}}})
    monkeypatch.setattr(trp, "measure_race", lambda year, race_name: None)

    assert trp.refresh_team_race_pace(2026, ["Chinese Grand Prix"], store=store) == []
    assert store.saved == []


def test_load_falls_back_to_the_committed_file_when_the_store_is_empty(monkeypatch, tmp_path):
    path = tmp_path / "pace.json"
    path.write_text('{"races": {"Australian Grand Prix": {"A": 0.0}}}')
    monkeypatch.setattr(trp, "committed_path", lambda year: path)

    payload = trp.load_team_race_pace(2026, store=_Store(None))

    assert payload["races"] == {"Australian Grand Prix": {"A": 0.0}}


@pytest.mark.real_team_race_pace
def test_warmup_stage_refreshes_with_completed_races_and_survives_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.utils.auto_updater.get_completed_races", lambda year: ["Azerbaijan Grand Prix"]
    )
    monkeypatch.setattr(trp, "refresh_team_race_pace", lambda year, races: calls.append(races))

    class _Summary:
        errors: list = []

    class _Ctx:
        dry_run = False
        year = 2026
        summary = _Summary()

    warmup._stage_refresh_team_race_pace(_Ctx())
    assert calls == [["Azerbaijan Grand Prix"]]

    def boom(year, races):
        raise OSError("fastf1 down")

    monkeypatch.setattr(trp, "refresh_team_race_pace", boom)
    ctx = _Ctx()
    ctx.summary = _Summary()
    ctx.summary.errors = []
    warmup._stage_refresh_team_race_pace(ctx)
    assert ctx.summary.errors == ["team_race_pace: fastf1 down"]


def test_file_mode_store_uses_the_committed_layout(tmp_path):
    from src.persistence.artifact_store import ArtifactStore

    store = ArtifactStore(data_root=tmp_path)
    store.save_artifact(trp.ARTIFACT_TYPE, trp.artifact_key(2026), {"races": {}})

    assert (tmp_path / "processed" / "team_race_pace" / "2026_team_race_pace.json").exists()
