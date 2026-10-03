"""The team strength seconds mapping refits itself after every Q and R."""

import json

import pandas as pd
import pytest

from src.models import team_strength_mapping as tsm
from src.models import team_strength_refresh as tsr


class _Store:
    """In-memory ArtifactStore stand-in keyed by (type, key)."""

    def __init__(self, items=None):
        self.items = dict(items or {})
        self.saved = []

    def load_artifact(self, artifact_type, artifact_key):
        return self.items.get((artifact_type, artifact_key))

    def save_artifact(self, artifact_type, artifact_key, data):
        self.saved.append((artifact_type, artifact_key))
        self.items[(artifact_type, artifact_key)] = data


def _copy_round(race_from: str, race_to: str, session_kind: str) -> pd.DataFrame:
    """Reuse a real 2026 round's rows under a new name, as a fake new session."""
    seed = tsr.seed_observations(2026)
    rows = seed[(seed.race_name == race_from) & (seed.session_kind == session_kind)].copy()
    rows["race_name"] = race_to
    return rows


@pytest.fixture
def fake_extract(monkeypatch):
    calls = []

    def extract(year, race_name, session_code, session_kind):
        calls.append((race_name, session_code))
        return _copy_round("Hungarian Grand Prix", race_name, session_kind)

    monkeypatch.setattr(tsr, "extract_session_rows", extract)
    return calls


def test_refresh_adds_missing_sessions_refits_and_is_idempotent(fake_extract):
    store = _Store()
    mus = tsr.driver_mu_by_kind()

    first = tsr.refresh_team_strength_mapping(
        2026, ["Hungarian Grand Prix", "Dutch Grand Prix"], store=store, mus=mus
    )

    assert first["added"] == ["Dutch Grand Prix Q", "Dutch Grand Prix R"]  # Hungary is seeded
    assert fake_extract == [("Dutch Grand Prix", "Q"), ("Dutch Grand Prix", "R")]
    assert first["saved_mapping"] is True
    mapping = store.items[(tsr.MAPPING_TYPE, tsr.MAPPING_KEY)]
    assert len(mapping["rounds"]) == 12
    assert mapping["mappings"]["race"]["slope_s_per_unit"] > 0

    second = tsr.refresh_team_strength_mapping(
        2026, ["Hungarian Grand Prix", "Dutch Grand Prix"], store=store, mus=mus
    )
    assert second["added"] == []
    assert len(fake_extract) == 2  # nothing re-extracted


def test_refit_matches_the_committed_mapping_on_the_committed_rows():
    """Same rows, same fit: the live path reproduces the hand-frozen artifact."""
    fit_rows = tsr.build_fit_rows(tsr.seed_observations(2026), tsr.driver_mu_by_kind())
    artifact = tsr.fit_mapping_artifact(fit_rows, 2026)
    committed = json.loads((tsr._MAPPING_DIR / "latest.json").read_text(encoding="utf-8"))
    for kind in ("race", "qualifying"):
        assert artifact["mappings"][kind]["slope_s_per_unit"] == pytest.approx(
            committed["mappings"][kind]["slope_s_per_unit"], rel=1e-9
        )


def test_guardrail_keeps_the_old_mapping_when_the_slope_jumps(fake_extract):
    old = {
        "mappings": {"race": {"slope_s_per_unit": 40.0}, "qualifying": {"slope_s_per_unit": 2.7}}
    }
    store = _Store({(tsr.MAPPING_TYPE, tsr.MAPPING_KEY): old})

    result = tsr.refresh_team_strength_mapping(
        2026, ["Dutch Grand Prix"], store=store, mus=tsr.driver_mu_by_kind()
    )

    assert "race slope moved" in result["rejected"]
    assert result["saved_mapping"] is False
    assert store.items[(tsr.MAPPING_TYPE, tsr.MAPPING_KEY)] is old
    assert (tsr.OBSERVATIONS_TYPE, tsr.observations_key(2026)) in store.saved  # rows still kept


def test_guardrail_needs_enough_rounds():
    assert tsr.guardrail_failure({"rounds": ["A", "B"], "mappings": {}}, None) == "only 2 rounds"


def test_session_without_published_laps_is_retried_later(monkeypatch):
    monkeypatch.setattr(tsr, "extract_session_rows", lambda *args: None)
    store = _Store()

    result = tsr.refresh_team_strength_mapping(2026, ["Dutch Grand Prix"], store=store, mus={})

    assert result == {"added": [], "saved_mapping": False, "rejected": None}
    assert store.saved == []


def test_loader_reads_the_store_and_caches(monkeypatch):
    payload = {
        "mappings": {
            "race": {"intercept_s": 0.0, "slope_s_per_unit": 5.0, "training_years": [2026]},
            "qualifying": {"intercept_s": 0.0, "slope_s_per_unit": 3.0, "training_years": [2026]},
        }
    }
    reads = []

    class _FakeStore:
        def __init__(self, data_root):
            pass

        def load_artifact(self, artifact_type, artifact_key):
            reads.append((artifact_type, artifact_key))
            return payload

    monkeypatch.setattr("src.persistence.artifact_store.ArtifactStore", _FakeStore)
    monkeypatch.delenv(tsm.TEAM_STRENGTH_SECONDS_MAPPING_PATH_ENV, raising=False)
    monkeypatch.setattr(tsm, "_store_mapping_cache", {"loaded_at": None, "mappings": None})

    first = tsm.load_live_team_strength_mappings()
    second = tsm.load_live_team_strength_mappings()

    assert first["race"].slope_s_per_unit == 5.0
    assert second["qualifying"].slope_s_per_unit == 3.0
    assert reads == [("team_strength_seconds_mapping", "latest")]  # cached
