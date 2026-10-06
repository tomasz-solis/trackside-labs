"""NaN and Infinity never reach a strict-JSON store."""

import json
import math

from src.persistence.artifact_store import ArtifactStore
from src.utils.json_io import json_safe


def test_json_safe_replaces_non_finite_floats_everywhere():
    value = {"a": float("nan"), "b": [1.0, float("inf"), {"c": -float("inf")}], "d": 2, "e": "x"}

    clean, count = json_safe(value)

    assert count == 3
    assert clean == {"a": None, "b": [1.0, None, {"c": None}], "d": 2, "e": "x"}
    json.dumps(clean, allow_nan=False)  # strict JSON accepts it


def test_save_artifact_sends_strict_json_to_the_database(tmp_path, monkeypatch, caplog):
    store = ArtifactStore(data_root=tmp_path)
    sent = []
    monkeypatch.setattr(
        store,
        "_write_db",
        lambda t, k, data, v, r: sent.append(data) or {"ok": True},
        raising=False,
    )
    monkeypatch.setattr("src.persistence.artifact_store.should_write_to_db", lambda: True)
    monkeypatch.setattr(store, "get_latest_version", lambda t, k: 0)

    store.save_artifact(
        "car_track_traits", "2026::car_track_traits", {"races": {"R": {"deg": math.nan}}}
    )

    written = (
        tmp_path / "processed" / "car_track_traits" / "2026_car_track_traits.json"
    ).read_text()
    assert "NaN" not in written
    assert len(sent) == 1
    json.dumps(sent[0], allow_nan=False)  # what reaches the database is strict JSON
    assert "Replaced 1 NaN/Infinity" in caplog.text


def test_committed_traits_seed_is_strict_json():
    from pathlib import Path

    text = Path("data/processed/car_track_traits/2026_car_track_traits.json").read_text()
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
