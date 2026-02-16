"""Dashboard caching and predictor bootstrap helpers."""

import logging
import os
from pathlib import Path

import fastf1
import streamlit as st

from src.persistence.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)
_FASTF1_CACHE_DIR = Path("data/raw/.fastf1_cache")


def enable_fastf1_cache() -> None:
    """Ensure FastF1 uses project-local cache (avoids default-cache warnings/noise)."""
    _FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(str(_FASTF1_CACHE_DIR))
    except Exception as exc:
        logger.warning(f"Could not enable FastF1 cache at {_FASTF1_CACHE_DIR}: {exc}")


def get_artifact_versions() -> dict[str, tuple[int, str]]:
    """
    Get version and updated_at timestamp for each artifact type.

    Returns dict of artifact_key -> (version, updated_at_timestamp).
    This replaces file mtime checks for DB-backed artifacts.
    """
    store = ArtifactStore(data_root="data")
    versions = {}

    # Track DB-backed artifacts
    artifacts_to_track = [
        ("car_characteristics", "2026::car_characteristics"),
        ("driver_characteristics", "2026::driver_characteristics"),
        ("track_characteristics", "2026::track_characteristics"),
    ]

    for artifact_type, artifact_key in artifacts_to_track:
        try:
            data = store.load_artifact(artifact_type, artifact_key)
            if data:
                version = data.get("version", 1)
                updated_at = data.get("last_updated", data.get("updated_at", ""))
                versions[f"{artifact_type}::{artifact_key}"] = (version, updated_at)
            else:
                # Artifact not found, use placeholder
                versions[f"{artifact_type}::{artifact_key}"] = (0, "")
        except Exception as e:
            logger.warning(f"Failed to load version for {artifact_type}::{artifact_key}: {e}")
            versions[f"{artifact_type}::{artifact_key}"] = (0, "")

    # Still track file-based artifacts (Pirelli info, config, code)
    file_timestamps = _get_file_timestamps()
    versions.update(file_timestamps)

    return versions


def _get_file_timestamps() -> dict[str, tuple[int, str]]:
    """Get timestamps for non-DB artifacts (config, code, Pirelli info)."""
    files = [
        "data/2025_pirelli_info.json",
        "data/2026_pirelli_info.json",
        "config/default.yaml",
        "src/predictors/baseline_2026.py",
    ]

    timestamps = {}
    for file in files:
        path = Path(file)
        if path.exists():
            mtime = os.path.getmtime(path)
            timestamps[file] = (int(mtime), str(mtime))
        else:
            timestamps[file] = (0, "")

    return timestamps


@st.cache_resource(show_spinner=False)
def get_predictor(_artifact_versions: dict[str, tuple[int, str]]):
    """
    Load and cache predictor instance (invalidates when tracked artifacts change).

    Args:
        _artifact_versions: Dict of artifact_key -> (version, updated_at).
                           Cache invalidates when any version/timestamp changes.
    """
    from src.predictors.baseline_2026 import Baseline2026Predictor

    # Temporarily suppress INFO logs during initialization to avoid clutter.
    original_level = logging.getLogger("src.utils.data_generator").level
    logging.getLogger("src.utils.data_generator").setLevel(logging.WARNING)

    predictor = Baseline2026Predictor()

    # Restore original level.
    logging.getLogger("src.utils.data_generator").setLevel(original_level)

    return predictor
