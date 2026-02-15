"""Dashboard caching and predictor bootstrap helpers."""

import logging
import os
from pathlib import Path

import fastf1
import streamlit as st

logger = logging.getLogger(__name__)
_FASTF1_CACHE_DIR = Path("data/raw/.fastf1_cache")


def enable_fastf1_cache() -> None:
    """Ensure FastF1 uses project-local cache (avoids default-cache warnings/noise)."""
    _FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(str(_FASTF1_CACHE_DIR))
    except Exception as exc:
        logger.warning(f"Could not enable FastF1 cache at {_FASTF1_CACHE_DIR}: {exc}")


def get_data_file_timestamps() -> dict[str, float]:
    """Get modification timestamps of data/config files used by cached prediction flow."""
    files = [
        "data/processed/car_characteristics/2026_car_characteristics.json",
        "data/processed/driver_characteristics.json",
        "data/processed/track_characteristics/2026_track_characteristics.json",
        "data/2025_pirelli_info.json",  # Tire characteristics (fallback for 2026)
        "data/2026_pirelli_info.json",  # Tire characteristics (if available)
        "config/default.yaml",  # Model weight tuning
        "src/predictors/baseline_2026.py",  # Predictor logic changes
    ]

    timestamps: dict[str, float] = {}
    for file in files:
        path = Path(file)
        if path.exists():
            timestamps[file] = os.path.getmtime(path)
        else:
            timestamps[file] = 0

    return timestamps


@st.cache_resource(show_spinner=False)
def get_predictor(_timestamps: dict[str, float]):
    """Load and cache predictor instance (invalidates when tracked files change)."""
    from src.predictors.baseline_2026 import Baseline2026Predictor

    # Temporarily suppress INFO logs during initialization to avoid clutter.
    original_level = logging.getLogger("src.utils.data_generator").level
    logging.getLogger("src.utils.data_generator").setLevel(logging.WARNING)

    predictor = Baseline2026Predictor()

    # Restore original level.
    logging.getLogger("src.utils.data_generator").setLevel(original_level)

    return predictor
