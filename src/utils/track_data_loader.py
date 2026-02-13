"""Track-specific data loader for race simulation parameters."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from src.utils import config_loader

logger = logging.getLogger(__name__)


def load_track_specific_params(race_name: Optional[str] = None) -> Dict:
    """Load track-specific parameters from track_characteristics.

    Returns dict with track-specific overrides for race simulation:
        - pit_stops.loss_duration: seconds (from track_characteristics)
        - sc_probability: safety car probability (from track_characteristics)
        - track_overtaking: overtaking difficulty (from track_characteristics)

    Falls back to config defaults if track_name not found or data missing.
    """
    track_params = {}

    if race_name:
        # Load track characteristics
        track_chars_path = (
            Path(config_loader.get("paths.processed", "data/processed"))
            / "track_characteristics"
            / "2026_track_characteristics.json"
        )

        try:
            with open(track_chars_path, "r") as f:
                track_data = json.load(f)

            tracks = track_data.get("tracks", {})
            track_info = tracks.get(race_name)

            if track_info:
                # Extract track-specific pit stop loss
                pit_loss = track_info.get("pit_stop_loss")
                if pit_loss is not None:
                    track_params["pit_stops"] = {"loss_duration": float(pit_loss)}
                    logger.info(
                        f"Loaded track-specific pit stop loss for {race_name}: {pit_loss}s"
                    )

                # Extract safety car probability
                sc_prob = track_info.get("safety_car_prob")
                if sc_prob is not None:
                    track_params["sc_probability"] = float(sc_prob)

                # Extract overtaking difficulty
                overtaking = track_info.get("overtaking_difficulty")
                if overtaking is not None:
                    track_params["track_overtaking"] = float(overtaking)

            else:
                logger.warning(
                    f"Track '{race_name}' not found in track_characteristics. "
                    "Using config defaults."
                )

        except FileNotFoundError:
            logger.warning(
                f"Track characteristics file not found at {track_chars_path}. "
                "Using config defaults."
            )
        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse track characteristics JSON at {track_chars_path}. "
                "Using config defaults."
            )
        except Exception as e:
            logger.error(
                f"Unexpected error loading track characteristics: {e}. "
                "Using config defaults."
            )

    return track_params


def get_tire_stress_score(race_name: Optional[str] = None) -> float:
    """Get tire stress score for race from Pirelli data.

    Returns average of traction + braking + lateral + abrasion.
    Defaults to 3.0 (medium stress) if data missing.
    """
    if not race_name:
        return config_loader.get(
            "baseline_predictor.compound_selection.default_stress_fallback", 3.0
        )

    # Load Pirelli tire stress data
    pirelli_path = Path("data") / "2025_pirelli_info.json"

    try:
        with open(pirelli_path, "r") as f:
            pirelli_data = json.load(f)

        # Normalize race name (lowercase, underscores)
        race_key = race_name.lower().replace(" ", "_")
        race_info = pirelli_data.get(race_key)

        if race_info and "tyre_stress" in race_info:
            tyre_stress = race_info["tyre_stress"]

            # Calculate average stress from key metrics
            stress_score = (
                tyre_stress.get("traction", 3.0)
                + tyre_stress.get("braking", 3.0)
                + tyre_stress.get("lateral", 3.0)
                + tyre_stress.get("asphalt_abrasion", 3.0)
            ) / 4.0

            return float(stress_score)
        else:
            logger.warning(
                f"Tire stress data not found for {race_name}. Using default (3.0)."
            )

    except FileNotFoundError:
        logger.warning(
            f"Pirelli data file not found at {pirelli_path}. Using default stress (3.0)."
        )
    except Exception as e:
        logger.error(f"Error loading Pirelli data: {e}. Using default stress (3.0).")

    # Fallback to config default
    return config_loader.get(
        "baseline_predictor.compound_selection.default_stress_fallback", 3.0
    )


def get_available_compounds(race_name: Optional[str] = None) -> list:
    """Get list of available tire compounds for race.

    Currently returns all dry compounds.
    Future: could be track/weather specific.
    """
    # For now, all dry compounds available
    return ["SOFT", "MEDIUM", "HARD"]
