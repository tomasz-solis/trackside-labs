"""
Race Data Updater System

Adaptive learning after each race:
- Updates team performance from telemetry
- Updates Bayesian driver ratings
- Reduces uncertainty as season progresses
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import fastf1
import numpy as np
import pandas as pd

from src.models.bayesian import BayesianDriverRanking
from src.utils.file_operations import atomic_json_write
from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger(__name__)


def load_race_session(year: int, race_name: str) -> tuple[pd.DataFrame, fastf1.core.Session]:
    """Load race results and session from FastF1."""
    logger.info(f"Loading {year} {race_name} results...")

    cache_dir = Path("data/raw/.fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(year, race_name, "R")
    session.load(laps=True, telemetry=False, weather=False)

    results = session.results
    results["race_name"] = race_name
    results["year"] = year

    return results, session


def extract_team_performance_from_telemetry(
    session: fastf1.core.Session, team_names: list[str]
) -> Dict[str, float]:
    """
    Extract team performance from race telemetry using median lap times.

    Filters:
    - Pit laps excluded
    - Lap 1 and last lap excluded
    - Outliers (>3σ) excluded

    Returns dict of team -> performance (0-1 scale, 1.0 = fastest)
    """
    race_pace = {}

    if not hasattr(session, "laps") or session.laps is None or session.laps.empty:
        logger.warning("No lap data available")
        return {}

    laps = session.laps
    known_teams = set(team_names)
    if "Team" not in laps.columns:
        logger.warning("Lap data does not include Team column")
        return {}

    laps = laps.copy()
    laps["_canonical_team"] = laps["Team"].apply(
        lambda raw: map_team_to_characteristics(raw, known_teams=known_teams)
    )

    for team in team_names:
        team_laps = laps[laps["_canonical_team"] == team]

        if len(team_laps) == 0:
            logger.warning(f"  No laps found for {team}")
            continue

        # Filter valid racing laps
        mask = team_laps["LapTime"].notna()
        if "PitOutTime" in team_laps.columns:
            mask &= team_laps["PitOutTime"].isna()
        if "PitInTime" in team_laps.columns:
            mask &= team_laps["PitInTime"].isna()
        if "LapNumber" in team_laps.columns:
            mask &= team_laps["LapNumber"] > 1
            mask &= team_laps["LapNumber"] < team_laps["LapNumber"].max()

        valid_laps = team_laps[mask]

        if len(valid_laps) < 5:
            logger.warning(f"  {team}: Only {len(valid_laps)} valid laps, skipping")
            continue

        # Get lap times in seconds
        lap_times_seconds = valid_laps["LapTime"].dt.total_seconds()

        # Remove outliers (>3 std devs)
        mean_time = lap_times_seconds.mean()
        std_time = lap_times_seconds.std()
        clean_times = lap_times_seconds[
            (lap_times_seconds > mean_time - 3 * std_time)
            & (lap_times_seconds < mean_time + 3 * std_time)
        ]

        if len(clean_times) == 0:
            clean_times = lap_times_seconds

        median_time = clean_times.median()
        race_pace[team] = median_time
        logger.debug(f"  {team}: Median lap time {median_time:.3f}s ({len(clean_times)} laps)")

    # Convert lap times to 0-1 performance scale
    if race_pace:
        fastest_time = min(race_pace.values())
        slowest_time = max(race_pace.values())

        if fastest_time < slowest_time:
            for team in race_pace:
                # Invert: faster time = higher score
                performance = 1.0 - (race_pace[team] - fastest_time) / (slowest_time - fastest_time)
                race_pace[team] = performance
        else:
            # All teams same pace
            for team in race_pace:
                race_pace[team] = 0.5

    return race_pace


def update_team_characteristics(
    race_results: pd.DataFrame, session: fastf1.core.Session, characteristics_file: Path
) -> None:
    """Update team performance ratings from race telemetry."""
    logger.info("Updating team characteristics from race telemetry...")

    with open(characteristics_file) as f:
        char_data = json.load(f)

    # Extract performance from telemetry
    team_names = list(char_data["teams"].keys())
    race_pace = extract_team_performance_from_telemetry(session, team_names)

    # Fallback to positions if no telemetry available
    if not race_pace:
        logger.warning("No telemetry data, using positions as fallback")
        known_teams = set(team_names)
        canonical_results = race_results.copy()
        if "TeamName" in canonical_results.columns:
            canonical_results["_canonical_team"] = canonical_results["TeamName"].apply(
                lambda raw: map_team_to_characteristics(raw, known_teams=known_teams)
            )
        else:
            canonical_results["_canonical_team"] = None

        for team in team_names:
            team_results = canonical_results[canonical_results["_canonical_team"] == team]
            if len(team_results) > 0:
                positions = pd.to_numeric(team_results["Position"], errors="coerce").dropna()
                if positions.empty:
                    continue
                avg_position = positions.mean()
                performance = 1.0 - (avg_position - 1) / 19
                race_pace[team] = performance

    # Update current season performance (running average)
    for team, new_performance in race_pace.items():
        if team in char_data["teams"]:
            team_data = char_data["teams"][team]

            if "current_season_performance" not in team_data:
                team_data["current_season_performance"] = []

            team_data["current_season_performance"].append(new_performance)

            running_avg = np.mean(team_data["current_season_performance"])
            old_uncertainty = team_data["uncertainty"]
            updated_uncertainty = max(0.10, old_uncertainty * 0.9)

            team_data["uncertainty"] = round(updated_uncertainty, 3)
            team_data["last_updated"] = datetime.now().isoformat()
            team_data["races_completed"] = len(team_data["current_season_performance"])

            logger.info(
                f"  {team}: Race {new_performance:.3f} → Avg {running_avg:.3f} "
                f"({team_data['races_completed']} races, uncertainty {old_uncertainty:.2f}→{updated_uncertainty:.2f})"
            )

    # Update metadata
    char_data["last_updated"] = datetime.now().isoformat()
    char_data["data_freshness"] = "LIVE_UPDATED"
    char_data["races_completed"] = char_data.get("races_completed", 0) + 1

    # Increment version
    current_version = char_data.get("version", 0)
    char_data["version"] = current_version + 1

    # Save with atomic write
    atomic_json_write(characteristics_file, char_data, create_backup=True)
    logger.info(
        f"✓ Updated team characteristics (v{char_data['version']}) in {characteristics_file}"
    )


def update_bayesian_driver_ratings(race_results: pd.DataFrame) -> None:
    """Update Bayesian driver skill ratings from race results."""
    logger.info("Updating Bayesian driver ratings...")

    # Create priors for drivers
    from src.models.priors_factory import PriorsFactory

    factory = PriorsFactory()
    priors = factory.create_priors()

    bayesian = BayesianDriverRanking(priors)
    drivers = race_results["Abbreviation"].tolist()
    positions = race_results["Position"].tolist()

    for driver, position in zip(drivers, positions):
        if pd.notna(position):
            bayesian.update(driver, int(position))

    logger.info(f"✓ Updated Bayesian ratings for {len(drivers)} drivers")


def update_from_race(year: int, race_name: str, data_dir: str = "data/processed") -> None:
    """
    Main entry point: Update all characteristics after a race.

    Workflow:
    1. Load race results from FastF1
    2. Update team performance from telemetry
    3. Update Bayesian driver ratings
    4. Reduce uncertainty
    """
    logger.info("=" * 60)
    logger.info(f"Updating from {year} {race_name}")
    logger.info("=" * 60)

    try:
        race_results, session = load_race_session(year, race_name)
        logger.info(f"✓ Loaded results for {len(race_results)} drivers\n")
    except Exception as e:
        logger.error(f"Failed to load race results: {e}")
        logger.error("Make sure race has completed and data is available via FastF1")
        raise

    # Update team characteristics
    char_file = Path(data_dir) / "car_characteristics" / f"{year}_car_characteristics.json"
    if char_file.exists():
        update_team_characteristics(race_results, session, char_file)
    else:
        logger.warning(f"Team characteristics file not found: {char_file}")

    # Update driver ratings
    update_bayesian_driver_ratings(race_results)

    logger.info("\n" + "=" * 60)
    logger.info("✓ Race Update Complete!")
    logger.info("=" * 60)
    logger.info("\nSystem learned from this race:")
    logger.info("- Team performance updated from telemetry")
    logger.info("- Driver skill confidence increased")
    logger.info("- Uncertainty reduced")
    logger.info("- Version incremented\n")
    logger.info("Next prediction will use these updated characteristics.")
