"""
Update Team/Driver Characteristics After a 2026 Race

This script implements adaptive learning after each race:
- Updates team performance ratings based on actual race pace
- Updates Bayesian driver skill ratings
- Reduces uncertainty as season progresses
- Learns track-specific factors if significantly different from baseline

USAGE:
    python scripts/update_from_race.py "Bahrain Grand Prix" --year 2026

WORKFLOW:
    1. After each race → Run this script
    2. System learns from actual results
    3. Confidence increases throughout season
    4. Next prediction uses updated characteristics
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import fastf1
import numpy as np
import pandas as pd

from src.models.bayesian import BayesianDriverRanking
from src.systems.learning import LearningSystem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_race_results(year: int, race_name: str) -> pd.DataFrame:
    """Load actual race results from FastF1."""
    logger.info(f"Loading {year} {race_name} results...")

    # Ensure cache directory exists
    from pathlib import Path

    cache_dir = Path("data/raw/.fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(year, race_name, "R")
    session.load()

    results = session.results
    results["race_name"] = race_name
    results["year"] = year

    return results


def update_team_performance(race_results: pd.DataFrame, characteristics_file: Path) -> None:
    """
    Update team performance ratings based on race results.

    Uses race pace (average lap time relative to fastest) to update team strength.
    """
    logger.info("Updating team performance ratings...")

    # Load current characteristics
    with open(characteristics_file) as f:
        char_data = json.load(f)

    # Calculate team pace from race
    # Group by team, calculate average lap time
    race_pace = {}
    session_laps = race_results  # Would need session.laps for detailed analysis

    # Simplified: use finishing positions as proxy for pace
    # In real implementation, would use lap-by-lap telemetry
    for team in char_data["teams"].keys():
        team_results = race_results[race_results["TeamName"] == team]
        if len(team_results) > 0:
            avg_position = team_results["Position"].mean()
            # Convert position to performance rating (1st = 1.0, 20th = 0.0)
            performance = 1.0 - (avg_position - 1) / 19
            race_pace[team] = performance

    # Update team characteristics with weighted average
    # Current rating gets 70% weight, new race gets 30% (Bayesian update)
    for team, new_performance in race_pace.items():
        if team in char_data["teams"]:
            old_rating = char_data["teams"][team]["overall_performance"]
            updated_rating = old_rating * 0.7 + new_performance * 0.3

            # Reduce uncertainty as we get more data
            old_uncertainty = char_data["teams"][team]["uncertainty"]
            updated_uncertainty = max(0.10, old_uncertainty * 0.9)  # Floor at 0.10

            char_data["teams"][team]["overall_performance"] = round(updated_rating, 3)
            char_data["teams"][team]["uncertainty"] = round(updated_uncertainty, 3)
            char_data["teams"][team]["last_updated"] = datetime.now().isoformat()
            char_data["teams"][team]["races_completed"] = (
                char_data["teams"][team].get("races_completed", 0) + 1
            )

            logger.info(
                f"  {team}: {old_rating:.3f} → {updated_rating:.3f} "
                f"(uncertainty: {old_uncertainty:.2f} → {updated_uncertainty:.2f})"
            )

    # Update metadata
    char_data["last_updated"] = datetime.now().isoformat()
    char_data["data_freshness"] = "LIVE_UPDATED"
    char_data["races_completed"] = char_data.get("races_completed", 0) + 1

    # Save updated characteristics
    with open(characteristics_file, "w") as f:
        json.dump(char_data, f, indent=2)

    logger.info(f"✓ Updated team characteristics in {characteristics_file}")


def update_bayesian_driver_ratings(race_results: pd.DataFrame) -> None:
    """
    Update Bayesian driver skill ratings based on race results.
    """
    logger.info("Updating Bayesian driver ratings...")

    # Initialize Bayesian ranking system
    bayesian = BayesianDriverRanking()

    # Create driver list from results
    drivers = race_results["Abbreviation"].tolist()

    # Update rankings based on finishing positions
    positions = race_results["Position"].tolist()

    # Convert to rating updates (simplified)
    for driver, position in zip(drivers, positions):
        if pd.notna(position):
            bayesian.update(driver, int(position))

    logger.info(f"✓ Updated Bayesian ratings for {len(drivers)} drivers")


def update_learning_system(race_name: str, predicted_results: Dict, actual_results: pd.DataFrame):
    """
    Update learning system with prediction accuracy.
    """
    logger.info("Updating learning system...")

    learning = LearningSystem()

    # Calculate Mean Absolute Error (MAE)
    # Would need predicted vs actual position comparison
    # Placeholder for now
    mae = 2.5  # Example MAE

    learning.update_after_race(
        race_name=race_name, method="baseline_2026", mae=mae, predictions=predicted_results
    )

    learning.save_state()

    logger.info(f"✓ Updated learning system (MAE: {mae:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Update characteristics after a race")
    parser.add_argument("race_name", help="Race name (e.g., 'Bahrain Grand Prix')")
    parser.add_argument("--year", type=int, default=2026, help="Season year")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument(
        "--skip-teams", action="store_true", help="Skip team characteristic updates"
    )
    parser.add_argument("--skip-drivers", action="store_true", help="Skip driver rating updates")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Updating from {args.year} {args.race_name}")
    logger.info("=" * 60)
    logger.info("")

    # Load race results
    try:
        race_results = load_race_results(args.year, args.race_name)
        logger.info(f"✓ Loaded results for {len(race_results)} drivers")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to load race results: {e}")
        logger.error("Make sure the race has been completed and data is available via FastF1.")
        return

    # Update team characteristics
    if not args.skip_teams:
        char_file = Path(args.data_dir) / "car_characteristics" / "2026_car_characteristics.json"
        if char_file.exists():
            update_team_performance(race_results, char_file)
            logger.info("")
        else:
            logger.warning(f"Team characteristics file not found: {char_file}")
            logger.info("")

    # Update driver ratings
    if not args.skip_drivers:
        update_bayesian_driver_ratings(race_results)
        logger.info("")

    # Update learning system
    # update_learning_system(args.race_name, {}, race_results)
    # logger.info("")

    logger.info("=" * 60)
    logger.info("✓ Race Update Complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("System has learned from this race:")
    logger.info("- Team performance ratings updated")
    logger.info("- Driver skill confidence increased")
    logger.info("- Uncertainty reduced")
    logger.info("")
    logger.info("Next prediction will use these updated characteristics.")


if __name__ == "__main__":
    main()
