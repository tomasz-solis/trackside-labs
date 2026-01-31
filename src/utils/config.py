"""
Production Config Helper

Uses comprehensive testing results to select best method.
NO hardcoded performance values!
"""

import json
import logging
from pathlib import Path
from typing import Dict, Literal

logger = logging.getLogger(__name__)


class ProductionConfig:
    """
    Load and use production configuration.

    Based on comprehensive testing (Notebook 21B, 24 races).
    """

    def __init__(self, config_path="config/production_config.json"):
        """Load production config."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(
                f"Production config not found at {config_path}\n"
                f"Run comprehensive testing (Notebook 21B) to generate it."
            )

        with open(config_file) as f:
            self.config = json.load(f)

    def get_qualifying_strategy(self, weekend_type: Literal["sprint", "conventional"]) -> Dict:
        """
        Get best qualifying prediction strategy.

        Args:
            weekend_type: 'sprint' or 'conventional'

        Returns:
            {
                'method': str,
                'session': str (optional),
                'blend_weight': float (optional),
                'expected_mae': float
            }
        """
        quali_config = self.config["qualifying_methods"]

        if weekend_type == "sprint":
            return quali_config["sprint_weekends"].copy()
        else:
            return quali_config["conventional_weekends"].copy()

    def get_expected_mae(
        self,
        prediction_type: Literal["qualifying", "race"],
        method: str = None,
        weekend_type: str = None,
    ) -> float:
        """
        Get expected MAE from config.

        Args:
            prediction_type: 'qualifying' or 'race'
            method: Specific method (optional)
            weekend_type: 'sprint' or 'conventional' (optional)

        Returns:
            Expected MAE
        """
        if prediction_type == "qualifying":
            if weekend_type:
                strategy = self.get_qualifying_strategy(weekend_type)
                return strategy["expected_mae"]
            else:
                # Return weighted average
                sprint = self.config["qualifying_methods"]["sprint_weekends"]
                conv = self.config["qualifying_methods"]["conventional_weekends"]

                # Assume 6 sprint, 18 conventional (2025 ratio)
                weighted_mae = (sprint["expected_mae"] * 6 + conv["expected_mae"] * 18) / 24
                return weighted_mae

        elif prediction_type == "race":
            return self.config["race_methods"]["default"]["expected_mae"]

        return 4.0  # Conservative fallback

    def get_performance_ranking(self) -> Dict:
        """Get performance ranking from 2025 testing."""
        return self.config["notes"]["performance_ranking_2025"].copy()

    def __str__(self):
        """Display config summary."""
        lines = []
        lines.append("PRODUCTION CONFIGURATION")
        lines.append("=" * 70)
        lines.append(f"Source: {self.config['notes']['comprehensive_testing_notebook']}")
        lines.append(f"Races analyzed: {self.config['notes']['total_races_analyzed']}")
        lines.append(f"Last updated: {self.config['notes']['last_updated']}")
        lines.append("")

        lines.append("QUALIFYING STRATEGY:")

        sprint = self.config["qualifying_methods"]["sprint_weekends"]
        lines.append(f"  Sprint weekends:")
        lines.append(f"    Method: {sprint['method']}")
        lines.append(f"    Session: {sprint.get('session', 'N/A')}")
        lines.append(f"    Expected MAE: {sprint['expected_mae']:.2f}")
        lines.append(f"    Confidence: {sprint['confidence']}")

        conv = self.config["qualifying_methods"]["conventional_weekends"]
        lines.append(f"  Conventional weekends:")
        lines.append(f"    Method: {conv['method']}")
        lines.append(f"    Blend weight: {conv.get('blend_weight', 'N/A')}")
        lines.append(f"    Expected MAE: {conv['expected_mae']:.2f}")
        lines.append(f"    Confidence: {conv['confidence']}")

        lines.append("")
        lines.append("PERFORMANCE RANKING (2025):")
        for rank, info in self.config["notes"]["performance_ranking_2025"].items():
            lines.append(f"  {rank}. {info}")

        return "\n".join(lines)


# Quick helper functions
def load_production_config(config_path="config/production_config.json") -> ProductionConfig:
    """Load production config."""
    return ProductionConfig(config_path)


def get_best_method(weekend_type: Literal["sprint", "conventional"]) -> Dict:
    """
    Get best method for weekend type.

    Quick helper that loads config and returns strategy.

    Args:
        weekend_type: 'sprint' or 'conventional'

    Returns:
        Strategy dict with method, expected_mae, etc.
    """
    config = load_production_config()
    return config.get_qualifying_strategy(weekend_type)


# Example usage
if __name__ == "__main__":
    config = load_production_config()

    logger.info(str(config))

    logger.info("=" * 70)
    logger.info("EXAMPLES")
    logger.info("=" * 70)

    # Sprint weekend
    sprint_strategy = config.get_qualifying_strategy("sprint")
    logger.info(f"Sprint weekend strategy:")
    logger.info(f"  Method: {sprint_strategy['method']}")
    logger.info(f"  Session: {sprint_strategy['session']}")
    logger.info(f"  Expected MAE: {sprint_strategy['expected_mae']:.2f}")

    # Conventional weekend
    conv_strategy = config.get_qualifying_strategy("conventional")
    logger.info(f"Conventional weekend strategy:")
    logger.info(f"  Method: {conv_strategy['method']}")
    logger.info(f"  Blend weight: {conv_strategy['blend_weight']}")
    logger.info(f"  Expected MAE: {conv_strategy['expected_mae']:.2f}")

    # Expected MAEs
    logger.info(f"Expected MAEs:")
    logger.info(
        f"  Sprint quali: {config.get_expected_mae('qualifying', weekend_type='sprint'):.2f}"
    )
    logger.info(
        f"  Conv quali: {config.get_expected_mae('qualifying', weekend_type='conventional'):.2f}"
    )
    logger.info(f"  Overall quali: {config.get_expected_mae('qualifying'):.2f}")
    logger.info(f"  Race: {config.get_expected_mae('race'):.2f}")
