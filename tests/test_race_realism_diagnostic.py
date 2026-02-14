"""Diagnostic tests to measure race prediction realism before fixes.

These tests quantify the current behavior to establish a baseline for improvements.
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.predictors.baseline_2026 import Baseline2026Predictor


class TestRaceRealismDiagnostic:
    """Diagnostic tests to measure current realism metrics."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return Baseline2026Predictor()

    def test_diagnostic_grid_to_race_correlation(self, predictor):
        """Measure Spearman correlation between quali grid and race finish (DRY conditions).

        Target: >= 0.75 for realistic race behavior
        """
        # Use high simulation count for stable measurement
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=500)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=500,
        )

        # Extract grid and race positions
        grid_positions = {entry["driver"]: entry["position"] for entry in quali["grid"]}
        race_positions = {entry["driver"]: entry["position"] for entry in race["finish_order"]}

        # Align drivers
        drivers = sorted(grid_positions.keys())
        grid_pos = [grid_positions[d] for d in drivers]
        race_pos = [race_positions[d] for d in drivers]

        # Calculate Spearman correlation
        correlation, p_value = spearmanr(grid_pos, race_pos)

        print("\n=== Grid-to-Race Correlation Diagnostic ===")
        print(f"Spearman correlation: {correlation:.3f} (p={p_value:.4f})")
        print("Target: >= 0.75")
        print(f"Status: {'✓ PASS' if correlation >= 0.75 else '✗ FAIL'}")

        # Show position changes
        changes = [abs(grid_positions[d] - race_positions[d]) for d in drivers]
        print(f"Mean absolute position change: {np.mean(changes):.2f}")
        print(f"Max position change: {max(changes)}")

        # This is diagnostic - we're measuring current behavior, not asserting
        # Uncomment after fixes:
        # assert correlation >= 0.75, f"Grid-race correlation too low: {correlation:.3f}"

    def test_diagnostic_mean_position_change(self, predictor):
        """Measure average position change from grid to race finish.

        Target: <= 3.0 positions for realistic races
        """
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=500)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=500,
        )

        grid_positions = {entry["driver"]: entry["position"] for entry in quali["grid"]}
        race_positions = {entry["driver"]: entry["position"] for entry in race["finish_order"]}

        position_changes = [
            abs(grid_positions[d] - race_positions[d]) for d in grid_positions.keys()
        ]

        mean_change = np.mean(position_changes)

        print("\n=== Mean Position Change Diagnostic ===")
        print(f"Mean absolute position change: {mean_change:.2f}")
        print("Target: <= 3.0")
        print(f"Status: {'✓ PASS' if mean_change <= 3.0 else '✗ FAIL'}")

        # Show distribution
        print(f"Median: {np.median(position_changes):.1f}")
        print(f"75th percentile: {np.percentile(position_changes, 75):.1f}")
        print(f"95th percentile: {np.percentile(position_changes, 95):.1f}")

        # Uncomment after fixes:
        # assert mean_change <= 3.0, f"Mean position change too high: {mean_change:.2f}"

    def test_diagnostic_pole_sitter_top3_probability(self, predictor):
        """Measure probability of pole sitter finishing in top 3.

        Target: >= 45% for realistic races
        """
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=500)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=500,
        )

        pole_driver = quali["grid"][0]["driver"]
        pole_finish = next(e for e in race["finish_order"] if e["driver"] == pole_driver)

        # Use podium probability from race prediction
        top3_prob = pole_finish["podium_probability"]

        print("\n=== Pole Sitter Top-3 Probability Diagnostic ===")
        print(f"Driver: {pole_driver}")
        print(f"Podium probability: {top3_prob:.1f}%")
        print("Target: >= 45%")
        print(f"Status: {'✓ PASS' if top3_prob >= 45.0 else '✗ FAIL'}")
        print(f"Predicted finish position: P{pole_finish['position']}")

        # Uncomment after fixes:
        # assert top3_prob >= 45.0, f"Pole sitter top-3 probability too low: {top3_prob:.1f}%"

    def test_diagnostic_top5_starters_podium_dominance(self, predictor):
        """Measure what fraction of podium probability comes from top-5 starters.

        Target: >= 50% (majority of podium outcomes from top grid positions)
        """
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=500)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=500,
        )

        # Get top 5 starters
        top5_drivers = {entry["driver"] for entry in quali["grid"][:5]}

        # Sum podium probabilities
        total_podium_prob = 0.0
        top5_podium_prob = 0.0

        for entry in race["finish_order"]:
            prob = entry["podium_probability"]
            total_podium_prob += prob
            if entry["driver"] in top5_drivers:
                top5_podium_prob += prob

        # Normalize (total should be ~300% since 3 drivers get 100% total)
        top5_fraction = (top5_podium_prob / total_podium_prob) * 100 if total_podium_prob > 0 else 0

        print("\n=== Top-5 Starters Podium Dominance Diagnostic ===")
        print(f"Top-5 starters' share of podium probability: {top5_fraction:.1f}%")
        print("Target: >= 50%")
        print(f"Status: {'✓ PASS' if top5_fraction >= 50.0 else '✗ FAIL'}")

        # Show individual top-5 podium probs
        print("\nTop-5 starter podium probabilities:")
        for entry in race["finish_order"]:
            if entry["driver"] in top5_drivers:
                grid_pos = next(
                    e["position"] for e in quali["grid"] if e["driver"] == entry["driver"]
                )
                print(f"  P{grid_pos} {entry['driver']}: {entry['podium_probability']:.1f}%")

        # Uncomment after fixes:
        # assert top5_fraction >= 50.0, f"Top-5 podium dominance too low: {top5_fraction:.1f}%"

    def test_diagnostic_top_grid_falloff_frequency(self, predictor):
        """Measure how often top-3 starters fall to P10+ without incidents.

        Target: Rare (< 10% of the time for each driver)
        """
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=500)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=500,
        )

        top3_drivers = [entry["driver"] for entry in quali["grid"][:3]]

        print("\n=== Top-3 Starter Falloff Diagnostic ===")
        print("Measuring P95 (95th percentile worst finish) for top-3 starters:")

        falloff_count = 0
        for driver in top3_drivers:
            entry = next(e for e in race["finish_order"] if e["driver"] == driver)
            grid_pos = next(e["position"] for e in quali["grid"] if e["driver"] == driver)
            p95 = entry["p95"]

            print(f"  P{grid_pos} {driver}: P95={p95}")
            if p95 >= 10:
                falloff_count += 1

        print(f"\nTop-3 drivers with P95 >= 10: {falloff_count}/3")
        print("Target: 0-1 drivers (rare falloffs)")
        print(f"Status: {'✓ PASS' if falloff_count <= 1 else '✗ FAIL'}")

        # Uncomment after fixes:
        # assert falloff_count <= 1, f"Too many top-3 falloffs: {falloff_count}/3"

    def test_diagnostic_podium_prob_ordering(self, predictor):
        """Measure whether podium probabilities decrease monotonically with finishing position.

        Target: At least for top 8, podium% should generally decrease
        """
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=500)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=500,
        )

        # Get top 8 finishers
        top8 = sorted(race["finish_order"], key=lambda x: x["position"])[:8]

        print("\n=== Podium Probability Ordering Diagnostic ===")
        print("Top-8 finishers with podium probabilities:")

        inversions = 0
        for i, entry in enumerate(top8):
            print(f"  P{entry['position']} {entry['driver']}: {entry['podium_probability']:.1f}%")
            if i > 0:
                prev_prob = top8[i - 1]["podium_probability"]
                curr_prob = entry["podium_probability"]
                if curr_prob > prev_prob + 5:  # Allow 5% tolerance
                    inversions += 1

        print(f"\nInversions (podium% increases): {inversions}")
        print("Target: <= 2 (mostly monotonic decrease)")
        print(f"Status: {'✓ PASS' if inversions <= 2 else '✗ FAIL'}")

        # Uncomment after fixes:
        # assert inversions <= 2, f"Too many podium probability inversions: {inversions}"


if __name__ == "__main__":
    # Run diagnostics directly
    test = TestRaceRealismDiagnostic()
    predictor = Baseline2026Predictor()

    print("=" * 70)
    print("RACE REALISM DIAGNOSTIC SUITE")
    print("=" * 70)

    test.test_diagnostic_grid_to_race_correlation(predictor)
    test.test_diagnostic_mean_position_change(predictor)
    test.test_diagnostic_pole_sitter_top3_probability(predictor)
    test.test_diagnostic_top5_starters_podium_dominance(predictor)
    test.test_diagnostic_top_grid_falloff_frequency(predictor)
    test.test_diagnostic_podium_prob_ordering(predictor)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
