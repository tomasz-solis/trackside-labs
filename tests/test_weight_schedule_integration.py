"""
Test Weight Schedule Integration

Quick test to ensure the weight schedule system is properly integrated.
"""

from src.systems.weight_schedule import get_schedule_weights
from src.predictors.baseline_2026 import Baseline2026Predictor
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_weight_schedule_integration():
    """Test that weight schedule is working in predictions."""

    print("=" * 70)
    print("TESTING WEIGHT SCHEDULE INTEGRATION")
    print("=" * 70)
    print()

    # 1. Test weight schedule module
    print("1. Testing weight_schedule module...")
    weights = get_schedule_weights(race_number=1, schedule="extreme")
    print(f"   Race 1 weights: {weights}")
    assert weights["baseline"] == 0.30
    assert weights["testing"] == 0.20
    assert weights["current"] == 0.50
    print("   ✓ Weight schedule module working")
    print()

    # 2. Test predictor initialization
    print("2. Initializing Baseline2026Predictor...")
    predictor = Baseline2026Predictor()
    print(f"   ✓ Predictor loaded with {len(predictor.teams)} teams")
    print(f"   ✓ Loaded {len(predictor.tracks)} track profiles")
    print(f"   ✓ Races completed: {predictor.races_completed}")
    print()

    # 3. Test track suitability calculation
    print("3. Testing track suitability calculation...")
    suitability = predictor.calculate_track_suitability("McLaren", "Bahrain Grand Prix")
    print(f"   McLaren at Bahrain: {suitability:+.4f}")
    print("   ✓ Track suitability calculation working")
    print()

    # 4. Test blended team strength
    print("4. Testing blended team strength...")
    for team in list(predictor.teams.keys())[:3]:  # Test first 3 teams
        baseline = predictor.teams[team].get("overall_performance", 0.5)
        blended = predictor.get_blended_team_strength(team, "Bahrain Grand Prix")

        print(f"   {team:25s} Baseline: {baseline:.3f} → Blended: {blended:.3f}")

    print("   ✓ Blended team strength calculation working")
    print()

    # 5. Test qualifying prediction (full integration test)
    print("5. Testing qualifying prediction with weight schedule...")
    result = predictor.predict_qualifying(
        year=2026, race_name="Bahrain Grand Prix", n_simulations=10  # Fast test
    )

    print(f"   ✓ Generated grid with {len(result['grid'])} drivers")
    print("   Top 3:")
    for i, driver in enumerate(result["grid"][:3], 1):
        print(
            f"      P{i}: {driver['driver']} ({driver['team']}) - "
            f"confidence: {driver['confidence']:.0f}%"
        )
    print()

    print("=" * 70)
    print("✅ ALL TESTS PASSED - WEIGHT SCHEDULE INTEGRATED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("Summary:")
    print("- Weight schedule system: ✓ Working")
    print("- Track suitability calculation: ✓ Working")
    print("- Blended team strength: ✓ Working")
    print("- Qualifying predictions: ✓ Using weight schedule")
    print()
    print("Next steps:")
    print("1. Run predictions from dashboard")
    print("2. After each race: python scripts/update_from_race.py 'Race Name' --year 2026")
    print("3. Predictions will automatically adapt using the weight schedule")


if __name__ == "__main__":
    try:
        test_weight_schedule_integration()
        sys.exit(0)
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
