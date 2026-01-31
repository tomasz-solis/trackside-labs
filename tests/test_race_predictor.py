"""
Tests for race prediction system.
"""
import pytest
import numpy as np
from src.predictors.race import RacePredictor


@pytest.mark.unit
class TestRacePredictor:
    """Test RacePredictor functionality."""

    def test_initialization(self, mock_driver_chars, temp_data_dir):
        """Should initialize with driver chars."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        assert predictor.year == 2026
        assert 'VER' in predictor.driver_chars
        assert predictor.driver_chars['VER']['racecraft']['skill_score'] == 0.95

    def test_predict_returns_valid_structure(self, mock_driver_chars, mock_qualifying_grid, temp_data_dir):
        """Prediction output should have correct structure."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        result = predictor.predict(
            year=2026,
            race_name='Test Grand Prix',
            qualifying_grid=mock_qualifying_grid,
            verbose=False
        )

        assert 'finish_order' in result
        assert 'metadata' in result
        assert len(result['finish_order']) == len(mock_qualifying_grid)

        # Check first place has required fields
        first = result['finish_order'][0]
        assert 'driver' in first
        assert 'position' in first
        assert 'confidence' in first
        assert 'podium_probability' in first

    def test_predict_assigns_positions_correctly(self, mock_driver_chars, mock_qualifying_grid, temp_data_dir):
        """Positions should be 1 through N."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        result = predictor.predict(
            year=2026,
            race_name='Test Grand Prix',
            qualifying_grid=mock_qualifying_grid,
            verbose=False
        )

        positions = [d['position'] for d in result['finish_order']]
        assert positions == [1, 2, 3, 4]

    def test_higher_skilled_driver_likely_beats_lower(self, mock_driver_chars, temp_data_dir):
        """VER (0.95 skill) should usually beat BOT (0.70 skill)."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        # Run simulation multiple times with different grids
        ver_better_positions = 0
        trials = 50

        np.random.seed(42)  # Consistency for test
        for _ in range(trials):
            result = predictor.predict(
                year=2026,
                race_name='Test GP',
                qualifying_grid=[
                    {'driver': 'VER', 'team': 'Red Bull', 'position': 10},
                    {'driver': 'BOT', 'team': 'Cadillac', 'position': 11}
                ],
                verbose=False
            )

            ver_pos = [d['position'] for d in result['finish_order'] if d['driver'] == 'VER'][0]
            bot_pos = [d['position'] for d in result['finish_order'] if d['driver'] == 'BOT'][0]

            if ver_pos < bot_pos:
                ver_better_positions += 1

        # VER should beat BOT in majority of sims
        assert ver_better_positions > trials * 0.5

    def test_wet_weather_penalizes_inconsistent_drivers(self, mock_driver_chars, temp_data_dir):
        """Rain should hurt drivers with low wet skill."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        grid = [
            {'driver': 'VER', 'team': 'Red Bull', 'position': 1},
            {'driver': 'BOT', 'team': 'Cadillac', 'position': 2}
        ]

        # Dry race
        dry_result = predictor.predict(
            year=2026,
            race_name='Test GP',
            qualifying_grid=grid,
            weather_forecast='dry',
            verbose=False
        )

        # Wet race
        wet_result = predictor.predict(
            year=2026,
            race_name='Test GP',
            qualifying_grid=grid,
            weather_forecast='rain',
            verbose=False
        )

        # BOT should have higher DNF risk in rain
        bot_dry = [d for d in dry_result['finish_order'] if d['driver'] == 'BOT'][0]
        bot_wet = [d for d in wet_result['finish_order'] if d['driver'] == 'BOT'][0]

        assert bot_wet['dnf_probability'] > bot_dry['dnf_probability']

    def test_lap_1_chaos_simulation(self, mock_driver_chars, temp_data_dir):
        """Lap 1 should introduce variance."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        # P10 should have some variance
        positions_after_lap1 = []

        for _ in range(30):
            pos = predictor._simulate_lap_1_chaos(
                start_pos=10,
                racecraft=0.8,
                consistency=0.8
            )
            positions_after_lap1.append(pos)

        # Should have some spread
        assert len(set(positions_after_lap1)) > 1
        # But shouldn't jump to P1 from P10
        assert max(positions_after_lap1) > 5

    def test_overtaking_difficulty_reduces_pace_gain(self, mock_driver_chars, temp_data_dir):
        """Monaco should limit overtaking vs Monza."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        pace_delta = -1.0  # Faster car
        skill = 0.85

        # Easy overtaking (Monza)
        gain_easy = predictor._calculate_effective_pace_gain(
            current_pos=10,
            pace_delta=pace_delta,
            difficulty=0.2,
            skill=skill
        )

        # Hard overtaking (Monaco)
        gain_hard = predictor._calculate_effective_pace_gain(
            current_pos=10,
            pace_delta=pace_delta,
            difficulty=0.9,
            skill=skill
        )

        # Should gain more positions at easy track
        assert abs(gain_easy) > abs(gain_hard)

    def test_tire_degradation_penalty_calculation(self, mock_driver_chars, temp_data_dir):
        """High deg should add position penalty."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        track_info = {'pit_stop_loss': 22.0}

        # Low deg (advantage)
        penalty_low = predictor._calculate_tire_pace_penalty(0.2, track_info)

        # High deg (penalty)
        penalty_high = predictor._calculate_tire_pace_penalty(0.8, track_info)

        # High deg should have positive penalty (lose positions)
        assert penalty_high > 0
        # Low deg should have negative penalty (gain positions)
        assert penalty_low < 0
        # Difference should be meaningful
        assert penalty_high - penalty_low > 1.0

    def test_dnf_probability_increases_with_risk_factors(self, mock_driver_chars, temp_data_dir):
        """DNF risk should stack properly."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        # Safe scenario
        dnf_safe = predictor._calculate_dnf_probability(
            team='Red Bull',
            consistency=0.95,
            weather='dry',
            track_info={'type': 'permanent'}
        )

        # Risky scenario
        dnf_risky = predictor._calculate_dnf_probability(
            team='Cadillac',
            consistency=0.60,
            weather='rain',
            track_info={'type': 'street'}
        )

        assert dnf_risky > dnf_safe
        assert dnf_risky > 0.15  # Should be meaningfully high

    def test_podium_probability_calculation(self, mock_driver_chars, temp_data_dir):
        """Podium prob should decrease with position."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        prob_p1 = predictor._calculate_podium_probability(1.5, 1.0)
        prob_p3 = predictor._calculate_podium_probability(3.0, 1.0)
        prob_p10 = predictor._calculate_podium_probability(10.0, 1.0)

        assert prob_p1 > prob_p3 > prob_p10
        assert prob_p10 == 0.0

    def test_get_driver_skills_returns_defaults_for_unknown(self, mock_driver_chars, temp_data_dir):
        """Unknown drivers should get default skills."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        skills = predictor._get_driver_skills('UNKNOWN')

        assert skills['racecraft'] == 0.5
        assert skills['consistency'] == 0.5
        assert skills['wet_weather'] == 0.5

    def test_weather_impact_calculation(self, mock_driver_chars, temp_data_dir):
        """Weather impact should scale with skill and intensity."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        # Good driver in rain (advantage)
        impact_good = predictor._calculate_weather_impact('rain', wet_skill=0.9)

        # Bad driver in rain (penalty)
        impact_bad = predictor._calculate_weather_impact('rain', wet_skill=0.2)

        # Good driver should gain, bad should lose
        assert impact_good < 0  # Negative = gain positions
        assert impact_bad > 0   # Positive = lose positions

    def test_safety_car_compresses_field(self, mock_driver_chars, temp_data_dir):
        """High SC probability should reduce gaps."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        # Monaco (high SC prob)
        impact_monaco = predictor._apply_safety_car_variance(
            {'safety_car_prob': 0.8},
            current_pos=15
        )

        # Bahrain (low SC prob)
        impact_bahrain = predictor._apply_safety_car_variance(
            {'safety_car_prob': 0.2},
            current_pos=15
        )

        # Monaco should compress more (pull back towards mean)
        assert abs(impact_monaco) > abs(impact_bahrain)

    def test_confidence_inversely_related_to_uncertainty(self, mock_driver_chars, mock_qualifying_grid, temp_data_dir):
        """Higher uncertainty should lower confidence."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        # Dry, easy overtaking (low uncertainty)
        result_certain = predictor.predict(
            year=2026,
            race_name='Monza GP',
            qualifying_grid=mock_qualifying_grid,
            overtaking_factor=0.2,
            weather_forecast='dry',
            verbose=False
        )

        # Wet, hard overtaking (high uncertainty)
        result_uncertain = predictor.predict(
            year=2026,
            race_name='Monaco GP',
            qualifying_grid=mock_qualifying_grid,
            overtaking_factor=0.9,
            weather_forecast='rain',
            verbose=False
        )

        avg_conf_certain = np.mean([d['confidence'] for d in result_certain['finish_order']])
        avg_conf_uncertain = np.mean([d['confidence'] for d in result_uncertain['finish_order']])

        assert avg_conf_certain > avg_conf_uncertain


@pytest.mark.integration
class TestRacePredictorIntegration:
    """Integration tests requiring multiple components."""

    def test_full_race_prediction_completes(self, mock_driver_chars, mock_qualifying_grid, temp_data_dir):
        """End-to-end prediction should complete without errors."""
        predictor = RacePredictor(
            year=2026,
            data_dir=temp_data_dir,
            driver_chars=mock_driver_chars
        )

        result = predictor.predict(
            year=2026,
            race_name='Bahrain Grand Prix',
            qualifying_grid=mock_qualifying_grid,
            fp2_pace={'Red Bull Racing': {'relative_pace': 0.1}},
            overtaking_factor=0.4,
            weather_forecast='dry',
            verbose=False
        )

        # Sanity checks
        assert len(result['finish_order']) == 4
        assert result['finish_order'][0]['position'] == 1
        assert all(0 <= d['confidence'] <= 100 for d in result['finish_order'])
        assert all(0 <= d['dnf_probability'] <= 1 for d in result['finish_order'])
