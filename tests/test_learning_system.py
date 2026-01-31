"""
Tests for adaptive learning system.
"""

import pytest
import json
from pathlib import Path
from src.systems.learning import LearningSystem


@pytest.mark.unit
class TestLearningSystem:
    """Test learning system functionality."""

    def test_initialization_creates_state(self, tmp_path):
        """Should create initial state structure."""
        system = LearningSystem(data_dir=tmp_path)

        # State exists in memory
        assert "history" in system.state
        assert "method_performance" in system.state

        # File created after first save
        system.save_state()
        assert system.state_file.exists()

    def test_is_race_analyzed_detects_existing(self, tmp_path):
        """Should detect already analyzed races."""
        system = LearningSystem(data_dir=tmp_path)

        # Initially not analyzed
        assert not system.is_race_analyzed("Bahrain GP")

        # Add to history
        system.state["history"].append(
            {"race": "Bahrain GP", "date": "2024-03-01", "comparisons": {}}
        )
        system.save_state()

        # Should now be detected
        assert system.is_race_analyzed("Bahrain GP")

    def test_update_after_race_adds_to_history(self, tmp_path):
        """Should log race in history."""
        system = LearningSystem(data_dir=tmp_path)

        system.update_after_race(
            race="Bahrain GP",
            actual_results={},
            prediction_comparison={"qualifying": {"method": "blend_70_30", "mae": 2.5}},
        )

        assert len(system.state["history"]) == 1
        assert system.state["history"][0]["race"] == "Bahrain GP"

    def test_update_tracks_method_performance(self, tmp_path):
        """Should update method stats."""
        system = LearningSystem(data_dir=tmp_path)

        system.update_after_race(
            race="Race 1",
            actual_results={},
            prediction_comparison={"qualifying": {"method": "blend_70_30", "mae": 2.0}},
        )

        assert "blend_70_30" in system.state["method_performance"]
        assert system.state["method_performance"]["blend_70_30"]["count"] == 1
        assert system.state["method_performance"]["blend_70_30"]["avg_mae"] == 2.0

    def test_running_average_calculation(self, tmp_path):
        """Should maintain rolling average of last 5."""
        system = LearningSystem(data_dir=tmp_path)

        # Add 7 results
        maes = [2.0, 3.0, 1.5, 2.5, 3.5, 1.0, 2.0]
        for i, mae in enumerate(maes):
            system.update_after_race(
                race=f"Race {i}",
                actual_results={},
                prediction_comparison={"q": {"method": "blend_70_30", "mae": mae}},
            )

        # Should only consider last 5: [2.5, 3.5, 1.0, 2.0]
        # Wait, that's 4. Let me recalculate: last 5 = [1.5, 2.5, 3.5, 1.0, 2.0]
        # avg = (1.5 + 2.5 + 3.5 + 1.0 + 2.0) / 5 = 10.5 / 5 = 2.1
        expected_avg = sum(maes[-5:]) / 5
        actual_avg = system.state["method_performance"]["blend_70_30"]["avg_mae"]

        assert abs(actual_avg - expected_avg) < 0.01

    def test_get_optimal_blend_weight_returns_best(self, tmp_path):
        """Should recommend blend weight with lowest MAE."""
        system = LearningSystem(data_dir=tmp_path)

        # Add performance for different methods
        system.update_after_race("R1", {}, {"q": {"method": "blend_70_30", "mae": 2.0}})
        system.update_after_race("R2", {}, {"q": {"method": "blend_80_20", "mae": 1.5}})
        system.update_after_race("R3", {}, {"q": {"method": "blend_50_50", "mae": 3.0}})

        optimal = system.get_optimal_blend_weight(default=0.7)

        # Should pick 80/20 (lowest MAE)
        assert optimal == 0.8

    def test_get_optimal_blend_weight_returns_default_if_no_data(self, tmp_path):
        """Should fallback to default."""
        system = LearningSystem(data_dir=tmp_path)

        optimal = system.get_optimal_blend_weight(default=0.65)

        assert optimal == 0.65

    def test_state_persists_across_instances(self, tmp_path):
        """State should save and reload."""
        system1 = LearningSystem(data_dir=tmp_path)
        system1.update_after_race("Bahrain", {}, {"q": {"method": "test", "mae": 1.0}})

        # Create new instance
        system2 = LearningSystem(data_dir=tmp_path)

        assert len(system2.state["history"]) == 1
        assert "test" in system2.state["method_performance"]

    def test_ignores_updates_without_mae(self, tmp_path):
        """Should skip method updates if no MAE provided."""
        system = LearningSystem(data_dir=tmp_path)

        system.update_after_race("Race 1", {}, {"qualifying": {"method": "blend_70_30"}})  # No MAE

        # Should log to history but not update method stats
        assert len(system.state["history"]) == 1
        assert "blend_70_30" not in system.state["method_performance"]

    def test_handles_malformed_method_names(self, tmp_path):
        """Should handle edge cases in weight extraction."""
        system = LearningSystem(data_dir=tmp_path)

        system.update_after_race("R1", {}, {"q": {"method": "weird_method_name", "mae": 2.0}})

        # Should fallback to default
        optimal = system.get_optimal_blend_weight(default=0.7)
        assert optimal == 0.7
