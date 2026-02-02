"""
Tests for src/utils/auto_updater.py - Automatic race detection and updating

Critical path testing for dashboard auto-update functionality.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir)

    # Create directory structure
    char_dir = data_dir / "processed" / "car_characteristics"
    char_dir.mkdir(parents=True, exist_ok=True)

    # Create initial characteristics file
    char_file = char_dir / "2026_car_characteristics.json"
    initial_data = {
        "year": 2026,
        "version": 1,
        "races_completed": 0,
        "last_updated": "2026-01-01T00:00:00",
        "last_learned_race": None,
        "teams": {}
    }

    with open(char_file, 'w') as f:
        json.dump(initial_data, f, indent=2)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestAutoUpdaterDetection:
    """Test automatic race completion detection."""

    def test_needs_update_no_races(self, temp_data_dir):
        """Test needs_update returns False when no races completed."""
        from src.utils.auto_updater import needs_update

        with patch('src.utils.auto_updater.get_completed_races') as mock_completed:
            mock_completed.return_value = []

            with patch('src.utils.auto_updater.get_learned_races') as mock_learned:
                mock_learned.return_value = []

                result, new_races = needs_update(data_dir=temp_data_dir)

        assert result is False
        assert new_races == []

    def test_needs_update_with_new_race(self, temp_data_dir):
        """Test needs_update returns True when new race completed."""
        from src.utils.auto_updater import needs_update

        with patch('src.utils.auto_updater.get_completed_races') as mock_completed:
            mock_completed.return_value = ["Bahrain Grand Prix"]

            with patch('src.utils.auto_updater.get_learned_races') as mock_learned:
                mock_learned.return_value = []

                result, new_races = needs_update(data_dir=temp_data_dir)

        assert result is True
        assert "Bahrain Grand Prix" in new_races

    def test_needs_update_all_learned(self, temp_data_dir):
        """Test needs_update returns False when all races already learned."""
        from src.utils.auto_updater import needs_update

        with patch('src.utils.auto_updater.get_completed_races') as mock_completed:
            mock_completed.return_value = ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"]

            with patch('src.utils.auto_updater.get_learned_races') as mock_learned:
                mock_learned.return_value = ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"]

                result, new_races = needs_update(data_dir=temp_data_dir)

        assert result is False
        assert new_races == []


class TestAutoUpdaterExecution:
    """Test automatic update execution."""

    def test_auto_update_from_races_success(self, temp_data_dir):
        """Test successful auto-update from completed races."""
        from src.utils.auto_updater import auto_update_from_races

        with patch('src.utils.auto_updater.needs_update') as mock_needs:
            mock_needs.return_value = (True, ["Bahrain Grand Prix"])

            with patch('src.utils.auto_updater.update_from_race') as mock_update:
                mock_update.return_value = None  # Success

                def progress_callback(current, total, message):
                    pass  # Mock progress callback

                updated_count = auto_update_from_races(
                    progress_callback=progress_callback,
                    data_dir=temp_data_dir
                )

        assert updated_count == 1
        mock_update.assert_called_once()

    def test_auto_update_multiple_races(self, temp_data_dir):
        """Test auto-update handles multiple races."""
        from src.utils.auto_updater import auto_update_from_races

        with patch('src.utils.auto_updater.needs_update') as mock_needs:
            mock_needs.return_value = (True, ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix"])

            with patch('src.utils.auto_updater.update_from_race') as mock_update:
                mock_update.return_value = None  # Success

                def progress_callback(current, total, message):
                    assert total == 3

                updated_count = auto_update_from_races(
                    progress_callback=progress_callback,
                    data_dir=temp_data_dir
                )

        assert updated_count == 3
        assert mock_update.call_count == 3

    def test_auto_update_handles_failure(self, temp_data_dir):
        """Test auto-update continues on single race failure."""
        from src.utils.auto_updater import auto_update_from_races

        with patch('src.utils.auto_updater.needs_update') as mock_needs:
            mock_needs.return_value = (True, ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"])

            with patch('src.utils.auto_updater.update_from_race') as mock_update:
                # First update succeeds, second fails
                mock_update.side_effect = [None, Exception("Update failed")]

                def progress_callback(current, total, message):
                    pass

                updated_count = auto_update_from_races(
                    progress_callback=progress_callback,
                    data_dir=temp_data_dir
                )

        # Should have updated 1 race (first one) before failure
        assert updated_count == 1


class TestCompletedRacesDetection:
    """Test detection of completed races from FastF1."""

    def test_get_completed_races_recent(self, temp_data_dir):
        """Test get_completed_races returns only recent races."""
        from src.utils.auto_updater import get_completed_races

        # Mock FastF1 schedule
        mock_schedule = pd.DataFrame({
            "EventName": ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix"],
            "Session5Date": [
                datetime.now() - timedelta(days=10),  # Completed
                datetime.now() - timedelta(days=5),   # Completed
                datetime.now() + timedelta(days=5),   # Future
            ]
        })

        with patch('fastf1.get_event_schedule') as mock_get_schedule:
            mock_get_schedule.return_value = mock_schedule

            completed = get_completed_races(year=2026, lookback_days=30)

        assert len(completed) == 2
        assert "Bahrain Grand Prix" in completed
        assert "Saudi Arabian Grand Prix" in completed
        assert "Australian Grand Prix" not in completed


class TestLearnedRacesTracking:
    """Test tracking of already-learned races."""

    def test_get_learned_races_from_file(self, temp_data_dir):
        """Test get_learned_races reads from characteristics file."""
        from src.utils.auto_updater import get_learned_races

        # Update characteristics file with learned race
        char_file = Path(temp_data_dir) / "processed" / "car_characteristics" / "2026_car_characteristics.json"
        with open(char_file) as f:
            data = json.load(f)

        data["last_learned_race"] = "Bahrain Grand Prix"
        data["races_completed"] = 1

        with open(char_file, 'w') as f:
            json.dump(data, f, indent=2)

        learned = get_learned_races(data_dir=temp_data_dir)

        # Should infer 1 race was learned
        assert len(learned) >= 1

    def test_get_learned_races_empty(self, temp_data_dir):
        """Test get_learned_races returns empty when no races learned."""
        from src.utils.auto_updater import get_learned_races

        learned = get_learned_races(data_dir=temp_data_dir)

        assert learned == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
