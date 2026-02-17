from __future__ import annotations

from src.utils.driver_validation import validate_driver_data


def test_validate_driver_data_reports_structure_and_range_issues():
    drivers = {
        "MIS": {"pace": {"quali_pace": 0.6, "race_pace": 0.6}},
        "NOP": {"racecraft": {"skill_score": 0.5}},
        "BAD": {
            "racecraft": {"skill_score": 1.2},
            "pace": {"quali_pace": 0.9, "race_pace": 0.5},
            "dnf_risk": {"dnf_rate": 0.8},
        },
    }

    errors = validate_driver_data(drivers)

    assert any("MIS: Missing 'racecraft' field" in item for item in errors)
    assert any("NOP: Missing 'pace' field" in item for item in errors)
    assert any("BAD: skill_score" in item for item in errors)
    assert any("BAD: Large pace gap" in item for item in errors)
    assert any("BAD: DNF rate" in item for item in errors)


def test_validate_driver_data_returns_empty_for_valid_payload():
    drivers = {
        "NOR": {
            "racecraft": {"skill_score": 0.7},
            "pace": {"quali_pace": 0.72, "race_pace": 0.70},
        }
    }

    errors = validate_driver_data(drivers)

    assert errors == []
