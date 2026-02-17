"""Tests for driver experience enrichment utilities."""

import csv

from src.features import driver_experience


def test_load_driver_debuts_from_csv(tmp_path):
    csv_path = tmp_path / "debuts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Driver", "First F1 season"])
        writer.writeheader()
        writer.writerow({"Driver": "Lewis Hamilton", "First F1 season": "2007"})
        writer.writerow({"Driver": "Unknown Name", "First F1 season": "2024"})

    debuts = driver_experience.load_driver_debuts_from_csv(csv_path)

    assert debuts == {"HAM": 2007}


def test_detect_first_season_and_calculate_experience():
    data = {"by_year": {"2022": {}, "2024": {}, "2023": {}}}

    first = driver_experience.detect_first_season(data)
    exp = driver_experience.calculate_experience("NOR", data, current_year=2026)

    assert first == 2022
    assert exp == 4


def test_calculate_experience_prefers_debuts_map():
    data = {"by_year": {"2023": {}, "2024": {}}}
    exp = driver_experience.calculate_experience(
        "NOR",
        data,
        current_year=2026,
        driver_debuts={"NOR": 2019},
    )
    assert exp == 7


def test_assign_experience_tier_boundaries():
    assert driver_experience.assign_experience_tier(None) == "unknown"
    assert driver_experience.assign_experience_tier(0) == "rookie"
    assert driver_experience.assign_experience_tier(2) == "developing"
    assert driver_experience.assign_experience_tier(5) == "established"
    assert driver_experience.assign_experience_tier(8) == "veteran"


def test_determine_confidence_flag_conditions():
    assert (
        driver_experience.determine_confidence_flag(
            {"sessions": 5},
            experience_tier="rookie",
            pace_delta=0.0,
            min_sessions=10,
        )
        == "low"
    )
    assert (
        driver_experience.determine_confidence_flag(
            {"sessions": 12, "std_ratio": 0.03, "teams": ["A", "B", "C"]},
            experience_tier="established",
            pace_delta=0.01,
        )
        == "gathering_info"
    )
    assert (
        driver_experience.determine_confidence_flag(
            {"sessions": 20, "std_ratio": 0.01, "teams": ["McLaren"]},
            experience_tier="veteran",
            pace_delta=0.01,
        )
        == "high"
    )


def test_enrich_driver_characteristics_and_distribution(tmp_path):
    debuts_csv = tmp_path / "debuts.csv"
    with open(debuts_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Driver", "First F1 season"])
        writer.writeheader()
        writer.writerow({"Driver": "Lando Norris", "First F1 season": "2019"})
        writer.writerow({"Driver": "Oliver Bearman", "First F1 season": "2025"})

    quali_data = {
        "extracted_at": "2026-02-01T00:00:00",
        "seasons": [2023, 2024, 2025],
        "drivers": {
            "NOR": {
                "avg_ratio": 1.02,
                "sessions": 30,
                "std_ratio": 0.01,
                "teammates": ["PIA"],
                "teams": ["McLaren"],
                "by_year": {"2023": {}, "2024": {}, "2025": {}},
            },
            "BEA": {
                "avg_ratio": 1.00,
                "sessions": 12,
                "std_ratio": 0.03,
                "teammates": ["HUL"],
                "teams": ["Haas"],
                "by_year": {"2025": {}},
            },
        },
    }
    race_data = {
        "drivers": {
            "NOR": {"avg_ratio": 1.01, "sessions": 28, "std_ratio": 0.012},
            "BEA": {"avg_ratio": 1.02, "sessions": 10, "std_ratio": 0.02},
        }
    }

    enriched = driver_experience.enrich_driver_characteristics(
        quali_data,
        race_data,
        current_year=2026,
        debuts_csv_path=str(debuts_csv),
    )

    assert enriched["total_drivers"] == 2
    assert enriched["drivers"]["NOR"]["experience"]["years_experience"] == 7
    assert enriched["drivers"]["NOR"]["experience"]["tier"] == "veteran"
    assert enriched["drivers"]["BEA"]["experience"]["tier"] == "developing"
    assert isinstance(enriched["drivers"]["NOR"]["pace_delta"], float)

    distribution = driver_experience.analyze_experience_distribution(enriched)
    assert distribution["by_tier"]["veteran"]["count"] == 1
    assert distribution["confidence_distribution"]["high"] >= 0
