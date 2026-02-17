"""Tests for track characteristic extraction utilities."""

import pandas as pd

from src.extractors import track


class _TelemetryFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return _TelemetryFrame

    def add_distance(self):
        return self


class _Lap:
    def __init__(self, telemetry: pd.DataFrame):
        self._telemetry = _TelemetryFrame(telemetry.copy())

    def get_car_data(self):
        return self._telemetry.copy()


class _Laps:
    def __init__(self, lap: _Lap | None, empty: bool = False):
        self._lap = lap
        self.empty = empty

    def pick_fastest(self):
        return self._lap


class _Session:
    def __init__(
        self, lap: _Lap | None, event_name: str = "Bahrain Grand Prix", empty: bool = False
    ):
        self.laps = _Laps(lap, empty=empty)
        self.event = {"EventName": event_name}


def _sample_telemetry(include_throttle: bool = True) -> pd.DataFrame:
    data = {
        "Speed": [250, 215, 175, 140, 120, 150, 190, 240, 210, 170, 130, 165, 210, 250],
        "Distance": [idx * 100 for idx in range(14)],
    }
    if include_throttle:
        data["Throttle"] = [100, 98, 95, 90, 80, 88, 96, 100, 97, 92, 85, 90, 98, 100]
    return pd.DataFrame(data)


def test_extract_track_metrics_returns_expected_values():
    session = _Session(_Lap(_sample_telemetry()))

    metrics = track.extract_track_metrics(session)

    assert metrics["avg_speed"] > 0
    assert metrics["top_speed"] == 250
    assert metrics["braking_events"] > 0
    assert 0 <= metrics["low_pct"] <= 1


def test_extract_track_metrics_returns_none_for_empty_laps():
    session = _Session(lap=None, empty=True)
    assert track.extract_track_metrics(session) is None


def test_identify_corners_detects_speed_minima():
    telemetry = _sample_telemetry()
    corners = track.identify_corners(telemetry, min_speed_drop=10)

    assert not corners.empty
    assert {"entry_speed", "apex_speed", "exit_speed", "speed_lost"}.issubset(corners.columns)


def test_extract_corner_characteristics_returns_corner_metrics():
    session = _Session(_Lap(_sample_telemetry()))

    corner_metrics = track.extract_corner_characteristics(session)

    assert corner_metrics["total_corners"] >= 1
    assert corner_metrics["corner_density"] > 0
    assert corner_metrics["max_speed_loss_kmh"] >= corner_metrics["avg_speed_loss_kmh"]


def test_extract_full_throttle_pct_handles_missing_column():
    session = _Session(_Lap(_sample_telemetry(include_throttle=False)))
    assert track.extract_full_throttle_pct(session) is None


def test_extract_tire_stress_proxy_returns_float():
    session = _Session(_Lap(_sample_telemetry()))
    value = track.extract_tire_stress_proxy(session)
    assert isinstance(value, float)
    assert value > 0


def test_extract_track_profile_builds_composite_profile():
    session = _Session(_Lap(_sample_telemetry()), event_name="Monaco Grand Prix")

    profile = track.extract_track_profile(2026, session)

    assert profile["track_name"] == "Monaco Grand Prix"
    assert profile["extracted_from"] == 2026
    assert "full_throttle_pct" in profile
    assert "corner_density" in profile


def test_identify_street_circuits():
    assert track.identify_street_circuits("Monaco Grand Prix") == 1
    assert track.identify_street_circuits("Silverstone") == 0


def test_calculate_track_z_scores_and_scaler_params():
    df = pd.DataFrame(
        [
            {"track": "A", "slow_corner_pct": 0.2, "energy_score": 3.0},
            {"track": "B", "slow_corner_pct": 0.5, "energy_score": 4.0},
            {"track": "C", "slow_corner_pct": 0.8, "energy_score": 5.0},
        ]
    )

    scored, params = track.calculate_track_z_scores(df, ["slow_corner_pct", "energy_score"])

    assert "slow_corner_pct_z" in scored.columns
    assert "energy_score_z" in scored.columns
    assert params["features"] == ["slow_corner_pct", "energy_score"]
    assert len(params["mean"]) == 2
    assert len(params["std"]) == 2


def test_describe_track_profile_generates_tags():
    row = pd.Series(
        {
            "slow_corner_pct_z": 1.2,
            "medium_corner_pct_z": 0.7,
            "fast_corner_pct_z": 1.6,
            "corner_density_z": 1.3,
            "min_corner_speed_kmh_z": -1.2,
            "avg_speed_loss_kmh_z": 1.1,
            "heavy_braking_pct_z": 1.2,
            "full_throttle_pct_z": 1.1,
            "energy_score_z": 1.7,
            "braking_zones_z": 1.2,
            "is_street_circuit": 1,
        }
    )

    description = track.describe_track_profile(row)

    assert "Heavy slow corners" in description
    assert "EXTREME high-speed" in description
    assert "High corner density" in description
    assert "Street circuit" in description
