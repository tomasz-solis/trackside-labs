"""Pydantic schemas for configuration validation."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PathsConfig(BaseModel):
    """File paths configuration."""

    data_dir: str = Field(default="data", min_length=1)
    processed: str = Field(default="data/processed", min_length=1)
    raw: str = Field(default="data/raw", min_length=1)
    driver_chars: str = Field(default="data/processed/driver_characteristics.json")
    track_chars: str = Field(default="data/processed/track_characteristics.json")
    lineups: str = Field(default="data/current_lineups.json")
    cache: str = Field(default=".fastf1_cache")


class BayesianConfig(BaseModel):
    """Bayesian model parameters."""

    base_volatility: float = Field(ge=0.0, le=1.0, default=0.1)
    base_observation_noise: float = Field(ge=0.0, default=2.0)
    shock_threshold: float = Field(ge=0.0, default=2.0)
    shock_multiplier: float = Field(ge=0.0, le=2.0, default=0.5)

    @field_validator("base_volatility")
    @classmethod
    def validate_volatility(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("base_volatility must be between 0.0 and 1.0")
        return v


class RaceWeightsConfig(BaseModel):
    """Race component weights."""

    pace_weight: float = Field(ge=0.0, le=1.0, default=0.4)
    grid_weight: float = Field(ge=0.0, le=1.0, default=0.3)
    overtaking_weight: float = Field(ge=0.0, le=1.0, default=0.15)
    tire_deg_weight: float = Field(ge=0.0, le=1.0, default=0.15)

    @field_validator("pace_weight", "grid_weight", "overtaking_weight", "tire_deg_weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v


class DNFConfig(BaseModel):
    """DNF risk parameters."""

    base_risk: float = Field(ge=0.0, le=1.0, default=0.05)
    driver_error_factor: float = Field(ge=0.0, le=1.0, default=0.15)
    street_circuit_risk: float = Field(ge=0.0, le=0.5, default=0.05)
    rain_risk: float = Field(ge=0.0, le=0.5, default=0.10)


class Lap1Config(BaseModel):
    """Lap 1 variance parameters."""

    midfield_variance: float = Field(ge=0.0, default=1.5)
    front_row_variance: float = Field(ge=0.0, default=0.0)


class TireConfig(BaseModel):
    """Tire degradation parameters."""

    degradation_multiplier: float = Field(ge=0.0, default=4.0)
    skill_reduction_factor: float = Field(ge=0.0, le=1.0, default=0.2)


class WeatherConfig(BaseModel):
    """Weather impact parameters."""

    rain_position_swing: float = Field(ge=0.0, default=6.0)
    mixed_intensity: float = Field(ge=0.0, le=1.0, default=0.5)


class SafetyCarConfig(BaseModel):
    """Safety car parameters."""

    compression_factor: float = Field(ge=0.0, le=1.0, default=0.1)


class PaceConfig(BaseModel):
    """Pace calculation parameters."""

    pace_delta_multiplier: float = Field(ge=0.0, default=3.0)


class RaceConfig(BaseModel):
    """Race simulation parameters."""

    weights: RaceWeightsConfig
    base_uncertainty: float = Field(ge=0.0, default=2.5)
    uncertainty_multipliers: dict[str, float] = Field(default_factory=dict)
    dnf: DNFConfig
    lap1: Lap1Config
    tire: TireConfig
    weather: WeatherConfig
    safety_car: SafetyCarConfig
    pace: PaceConfig
    dnf_position_penalty: int = Field(ge=20, le=30, default=22)


class BlendConfig(BaseModel):
    """Blend weight configuration."""

    default: float = Field(ge=0.0, le=1.0, default=0.7)
    fp3_only: float = Field(ge=0.0, le=1.0, default=0.8)
    fp1_only: float = Field(ge=0.0, le=1.0, default=0.4)


class SessionConfidenceConfig(BaseModel):
    """Session confidence weights."""

    fp1: float = Field(ge=0.0, le=1.0, default=0.2)
    fp2: float = Field(ge=0.0, le=1.0, default=0.5)
    fp3: float = Field(ge=0.0, le=1.0, default=0.9)
    sprint_quali: float = Field(ge=0.0, le=1.0, default=0.85)


class QualifyingConfig(BaseModel):
    """Qualifying prediction parameters."""

    blend: BlendConfig
    session_confidence: SessionConfidenceConfig
    base_uncertainty: float = Field(ge=0.0, default=1.5)


class LearningConfig(BaseModel):
    """Learning system parameters."""

    performance_window: int = Field(ge=1, le=20, default=5)
    min_races_for_blend: int = Field(ge=1, le=10, default=3)


class BaselinePredictorConfig(BaseModel):
    """Top-level baseline predictor configuration."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields for extensibility

    paths: PathsConfig
    bayesian: BayesianConfig
    race: RaceConfig
    qualifying: QualifyingConfig
    learning: LearningConfig


def validate_config(config_dict: dict) -> BaselinePredictorConfig:
    """
    Validate configuration dictionary against schema.

    Args:
        config_dict: Raw configuration dictionary from YAML

    Returns:
        Validated configuration object

    Raises:
        ValidationError: If configuration is invalid
    """
    return BaselinePredictorConfig(**config_dict)
