"""Tests for configuration schema validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config_schema import (
    BayesianConfig,
    BlendConfig,
    validate_config,
)


def test_bayesian_config_validates_volatility_range():
    """Volatility must stay between 0.0 and 1.0."""
    valid = BayesianConfig(base_volatility=0.5)
    assert valid.base_volatility == 0.5

    with pytest.raises(ValidationError):
        BayesianConfig(base_volatility=1.5)

    with pytest.raises(ValidationError):
        BayesianConfig(base_volatility=-0.1)


def test_blend_config_validates_weights():
    """Qualifying blend weights must stay in the unit interval."""
    valid = BlendConfig(default=0.7, fp3_only=0.8, fp1_only=0.4)
    assert valid.default == 0.7

    with pytest.raises(ValidationError):
        BlendConfig(default=1.2)


def test_validate_config_accepts_default_yaml():
    """The shipped YAML config should validate cleanly against the strict schema."""
    config_path = Path("config/default.yaml")
    config_dict = yaml.safe_load(config_path.read_text())

    validated = validate_config(config_dict)

    assert validated.grid.size == 22
    assert validated.learning.min_samples == 3
    assert validated.baseline_predictor.qualifying.fp_blend_weight == pytest.approx(0.62)
    assert (
        validated.baseline_predictor.race.overtake_model.zone_front_probability_scale
        == pytest.approx(0.55)
    )
    assert validated.dashboard.prediction_precompute.reconcile_accuracy_after_warmup is True
    assert validated.dashboard.prediction_precompute.learn_completed_races_before_warmup is True
    assert validated.dashboard.prediction_precompute.accuracy_reconcile_lookback_days == 14


def test_validate_config_rejects_unknown_nested_keys():
    """Any YAML key missing from the schema should fail validation."""
    config_dict = yaml.safe_load(Path("config/default.yaml").read_text())
    config_dict["baseline_predictor"]["race"]["unknown_new_knob"] = 123

    with pytest.raises(ValidationError) as exc_info:
        validate_config(config_dict)

    assert "unknown_new_knob" in str(exc_info.value)


def test_validate_config_rejects_invalid_numeric_values():
    """Strict schema validation should still reject bad scalar values."""
    config_dict = yaml.safe_load(Path("config/default.yaml").read_text())
    config_dict["bayesian"]["base_volatility"] = 2.5

    with pytest.raises(ValidationError) as exc_info:
        validate_config(config_dict)

    assert "base_volatility" in str(exc_info.value)


def test_config_loader_integration():
    """Config loader should continue to validate the real config file."""
    from src.utils.config_loader import Config

    config = Config()
    assert config._config is not None

    base_volatility = config.get("bayesian.base_volatility")
    assert isinstance(base_volatility, int | float)
    assert 0.0 <= base_volatility <= 1.0


def test_model_version_defaults_agree():
    """The three model-version literals must not drift apart.

    `config/default.yaml` holds the live value; `ModelConfig.version` is the schema
    default when the key is absent; `_DEFAULT_MODEL_VERSION` is the fallback used when
    config fails to load entirely. A stale fallback silently stamps generated artifacts
    with a version that was never active - that is how "2.3" survived into README.md,
    CONFIGURATION.md and two source files while the model was on 3.0.
    """
    from src.utils.config_schema import ModelConfig
    from src.utils.model_version import _DEFAULT_MODEL_VERSION

    live = yaml.safe_load(Path("config/default.yaml").read_text())["model"]["version"]
    schema_default = ModelConfig.model_fields["version"].default

    assert live == schema_default == _DEFAULT_MODEL_VERSION, (
        f"model version drifted: default.yaml={live!r} "
        f"schema={schema_default!r} fallback={_DEFAULT_MODEL_VERSION!r}"
    )


def test_recency_exponent_defaults_agree():
    """Every recency_exponent fallback must match the live config value.

    Two call sites pass their own literal to `cfg.get(...)`, so a config miss uses that
    literal rather than the schema default. They sat at 1.5 while the live value moved to
    0.3 - a 5x divergence on any path where the key is absent.
    """
    import re

    from src.utils.config_schema import CurrentSeasonFormConfig

    live = yaml.safe_load(Path("config/default.yaml").read_text())["baseline_predictor"][
        "current_season_form"
    ]["recency_exponent"]
    schema_default = CurrentSeasonFormConfig.model_fields["recency_exponent"].default
    assert live == schema_default, f"default.yaml={live} schema={schema_default}"

    pattern = re.compile(r"recency_exponent\"?,\s*([0-9]+\.[0-9]+)\s*\)")
    for source in (
        "src/predictors/baseline/data_mixin.py",
        "src/systems/updater_flow.py",
    ):
        found = pattern.findall(Path(source).read_text(encoding="utf-8"))
        assert found, f"no recency_exponent fallback literal found in {source}"
        for literal in found:
            assert float(literal) == live, (
                f"{source} falls back to {literal} but the live value is {live}"
            )
