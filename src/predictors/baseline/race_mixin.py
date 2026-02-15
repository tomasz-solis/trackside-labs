"""Race simulation mixin composition for Baseline2026Predictor."""

from src.predictors.baseline.race import (
    BaselineRaceParamsMixin,
    BaselineRacePredictionMixin,
    BaselineRacePreparationMixin,
)


class BaselineRaceMixin(
    BaselineRacePreparationMixin,
    BaselineRaceParamsMixin,
    BaselineRacePredictionMixin,
):
    """Composed race mixin for Baseline2026Predictor."""

    pass
