"""Predictor package exports.

``Baseline2026Predictor`` loads on first use. The web imports helpers from
``src.predictors.baseline`` and never runs a predictor, so an eager import here
loaded the whole predictor and FastF1 into every web process.
"""

from typing import Any

__all__ = ["Baseline2026Predictor"]


def __getattr__(name: str) -> Any:
    if name == "Baseline2026Predictor":
        from .baseline_2026 import Baseline2026Predictor

        return Baseline2026Predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
