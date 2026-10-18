"""Challenger models that forecast beside the champion and are scored, never served.

A challenger is a function from one checkpoint's context to a forecast per target. It
sees what the champion saw at that checkpoint: the champion's forecast, the checkpoint
predictor (same data cutoff) and earlier events' actual results. It must not fetch
actuals itself; ``ChallengerContext.prior_actuals`` is the leak-safe history.

``PredictionLogger.save_prediction`` runs the challengers named in
``challengers.enabled`` and stores them under ``challengers`` in the saved forecast.
A challenger that raises is recorded as ``status: error``; the champion is untouched.
The replay also writes each challenger's accuracy snapshots to
``<replay root>/challengers/<name>``, so ``scripts/compare_replay_arms.py`` scores it
against the champion with the seed floor.

Add a challenger: write ``fn(ctx) -> {target_key: rows}`` and register it in
``CHALLENGERS``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.models.shadow_challenger import (
    _calendar_order_map,
    _coerce_int,
    _normalize_label,
    _parse_saved_datetime,
    _prior_target_actual_history,
)
from src.utils.accuracy_targets import explicit_target_predictions, sanitize_prediction_rows

logger = logging.getLogger(__name__)

CHALLENGER_SCHEMA_VERSION = 1

Rows = list[dict[str, Any]]


@dataclass
class ChallengerContext:
    """What a challenger may read for one checkpoint forecast."""

    year: int
    race_name: str
    checkpoint_session: str
    champion_targets: dict[str, Rows]
    historical_predictions: list[dict[str, Any]]
    predictor: Any = None
    prediction_data: dict[str, Any] | None = None

    def prior_actuals(self, target_key: str) -> list[Rows]:
        """Actual results for ``target_key`` from earlier events, oldest first."""
        metadata = (self.prediction_data or {}).get("metadata", {})
        order_map = _calendar_order_map(self.year)
        return _prior_target_actual_history(
            self.historical_predictions,
            target_key=target_key,
            current_race=self.race_name,
            current_order=order_map.get(_normalize_label(self.race_name)),
            current_time=_parse_saved_datetime(
                metadata.get("information_cutoff_at") or metadata.get("predicted_at")
            ),
            order_map=order_map,
        )

    def prior_predictions(self) -> list[dict[str, Any]]:
        """Saved forecasts of earlier events, oldest event first, same rule as ``prior_actuals``."""
        metadata = (self.prediction_data or {}).get("metadata", {})
        order_map = _calendar_order_map(self.year)
        current_label = _normalize_label(self.race_name)
        current_order = order_map.get(current_label)
        current_time = _parse_saved_datetime(
            metadata.get("information_cutoff_at") or metadata.get("predicted_at")
        )
        earlier = []
        for prediction in self.historical_predictions:
            saved = prediction.get("metadata", {})
            label = _normalize_label(saved.get("race_name", ""))
            if not label or label == current_label:
                continue
            order = order_map.get(label)
            saved_time = _parse_saved_datetime(
                saved.get("information_cutoff_at") or saved.get("predicted_at")
            )
            if current_order is not None and order is not None:
                if order >= current_order:
                    continue
            elif current_time is None or saved_time is None or saved_time >= current_time:
                continue
            earlier.append((order if order is not None else 10_000, label, prediction))
        earlier.sort(key=lambda item: (item[0], item[1]))
        return [prediction for _, _, prediction in earlier]


ChallengerFn = Callable[[ChallengerContext], dict[str, Rows]]


@dataclass(frozen=True)
class ChallengerSpec:
    """One registered challenger."""

    name: str
    version: str
    fn: ChallengerFn
    description: str


def _rank(drivers_in_order: list[str], champion_rows: Rows) -> Rows:
    """Rows in the given driver order, carrying the champion's team and position."""
    by_driver = {str(row["driver"]): row for row in champion_rows}
    return [
        {
            "driver": driver,
            "team": by_driver[driver].get("team"),
            "position": position,
            "champion_position": _coerce_int(by_driver[driver].get("position")),
        }
        for position, driver in enumerate(drivers_in_order, start=1)
    ]


def previous_event_order(ctx: ChallengerContext) -> dict[str, Rows]:
    """Naive baseline: repeat the previous event's actual order for the same target.

    Drivers missing from that result follow in the champion's order. A target with no
    earlier result is left out.
    """
    forecasts: dict[str, Rows] = {}
    for target_key, champion_rows in ctx.champion_targets.items():
        history = ctx.prior_actuals(target_key)
        if not history:
            continue
        previous = {
            str(row["driver"]): int(row["position"])
            for row in history[-1]
            if row.get("driver") and row.get("position") is not None
        }
        champion_order = [str(row["driver"]) for row in champion_rows]
        ordered = sorted(
            champion_order,
            key=lambda driver: (
                driver not in previous,
                previous.get(driver, 0),
                champion_order.index(driver),
            ),
        )
        forecasts[target_key] = _rank(ordered, champion_rows)
    return forecasts


def qualifying_ridge(ctx: ChallengerContext) -> dict[str, Rows]:
    """Stacked qualifying ridge from the 2026-10-08 ML probe (``challenger_ridge``)."""
    from src.models.challenger_ridge import qualifying_ridge as fit_and_forecast

    return fit_and_forecast(ctx)


def qualifying_laptime(ctx: ChallengerContext) -> dict[str, Rows]:
    """Qualifying as a lap-time model in seconds (``challenger_laptime``)."""
    from src.models.challenger_laptime import qualifying_laptime as forecast

    return forecast(ctx)


CHALLENGERS: dict[str, ChallengerSpec] = {
    spec.name: spec
    for spec in (
        ChallengerSpec(
            name="previous_event_order",
            version="1",
            fn=previous_event_order,
            description="Previous event's actual order for the same target (naive baseline).",
        ),
        ChallengerSpec(
            name="qualifying_ridge",
            version="1",
            fn=qualifying_ridge,
            description="Ridge on the champion forecast, form, teammate gap and practice pace.",
        ),
        ChallengerSpec(
            name="qualifying_laptime",
            version="1",
            fn=qualifying_laptime,
            description="Lap-time gaps: car pace, track fit, practice and driver, in seconds.",
        ),
    )
}


def enabled_challengers(config: Any = None) -> list[str]:
    """Names from ``challengers.enabled`` that are registered, in config order."""
    from src.utils import config_loader

    source = config if config is not None else config_loader
    names = source.get("challengers.enabled", []) or []
    unknown = [name for name in names if name not in CHALLENGERS]
    if unknown:
        logger.warning("Unknown challengers in config, skipped: %s", ", ".join(unknown))
    return [name for name in names if name in CHALLENGERS]


def run_challengers(
    prediction_data: dict[str, Any],
    *,
    historical_predictions: list[dict[str, Any]],
    predictor: Any = None,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Run each challenger on one saved forecast; failures are recorded, never raised."""
    if names is None:
        names = enabled_challengers(getattr(predictor, "config", None))
    if not names:
        return {}
    metadata = prediction_data.get("metadata", {})
    champion_targets = {
        target_key: rows
        for target_key, payload in explicit_target_predictions(prediction_data).items()
        # Only targets the champion forecast; an ACTUAL section is a result, not a forecast.
        if payload.get("result_mode", "PREDICTED") == "PREDICTED"
        and (rows := sanitize_prediction_rows(payload.get("predicted_order")))
    }
    ctx = ChallengerContext(
        year=_coerce_int(metadata.get("year")) or 0,
        race_name=str(metadata.get("race_name", "")).strip(),
        checkpoint_session=str(metadata.get("session_name", "")).strip().upper(),
        champion_targets=champion_targets,
        historical_predictions=historical_predictions,
        predictor=predictor,
        prediction_data=prediction_data,
    )
    results: dict[str, Any] = {}
    for name in names:
        spec = CHALLENGERS[name]
        payload: dict[str, Any] = {
            "schema_version": CHALLENGER_SCHEMA_VERSION,
            "version": spec.version,
        }
        try:
            targets = spec.fn(ctx)
        except Exception as exc:  # noqa: BLE001 - a challenger must never break the champion
            logger.warning("Challenger %s failed for %s: %s", name, ctx.race_name, exc)
            results[name] = {**payload, "status": "error", "error": str(exc)[:300], "targets": {}}
            continue
        results[name] = {
            **payload,
            "status": "active" if targets else "insufficient_history",
            "targets": {key: {"predicted_order": rows} for key, rows in targets.items()},
        }
    return results


def challenger_prediction_view(prediction: dict[str, Any], name: str) -> dict[str, Any] | None:
    """The saved forecast with the champion's targets replaced by one challenger's.

    Metadata and actuals stay, so the champion's scorers score the challenger. Targets
    the challenger did not forecast are dropped. ``None`` when it forecast nothing.
    """
    challenger = (prediction.get("challengers") or {}).get(name)
    if not isinstance(challenger, dict) or not challenger.get("targets"):
        return None
    champion_targets = prediction.get("targets") or {}
    targets = {
        key: {**champion_targets.get(key, {}), "predicted_order": payload["predicted_order"]}
        for key, payload in challenger["targets"].items()
        if key in champion_targets
    }
    if not targets:
        return None
    return {**prediction, "targets": targets}
