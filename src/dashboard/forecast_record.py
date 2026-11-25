"""Append-only record of issued forecasts (plan row 21).

The warmup writes one row per still-forecast target each time it computes and saves a
served payload, beside the existing stores; nothing reads the record yet. A row holds
the forecast rows exactly as served and what produced them, never results. The table
refuses updates and deletes (``migrations/007_create_forecast_record.sql``), so this
module only inserts:

- a repeat of the latest row for the identity stores nothing; A, B, then A stores three
  (migration 008 makes ``content_hash`` unique per ``replaces``);
- a changed forecast for the same identity gets ``replaces`` and a reason;
- a forecast issued after its information cutoff, or with no cutoff, is not written.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from src.analysis.canonical_rows import content_hash, copy_canonical_rows
from src.dashboard.prediction_checkpointing import prediction_targets_for_checkpoint
from src.utils.operational_observability import record_counter

logger = logging.getLogger(__name__)

TABLE = "forecast_record"
_UNIQUE_VIOLATION = "23505"  # PostgreSQL unique_violation, as PostgREST reports it
_IDENTITY = ("season", "event", "target", "checkpoint", "weather")
_PRODUCED_BY = (
    "model_version",
    "config_hash",
    "learning_state_hash",
    "artifact_hash",
    "boundary_signature",
)


def build_forecast_records(
    *,
    year: int,
    race_name: str,
    checkpoint: str,
    weather: str,
    is_sprint: bool,
    prediction_results: Mapping[str, Any],
    information_cutoff_at: str | None,
    boundary_signature: str,
    model_version: str,
    artifact_hash: str,
    artifact_versions: Mapping[str, tuple[int, str]],
) -> list[dict[str, Any]]:
    """Return one insertable row per still-forecast target, its rows copied exactly as served."""
    produced_by = {
        "model_version": str(model_version),
        "config_hash": str(artifact_versions.get("config/default.yaml", (0, ""))[1]),
        "learning_state_hash": str(
            artifact_versions.get("learning_state::learning_state", (0, ""))[1]
        ),
        "artifact_hash": str(artifact_hash),
        "boundary_signature": str(boundary_signature),
    }
    targets = prediction_targets_for_checkpoint(
        prediction_results=dict(prediction_results),
        is_sprint=is_sprint,
        session_name=checkpoint,
    )
    records = []
    for target, payload in sorted(targets.items()):
        identity = {
            "season": int(year),
            "event": str(race_name),
            "target": target,
            "checkpoint": str(checkpoint),
            "weather": str(weather).strip().lower(),
        }
        rows = copy_canonical_rows(
            payload["predicted_order"], label=f"{target} rows", ranked_order=True
        )
        core = {**identity, "rows": rows}
        records.append(
            {
                **identity,
                **produced_by,
                "information_cutoff_at": information_cutoff_at,
                "core": core,
                "content_hash": content_hash(core),
            }
        )
    return records


def write_forecast_records(
    records: list[dict[str, Any]], *, client: Any, now: datetime | None = None
) -> int:
    """Insert the records that are new and on time; return how many were stored.

    Only a repeat of the latest record for the identity is skipped, so A, B, then A
    stores three issuances. Migration 008 lets the table hold the returning A (unique
    per ``content_hash`` and ``replaces``); before it, the table refuses that row as a
    duplicate key, which is skipped, as is a row a parallel run stored first.
    """
    now = now or datetime.now(UTC)
    sent = 0
    for record in records:
        labels = {key: record[key] for key in ("season", "event", "target", "checkpoint")}
        cutoff = record.get("information_cutoff_at")
        if not cutoff or now > datetime.fromisoformat(str(cutoff)):
            reason = "after_cutoff" if cutoff else "no_cutoff"
            record_counter("forecast_record_refused_total", labels={**labels, "reason": reason})
            continue
        query = client.table(TABLE).select(",".join(("id", "content_hash", *_PRODUCED_BY)))
        for key in _IDENTITY:
            query = query.eq(key, record[key])
        previous = query.order("id", desc=True).limit(1).execute().data
        row = dict(record)
        if previous:
            if previous[0]["content_hash"] == record["content_hash"]:
                continue
            changed = [key for key in _PRODUCED_BY if previous[0].get(key) != record[key]]
            row["replaces"] = previous[0]["id"]
            row["replace_reason"] = (
                "changed: " + ", ".join(changed) if changed else "recomputed from the same inputs"
            )
        try:
            client.table(TABLE).insert(row).execute()
        except Exception as exc:
            if getattr(exc, "code", None) != _UNIQUE_VIOLATION:
                raise
            record_counter("forecast_record_duplicate_total", labels=labels)
            continue
        sent += 1
    return sent


def record_served_forecast(
    *,
    year: int,
    race_name: str,
    checkpoint: str,
    weather: str,
    is_sprint: bool,
    prediction_results: Mapping[str, Any],
    boundary_signature: str,
    artifact_hash: str,
    artifact_versions: Mapping[str, tuple[int, str]],
) -> None:
    """Append one served payload to the record; a failure is logged and never stops serving."""
    from src.persistence.config import should_write_to_db

    if not should_write_to_db():
        return
    labels = {"season": int(year), "event": str(race_name), "checkpoint": str(checkpoint)}
    try:
        from src.persistence.db import get_supabase_client
        from src.utils.checkpoint_reconstruction import compute_information_cutoff_at
        from src.utils.model_version import get_model_version

        records = build_forecast_records(
            year=year,
            race_name=race_name,
            checkpoint=checkpoint,
            weather=weather,
            is_sprint=is_sprint,
            prediction_results=prediction_results,
            information_cutoff_at=compute_information_cutoff_at(
                year=year, race_name=race_name, checkpoint_session=checkpoint, is_sprint=is_sprint
            ),
            boundary_signature=boundary_signature,
            model_version=get_model_version(),
            artifact_hash=artifact_hash,
            artifact_versions=artifact_versions,
        )
        sent = write_forecast_records(records, client=get_supabase_client())
        record_counter("forecast_record_written_total", labels={**labels, "count": sent})
    except Exception as exc:  # noqa: BLE001 - the record sits beside serving and must never break it
        logger.warning("Could not record forecast %s: %s", labels, exc)
        record_counter("forecast_record_write_failed_total", labels=labels)
