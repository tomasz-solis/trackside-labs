#!/usr/bin/env python3
"""Inspect local forecast JSON files without changing or interpreting them as published."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

KNOWN_TARGETS = (
    "grand_prix_race",
    "main_qualifying",
    "sprint_qualifying",
    "sprint_race",
)
LEGACY_TARGET_KEYS = {
    "grand_prix_race": "race",
    "main_qualifying": "qualifying",
    "sprint_qualifying": "qualifying",
    "sprint_race": "race",
}
KNOWN_CHECKPOINTS = {"PRE", "FP1", "FP2", "FP3", "SQ", "Sprint", "Q"}
MODEL_VERSION_RE = re.compile(r"v?\d{1,4}(?:\.\d{1,4}){1,3}")
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
)
HISTORICAL_SOURCES = {"historical_replay", "checkpoint_reconstruction"}


class _NonFiniteJSON(ValueError):
    """Signal a non-standard JSON number such as NaN or Infinity."""


class _DuplicateJSONKey(ValueError):
    """Signal ambiguous JSON objects containing the same key more than once."""


def _reject_constant(value: str) -> None:
    """Reject JavaScript numeric constants accepted by Python's JSON decoder."""
    raise _NonFiniteJSON(value)


def _finite_float(value: str) -> float:
    """Parse a JSON float while rejecting exponents that overflow to infinity."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _NonFiniteJSON(value)
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object and reject duplicate keys instead of silently choosing one."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey
        result[key] = value
    return result


def _input_files(inputs: list[Path]) -> list[Path]:
    """Expand explicit files and directories to a stable, de-duplicated file list."""
    files: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.expanduser()
        if not path.exists():
            raise ValueError("input path unavailable")
        if path.is_dir():
            try:
                files.update(
                    candidate.resolve()
                    for candidate in path.rglob("*.json")
                    if candidate.is_file() and not candidate.is_symlink()
                )
            except OSError as exc:
                raise ValueError("input path unavailable") from exc
        elif path.is_file():
            files.add(path.resolve())
        else:
            raise ValueError("input path unavailable")
    if not files:
        raise ValueError("no JSON input files found")
    return sorted(files, key=lambda item: item.as_posix())


def _known_text(value: Any, allowed: set[str] | None = None) -> str | None:
    """Return a bounded allowlisted text value, never arbitrary payload text."""
    if not isinstance(value, str) or len(value) > 128:
        return None
    if allowed is not None and value not in allowed:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.+:-]+", value):
        return None
    return value


def _as_of(metadata: dict[str, Any]) -> dict[str, Any]:
    """Report only a valid explicit information cutoff, never a generation timestamp."""
    value = metadata.get("information_cutoff_at")
    if not isinstance(value, str) or len(value) > 64:
        return {"status": "unknown", "value": None}
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return {"status": "unknown", "value": None}
    if parsed.utcoffset() is None:
        return {"status": "unknown", "value": None}
    return {"status": "known", "value": value}


def _has_forecast(value: Any) -> str:
    """Classify whether a supported forecast row collection is present."""
    if not isinstance(value, dict):
        return "unsupported"
    rows = value.get("predicted_order")
    if isinstance(rows, list):
        return _collection_status(rows)
    return "unsupported"


def _collection_status(value: Any) -> str:
    """Require list rows with nonempty driver and team names.

    Saved actuals use lists of row objects or ``None``. A mapping has no verified
    interpretation here, and a row without its driver/team identity is unusable by
    the consumer sanitizer. Position is optional because the consumer supplies a
    stable row-order default.
    """
    if isinstance(value, list):
        if not value:
            return "present_empty"
        valid_rows = all(
            isinstance(row, dict)
            and isinstance(row.get("driver"), str)
            and bool(row["driver"].strip())
            and isinstance(row.get("team"), str)
            and bool(row["team"].strip())
            for row in value
        )
        return "present" if valid_rows else "unsupported"
    return "unsupported"


def _explicitly_not_forecast(
    document: dict[str, Any], metadata: dict[str, Any], target: str
) -> bool:
    """Recognize only explicit target ineligibility or an actual-result mode."""
    targets = document.get("targets")
    target_record = targets.get(target) if isinstance(targets, dict) else None
    if isinstance(target_record, dict):
        result_mode = target_record.get("result_mode")
        return target_record.get("eligible_at_save") is False or (
            isinstance(result_mode, str) and result_mode.upper() == "ACTUAL"
        )

    legacy_key = LEGACY_TARGET_KEYS[target]
    field_prefix = "top_level_race" if legacy_key == "race" else "top_level_qualifying"
    if metadata.get(f"{field_prefix}_target") not in (None, target):
        return False
    eligible_at_save = metadata.get(f"{field_prefix}_eligible_at_save")
    result_mode = metadata.get(f"{field_prefix}_result_mode")
    return eligible_at_save is False or (
        isinstance(result_mode, str) and result_mode.upper() == "ACTUAL"
    )


def _target_rows(document: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize target rows while keeping forecast display and scoreability conservative."""
    targets = document.get("targets")
    targets = targets if isinstance(targets, dict) else {}
    actuals = document.get("actuals")
    actuals = actuals if isinstance(actuals, dict) else {}
    actual_target_map = actuals.get("targets")
    actual_target_map = actual_target_map if isinstance(actual_target_map, dict) else {}

    rows: list[dict[str, str]] = []
    for target in KNOWN_TARGETS:
        if target in targets:
            forecast_status = _has_forecast(targets[target])
        else:
            legacy_key = LEGACY_TARGET_KEYS[target]
            target_field = (
                "top_level_race_target" if legacy_key == "race" else "top_level_qualifying_target"
            )
            declared_target = metadata.get(target_field)
            default_target = "grand_prix_race" if legacy_key == "race" else "main_qualifying"
            if declared_target is not None and declared_target != target:
                continue
            if declared_target is None and target != default_target:
                continue
            legacy_section = document.get(legacy_key)
            legacy_rows_key = "predicted_results" if legacy_key == "race" else "predicted_grid"
            legacy_rows = (
                legacy_section.get(legacy_rows_key) if isinstance(legacy_section, dict) else None
            )
            forecast_status = (
                _collection_status(legacy_rows) if isinstance(legacy_rows, list) else "missing"
            )

        if target in actual_target_map:
            actual_value = actual_target_map[target]
        elif target in targets:
            actual_value = None
        else:
            legacy_key = LEGACY_TARGET_KEYS[target]
            target_field = (
                "top_level_race_target" if legacy_key == "race" else "top_level_qualifying_target"
            )
            declared_target = metadata.get(target_field)
            default_target = "grand_prix_race" if legacy_key == "race" else "main_qualifying"
            if declared_target is not None and declared_target != target:
                continue
            if declared_target is None and target != default_target:
                continue
            actual_value = actuals.get(legacy_key)
        if actual_value is None:
            actuals_status = "missing"
        else:
            actuals_status = _collection_status(actual_value)

        explicitly_not_forecast = _explicitly_not_forecast(document, metadata, target)
        if explicitly_not_forecast:
            displayable_state = "not_a_forecast"
        elif forecast_status == "present":
            displayable_state = "historical_inspection_only"
        else:
            displayable_state = "unavailable"
        scoreability_status = (
            "provenance_unknown"
            if forecast_status == "present"
            and actuals_status == "present"
            and not explicitly_not_forecast
            else "not_scoreable"
        )

        if forecast_status != "missing" or actuals_status != "missing":
            rows.append(
                {
                    "target": target,
                    "forecast_status": forecast_status,
                    "actuals_status": actuals_status,
                    "displayable_state": displayable_state,
                    "scoreability_status": scoreability_status,
                }
            )
    return rows


def _target_coverage(
    document: dict[str, Any], metadata: dict[str, Any], target_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Count unsupported target declarations without exposing their names or values."""
    unsupported_key_count = 0
    recognized_target_seen = bool(target_rows)
    targets = document.get("targets")
    if isinstance(targets, dict):
        for target in targets:
            if target in KNOWN_TARGETS:
                recognized_target_seen = True
            else:
                unsupported_key_count += 1

    actuals = document.get("actuals")
    actual_targets = actuals.get("targets") if isinstance(actuals, dict) else None
    if isinstance(actual_targets, dict):
        for target in actual_targets:
            if target in KNOWN_TARGETS:
                recognized_target_seen = True
            else:
                unsupported_key_count += 1

    for field in ("top_level_race_target", "top_level_qualifying_target"):
        target = metadata.get(field)
        if target is None:
            continue
        if isinstance(target, str) and target in KNOWN_TARGETS:
            recognized_target_seen = True
        else:
            unsupported_key_count += 1

    if unsupported_key_count:
        status = "partial" if recognized_target_seen else "unsupported"
    else:
        status = "complete" if recognized_target_seen else "unknown"
    return {"status": status, "unsupported_key_count": unsupported_key_count}


def _inspect_file(path: Path) -> dict[str, Any]:
    """Return a redacted, non-publishing lineage summary of original file bytes."""
    result: dict[str, Any] = {
        "sha256": None,
        "artifact_status": "unreadable",
        "targets": [],
        "checkpoint": {"status": "unknown", "value": None},
        "model_version": {"status": "unknown", "value": None},
        "run_id": {"status": "unknown", "value": None},
        "forecast_id": {"status": "unknown", "value": None},
        "as_of": {"status": "unknown", "value": None},
        "actuals_status": "missing",
        "target_coverage": {"status": "unknown", "unsupported_key_count": 0},
        "publication_status": "unknown",
        "provenance_status": "unknown",
        "scoreability_status": "not_scoreable",
        "prospective_eligibility_status": "unknown",
        "record_state": "unknown",
        "displayable_state": "unavailable",
    }
    try:
        source_bytes = path.read_bytes()
    except OSError:
        return result
    result["sha256"] = hashlib.sha256(source_bytes).hexdigest()
    result["artifact_status"] = "ok"
    try:
        document = json.loads(
            source_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=_unique_object,
        )
    except UnicodeDecodeError:
        result["artifact_status"] = "invalid_encoding"
        return result
    except _NonFiniteJSON:
        result["artifact_status"] = "invalid_json_nonfinite"
        return result
    except _DuplicateJSONKey:
        result["artifact_status"] = "invalid_json_duplicate_key"
        return result
    except RecursionError:
        result["artifact_status"] = "invalid_json_depth"
        return result
    except (json.JSONDecodeError, ValueError):
        result["artifact_status"] = "invalid_json"
        return result

    if not isinstance(document, dict):
        result["artifact_status"] = "unsupported_schema"
        return result

    metadata = document.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    checkpoint = _known_text(metadata.get("session_name"), KNOWN_CHECKPOINTS)
    if checkpoint is not None:
        result["checkpoint"] = {"status": "known", "value": checkpoint}
    model_version = metadata.get("model_version")
    if isinstance(model_version, str) and MODEL_VERSION_RE.fullmatch(model_version):
        result["model_version"] = {"status": "known", "value": model_version}
    for identity_field in ("run_id", "forecast_id"):
        identity = metadata.get(identity_field)
        if isinstance(identity, str) and UUID_RE.fullmatch(identity):
            result[identity_field] = {"status": "known", "value": identity}
    source = metadata.get("source")
    if isinstance(source, str) and source in HISTORICAL_SOURCES:
        result["record_state"] = "historical_only"
    result["as_of"] = _as_of(metadata)
    target_rows = _target_rows(document, metadata)
    target_coverage = _target_coverage(document, metadata, target_rows)
    result["target_coverage"] = target_coverage
    result["targets"] = target_rows
    if target_coverage["unsupported_key_count"]:
        result["artifact_status"] = (
            "partial_coverage" if target_coverage["status"] == "partial" else "unsupported_schema"
        )
    if not target_rows:
        result["artifact_status"] = (
            "partial_coverage" if target_coverage["status"] == "partial" else "unsupported_schema"
        )
        result["actuals_status"] = "unsupported"
        result["displayable_state"] = "unavailable"
        result["scoreability_status"] = "not_scoreable"
    else:
        displayable_states = {row["displayable_state"] for row in target_rows}
        result["displayable_state"] = (
            "partial"
            if target_coverage["status"] == "partial"
            else next(iter(displayable_states))
            if len(displayable_states) == 1
            else "mixed"
        )
        target_scoreability = {row["scoreability_status"] for row in target_rows}
        result["scoreability_status"] = (
            "provenance_unknown"
            if target_scoreability == {"provenance_unknown"}
            and target_coverage["unsupported_key_count"] == 0
            else "not_scoreable"
        )
        statuses = {row["actuals_status"] for row in target_rows}
        if "present" in statuses:
            result["actuals_status"] = (
                "partial"
                if statuses.intersection({"missing", "present_empty", "unsupported"})
                else "present"
            )
        elif "present_empty" in statuses:
            result["actuals_status"] = (
                "partial" if statuses.intersection({"missing", "unsupported"}) else "present_empty"
            )
        elif "unsupported" in statuses:
            result["actuals_status"] = "unsupported"
    return result


def audit_forecast_lineage(inputs: list[Path]) -> dict[str, Any]:
    """Inventory local forecast rows, never certify readiness or public score eligibility.

    ``historical_inspection_only`` means row-shaped forecast data exists for manual
    inspection. A paired actual remains ``provenance_unknown`` until independent
    publication and source lineage evidence exists; missing or malformed pairs are
    ``not_scoreable``. These states never imply a current or prospective publication.
    """
    artifacts = [_inspect_file(path) for path in _input_files(inputs)]
    artifacts.sort(key=lambda item: (item["sha256"] or "", item["artifact_status"]))
    return {"schema_version": 1, "artifact_count": len(artifacts), "artifacts": artifacts}


def main(argv: list[str] | None = None) -> int:
    """Run the read-only audit CLI for one or more local paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Local JSON file or directory to inspect"
    )
    args = parser.parse_args(argv)
    try:
        report = audit_forecast_lineage(args.inputs)
    except ValueError:
        print(json.dumps({"schema_version": 1, "error": "input_unavailable"}, sort_keys=True))
        return 2
    print(json.dumps(report, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
