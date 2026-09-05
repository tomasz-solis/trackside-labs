"""Shared payload helpers for baseline predictor data loading."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.utils.accuracy_targets import (
    explicit_target_actuals,
    row_is_dnf,
    synthesize_legacy_actuals,
)
from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger("src.predictors.baseline_2026")


def driver_characteristics_fallback_paths(data_dir: Path, year: int) -> tuple[Path, ...]:
    """Return season-aware driver-characteristics fallback candidates."""
    candidates: list[Path] = [
        data_dir / "driver_characteristics" / f"{year}_driver_characteristics.json"
    ]
    nearest = nearest_season_payload_path(
        data_dir / "driver_characteristics",
        suffix="driver_characteristics",
        target_year=year,
    )
    if nearest is not None:
        _, nearest_path = nearest
        if nearest_path not in candidates:
            candidates.append(nearest_path)
    candidates.append(data_dir / "driver_characteristics.json")
    return tuple(candidates)


def nearest_season_payload_path(
    directory: Path,
    *,
    suffix: str,
    target_year: int,
) -> tuple[int, Path] | None:
    """Return the closest season-scoped payload file under one directory."""
    exact_path = directory / f"{target_year}_{suffix}.json"
    if exact_path.exists():
        return target_year, exact_path

    candidates: list[tuple[int, Path]] = []
    for path in directory.glob(f"*_{suffix}.json"):
        prefix = path.name.split("_", 1)[0].strip()
        if prefix.isdigit():
            candidates.append((int(prefix), path))

    if not candidates:
        return None

    return min(candidates, key=lambda item: (abs(item[0] - target_year), item[0]))


def infer_payload_year_from_path(path: Path, *, suffix: str) -> int | None:
    """Extract the season year from a `YYYY_<suffix>.json` filename."""
    prefix = path.name.removesuffix(f"_{suffix}.json").split("_", 1)[0].strip()
    if prefix.isdigit():
        return int(prefix)
    return None


def coerce_non_negative_int(value: object) -> int | None:
    """Convert an int-like value into a non-negative integer when possible."""
    if isinstance(value, bool):
        parsed = int(value)
    elif isinstance(value, int | float | np.integer | np.floating):
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    elif isinstance(value, str | bytes | bytearray):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
    else:
        return None
    return max(parsed, 0)


def sanitize_performance_observations(observations: object) -> list[float]:
    """Return a finite 0-1 performance series from a raw observations payload."""
    if not isinstance(observations, list):
        return []

    sanitized: list[float] = []
    for value in observations:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_value):
            continue
        sanitized.append(float(np.clip(numeric_value, 0.0, 1.0)))
    return sanitized


def extract_target_actual_rows(
    prediction_data: dict[str, object],
    *,
    target_key: str,
) -> list[dict[str, object]]:
    """Return canonical actual rows for one target from a saved prediction payload."""
    explicit_rows = explicit_target_actuals(prediction_data).get(target_key)
    if explicit_rows:
        return explicit_rows

    metadata = prediction_data.get("metadata", {})
    weekend_format = ""
    if isinstance(metadata, dict):
        weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
    synthesized_targets = synthesize_legacy_actuals(
        prediction_data,
        is_sprint=weekend_format == "sprint",
    )
    return synthesized_targets.get(target_key, [])


def score_teams_from_actual_rows(
    actual_rows: list[dict[str, object]],
    *,
    known_teams: set[str],
) -> dict[str, float]:
    """Convert classified positions into rank-based team-form scores.

    Scores are ranks, so excluding a retirement is not monotonically favourable: if a
    retired car was classified AHEAD of its surviving teammate, dropping it raises the
    team mean and can cost a rank step. That is rare, because classification normally
    places non-classified cars behind finishers, but the exclusion is not a guaranteed
    improvement for every team.

    ponytail: a team scored on one surviving car is not comparable to one scored on
    two, and unlike the telemetry path in `updater_flow._build_position_fallback_race_pace`
    there is no entered-field guard here. Shrink a single-car mean toward the field
    mean if high-attrition races start distorting season form.

    Retirements (per `row_is_dnf`, which reads the `dnf`, `status`, or `classified`
    shape a row carries) are excluded from a team's position mean, so a mechanical
    DNF does not score a fast car as if it were a slow one. Rows with no DNF signal
    of any kind are always kept, so legacy actuals that predate the flag score
    identically to before. If excluding retirements would leave a team with no rows
    to score, that team's original (unfiltered) rows are kept instead of dropping the
    team entirely - mirrors the guard in `updater_flow.extract_dnf_drivers` usage.

    A row whose team does not resolve into `known_teams` is excluded entirely rather
    than kept under its raw name: an unmapped team taking a rank slot shifts every
    other team's rank-spaced score. Rows with an invalid `position` (missing,
    non-integer, or < 1) are also excluded, as before, but now counted and logged.
    A single resolvable team carries no relative ranking information, so it scores
    `{}` instead of a fabricated 0.5.
    """
    team_positions: dict[str, list[int]] = {}
    team_positions_excluding_dnf: dict[str, list[int]] = {}
    unresolved_team_names: set[str] = set()
    unresolved_row_count = 0
    invalid_position_count = 0
    rows_with_team = 0

    for row in actual_rows:
        raw_team = row.get("team")
        if not isinstance(raw_team, str) or not raw_team.strip():
            continue
        raw_team_name = raw_team.strip()
        rows_with_team += 1

        canonical_team = map_team_to_characteristics(raw_team_name, known_teams=known_teams)
        if canonical_team is None:
            unresolved_team_names.add(raw_team_name)
            unresolved_row_count += 1
            continue

        position = coerce_non_negative_int(row.get("position"))
        if position is None or position < 1:
            invalid_position_count += 1
            continue

        team_positions.setdefault(canonical_team, []).append(position)
        if not row_is_dnf(row):
            team_positions_excluding_dnf.setdefault(canonical_team, []).append(position)

    if unresolved_team_names:
        logger.warning(
            "score_teams_from_actual_rows: excluded %s row(s) for unresolved team "
            "name(s) not in known_teams: %s",
            unresolved_row_count,
            sorted(unresolved_team_names),
        )

    if invalid_position_count:
        logger.warning(
            "score_teams_from_actual_rows: skipped %s row(s) with an invalid "
            "position (missing, non-integer, or < 1)",
            invalid_position_count,
        )

    if not team_positions:
        if rows_with_team and unresolved_row_count == rows_with_team:
            logger.error(
                "score_teams_from_actual_rows: every row's team failed to resolve "
                "into known_teams; check the alias map. Unresolved name(s): %s",
                sorted(unresolved_team_names),
            )
        return {}

    scored_positions = {
        team_name: team_positions_excluding_dnf.get(team_name) or positions
        for team_name, positions in team_positions.items()
    }

    if len(scored_positions) == 1:
        logger.warning(
            "score_teams_from_actual_rows: only one resolvable team in this set of "
            "rows; a single team carries no relative ranking information"
        )
        return {}

    team_avg = {
        team_name: float(np.mean(positions))
        for team_name, positions in scored_positions.items()
        if positions
    }
    if not team_avg:
        return {}

    sorted_teams = sorted(team_avg, key=lambda team_name: team_avg[team_name])
    team_count = len(sorted_teams)
    scored_teams: dict[str, float] = {}
    for rank_index, team_name in enumerate(sorted_teams):
        scored_teams[team_name] = float(1.0 - (rank_index / max(team_count - 1, 1)))

    return scored_teams


def canonicalize_team_payload_keys(teams_payload: dict[str, object]) -> dict[str, dict]:
    """Canonicalize team payload keys and safely merge overlapping aliases."""
    canonical_payload: dict[str, dict] = {}
    for raw_team_name, raw_team_data in teams_payload.items():
        if not isinstance(raw_team_data, dict):
            continue

        mapped_name = map_team_to_characteristics(str(raw_team_name))
        team_name = (
            mapped_name if isinstance(mapped_name, str) and mapped_name else str(raw_team_name)
        )

        existing = canonical_payload.get(team_name)
        if existing is None:
            canonical_payload[team_name] = deepcopy(raw_team_data)
            continue

        merged = _merge_team_payload(existing, raw_team_data)
        canonical_payload[team_name] = merged if isinstance(merged, dict) else existing

    return canonical_payload


def _is_missing_payload_value(value: object) -> bool:
    """Return True when payload value should be treated as missing during merge."""
    if value is None:
        return True
    if isinstance(value, float):
        return not np.isfinite(value)
    return False


def _merge_team_payload(existing: object, incoming: object) -> object:
    """Merge team payload fragments while preserving existing non-missing values."""
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = deepcopy(existing)
        for key, incoming_value in incoming.items():
            if key not in merged:
                merged[key] = deepcopy(incoming_value)
                continue
            merged[key] = _merge_team_payload(merged[key], incoming_value)
        return merged

    if _is_missing_payload_value(existing) and not _is_missing_payload_value(incoming):
        return deepcopy(incoming)
    return deepcopy(existing)
