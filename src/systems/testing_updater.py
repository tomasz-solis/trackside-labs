"""
Testing/Practice Directionality Updater

Updates team car directionality metrics from pre-season testing or weekend
practice sessions. It can be run manually (CLI) and is also invoked by the
dashboard when new FP sessions are completed.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

from src.extractors.performance import extract_all_teams_performance
from src.systems.compound_analyzer import (
    aggregate_compound_samples,
    extract_compound_metrics,
    normalize_compound_metrics_across_teams,
)
from src.utils.file_operations import atomic_json_write
from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger(__name__)
logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logging.getLogger("fastf1.logger").setLevel(logging.CRITICAL)
logging.getLogger("requests_cache").setLevel(logging.CRITICAL)
try:
    fastf1.set_log_level("CRITICAL")
except (AttributeError, TypeError):
    pass


DEFAULT_SESSION_CANDIDATES = [
    "FP1",
    "FP2",
    "FP3",
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Day 1",
    "Day 2",
    "Day 3",
]

_TESTING_BACKENDS = ("f1timing", "fastf1", None)
_TESTING_CACHE_ROOT = Path("data/raw")
_DEFAULT_TESTING_CACHE_DIR = _TESTING_CACHE_ROOT / ".fastf1_cache_testing"

_DIRECTIONALITY_KEYS = (
    "max_speed",
    "slow_corner_speed",
    "medium_corner_speed",
    "high_corner_speed",
)

_SESSION_AGGREGATION_MODES = ("mean", "median", "laps_weighted")
_RUN_PROFILE_MODES = ("balanced", "all", "short_run", "long_run")
_PROFILES_FOR_STORAGE = ("balanced", "short_run", "long_run")
_SHORT_STINT_MAX_LAPS = 5
_LONG_STINT_MIN_LAPS = 8


def _normalize_name(value: str) -> str:
    """Normalize names for robust matching."""
    return "".join(char for char in value.lower() if char.isalnum())


def _is_testing_event(event_name: str) -> bool:
    """Best-effort detection of testing events from user-provided name."""
    normalized = _normalize_name(event_name)
    return "test" in normalized


def _extract_testing_day(session_name: str) -> int | None:
    """Map session label to a testing day number (1..3) if possible."""
    normalized = _normalize_name(session_name)
    for day in (1, 2, 3):
        if str(day) in normalized:
            return day
    return None


def _extract_testing_number(event_name: str) -> int | None:
    """Parse explicit test number from event name (e.g., 'Testing 2')."""
    normalized = _normalize_name(event_name)
    for number in (1, 2, 3):
        if f"test{number}" in normalized or f"testing{number}" in normalized:
            return number
    return None


def _resolve_testing_backends(
    preferred_backend: str | None = "auto",
) -> tuple[str | None, ...]:
    """Resolve backend preference into an ordered list of backends to try."""
    if preferred_backend in (None, "auto"):
        return _TESTING_BACKENDS
    if preferred_backend in ("fastf1", "f1timing"):
        return (preferred_backend,)

    raise ValueError("Invalid testing backend. Use one of: auto, fastf1, f1timing.")


def _resolve_testing_cache_dir(cache_dir: str | None = None) -> Path:
    """
    Resolve testing cache location.

    Relative paths are kept under data/raw to avoid repository root clutter.
    """
    if not cache_dir:
        return _DEFAULT_TESTING_CACHE_DIR

    candidate = Path(cache_dir).expanduser()
    if candidate.is_absolute():
        return candidate

    cleaned_parts = tuple(part for part in candidate.parts if part not in ("", "."))
    if not cleaned_parts:
        return _DEFAULT_TESTING_CACHE_DIR

    relative_candidate = Path(*cleaned_parts)
    if relative_candidate.parts[:2] == ("data", "raw"):
        return relative_candidate

    return _TESTING_CACHE_ROOT / relative_candidate


def _coerce_utc_datetime(value) -> datetime | None:
    """Convert FastF1 event datetime values to UTC-aware datetime."""
    if value is None or pd.isna(value):
        return None

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.to_pydatetime()


def _testing_session_has_started(
    event: fastf1.events.Event, day_number: int, now_utc: datetime | None = None
) -> bool:
    """Check whether a testing day has started based on UTC session timestamp."""
    session_dt_utc = _coerce_utc_datetime(event.get(f"Session{day_number}DateUtc"))
    if session_dt_utc is None:
        return True

    now = now_utc or datetime.now(UTC)
    # Keep a small tolerance for clock skew between systems.
    return session_dt_utc <= (now + timedelta(minutes=15))


def _get_testing_event_with_backends(
    year: int,
    test_number: int,
    testing_backends: tuple[str | None, ...],
    error_messages: list[str] | None = None,
) -> fastf1.events.Event | None:
    """Load a testing event, trying explicit backends before auto mode."""
    for backend in testing_backends:
        kwargs = {"backend": backend} if backend is not None else {}
        backend_label = backend or "auto"
        try:
            return fastf1.get_testing_event(year, test_number, **kwargs)
        except Exception as exc:
            logger.debug(
                "Unable to load testing event %s/%s via backend %s: %s",
                year,
                test_number,
                backend_label,
                exc,
            )
            if error_messages is not None:
                error_messages.append(
                    f"testing_event#{test_number} backend={backend_label} -> "
                    f"{type(exc).__name__}: {exc}"
                )

    return None


def _normalize_testing_event_sessions(event: fastf1.events.Event) -> None:
    """
    Normalize testing session labels to FastF1-compatible names.

    Some schedules expose "Day 1/2/3". FastF1 Session initialization expects
    canonical names like "Practice 1/2/3".
    """
    for day_number in (1, 2, 3):
        key = f"Session{day_number}"
        value = event.get(key)
        if not isinstance(value, str):
            continue
        if _normalize_name(value) == f"day{day_number}":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SettingWithCopyWarning)
                event[key] = f"Practice {day_number}"


def _load_testing_session_with_backends(
    year: int,
    test_number: int,
    day_number: int,
    testing_backends: tuple[str | None, ...],
    error_messages: list[str] | None = None,
) -> fastf1.core.Session | None:
    """
    Load a testing session and verify laps are actually accessible.

    This avoids reporting sessions as discovered when `session.laps` would still
    raise DataNotLoadedError after `load()`.
    """
    for backend in testing_backends:
        kwargs = {"backend": backend} if backend is not None else {}
        backend_label = backend or "auto"
        try:
            event = fastf1.get_testing_event(year, test_number, **kwargs)
            _normalize_testing_event_sessions(event)
            session = event.get_session(day_number)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = session.laps
            if laps is None:
                raise ValueError("laps are None after session.load()")
            # Access row count to force DataNotLoadedError if load is incomplete.
            _ = len(laps)
            return session
        except Exception as exc:
            logger.debug(
                "Unable to load testing session %s/%s day %s via backend %s: %s",
                year,
                test_number,
                day_number,
                backend_label,
                exc,
            )
            if error_messages is not None:
                error_messages.append(
                    f"testing#{test_number}/day{day_number} backend={backend_label} -> "
                    f"{type(exc).__name__}: {exc}"
                )

    return None


def _load_sessions_for_event(
    year: int,
    event_name: str,
    session_candidates: list[str],
    testing_backends: tuple[str | None, ...] = _TESTING_BACKENDS,
    error_messages: list[str] | None = None,
) -> list[tuple[str, fastf1.core.Session]]:
    """
    Load available sessions for an event.

    Strategy:
    1) For non-testing events: use regular `get_session(event_name, session_name)`.
    2) For testing events: use `get_testing_event` + `get_testing_session`.
    """
    loaded: list[tuple[str, fastf1.core.Session]] = []

    if not _is_testing_event(event_name):
        for session_name in session_candidates:
            try:
                session = fastf1.get_session(year, event_name, session_name)
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                loaded.append((session_name, session))
            except Exception as exc:
                logger.debug(
                    f"Skipping unavailable session {year} {event_name} {session_name}: {exc}"
                )
                if error_messages is not None:
                    error_messages.append(
                        f"{event_name}::{session_name} -> {type(exc).__name__}: {exc}"
                    )
        return loaded

    explicit_test_number = _extract_testing_number(event_name)
    test_numbers = [explicit_test_number] if explicit_test_number else [1, 2, 3]

    day_candidates = []
    for session_name in session_candidates:
        maybe_day = _extract_testing_day(session_name)
        if maybe_day is not None and maybe_day not in day_candidates:
            day_candidates.append(maybe_day)
    if not day_candidates:
        day_candidates = [1, 2, 3]

    now_utc = datetime.now(UTC)

    for test_number in test_numbers:
        event = _get_testing_event_with_backends(
            year=year,
            test_number=test_number,
            testing_backends=testing_backends,
            error_messages=error_messages,
        )
        if event is None:
            continue

        for day_number in day_candidates:
            if not _testing_session_has_started(event, day_number, now_utc=now_utc):
                if error_messages is not None:
                    error_messages.append(
                        f"testing#{test_number}/day{day_number} -> session has not started yet"
                    )
                continue

            session = _load_testing_session_with_backends(
                year=year,
                test_number=test_number,
                day_number=day_number,
                testing_backends=testing_backends,
                error_messages=error_messages,
            )
            if session is None:
                continue

            label = f"Testing {test_number} Day {day_number}"
            loaded.append((label, session))

    return loaded


def _canonicalize_team_name(raw_team: str, known_teams: set[str]) -> str | None:
    """Map session team name to canonical team key used in characteristics JSON."""
    return map_team_to_characteristics(raw_team, known_teams=known_teams)


def _filter_valid_laps(team_laps: pd.DataFrame) -> pd.DataFrame:
    """Filter to representative non-pit laps."""
    if team_laps.empty:
        return team_laps
    if "LapTime" not in team_laps.columns:
        return team_laps.iloc[0:0].copy()

    # For testing updates we prioritize availability over strict pit filtering.
    # Timed laps are enough to infer early directionality.
    mask = team_laps["LapTime"].notna()
    # Testing sessions often have sparse/inconsistent IsAccurate flags.
    # Enforce this only when explicit True rows exist.
    if "IsAccurate" in team_laps.columns:
        accurate = team_laps["IsAccurate"]
        if accurate.notna().any() and bool(accurate.fillna(False).any()):
            mask &= accurate.fillna(False)

    return team_laps[mask].copy()


def _strip_in_out_laps(team_laps: pd.DataFrame) -> pd.DataFrame:
    """Remove in-laps/out-laps when pit timing columns are available."""
    if team_laps.empty:
        return team_laps

    filtered = team_laps.copy()
    if "PitOutTime" in filtered.columns:
        filtered = filtered[filtered["PitOutTime"].isna()]
    if "PitInTime" in filtered.columns:
        filtered = filtered[filtered["PitInTime"].isna()]

    return filtered


def _classify_run_laps(team_laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split team laps into short-run and long-run candidate subsets.

    Uses stint lengths when available; otherwise falls back to lap-time quantiles.
    """
    if team_laps.empty:
        return team_laps, team_laps

    cleaned = _strip_in_out_laps(team_laps)
    if cleaned.empty:
        cleaned = team_laps

    short_chunks: list[pd.DataFrame] = []
    long_chunks: list[pd.DataFrame] = []

    has_stint = "Stint" in cleaned.columns and bool(cleaned["Stint"].notna().any())

    if has_stint:
        grouping_cols = ["Driver", "Stint"]
        for _, stint_laps in cleaned.groupby(grouping_cols, dropna=False):
            timed = stint_laps[stint_laps["LapTime"].notna()].copy()
            if len(timed) < 2:
                continue
            stint_len = len(timed)
            if stint_len <= _SHORT_STINT_MAX_LAPS:
                short_chunks.append(timed)
            if stint_len >= _LONG_STINT_MIN_LAPS:
                long_chunks.append(timed)
    else:
        lap_seconds = pd.to_timedelta(cleaned["LapTime"], errors="coerce").dt.total_seconds()
        lap_seconds = lap_seconds.dropna()
        if not lap_seconds.empty:
            short_threshold = lap_seconds.quantile(0.35)
            long_threshold = lap_seconds.quantile(0.65)
            short_chunks.append(cleaned[lap_seconds <= short_threshold].copy())
            long_chunks.append(cleaned[lap_seconds >= long_threshold].copy())

    short_laps = (
        pd.concat(short_chunks, ignore_index=False) if short_chunks else cleaned.iloc[0:0].copy()
    )
    long_laps = (
        pd.concat(long_chunks, ignore_index=False) if long_chunks else cleaned.iloc[0:0].copy()
    )

    return short_laps, long_laps


def _select_stint_representative_laps(team_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce laps to one representative lap per driver/stint(/compound) slice.

    This avoids over-weighting teams with longer programs in the same session.
    """
    if team_laps.empty:
        return team_laps

    grouping_cols = ["Driver"]
    if "Stint" in team_laps.columns and bool(team_laps["Stint"].notna().any()):
        grouping_cols.append("Stint")
    if "Compound" in team_laps.columns and bool(team_laps["Compound"].notna().any()):
        grouping_cols.append("Compound")

    rows = []
    for _, laps in team_laps.groupby(grouping_cols, dropna=False):
        timed = laps[laps["LapTime"].notna()].copy()
        if timed.empty:
            continue

        lap_seconds = pd.to_timedelta(timed["LapTime"], errors="coerce").dt.total_seconds()
        valid_idx = lap_seconds.dropna().index
        if valid_idx.empty:
            continue

        median_value = float(lap_seconds.loc[valid_idx].median())
        representative_idx = (lap_seconds.loc[valid_idx] - median_value).abs().idxmin()
        rows.append(timed.loc[representative_idx])

    if not rows:
        return team_laps

    return pd.DataFrame(rows).copy()


def _select_program_aware_laps(team_laps: pd.DataFrame, run_profile: str) -> pd.DataFrame:
    """
    Select representative laps with program-aware run filtering.

    Modes:
    - all: use all valid laps
    - short_run: prefer short stints
    - long_run: prefer long stints
    - balanced: blend short + long stints
    """
    if team_laps.empty:
        return team_laps

    if run_profile not in _RUN_PROFILE_MODES:
        raise ValueError(
            f"Invalid run_profile '{run_profile}'. Use one of: {', '.join(_RUN_PROFILE_MODES)}"
        )

    if run_profile == "all":
        selected = team_laps
    else:
        short_laps, long_laps = _classify_run_laps(team_laps)
        if run_profile == "short_run":
            selected = short_laps if not short_laps.empty else team_laps
        elif run_profile == "long_run":
            selected = long_laps if not long_laps.empty else team_laps
        else:
            if not short_laps.empty and not long_laps.empty:
                selected = pd.concat([short_laps, long_laps], ignore_index=False)
            elif not short_laps.empty:
                selected = short_laps
            elif not long_laps.empty:
                selected = long_laps
            else:
                selected = team_laps

    representative = _select_stint_representative_laps(selected)
    return representative if not representative.empty else selected


def _count_team_selected_laps(
    session: fastf1.core.Session,
    known_teams: set[str],
    run_profile: str = "all",
) -> dict[str, float]:
    """Count selected laps per team for a specific run-profile strategy."""
    try:
        laps = session.laps
    except Exception:
        return {}

    if laps is None or laps.empty or "Team" not in laps.columns:
        return {}

    if run_profile not in _RUN_PROFILE_MODES:
        raise ValueError(
            f"Invalid run_profile '{run_profile}'. Use one of: {', '.join(_RUN_PROFILE_MODES)}"
        )

    counts: dict[str, float] = {}
    raw_teams = laps["Team"].dropna().unique()
    for raw_team in raw_teams:
        canonical_team = _canonicalize_team_name(str(raw_team), known_teams)
        if not canonical_team:
            continue

        team_laps = laps[laps["Team"] == raw_team]
        valid_laps = _filter_valid_laps(team_laps)
        if valid_laps.empty:
            continue

        selected_laps = _select_program_aware_laps(valid_laps, run_profile=run_profile)
        if selected_laps.empty:
            selected_laps = valid_laps

        counts[canonical_team] = counts.get(canonical_team, 0.0) + float(len(selected_laps))

    return counts


def _median_timedelta_seconds(series: pd.Series) -> float | None:
    """Get median timedelta in seconds if available."""
    if series is None or series.empty:
        return None

    values = pd.to_timedelta(series, errors="coerce").dropna()
    if values.empty:
        return None

    return float(values.dt.total_seconds().median())


def _median_lap_seconds(team_laps: pd.DataFrame) -> float | None:
    """Get median lap time in seconds for a team slice."""
    if "LapTime" not in team_laps.columns or team_laps.empty:
        return None

    lap_seconds = pd.to_timedelta(team_laps["LapTime"], errors="coerce").dt.total_seconds()
    lap_seconds = lap_seconds.dropna()
    if lap_seconds.empty:
        return None

    return float(lap_seconds.median())


def _estimate_tire_deg_slope(team_laps: pd.DataFrame) -> float | None:
    """
    Estimate team tire degradation slope from same-stint runs.

    Returns slope in seconds/lap (higher means more degradation).
    """
    if team_laps.empty or "LapNumber" not in team_laps.columns:
        return None

    grouping_cols = ["Driver"]
    if "Stint" in team_laps.columns:
        grouping_cols.append("Stint")
    if "Compound" in team_laps.columns:
        grouping_cols.append("Compound")

    slopes = []
    for _, stint_laps in team_laps.groupby(grouping_cols, dropna=False):
        stint = stint_laps.sort_values("LapNumber")
        if len(stint) < 3:
            continue

        lap_seconds = pd.to_timedelta(stint["LapTime"], errors="coerce").dt.total_seconds()
        lap_seconds = lap_seconds.dropna()
        if len(lap_seconds) < 3:
            continue

        x = np.arange(len(lap_seconds), dtype=float)
        y = lap_seconds.to_numpy(dtype=float)

        slope = float(np.polyfit(x, y, 1)[0])
        if -0.3 <= slope <= 1.0:
            slopes.append(slope)

    if not slopes:
        return None

    return float(np.median(slopes))


def _normalize_tire_deg_scores(
    tire_deg_slopes: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Normalize tire degradation to 0-1 performance scale (1.0 = best tire life)."""
    if not tire_deg_slopes:
        return {}

    min_slope = min(tire_deg_slopes.values())
    max_slope = max(tire_deg_slopes.values())

    normalized = {}
    for team, slope in tire_deg_slopes.items():
        if max_slope > min_slope:
            perf = 1.0 - ((slope - min_slope) / (max_slope - min_slope))
        else:
            perf = 0.5

        normalized[team] = {
            "tire_deg_slope": float(slope),
            "tire_deg_performance": float(np.clip(perf, 0.0, 1.0)),
        }

    return normalized


def _normalize_lower_better(metric_values: dict[str, float]) -> dict[str, float]:
    """Normalize a lower-is-better metric into 0-1 scale."""
    if not metric_values:
        return {}

    best = min(metric_values.values())
    worst = max(metric_values.values())
    if worst <= best:
        return {team: 0.5 for team in metric_values}

    normalized = {}
    for team, value in metric_values.items():
        score = 1.0 - ((value - best) / (worst - best))
        normalized[team] = float(np.clip(score, 0.0, 1.0))

    return normalized


def _extract_team_payload(valid_laps: pd.DataFrame) -> dict:
    """Build payload expected by extract_all_teams_performance()."""
    payload = {}

    sector_times = {}
    if "Sector1Time" in valid_laps.columns:
        s1 = _median_timedelta_seconds(valid_laps["Sector1Time"])
        if s1 is not None:
            sector_times["s1"] = s1
    if "Sector2Time" in valid_laps.columns:
        s2 = _median_timedelta_seconds(valid_laps["Sector2Time"])
        if s2 is not None:
            sector_times["s2"] = s2
    if "Sector3Time" in valid_laps.columns:
        s3 = _median_timedelta_seconds(valid_laps["Sector3Time"])
        if s3 is not None:
            sector_times["s3"] = s3
    if sector_times:
        payload["sector_times"] = sector_times

    speed_columns = [
        col for col in ("SpeedST", "SpeedFL", "SpeedI2", "SpeedI1") if col in valid_laps
    ]
    if speed_columns:
        speed_values = []
        for col in speed_columns:
            speed_values.extend(valid_laps[col].dropna().tolist())
        if speed_values:
            payload["speed_profile"] = {"top_speed": float(np.nanmedian(speed_values))}

    lap_seconds = pd.to_timedelta(valid_laps["LapTime"], errors="coerce").dt.total_seconds()
    lap_seconds = lap_seconds.dropna()
    if len(lap_seconds) >= 2:
        payload["consistency"] = {"std_lap_time": float(lap_seconds.std(ddof=0))}

    return payload


def _collect_session_metrics(
    session: fastf1.core.Session,
    session_key: str,
    known_teams: set[str],
    run_profile: str = "balanced",
    diagnostics: list[str] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Collect normalized directionality metrics and tire degradation metrics per team."""
    try:
        laps = session.laps
    except Exception as exc:
        logger.debug(f"Session laps unavailable for {session_key}: {exc}")
        if diagnostics is not None:
            diagnostics.append(f"{session_key}: laps unavailable ({type(exc).__name__})")
        return {}, {}

    if laps is None or laps.empty:
        if diagnostics is not None:
            diagnostics.append(f"{session_key}: no laps loaded")
        return {}, {}

    if "Team" not in laps.columns:
        if diagnostics is not None:
            diagnostics.append(f"{session_key}: laps missing Team column")
        return {}, {}

    per_team_payload = {}
    tire_deg_slopes = {}
    lap_pace_seconds = {}
    raw_teams = laps["Team"].dropna().unique()
    mapped_team_count = 0
    selected_lap_count = 0

    for raw_team in raw_teams:
        canonical_team = _canonicalize_team_name(str(raw_team), known_teams)
        if not canonical_team:
            continue
        mapped_team_count += 1

        team_laps = laps[laps["Team"] == raw_team]
        valid_laps = _filter_valid_laps(team_laps)
        # Allow early-session partial data (e.g., testing day in progress).
        if len(valid_laps) < 1:
            continue

        representative_laps = _select_program_aware_laps(valid_laps, run_profile=run_profile)
        if representative_laps.empty:
            representative_laps = valid_laps
        selected_lap_count += len(representative_laps)

        median_lap_seconds = _median_lap_seconds(representative_laps)
        if median_lap_seconds is not None:
            lap_pace_seconds[canonical_team] = median_lap_seconds

        payload = _extract_team_payload(representative_laps)
        if payload:
            per_team_payload.setdefault(canonical_team, {})[session_key] = payload

        if run_profile in ("balanced", "long_run"):
            _, long_laps = _classify_run_laps(valid_laps)
            tire_source = long_laps if not long_laps.empty else valid_laps
        else:
            tire_source = valid_laps

        slope = _estimate_tire_deg_slope(tire_source)
        if slope is not None:
            tire_deg_slopes[canonical_team] = slope

    normalized_perf = extract_all_teams_performance(per_team_payload, session_name=session_key)
    normalized_pace = _normalize_lower_better(lap_pace_seconds)
    for team, pace_score in normalized_pace.items():
        normalized_perf.setdefault(team, {})["overall_pace"] = pace_score

    normalized_tire = _normalize_tire_deg_scores(tire_deg_slopes)

    if diagnostics is not None:
        diagnostics.append(
            f"{session_key}: teams={len(raw_teams)} mapped={mapped_team_count} "
            f"perf_teams={len(normalized_perf)} tire_teams={len(normalized_tire)} "
            f"selected_laps={selected_lap_count} profile={run_profile}"
        )

    return normalized_perf, normalized_tire


def _build_directionality_from_metrics(
    metrics: dict[str, float], directionality_scale: float = 0.10
) -> dict[str, float]:
    """
    Convert 0-1 relative performance metrics into centered directionality deltas.

    Centered around 0 so testing modifier remains small in weight schedule blending.
    """
    metric_map = {
        "max_speed": "top_speed",
        "slow_corner_speed": "slow_corner_performance",
        "medium_corner_speed": "medium_corner_performance",
        "high_corner_speed": "fast_corner_performance",
    }

    fallback_pace = metrics.get("overall_pace")
    directionality = {}
    for key, metric_name in metric_map.items():
        if metric_name in metrics:
            value = float(metrics[metric_name])
        elif fallback_pace is not None and metric_name != "top_speed":
            # Conservative fallback: use overall pace only for corner directionality
            # when granular sector telemetry is still sparse.
            value = float(fallback_pace)
        else:
            value = 0.5
        centered = (value - 0.5) * directionality_scale
        directionality[key] = round(float(np.clip(centered, -0.2, 0.2)), 4)

    return directionality


def _blend_directionality(
    old_directionality: dict[str, float],
    new_directionality: dict[str, float],
    new_weight: float,
) -> dict[str, float]:
    """Blend current and newly extracted directionality to reduce noise."""
    bounded_weight = float(np.clip(new_weight, 0.0, 1.0))

    blended = {}
    for key in _DIRECTIONALITY_KEYS:
        old_value = float(old_directionality.get(key, 0.0))
        new_value = float(new_directionality.get(key, 0.0))
        blended[key] = round(((1.0 - bounded_weight) * old_value) + (bounded_weight * new_value), 4)

    return blended


def _count_team_valid_laps(session: fastf1.core.Session, known_teams: set[str]) -> dict[str, float]:
    """Count valid timed laps per canonical team for session weighting."""
    try:
        laps = session.laps
    except Exception:
        return {}

    if laps is None or laps.empty or "Team" not in laps.columns:
        return {}

    counts: dict[str, float] = {}
    raw_teams = laps["Team"].dropna().unique()
    for raw_team in raw_teams:
        canonical_team = _canonicalize_team_name(str(raw_team), known_teams)
        if not canonical_team:
            continue

        team_laps = laps[laps["Team"] == raw_team]
        valid_laps = _filter_valid_laps(team_laps)
        if valid_laps.empty:
            continue

        counts[canonical_team] = counts.get(canonical_team, 0.0) + float(len(valid_laps))

    return counts


def _aggregate_metric_samples(
    samples: list[tuple[float, float]],
    session_aggregation: str,
) -> float | None:
    """Aggregate session metric samples with explicit strategy."""
    if not samples:
        return None

    values = np.array([float(value) for value, _ in samples], dtype=float)
    if values.size == 0:
        return None

    if session_aggregation == "median":
        return float(np.median(values))

    if session_aggregation == "laps_weighted":
        weights = np.array([max(0.0, float(weight)) for _, weight in samples], dtype=float)
        total_weight = float(np.sum(weights))
        if total_weight > 0:
            return float(np.average(values, weights=weights))
        return float(np.mean(values))

    # Default and backward-compatible behavior.
    return float(np.mean(values))


def update_from_testing_sessions(
    year: int,
    events: list[str],
    data_dir: str = "data/processed",
    sessions: list[str] | None = None,
    characteristics_year: int | None = None,
    testing_backend: str | None = "auto",
    cache_dir: str = str(_DEFAULT_TESTING_CACHE_DIR),
    force_renew_cache: bool = False,
    new_weight: float = 0.7,
    directionality_scale: float = 0.10,
    session_aggregation: str = "mean",
    run_profile: str = "balanced",
    dry_run: bool = False,
) -> dict:
    """
    Update car directionality from testing or practice sessions.

    This function only updates testing-related fields:
    - teams[*].directionality
    - teams[*].testing_characteristics
    """
    if not events:
        raise ValueError("At least one event name is required")

    target_year = characteristics_year or year
    characteristics_file = (
        Path(data_dir) / "car_characteristics" / f"{target_year}_car_characteristics.json"
    )
    if not characteristics_file.exists():
        raise FileNotFoundError(f"Characteristics file not found: {characteristics_file}")

    with open(characteristics_file) as f:
        characteristics = json.load(f)

    if "teams" not in characteristics:
        raise ValueError(
            f"Invalid characteristics format in {characteristics_file}: missing 'teams'"
        )

    if session_aggregation not in _SESSION_AGGREGATION_MODES:
        raise ValueError(
            f"Invalid session aggregation mode. Use one of: {', '.join(_SESSION_AGGREGATION_MODES)}"
        )
    if run_profile not in _RUN_PROFILE_MODES:
        raise ValueError(f"Invalid run profile mode. Use one of: {', '.join(_RUN_PROFILE_MODES)}")

    session_candidates = sessions or DEFAULT_SESSION_CANDIDATES
    known_teams = set(characteristics["teams"].keys())
    testing_backends = _resolve_testing_backends(testing_backend)

    cache_path = _resolve_testing_cache_dir(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path), force_renew=force_renew_cache)

    metric_samples: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    profile_metric_samples: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {
        profile: defaultdict(lambda: defaultdict(list)) for profile in _PROFILES_FOR_STORAGE
    }
    team_sessions_used: dict[str, set[str]] = defaultdict(set)
    team_profile_sessions_used: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    loaded_sessions = []
    discovered_sessions = []
    load_errors: list[str] = []
    extraction_diagnostics: list[str] = []
    compound_metrics_by_session: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for event_name in events:
        event_sessions = _load_sessions_for_event(
            year=year,
            event_name=event_name,
            session_candidates=session_candidates,
            testing_backends=testing_backends,
            error_messages=load_errors,
        )
        for session_name, session in event_sessions:
            session_id = f"{event_name}::{session_name}"
            discovered_sessions.append(session_id)

            profiles_to_collect = []
            for profile in (*_PROFILES_FOR_STORAGE, run_profile):
                if profile not in profiles_to_collect:
                    profiles_to_collect.append(profile)

            profile_results: dict[
                str, tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]
            ] = {}
            for profile in profiles_to_collect:
                perf_by_profile, tire_by_profile = _collect_session_metrics(
                    session=session,
                    session_key=session_name,
                    known_teams=known_teams,
                    run_profile=profile,
                    diagnostics=(extraction_diagnostics if profile == run_profile else None),
                )
                profile_results[profile] = (perf_by_profile, tire_by_profile)

                if profile in _PROFILES_FOR_STORAGE and (perf_by_profile or tire_by_profile):
                    profile_weights = _count_team_selected_laps(
                        session=session,
                        known_teams=known_teams,
                        run_profile=profile,
                    )
                    for team, metrics in perf_by_profile.items():
                        for metric_name, value in metrics.items():
                            profile_metric_samples[profile][team][metric_name].append(
                                (float(value), float(profile_weights.get(team, 1.0)))
                            )
                            team_profile_sessions_used[team][profile].add(session_id)
                    for team, metrics in tire_by_profile.items():
                        for metric_name, value in metrics.items():
                            profile_metric_samples[profile][team][metric_name].append(
                                (float(value), float(profile_weights.get(team, 1.0)))
                            )
                            team_profile_sessions_used[team][profile].add(session_id)

            normalized_perf, normalized_tire = profile_results.get(run_profile, ({}, {}))

            if not normalized_perf and not normalized_tire:
                continue

            loaded_sessions.append(session_id)
            team_lap_weights = _count_team_selected_laps(
                session=session,
                known_teams=known_teams,
                run_profile=run_profile,
            )

            for team, metrics in normalized_perf.items():
                for metric_name, value in metrics.items():
                    metric_samples[team][metric_name].append(
                        (float(value), float(team_lap_weights.get(team, 1.0)))
                    )
                    team_sessions_used[team].add(session_id)

            for team, metrics in normalized_tire.items():
                for metric_name, value in metrics.items():
                    metric_samples[team][metric_name].append(
                        (float(value), float(team_lap_weights.get(team, 1.0)))
                    )
                    team_sessions_used[team].add(session_id)

            # Extract compound-specific metrics
            try:
                laps = session.laps
                if laps is not None and not laps.empty and "Team" in laps.columns:
                    session_compound_metrics = {}
                    raw_teams = laps["Team"].dropna().unique()

                    for raw_team in raw_teams:
                        canonical_team = _canonicalize_team_name(str(raw_team), known_teams)
                        if not canonical_team:
                            continue

                        team_laps = laps[laps["Team"] == raw_team]
                        compound_data = extract_compound_metrics(
                            team_laps, canonical_team, event_name
                        )

                        if compound_data:
                            session_compound_metrics[canonical_team] = compound_data

                    # Normalize compound metrics across teams for this session (track-aware)
                    if session_compound_metrics:
                        normalized_compound_metrics = normalize_compound_metrics_across_teams(
                            session_compound_metrics, event_name
                        )
                        compound_metrics_by_session[session_id] = normalized_compound_metrics
                        logger.debug(
                            f"  Extracted compound metrics for {len(normalized_compound_metrics)} teams"
                        )
            except Exception as exc:
                logger.warning(f"  Failed to extract compound metrics from {session_id}: {exc}")

    if not loaded_sessions:
        if discovered_sessions:
            unique_discovered = []
            seen_discovered = set()
            for session_id in discovered_sessions:
                if session_id not in seen_discovered:
                    seen_discovered.add(session_id)
                    unique_discovered.append(session_id)
                if len(unique_discovered) >= 5:
                    break

            raise ValueError(
                "Sessions were found, but no usable team telemetry could be extracted yet. "
                "This usually means the session has too little completed running. "
                f"Detected sessions: {unique_discovered}. "
                f"Extraction diagnostics: {extraction_diagnostics[:3]}"
            )

        unique_errors = []
        seen = set()
        for msg in load_errors:
            if msg not in seen:
                seen.add(msg)
                unique_errors.append(msg)
            if len(unique_errors) >= 3:
                break

        details = f" First errors: {unique_errors}" if unique_errors else ""
        all_data_not_loaded = bool(unique_errors) and all(
            "DataNotLoadedError" in error for error in unique_errors
        )
        cache_hint = ""
        if all_data_not_loaded:
            cache_hint = (
                " Likely cache issue; retry with a fresh cache directory "
                "(e.g. --cache-dir _tmp_fastf1_cache_testing_2026 "
                "--force-renew-cache; it will be created under data/raw)."
            )
        raise ValueError(
            "No loadable sessions found. Verify event names and data availability in FastF1 cache/API."
            + cache_hint
            + details
        )

    now_iso = datetime.now().isoformat()
    updated_teams = []

    for team_name, samples in metric_samples.items():
        if team_name not in characteristics["teams"]:
            continue

        averaged_metrics: dict[str, float] = {}
        for metric_name, values in samples.items():
            aggregated = _aggregate_metric_samples(values, session_aggregation=session_aggregation)
            if aggregated is not None:
                averaged_metrics[metric_name] = aggregated

        if not averaged_metrics:
            continue

        extracted_directionality = _build_directionality_from_metrics(
            averaged_metrics,
            directionality_scale=directionality_scale,
        )

        team_data = characteristics["teams"][team_name]
        current_directionality = team_data.get("directionality")
        if not isinstance(current_directionality, dict):
            current_directionality = {}
        blended_directionality = _blend_directionality(
            old_directionality=current_directionality,
            new_directionality=extracted_directionality,
            new_weight=new_weight,
        )

        team_data["directionality"] = blended_directionality
        team_data["last_updated"] = now_iso

        testing_characteristics = team_data.get("testing_characteristics")
        if not isinstance(testing_characteristics, dict):
            testing_characteristics = {}
        for metric_name in (
            "slow_corner_performance",
            "medium_corner_performance",
            "fast_corner_performance",
            "braking_performance",
            "top_speed",
            "overall_pace",
            "consistency",
            "tire_deg_slope",
            "tire_deg_performance",
        ):
            if metric_name in averaged_metrics:
                testing_characteristics[metric_name] = round(
                    float(averaged_metrics[metric_name]), 4
                )

        testing_characteristics["last_updated"] = now_iso
        testing_characteristics["sessions_used"] = len(team_sessions_used.get(team_name, set()))
        testing_characteristics["session_aggregation"] = session_aggregation
        testing_characteristics["run_profile"] = run_profile
        team_data["testing_characteristics"] = testing_characteristics

        existing_profiles = team_data.get("testing_characteristics_profiles")
        if not isinstance(existing_profiles, dict):
            existing_profiles = {}

        for profile in _PROFILES_FOR_STORAGE:
            profile_samples = profile_metric_samples.get(profile, {}).get(team_name, {})
            profile_metrics: dict[str, float] = {}
            for metric_name, values in profile_samples.items():
                aggregated = _aggregate_metric_samples(
                    values, session_aggregation=session_aggregation
                )
                if aggregated is not None:
                    profile_metrics[metric_name] = aggregated

            if not profile_metrics:
                continue

            profile_data = existing_profiles.get(profile)
            if not isinstance(profile_data, dict):
                profile_data = {}

            for metric_name in (
                "slow_corner_performance",
                "medium_corner_performance",
                "fast_corner_performance",
                "braking_performance",
                "top_speed",
                "overall_pace",
                "consistency",
                "tire_deg_slope",
                "tire_deg_performance",
            ):
                if metric_name in profile_metrics:
                    profile_data[metric_name] = round(float(profile_metrics[metric_name]), 4)

            profile_data["last_updated"] = now_iso
            profile_data["sessions_used"] = len(
                team_profile_sessions_used[team_name].get(profile, set())
            )
            profile_data["session_aggregation"] = session_aggregation
            profile_data["run_profile"] = profile
            existing_profiles[profile] = profile_data

        team_data["testing_characteristics_profiles"] = existing_profiles

        # Update compound-specific characteristics
        existing_compound_chars = team_data.get("compound_characteristics")
        if not isinstance(existing_compound_chars, dict):
            existing_compound_chars = {}

        # Aggregate compound metrics from all sessions
        for _session_id, session_compounds in compound_metrics_by_session.items():
            if team_name in session_compounds:
                new_compound_data = session_compounds[team_name]
                # Blend with existing data
                existing_compound_chars = aggregate_compound_samples(
                    existing_compound_chars,
                    new_compound_data,
                    blend_weight=new_weight,
                    race_name=event_name,
                )

        # Update last_updated timestamp for each compound
        if existing_compound_chars:
            for compound_data in existing_compound_chars.values():
                compound_data["last_updated"] = now_iso

        team_data["compound_characteristics"] = existing_compound_chars
        updated_teams.append(team_name)

    if not updated_teams:
        raise ValueError(
            "Sessions loaded but no teams were matched to characteristics file team names."
        )

    characteristics["directionality_source"] = "SESSION_EXTRACTION"
    characteristics["directionality_last_updated"] = now_iso
    characteristics["directionality_meta"] = {
        "year": year,
        "characteristics_year": target_year,
        "events": events,
        "sessions_loaded": loaded_sessions,
        "testing_backend": testing_backend or "auto",
        "cache_dir": str(cache_path),
        "force_renew_cache": force_renew_cache,
        "new_weight": new_weight,
        "directionality_scale": directionality_scale,
        "session_aggregation": session_aggregation,
        "run_profile": run_profile,
        "profiles_captured": list(_PROFILES_FOR_STORAGE),
    }

    if not dry_run:
        atomic_json_write(characteristics_file, characteristics, create_backup=True)

    return {
        "year": year,
        "characteristics_year": target_year,
        "events": events,
        "loaded_sessions": loaded_sessions,
        "updated_teams": sorted(updated_teams),
        "characteristics_file": str(characteristics_file),
        "testing_backend": testing_backend or "auto",
        "cache_dir": str(cache_path),
        "force_renew_cache": force_renew_cache,
        "session_aggregation": session_aggregation,
        "run_profile": run_profile,
        "profiles_captured": list(_PROFILES_FOR_STORAGE),
        "dry_run": dry_run,
    }
