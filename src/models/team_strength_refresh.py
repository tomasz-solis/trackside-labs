"""Keep the team-strength seconds mapping fitted on every completed session.

The mapping converts the team-strength scalar into seconds (one slope for race, one
for qualifying). It used to be frozen by hand (``latest.json``, 11 rounds). Now each
refresh:

1. extracts matched-lap driver observations for every completed Q and R the store
   lacks (same construct as ``scripts/build_matched_lap_observations.py``),
2. attaches driver mus (the per-session team-strength proxy is stored with each row),
3. refits both mappings on this season's rows (same policy as
   ``scripts/freeze_team_strength_seconds_mapping.py``),
4. saves the mapping unless a guardrail rejects it.

Store artifacts: ``team_strength_observations`` (this season's base rows) and
``team_strength_seconds_mapping`` (key ``latest``; in file mode the committed
``latest.json``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.models.team_strength_mapping import (
    CALIBRATION_DRIVER_COLUMNS,
    attach_driver_rating_mus,
    build_construct_aligned_driver_observations,
    evaluate_within_season_folds,
    fit_linear_team_strength_mapping,
)
from src.persistence.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

OBSERVATIONS_TYPE = "team_strength_observations"
MAPPING_TYPE = "team_strength_seconds_mapping"
MAPPING_KEY = "latest"
POLICY = "same_session_construct"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAPPING_DIR = _PROJECT_ROOT / "data" / "processed" / "team_strength_seconds_mapping"
_PRIOR_PATH = _PROJECT_ROOT / "data" / "processed" / "teammate_network_prior" / "latest.json"
_SESSIONS = (("Q", "qualifying"), ("R", "race"))
# Rows from fewer rounds than this make the slope a guess; keep the old mapping.
MIN_ROUNDS = 3
# A refit whose slope moves more than this factor is treated as broken input.
MAX_SLOPE_RATIO = 2.0
BASE_COLUMNS = (
    *CALIBRATION_DRIVER_COLUMNS,
    "driver_median_s",
    "n_construct_laps",
    "field_median_s",
    "n_field_drivers",
    "n_field_teams",
    "observed_driver_to_field_s",
    # The policy proxy is per session and must be computed on the full field before
    # drivers without a prior are dropped, so it is stored, never recomputed.
    "team_strength_same_session",
)
_LOAD_ERRORS = (
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def observations_key(year: int) -> str:
    """Return the store key for one season's base observation rows."""
    return f"{int(year)}::{OBSERVATIONS_TYPE}"


def driver_mu_by_kind(prior_path: Path = _PRIOR_PATH) -> dict[str, dict[str, float]]:
    """Return race and qualifying driver means from the teammate-network prior."""
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    return {
        kind: {code: float(row["mu_s"]) for code, row in prior[network]["drivers"].items()}
        for kind, network in (("race", "race_network"), ("qualifying", "quali_network"))
    }


def seed_observations(year: int) -> pd.DataFrame:
    """Return this season's base rows from the committed calibration CSV."""
    path = _MAPPING_DIR / "calibration_observations.csv"
    if not path.exists():
        return pd.DataFrame(columns=list(BASE_COLUMNS))
    frame = pd.read_csv(path)
    return frame.loc[frame["year"].eq(int(year)), list(BASE_COLUMNS)].reset_index(drop=True)


def load_observations(year: int, store: ArtifactStore) -> pd.DataFrame:
    """Return this season's base rows from the store, seeded from the committed CSV."""
    try:
        payload = store.load_artifact(OBSERVATIONS_TYPE, observations_key(year))
    except Exception as exc:  # store outage falls back to the seed
        logger.warning("Could not load %s: %s", OBSERVATIONS_TYPE, exc)
        payload = None
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if isinstance(rows, list):
        return pd.DataFrame(rows, columns=list(BASE_COLUMNS))
    return seed_observations(year)


def extract_session_rows(
    year: int, race_name: str, session_code: str, session_kind: str
) -> pd.DataFrame | None:
    """Return base observation rows for one session, or None when laps are not ready."""
    import fastf1

    from src.extractors.matched_laps import (
        MatchedLapConfig,
        SessionKind,
        extract_matched_teammate_laps,
    )

    try:
        session = fastf1.get_session(year, race_name, session_code)
        session.load(laps=True, weather=True, telemetry=False, messages=False)
        raw = extract_matched_teammate_laps(
            session,
            session_kind=cast(SessionKind, session_kind),
            weather_mode="mixed",
            config=MatchedLapConfig(),
        )
    except _LOAD_ERRORS as exc:
        logger.info("Skipping %s %s: %s", race_name, session_code, exc)
        return None
    rows = build_construct_aligned_driver_observations(raw)
    if rows.empty:
        return None
    return rows.loc[:, list(BASE_COLUMNS)]


def build_fit_rows(base_rows: pd.DataFrame, mus: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Attach driver mus and the team target, then keep usable rows."""
    rows = attach_driver_rating_mus(base_rows.loc[:, list(BASE_COLUMNS)], driver_mu_by_kind=mus)
    return rows.dropna(subset=["driver_rating_mu_s", "team_target_s"]).reset_index(drop=True)


def fit_mapping_artifact(fit_rows: pd.DataFrame, year: int) -> dict[str, Any]:
    """Fit race and qualifying mappings on this season and return the artifact payload."""
    mappings = {
        kind: fit_linear_team_strength_mapping(
            fit_rows, session_kind=kind, policy=POLICY, training_years=(int(year),)
        )
        for kind in ("race", "qualifying")
    }
    return {
        "artifact_type": MAPPING_TYPE,
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "last_updated": datetime.now(UTC).isoformat(),
        "policy": POLICY,
        "training_years": [int(year)],
        "rounds": sorted(fit_rows.loc[fit_rows["year"].eq(int(year)), "race_name"].unique()),
        "sign_convention": "positive_seconds_means_faster_than_field",
        "mappings": {
            kind: {
                "session_kind": mapping.session_kind,
                "policy": mapping.policy,
                "intercept_s": mapping.intercept_s,
                "slope_s_per_unit": mapping.slope_s_per_unit,
                "training_years": list(mapping.training_years),
            }
            for kind, mapping in mappings.items()
        },
        "validation": {
            "primary_folds": "within_season_leave_one_round_out",
            "within_season_folds": evaluate_within_season_folds(
                fit_rows, policy=POLICY, year=int(year)
            ).get("folds", []),
        },
    }


def guardrail_failure(new: Mapping[str, Any], old: Mapping[str, Any] | None) -> str | None:
    """Return why a refit must not ship, or None when it may."""
    rounds = len(new.get("rounds", []))
    if rounds < MIN_ROUNDS:
        return f"only {rounds} rounds"
    old_mappings = (old or {}).get("mappings", {})
    for kind, mapping in new["mappings"].items():
        slope = float(mapping["slope_s_per_unit"])
        if slope <= 0.0:
            return f"{kind} slope {slope:.3f} is not positive"
        previous = old_mappings.get(kind, {}).get("slope_s_per_unit")
        if previous:
            ratio = slope / float(previous)
            if not 1.0 / MAX_SLOPE_RATIO <= ratio <= MAX_SLOPE_RATIO:
                return f"{kind} slope moved {ratio:.2f}x ({float(previous):.3f} to {slope:.3f})"
    return None


def refresh_team_strength_mapping(
    year: int,
    completed_races: list[str],
    store: ArtifactStore | None = None,
    mus: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Add missing Q and R observations, refit, and save; return what happened.

    Idempotent: a session already stored is never re-extracted, and a session whose
    laps are not published yet is retried on the next refresh.
    """
    store = store or ArtifactStore(data_root="data")
    base = load_observations(year, store)
    have = set(zip(base["race_name"], base["session_kind"], strict=False))

    new_frames: list[pd.DataFrame] = []
    added: list[str] = []
    for race_name in completed_races:
        for session_code, session_kind in _SESSIONS:
            if (race_name, session_kind) in have:
                continue
            rows = extract_session_rows(year, race_name, session_code, session_kind)
            if rows is not None:
                new_frames.append(rows)
                added.append(f"{race_name} {session_code}")

    result: dict[str, Any] = {"added": added, "saved_mapping": False, "rejected": None}
    if not added:
        return result

    base = pd.concat([base, *new_frames], ignore_index=True)
    store.save_artifact(
        OBSERVATIONS_TYPE,
        observations_key(year),
        {
            "year": int(year),
            "last_updated": datetime.now(UTC).isoformat(),
            "rows": json.loads(base.to_json(orient="records")),
        },
    )

    artifact = fit_mapping_artifact(build_fit_rows(base, mus or driver_mu_by_kind()), year)
    old = store.load_artifact(MAPPING_TYPE, MAPPING_KEY)
    reason = guardrail_failure(artifact, old if isinstance(old, Mapping) else None)
    if reason:
        logger.warning("Team strength mapping refit rejected (%s); keeping the old one", reason)
        result["rejected"] = reason
        return result

    store.save_artifact(MAPPING_TYPE, MAPPING_KEY, artifact)
    result["saved_mapping"] = True
    logger.info("Team strength mapping refit on %d rounds after %s", len(artifact["rounds"]), added)
    return result
