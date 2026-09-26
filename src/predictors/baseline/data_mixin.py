"""Data-loading and team-strength mixin for Baseline2026Predictor."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.data.compound_performance import (
    get_compound_performance_modifier,
    should_use_compound_adjustments,
)
from src.persistence.artifact_store import ArtifactStore
from src.predictors.baseline.compound_updates import update_compound_characteristics_from_session
from src.predictors.baseline.data_support import (
    canonicalize_team_payload_keys,
    coerce_non_negative_int,
    driver_characteristics_fallback_paths,
    extract_target_actual_rows,
    infer_payload_year_from_path,
    nearest_season_payload_path,
    sanitize_performance_observations,
    score_teams_from_actual_rows,
)
from src.predictors.baseline.team_strength import (
    calculate_track_suitability as calculate_track_suitability_helper,
)
from src.predictors.baseline.team_strength import (
    get_blended_team_strength as get_blended_team_strength_helper,
)
from src.predictors.baseline.team_strength import (
    get_compound_adjusted_team_strength as get_compound_adjusted_team_strength_helper,
)
from src.predictors.baseline.team_strength import (
    resolve_team_data as resolve_team_data_helper,
)
from src.predictors.baseline.team_strength import (
    select_race_compound,
)
from src.predictors.baseline.testing_profiles import (
    compute_testing_profile_modifier,
    get_testing_characteristics_for_profile,
)
from src.systems.weight_schedule import (
    SCHEDULES,
    calculate_blended_performance,
    get_recommended_schedule,
)
from src.utils import config_loader
from src.utils.prediction_context import get_active_prediction_context
from src.utils.schema_validation import (
    validate_driver_characteristics,
    validate_team_characteristics,
    validate_track_characteristics,
)

logger = logging.getLogger("src.predictors.baseline_2026")
_TRACK_LAYOUT_FIELDS = (
    "straights_pct",
    "slow_corners_pct",
    "medium_corners_pct",
    "high_corners_pct",
)


class BaselineDataMixin:
    """Shared data and team-strength methods for Baseline2026Predictor."""

    if TYPE_CHECKING:
        artifact_store: ArtifactStore | None
        car_characteristics_snapshot: dict[str, Any]
        config: Any
        data_dir: Path
        drivers: dict[str, dict[str, Any]]
        season_year: int
        teams: dict[str, dict[str, Any]]
        tracks: dict[str, dict[str, Any]]
        year: int

    def __init__(self) -> None:
        """Initialize data mixin with compound extraction cache."""
        if not hasattr(self, "_compound_cache"):
            self._compound_cache: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
        if not hasattr(self, "_replayed_actual_race_scores"):
            self._replayed_actual_race_scores: dict[int, list[dict[str, object]]] = {}

    def _resolve_predictions_data_root(self) -> Path:
        """Return the data root used when reading saved prediction artifacts."""
        store = getattr(self, "artifact_store", None)
        store_root = getattr(store, "data_root", None)
        if isinstance(store_root, Path):
            return store_root
        if isinstance(store_root, str) and store_root:
            return Path(store_root)
        return self.data_dir.parent if self.data_dir.name == "processed" else self.data_dir

    def _resolve_prediction_target_year(self) -> int:
        """Return the season year that the active prediction should reason about."""
        active_context = get_active_prediction_context()
        contextual_year = getattr(active_context, "season_year", None)
        if isinstance(contextual_year, int) and contextual_year > 0:
            return int(contextual_year)
        return int(getattr(self, "season_year", getattr(self, "year", 2026)))

    def _get_replayed_actual_race_scores(self, target_year: int) -> list[dict[str, object]]:
        """Return actual-result snapshots recorded during one replayed season run."""
        records = getattr(self, "_replayed_actual_race_scores", {}).get(target_year, [])
        return list(records) if isinstance(records, list) else []

    def _get_race_order_map(self, target_year: int) -> dict[str, int]:
        """Return season race order for contextual current-form cutoffs."""
        cache = getattr(self, "_race_order_map_cache", {})
        if target_year in cache:
            return cache[target_year]

        try:
            from src.utils.weekend import get_schedule_rows

            schedule_rows = get_schedule_rows(target_year)
        except Exception as exc:
            logger.debug("Could not load schedule rows for %s: %s", target_year, exc)
            schedule_rows = ()

        race_order_map: dict[str, int] = {}
        race_index = 0
        for raw_race_name, raw_event_format in schedule_rows:
            race_name = str(raw_race_name).strip()
            event_format = str(raw_event_format).strip().lower()
            if not race_name:
                continue
            if "testing" in race_name.lower() or "testing" in event_format:
                continue
            race_index += 1
            race_order_map.setdefault(race_name, race_index)

        cache[target_year] = race_order_map
        self._race_order_map_cache = cache
        return race_order_map

    def _prediction_race_sort_key(
        self,
        prediction: dict[str, object],
        *,
        race_order_map: dict[str, int],
    ) -> tuple[int, str, str]:
        """Build a stable race ordering key for saved predictions."""
        metadata = prediction.get("metadata", {})
        if not isinstance(metadata, dict):
            return (10_000, "", "")

        race_name = str(metadata.get("race_name", "")).strip()
        predicted_at = str(metadata.get("predicted_at", "")).strip()
        return (race_order_map.get(race_name, 10_000), predicted_at, race_name)

    def _blend_saved_actual_team_scores(
        self,
        *,
        qualifying_rows: list[dict[str, object]],
        race_rows: list[dict[str, object]],
        known_teams: set[str],
    ) -> dict[str, float]:
        """Blend saved qualifying and race actuals into one team-form snapshot."""
        qualifying_scores = (
            score_teams_from_actual_rows(qualifying_rows, known_teams=known_teams)
            if qualifying_rows
            else {}
        )
        race_scores = (
            score_teams_from_actual_rows(race_rows, known_teams=known_teams) if race_rows else {}
        )
        if race_scores and not qualifying_scores:
            return race_scores
        if qualifying_scores and not race_scores:
            return qualifying_scores
        if not qualifying_scores and not race_scores:
            return {}

        cfg = getattr(self, "config", config_loader)
        race_weight = float(
            cfg.get("baseline_predictor.current_season_form.saved_actual_race_weight", 0.70)
        )
        race_weight = float(np.clip(race_weight, 0.0, 1.0))
        qualifying_weight = 1.0 - race_weight

        blended_scores: dict[str, float] = {}
        for team_name in set(qualifying_scores) | set(race_scores):
            qualifying_score = qualifying_scores.get(team_name)
            race_score = race_scores.get(team_name)
            if qualifying_score is None and race_score is None:
                continue
            if qualifying_score is None:
                if race_score is None:
                    continue
                blended_scores[team_name] = race_score
                continue
            if race_score is None:
                blended_scores[team_name] = qualifying_score
                continue
            blended_scores[team_name] = float(
                np.clip(
                    (qualifying_score * qualifying_weight) + (race_score * race_weight),
                    0.0,
                    1.0,
                )
            )

        return blended_scores

    def _load_saved_actual_race_scores(
        self,
        target_year: int,
    ) -> list[dict[str, object]]:
        """Load one team-score snapshot per completed race from saved actuals."""
        cache = getattr(self, "_saved_actual_race_scores_cache", {})
        if target_year in cache:
            cached_records = cache[target_year]
            return cached_records if isinstance(cached_records, list) else []

        cfg = getattr(self, "config", config_loader)
        if not bool(
            cfg.get("baseline_predictor.current_season_form.infer_from_saved_actuals", True)
        ):
            return []

        try:
            from src.utils.prediction_logger import PredictionLogger
        except Exception as exc:
            logger.debug("Could not import PredictionLogger for current-form inference: %s", exc)
            return []

        predictions_dir = self._resolve_predictions_data_root() / "predictions"
        try:
            prediction_logger = PredictionLogger(predictions_dir=str(predictions_dir))
            predictions = prediction_logger.get_all_predictions(target_year)
        except Exception as exc:
            logger.debug("Could not load saved predictions for %s: %s", target_year, exc)
            return []

        if not predictions:
            return []

        known_teams = set(self.teams.keys())
        race_order_map = self._get_race_order_map(target_year)
        seen_races: set[str] = set()
        race_records: list[dict[str, object]] = []

        for prediction in sorted(
            predictions,
            key=lambda payload: self._prediction_race_sort_key(
                payload,
                race_order_map=race_order_map,
            ),
        ):
            metadata = prediction.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            race_name = str(metadata.get("race_name", "")).strip()
            if not race_name or race_name in seen_races:
                continue

            qualifying_rows = extract_target_actual_rows(
                prediction,
                target_key="main_qualifying",
            )
            race_rows = extract_target_actual_rows(
                prediction,
                target_key="grand_prix_race",
            )
            team_scores = self._blend_saved_actual_team_scores(
                qualifying_rows=qualifying_rows,
                race_rows=race_rows,
                known_teams=known_teams,
            )
            if not team_scores:
                continue

            seen_races.add(race_name)
            race_records.append(
                {
                    "race_name": race_name,
                    "team_scores": team_scores,
                }
            )

        cache[target_year] = race_records
        self._saved_actual_race_scores_cache = cache
        return race_records

    def _resolve_saved_actual_races_completed(
        self,
        *,
        target_year: int,
        data_freshness: str,
        races_completed: int,
    ) -> int:
        """Resolve available completed-race count from live state or saved actuals."""
        has_live_current_form = any(
            sanitize_performance_observations(team_data.get("current_season_performance"))
            for team_data in self.teams.values()
        )
        if data_freshness == "LIVE_UPDATED" or has_live_current_form:
            return races_completed

        race_records = self._load_saved_actual_race_scores(target_year)
        inferred_races = len(race_records)
        if inferred_races <= 0:
            return races_completed

        teams_with_scores = set()
        for record in race_records:
            team_scores = record.get("team_scores", {})
            if not isinstance(team_scores, dict):
                continue
            for team_name in team_scores:
                if team_name in self.teams:
                    teams_with_scores.add(str(team_name))

        logger.info(
            "Recovered current-season team form from saved actuals: %s race(s), %s team(s)",
            inferred_races,
            len(teams_with_scores),
        )
        return max(races_completed, inferred_races)

    def _count_known_prior_races(self, target_year: int, race_name: str | None) -> int | None:
        """Return number of scheduled races before the target race.

        ``None`` means no target race was given, so no cap applies. A named target
        missing from the schedule (an alias, an unlisted venue, or a schedule that
        failed to load) cannot be ordered, so it fails closed at zero prior races
        rather than admitting every other race, later ones included.
        """
        normalized_race_name = str(race_name or "").strip()
        if not normalized_race_name:
            return None

        race_order_map = self._get_race_order_map(target_year)
        target_order = race_order_map.get(normalized_race_name)
        if target_order is None:
            logger.warning(
                "Race %r is not in the %s schedule; using no current-season form for it",
                normalized_race_name,
                target_year,
            )
            return 0
        return max(target_order - 1, 0)

    def _race_precedes_target(
        self,
        *,
        target_year: int,
        candidate_race_name: str,
        target_race_name: str | None,
    ) -> bool:
        """Return True when one race is known to be earlier than the target race."""
        normalized_target = str(target_race_name or "").strip()
        normalized_candidate = str(candidate_race_name).strip()
        if not normalized_target:
            return True
        if normalized_candidate == normalized_target:
            return False

        race_order_map = self._get_race_order_map(target_year)
        target_order = race_order_map.get(normalized_target)
        candidate_order = race_order_map.get(normalized_candidate)
        if target_order is None or candidate_order is None:
            return False
        return candidate_order < target_order

    def _get_saved_actual_observations(
        self,
        *,
        team_name: str,
        target_year: int,
        race_name: str | None,
    ) -> list[tuple[float, int]]:
        """Return (score, race_ordinal) saved-actual observations for one team before the target race.

        The race ordinal comes from `_get_race_order_map`, so a team that missed a
        race is not weighted by `_get_current_season_score` as if its later
        observations were one race earlier than they are. A race missing from the
        order map (or one that maps to an ordinal not after the previous
        observation - e.g. an accented-key mismatch like the recorded Sao Paulo
        defect) falls back to `previous_ordinal + 1`, so ordinals stay strictly
        increasing and the recency weighting can never treat an older race as more
        recent than a newer one.
        """
        race_order_map = self._get_race_order_map(target_year)
        observations: list[tuple[float, int]] = []
        for race_record in self._load_saved_actual_race_scores(target_year):
            candidate_race_name = str(race_record.get("race_name", "")).strip()
            if not self._race_precedes_target(
                target_year=target_year,
                candidate_race_name=candidate_race_name,
                target_race_name=race_name,
            ):
                continue
            team_scores = race_record.get("team_scores", {})
            if not isinstance(team_scores, dict):
                continue
            score = team_scores.get(team_name)
            if score is None:
                continue
            previous_ordinal = observations[-1][1] if observations else 0
            race_ordinal = race_order_map.get(candidate_race_name, previous_ordinal + 1)
            if race_ordinal <= previous_ordinal:
                race_ordinal = previous_ordinal + 1
            observations.append((float(np.clip(float(score), 0.0, 1.0)), race_ordinal))
        return observations

    def _get_replayed_actual_observations(
        self,
        *,
        team_name: str,
        target_year: int,
        race_name: str | None,
    ) -> list[tuple[float, int]]:
        """Return (score, race_ordinal) observations recorded during the active historical replay.

        See `_get_saved_actual_observations` for the race-ordinal fallback rule.
        """
        race_order_map = self._get_race_order_map(target_year)
        observations: list[tuple[float, int]] = []
        for race_record in self._get_replayed_actual_race_scores(target_year):
            candidate_race_name = str(race_record.get("race_name", "")).strip()
            if not self._race_precedes_target(
                target_year=target_year,
                candidate_race_name=candidate_race_name,
                target_race_name=race_name,
            ):
                continue
            team_scores = race_record.get("team_scores", {})
            if not isinstance(team_scores, dict):
                continue
            score = team_scores.get(team_name)
            if score is None:
                continue
            previous_ordinal = observations[-1][1] if observations else 0
            race_ordinal = race_order_map.get(candidate_race_name, previous_ordinal + 1)
            if race_ordinal <= previous_ordinal:
                race_ordinal = previous_ordinal + 1
            observations.append((float(np.clip(float(score), 0.0, 1.0)), race_ordinal))
        return observations

    def record_completed_weekend_actuals(
        self,
        *,
        year: int,
        race_name: str,
        qualifying_actual: list[dict[str, object]],
        race_actual: list[dict[str, object]],
    ) -> dict[str, object]:
        """Record one completed weekend so later replayed races can reuse its team-form signal."""
        known_teams = set(self.teams.keys())
        team_scores = self._blend_saved_actual_team_scores(
            qualifying_rows=qualifying_actual,
            race_rows=race_actual,
            known_teams=known_teams,
        )
        if not team_scores:
            return {
                "race_name": race_name,
                "teams_updated": 0,
                "races_recorded": len(self._get_replayed_actual_race_scores(year)),
            }

        replay_store = getattr(self, "_replayed_actual_race_scores", {})
        if not isinstance(replay_store, dict):
            replay_store = {}

        normalized_race_name = str(race_name).strip()
        year_records = [
            record
            for record in self._get_replayed_actual_race_scores(year)
            if str(record.get("race_name", "")).strip() != normalized_race_name
        ]
        year_records.append(
            {
                "race_name": normalized_race_name,
                "team_scores": team_scores,
            }
        )

        race_order_map = self._get_race_order_map(year)
        year_records.sort(
            key=lambda record: (
                race_order_map.get(str(record.get("race_name", "")).strip(), 10_000),
                str(record.get("race_name", "")).strip(),
            )
        )
        replay_store[int(year)] = year_records
        self._replayed_actual_race_scores = replay_store
        return {
            "race_name": normalized_race_name,
            "teams_updated": len(team_scores),
            "races_recorded": len(year_records),
        }

    def _resolve_field_observation_source(
        self,
        target_year: int,
        race_name: str | None,
    ) -> str:
        """Resolve and memoize which observation source the whole field reads from.

        `live_observations` (telemetry/position-derived pace), `saved_actual_observations`
        (rank scores from saved predictions), and `replayed_actual_observations` (rank
        scores from an active historical replay) share one 0-1 scale but are different
        constructs. Choosing a source per team lets teams in the same race be ranked
        against each other on different scales, so the source is chosen once for the
        whole field here, by total observation coverage summed across every team in
        `self.teams` - not by any single team's list length. Ties break toward
        replayed > saved > live, the same order the previous per-team tie-break used.
        `live` is dropped from the candidate set entirely when `target_year` is not the
        loaded season, mirroring the previous branch's intent.
        """
        cache = getattr(self, "_field_observation_source_cache", {})
        cache_key = (int(target_year), str(race_name or ""))
        if cache_key in cache:
            return cache[cache_key]

        loaded_season_year = int(getattr(self, "season_year", getattr(self, "year", 2026)))
        prior_race_limit = coerce_non_negative_int(
            self._count_known_prior_races(target_year, race_name)
        )

        totals = {"live": 0, "saved": 0, "replayed": 0}
        for field_team_name, field_team_data in self.teams.items():
            live_observations = sanitize_performance_observations(
                field_team_data.get("current_season_performance")
            )
            saved_observations = self._get_saved_actual_observations(
                team_name=field_team_name, target_year=target_year, race_name=race_name
            )
            replayed_observations = self._get_replayed_actual_observations(
                team_name=field_team_name, target_year=target_year, race_name=race_name
            )
            if prior_race_limit is not None:
                live_observations = live_observations[:prior_race_limit]
                saved_observations = saved_observations[:prior_race_limit]
                replayed_observations = replayed_observations[:prior_race_limit]
            totals["live"] += len(live_observations)
            totals["saved"] += len(saved_observations)
            totals["replayed"] += len(replayed_observations)

        candidate_sources = (
            ("live", "saved", "replayed")
            if target_year == loaded_season_year
            else ("saved", "replayed")
        )
        chosen_source = candidate_sources[0]
        for source in candidate_sources[1:]:
            if totals[source] >= totals[chosen_source]:
                chosen_source = source

        logger.info(
            "Current-season observation source for %s race %r: %s (coverage=%s across %s team(s))",
            target_year,
            race_name,
            chosen_source,
            totals[chosen_source],
            len(self.teams),
        )

        cache[cache_key] = chosen_source
        self._field_observation_source_cache = cache
        return chosen_source

    def _get_current_season_observations_with_ordinals(
        self,
        *,
        team_name: str,
        team_data: dict[str, object],
        race_name: str | None,
    ) -> tuple[list[float], list[int] | None]:
        """Return one team's current-season scores alongside a per-observation race ordinal.

        The field-wide source resolved by `_resolve_field_observation_source` is the
        only source this team is read from; a team absent from that source returns no
        observations here; `_get_current_season_score` then falls back to the preseason
        baseline rather than silently reading a different source for that one team.

        Saved and replayed observations carry a race ordinal (see
        `_get_saved_actual_observations`), so `_get_current_season_score` can weight by
        actual race number instead of list position. `live_observations` are bare floats
        from the updater with no race labels, so the second element is `None` for that
        source and callers keep index-based weighting.
        """
        target_year = self._resolve_prediction_target_year()
        chosen_source = self._resolve_field_observation_source(target_year, race_name)
        prior_race_limit = coerce_non_negative_int(
            self._count_known_prior_races(target_year, race_name)
        )

        if chosen_source == "live":
            observations = sanitize_performance_observations(
                team_data.get("current_season_performance")
            )
            if prior_race_limit is not None:
                observations = observations[:prior_race_limit]
            return observations, None

        if chosen_source == "saved":
            scored_observations = self._get_saved_actual_observations(
                team_name=team_name, target_year=target_year, race_name=race_name
            )
        else:
            scored_observations = self._get_replayed_actual_observations(
                team_name=team_name, target_year=target_year, race_name=race_name
            )
        if prior_race_limit is not None:
            scored_observations = scored_observations[:prior_race_limit]

        observations = [score for score, _race_ordinal in scored_observations]
        race_ordinals = [race_ordinal for _score, race_ordinal in scored_observations]
        return observations, race_ordinals

    def _get_current_season_observations(
        self,
        *,
        team_name: str,
        team_data: dict[str, object],
        race_name: str | None,
    ) -> list[float]:
        """Return race-context-aware current-season observations for one team.

        See `_resolve_field_observation_source` for how the source (live telemetry
        pace, saved-actual rank scores, or replayed-actual rank scores) is chosen once
        for the whole field rather than per team.
        """
        observations, _race_ordinals = self._get_current_season_observations_with_ordinals(
            team_name=team_name,
            team_data=team_data,
            race_name=race_name,
        )
        return observations

    def _get_contextual_races_completed(self, race_name: str | None) -> int:
        """Return completed-race count capped to what the target race could have known."""
        target_year = self._resolve_prediction_target_year()
        loaded_season_year = int(getattr(self, "season_year", getattr(self, "year", 2026)))
        live_races_completed = (
            coerce_non_negative_int(getattr(self, "races_completed", 0)) or 0
            if target_year == loaded_season_year
            else 0
        )
        available_races = max(
            live_races_completed,
            len(self._load_saved_actual_race_scores(target_year)),
            len(self._get_replayed_actual_race_scores(target_year)),
        )
        prior_race_limit = self._count_known_prior_races(target_year, race_name)
        if prior_race_limit is None:
            return available_races
        return min(available_races, prior_race_limit)

    def _get_current_season_score(
        self,
        team_name: str,
        team_data: dict[str, object],
        *,
        fallback: float,
        race_name: str | None,
    ) -> float:
        """Return the current-season score using a recency-weighted average.

        Weights use each observation's race ordinal when the resolved source carries
        one (saved/replayed actuals), so a team that missed a race is not weighted as
        if its later observations were one race earlier. `live_observations` have no
        race labels and keep the previous index-based weighting.
        """
        observations, race_ordinals = self._get_current_season_observations_with_ordinals(
            team_name=team_name,
            team_data=team_data,
            race_name=race_name,
        )
        if not observations:
            return float(fallback)

        cfg = getattr(self, "config", config_loader)
        recency_exponent = float(
            cfg.get("baseline_predictor.current_season_form.recency_exponent", 0.3)
        )
        recency_exponent = max(0.0, recency_exponent)
        if len(observations) == 1 or recency_exponent == 0.0:
            weighted_score = float(np.mean(observations))
        else:
            recency_positions = (
                np.array(race_ordinals, dtype=float)
                if race_ordinals is not None
                else np.arange(1, len(observations) + 1, dtype=float)
            )
            weights = np.power(recency_positions, recency_exponent)
            weighted_score = float(np.average(observations, weights=weights))

        stabilization_strength = float(
            cfg.get("baseline_predictor.current_season_form.stabilization_strength", 1.5)
        )
        stabilization_strength = max(0.0, stabilization_strength)
        if stabilization_strength == 0.0:
            return weighted_score

        observation_weight = len(observations) / (len(observations) + stabilization_strength)
        stabilized_score = float(fallback) + (
            (weighted_score - float(fallback)) * observation_weight
        )
        return float(np.clip(stabilized_score, 0.0, 1.0))

    def _load_team_characteristics(
        self,
        store: ArtifactStore,
        target_year: int,
    ) -> dict:
        """Load, validate, and return the raw car-characteristics payload.

        Tries the artifact store first, falls back to the file on disk.
        Raises ValueError if schema validation fails.
        """
        data = store.load_artifact(
            artifact_type="car_characteristics",
            artifact_key=f"{target_year}::car_characteristics",
        )
        source_year = target_year
        if not data:
            logger.warning("Could not load car characteristics from DB, falling back to file")
            fallback = nearest_season_payload_path(
                self.data_dir / "car_characteristics",
                suffix="car_characteristics",
                target_year=target_year,
            )
            if fallback is None:
                raise FileNotFoundError(
                    f"Could not locate car characteristics fallback for season {target_year} "
                    f"under {self.data_dir}"
                )
            source_year, car_file = fallback
            with open(car_file) as f:
                data = json.load(f)
            if source_year != target_year:
                logger.warning(
                    "Using %s car characteristics as preseason proxy for season %s.",
                    source_year,
                    target_year,
                )

        try:
            validate_team_characteristics(data, expected_year=source_year)
        except ValueError as e:
            logger.error("Failed to load team characteristics: %s", e)
            raise
        return self._build_cross_season_team_proxy_payload(
            data,
            source_year=source_year,
            target_year=target_year,
        )

    def _apply_team_payload(self, data: dict, target_year: int) -> int:
        """Set self.teams and self.car_characteristics_snapshot from validated payload.

        Returns the resolved races_completed count after contextual fallback.
        """
        raw_teams = data.get("teams", {})
        if not isinstance(raw_teams, dict):
            raise ValueError("Team characteristics payload is missing a valid `teams` mapping")
        self.teams = canonicalize_team_payload_keys(raw_teams)

        checkpoint_snapshot = data.get("checkpoint_snapshot", {})
        self.car_characteristics_snapshot = (
            checkpoint_snapshot if isinstance(checkpoint_snapshot, dict) else {}
        )
        directionality_meta = data.get("directionality_meta", {})
        self.car_characteristics_meta = (
            deepcopy(directionality_meta) if isinstance(directionality_meta, dict) else {}
        )

        data_freshness = data.get("data_freshness", "UNKNOWN")
        races_completed_raw = data.get("races_completed", 0)
        if data_freshness == "BASELINE_PRESEASON":
            logger.warning(
                "Using pre-season baseline data; team performance remains uncertain until races complete."
            )
        elif data_freshness == "LIVE_UPDATED":
            logger.info("Using live-updated data from %s race(s)", races_completed_raw)
        else:
            logger.warning(
                "Data freshness unknown (%s); predictions may be outdated", data_freshness
            )

        try:
            races_completed_value = int(races_completed_raw)
        except (TypeError, ValueError):
            races_completed_value = 0

        races_completed = self._resolve_saved_actual_races_completed(
            target_year=target_year,
            data_freshness=str(data_freshness),
            races_completed=max(0, races_completed_value),
        )
        if data_freshness == "BASELINE_PRESEASON" and races_completed > races_completed_value:
            logger.info(
                "Pre-season payload will use contextual saved-actual fallback from %s completed race(s).",
                races_completed,
            )
        return races_completed

    def _uses_testing_model_team_seed(self) -> bool:
        """Return whether the active team payload came from the testing seed model."""
        meta = getattr(self, "car_characteristics_meta", {})
        if not isinstance(meta, dict):
            return False
        return str(meta.get("seed_mode", "")).strip() == "testing_model"

    def _load_driver_characteristics(
        self,
        store: ArtifactStore,
        target_year: int,
    ) -> dict:
        """Load, clean, validate, and return the driver-characteristics payload.

        Tries the artifact store first, then walks file fallback paths.
        Strips legacy Bayesian fields from older artifacts before validation.
        Raises FileNotFoundError when no file fallback exists.
        Raises ValueError if schema validation fails.
        """
        driver_data = store.load_artifact(
            artifact_type="driver_characteristics",
            artifact_key=f"{target_year}::driver_characteristics",
        )
        source_year = target_year
        if not driver_data:
            logger.warning("Could not load driver characteristics from DB, falling back to file")
            driver_data = None
            for driver_file in driver_characteristics_fallback_paths(self.data_dir, target_year):
                if not driver_file.exists():
                    continue
                with open(driver_file) as f:
                    driver_data = json.load(f)
                source_year = infer_payload_year_from_path(
                    driver_file,
                    suffix="driver_characteristics",
                ) or int(driver_data.get("year", target_year))
                logger.info(
                    "Loaded driver characteristics fallback from %s for season %s",
                    driver_file,
                    target_year,
                )
                break
            if driver_data is None:
                raise FileNotFoundError(
                    f"Could not locate driver characteristics fallback for season {target_year} "
                    f"under {self.data_dir}"
                )

        # Strip legacy bayesian fields that older artifacts may contain.
        from src.utils.schema_validation import strip_legacy_bayesian_fields

        drivers_section = driver_data.get("drivers") if isinstance(driver_data, dict) else None
        if isinstance(drivers_section, dict):
            stripped = strip_legacy_bayesian_fields(drivers_section)
            if stripped:
                logger.info(
                    "Stripped %d legacy bayesian field(s) from loaded driver characteristics",
                    stripped,
                )

        driver_data = self._build_cross_season_driver_proxy_payload(
            driver_data,
            source_year=source_year,
            target_year=target_year,
        )
        try:
            validate_driver_characteristics(driver_data, expected_year=target_year)
        except ValueError as e:
            logger.error("Failed to load driver characteristics: %s", e)
            raise

        from src.utils.driver_validation import validate_driver_data

        errors = validate_driver_data(driver_data["drivers"])
        if errors:
            logger.warning(
                "Driver data has %s validation errors. Consider re-running extraction: "
                "python scripts/extract_driver_characteristics.py --years 2023,2024,2025,2026",
                len(errors),
            )

        return driver_data

    def _load_track_characteristics(
        self,
        store: ArtifactStore,
        target_year: int,
    ) -> dict:
        """Load, validate, and return the track-characteristics payload.

        Returns an empty dict when no track file exists (non-fatal).
        Raises ValueError if a file is found but fails schema validation.
        """
        track_data = store.load_artifact(
            artifact_type="track_characteristics",
            artifact_key=f"{target_year}::track_characteristics",
        )
        source_year = target_year
        if not track_data:
            logger.warning("Could not load track characteristics from DB, falling back to file")
            fallback = nearest_season_payload_path(
                self.data_dir / "track_characteristics",
                suffix="track_characteristics",
                target_year=target_year,
            )
            if fallback is None:
                logger.warning("Track characteristics not found")
                return {}
            source_year, track_file = fallback
            with open(track_file) as f:
                track_data = json.load(f)
            if source_year != target_year:
                logger.warning(
                    "Using %s track characteristics as preseason proxy for season %s.",
                    source_year,
                    target_year,
                )

        if track_data:
            track_data = self._build_cross_season_track_proxy_payload(
                track_data,
                source_year=source_year,
                target_year=target_year,
            )
            try:
                validate_track_characteristics(track_data, expected_year=target_year)
            except ValueError as e:
                logger.error("Failed to load track characteristics: %s", e)
                raise

        return self._supplement_track_layout_fields(track_data)

    @staticmethod
    def _build_proxy_note(
        existing_note: object,
        *,
        source_year: int,
        target_year: int,
        label: str,
    ) -> str:
        """Append one short note explaining a cross-season proxy fallback."""
        base_note = str(existing_note).strip() if isinstance(existing_note, str) else ""
        proxy_note = (
            f"Proxy fallback: reused {source_year} {label} as preseason priors for season "
            f"{target_year}."
        )
        return f"{base_note} {proxy_note}".strip() if base_note else proxy_note

    def _build_cross_season_team_proxy_payload(
        self,
        data: dict[str, Any],
        *,
        source_year: int,
        target_year: int,
    ) -> dict[str, Any]:
        """Rewrite cross-season team payloads into preseason proxy state."""
        if source_year == target_year:
            return data

        proxy = deepcopy(data)
        proxy["year"] = target_year
        proxy["data_freshness"] = "BASELINE_PRESEASON"
        proxy["races_completed"] = 0
        proxy["checkpoint_snapshot"] = {}
        proxy["note"] = self._build_proxy_note(
            proxy.get("note"),
            source_year=source_year,
            target_year=target_year,
            label="car characteristics",
        )

        teams = proxy.get("teams", {})
        if isinstance(teams, dict):
            for team_payload in teams.values():
                if not isinstance(team_payload, dict):
                    continue
                team_payload["current_season_performance"] = []
                team_payload["races_completed"] = 0

        return proxy

    def _build_cross_season_driver_proxy_payload(
        self,
        data: dict[str, Any],
        *,
        source_year: int,
        target_year: int,
    ) -> dict[str, Any]:
        """Rewrite cross-season driver priors so schema and notes match the replay year."""
        if source_year == target_year:
            return data

        proxy = deepcopy(data)
        proxy["year"] = target_year
        proxy["carried_over_from"] = source_year
        proxy["note"] = self._build_proxy_note(
            proxy.get("note"),
            source_year=source_year,
            target_year=target_year,
            label="driver characteristics",
        )
        drivers = proxy.get("drivers", {})
        if isinstance(drivers, dict):
            for driver_payload in drivers.values():
                if not isinstance(driver_payload, dict):
                    continue
                driver_payload["bayesian"] = {}
        return proxy

    def _build_cross_season_track_proxy_payload(
        self,
        data: dict[str, Any],
        *,
        source_year: int,
        target_year: int,
    ) -> dict[str, Any]:
        """Rewrite cross-season track priors so historical replay stays explicit."""
        if source_year == target_year:
            return data

        proxy = deepcopy(data)
        proxy["year"] = target_year
        proxy["data_freshness"] = "BASELINE_PRESEASON"
        proxy["note"] = self._build_proxy_note(
            proxy.get("note"),
            source_year=source_year,
            target_year=target_year,
            label="track characteristics",
        )
        return proxy

    def _supplement_track_layout_fields(self, track_data: dict) -> dict:
        """Fill missing layout-mix fields from the extracted track-profile cache.

        The committed 2026 track-characteristics payload carries race-strategy
        metadata such as pit-loss and overtaking, while the extracted
        ``track_profiles_cache.json`` carries the layout percentages used by the
        car directionality model. Keep both sources in sync at load time so the
        team-strength path does not silently lose track-suitability inputs.
        """
        tracks_payload = track_data.get("tracks")
        if not isinstance(track_data, dict) or not isinstance(tracks_payload, dict):
            return track_data

        cache_file = self.data_dir / "track_characteristics" / "track_profiles_cache.json"
        if not cache_file.exists():
            return track_data

        try:
            with open(cache_file) as handle:
                cache_payload = json.load(handle)
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Could not read track profile cache %s: %s", cache_file, exc)
            return track_data

        if not isinstance(cache_payload, dict):
            return track_data

        supplemented = 0
        for race_name, track_profile in tracks_payload.items():
            if not isinstance(track_profile, dict):
                continue
            cache_profile = cache_payload.get(race_name)
            if not isinstance(cache_profile, dict):
                continue

            for field_name in _TRACK_LAYOUT_FIELDS:
                if field_name in track_profile:
                    continue
                raw_value = cache_profile.get(field_name)
                if not isinstance(raw_value, int | float | str):
                    continue
                try:
                    track_profile[field_name] = float(raw_value)
                except (TypeError, ValueError):
                    continue
                supplemented += 1

        if supplemented > 0:
            logger.info(
                "Supplemented %s missing track layout field(s) from %s",
                supplemented,
                cache_file,
            )

        return track_data

    def load_data(self) -> None:
        """Load season data from the artifact store and apply it to this predictor.

        Delegates to three focused loaders, each following the same pattern:
        try the artifact store → fall back to file → validate schema → apply.

        Sets self.teams, self.drivers, self.tracks, self.races_completed,
        self.car_characteristics_snapshot, and self.year.
        """
        target_year = int(getattr(self, "season_year", 2026))
        store = getattr(self, "artifact_store", None) or ArtifactStore(data_root=self.data_dir)

        team_payload = self._load_team_characteristics(store, target_year)
        races_completed = self._apply_team_payload(team_payload, target_year)

        driver_payload = self._load_driver_characteristics(store, target_year)
        self.drivers = driver_payload["drivers"]

        track_payload = self._load_track_characteristics(store, target_year)
        self.tracks = track_payload.get("tracks", {})
        if self.tracks:
            logger.info("Loaded track characteristics for %s circuits", len(self.tracks))

        self.races_completed = races_completed
        self.year = team_payload.get("year", target_year)

    def _resolve_team_data(self, team: str) -> dict:
        """Resolve team payload using alias-aware mapping before fallback."""
        return resolve_team_data_helper(teams=self.teams, team=team)

    def calculate_track_suitability(self, team: str, race_name: str) -> float:
        """Calculate track-car suitability from directionality and layout mix."""
        return calculate_track_suitability_helper(
            team_data=self._resolve_team_data(team),
            track_profile=self.tracks.get(race_name, {}),
        )

    def get_blended_team_strength(self, team: str, race_name: str) -> float:
        """Blend preseason baseline and current-season form into one score."""
        cfg = getattr(self, "config", config_loader)
        return get_blended_team_strength_helper(
            context=self,
            team=team,
            race_name=race_name,
            cfg=cfg,
            schedules=SCHEDULES,
            get_recommended_schedule_fn=get_recommended_schedule,
            calculate_blended_performance_fn=calculate_blended_performance,
        )

    def _select_race_compound(self, race_name: str) -> str:
        """Select the likely primary race compound from Pirelli tire-stress data."""
        cfg = getattr(self, "config", config_loader)
        season_year = self._resolve_prediction_target_year()
        return select_race_compound(race_name=race_name, season_year=season_year, cfg=cfg)

    def get_compound_adjusted_team_strength(
        self, team: str, race_name: str, compound: str = "MEDIUM"
    ) -> float:
        """Return team strength adjusted for compound-specific performance data."""
        cfg = getattr(self, "config", config_loader)
        return get_compound_adjusted_team_strength_helper(
            context=self,
            team=team,
            race_name=race_name,
            compound=compound,
            cfg=cfg,
            should_use_compound_adjustments_fn=should_use_compound_adjustments,
            get_compound_performance_modifier_fn=get_compound_performance_modifier,
        )

    def _get_testing_characteristics_for_profile(self, team: str, profile: str) -> dict[str, float]:
        """Return testing characteristics for one run profile with older-file fallbacks."""
        return get_testing_characteristics_for_profile(
            resolve_team_data=self._resolve_team_data,
            team=team,
            profile=profile,
        )

    def _get_checkpoint_driver_delta_seconds(
        self,
        team: str,
        driver: str,
        preferred_profiles: tuple[str, ...] = ("short_run", "balanced", "long_run"),
    ) -> float | None:
        """Return one checkpoint-backed driver lap-time delta when a snapshot provides it."""
        team_data = self._resolve_team_data(team)
        checkpoint_driver_deltas = team_data.get("checkpoint_driver_deltas_seconds")
        if not isinstance(checkpoint_driver_deltas, dict) or not checkpoint_driver_deltas:
            return None

        driver_code = str(driver).strip().upper()
        if not driver_code:
            return None

        cfg = getattr(self, "config", config_loader)
        # Reject implausible single-lap teammate gaps. A snapshot delta of several seconds
        # means an unrepresentative lap survived extraction; consuming it would saturate the
        # downstream per-driver adjustment cap and flip teammate order on noise. Skipping it
        # falls back to the model's own driver skill/pace signal.
        max_delta_seconds = float(
            cfg.get("baseline_predictor.qualifying.checkpoint_driver_delta_max_seconds", 2.0)
        )

        for profile_name in preferred_profiles:
            profile_deltas = checkpoint_driver_deltas.get(profile_name)
            if not isinstance(profile_deltas, dict):
                continue
            raw_delta = profile_deltas.get(driver_code)
            if not isinstance(raw_delta, int | float | str):
                continue
            try:
                delta_seconds = float(raw_delta)
            except (TypeError, ValueError):
                continue
            if np.isfinite(delta_seconds) and abs(delta_seconds) <= max_delta_seconds:
                return delta_seconds

        return None

    def _compute_testing_profile_modifier(
        self,
        team: str,
        profile: str,
        metric_weights: dict[str, float],
        scale: float,
    ) -> tuple[float, bool]:
        """Compute a small bounded modifier from testing or practice profile data."""
        cfg = getattr(self, "config", config_loader)
        return compute_testing_profile_modifier(
            team=team,
            profile=profile,
            metric_weights=metric_weights,
            scale=scale,
            get_testing_characteristics_for_profile_fn=self._get_testing_characteristics_for_profile,
            cfg=cfg,
        )

    def _update_compound_characteristics_from_session(
        self,
        session_laps: pd.DataFrame,
        race_name: str,
        year: int,
        is_sprint: bool,
    ) -> None:
        """Extract compound characteristics with session-level caching and persistence."""
        cfg = getattr(self, "config", config_loader)
        update_compound_characteristics_from_session(
            context=self,
            session_laps=session_laps,
            race_name=race_name,
            year=year,
            is_sprint=is_sprint,
            cfg=cfg,
        )
