"""Start a new season from the previous season's end state in the artifact store.

Every reader that finds no ``{year}`` car or driver artifact falls back to the nearest
season file in the repo, which is a mid-season snapshot (2026's was saved after 11 of
24 races). Seeding ``{year}`` from the store's ``{year - 1}`` keeps everything the
previous season learned, rewritten as preseason priors, so testing and race 1 build on it.
"""

from __future__ import annotations

import logging
from typing import Any

from src.predictors.baseline.data_support import (
    cross_season_driver_proxy,
    cross_season_team_proxy,
)
from src.utils.schema_validation import (
    validate_driver_characteristics,
    validate_team_characteristics,
)

logger = logging.getLogger(__name__)

_SEASON_ARTIFACTS = (
    ("car_characteristics", cross_season_team_proxy, validate_team_characteristics),
    ("driver_characteristics", cross_season_driver_proxy, validate_driver_characteristics),
)


def seed_season_from_previous(year: int, store: Any = None) -> list[str]:
    """Seed ``year``'s car and driver state from ``year - 1`` when ``year`` has none.

    Idempotent: a season that already has state is left alone. Returns the artifact
    types it seeded.
    """
    if store is None:
        from src.persistence.artifact_store import ArtifactStore

        store = ArtifactStore()

    seeded: list[str] = []
    for artifact_type, to_preseason, validate in _SEASON_ARTIFACTS:
        if store.load_artifact(artifact_type, f"{year}::{artifact_type}"):
            continue
        previous = store.load_artifact(artifact_type, f"{year - 1}::{artifact_type}")
        if not previous:
            continue
        payload = to_preseason(previous, source_year=year - 1, target_year=year)
        payload["version"] = 1
        if artifact_type == "car_characteristics":
            # Turns on per-team scoring of the carried-over testing profile
            # (src/systems/testing_prior_selection.py).
            payload["seeded_from"] = year - 1
        validate(payload, expected_year=year)
        store.save_artifact(artifact_type, f"{year}::{artifact_type}", payload, version=1)
        logger.info("Seeded %s %s from the %s end state", year, artifact_type, year - 1)
        seeded.append(artifact_type)
    return seeded
