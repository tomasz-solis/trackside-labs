"""Backfill DNF data into stored prediction artifacts so the historical report is complete.

For every prediction artifact under ``data/predictions/<year>/`` this script:

  1. Re-fetches the actual race/sprint classification (which now carries a per-driver
     ``dnf`` flag) and writes it back into the stored ``actuals`` blocks, so
     finisher-only MAE and DNF calibration (Brier) can be scored historically.
  2. Fills any *missing* predicted ``dnf_probability`` on race-like predicted rows by
     re-running the race forecast. Existing recorded probabilities and all finishing
     positions are left untouched.

Re-fetching actuals is exact (the classification is ground truth). Re-derived
probabilities use current model state and only ever fill gaps left by the old
save bug; rows that already carry a probability are preserved.

``--store`` labels the stored predictions instead (production with
``--env-file .env.local``). It reads the ArtifactStore listing only, never local files,
attaches DNF flags only, never refills probabilities or runs learning, and writes every
original record it changes to ``--export`` before saving anything. It writes no
operational events, skips races with no stored result, and stops if a stored flag
would flip. Check the summary line of the dry run: it must say ``0 flips``.

Usage:
  python scripts/backfill_dnf_data.py --year 2026
  python scripts/backfill_dnf_data.py --year 2026 --dry-run
  python scripts/backfill_dnf_data.py --year 2026 --store --env-file .env.local --dry-run
  python scripts/backfill_dnf_data.py --year 2026 --store --env-file .env.local --export <file>
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

import fastf1

from src.data.actual_results_fetcher import fetch_actual_session_results
from src.utils.accuracy_targets import TARGET_SESSION_BY_KEY
from src.utils.env_file import load_env_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backfill_dnf")

# Race-like targets get a DNF flag; qualifying targets do not have a DNF concept.
_RACE_LIKE_TARGETS = {"grand_prix_race": "R", "sprint_race": "Sprint"}
_FASTF1_CACHE_DIRS = ("data/raw/.fastf1_cache", "data/raw/.fastf1_cache_testing")


def _enable_cache() -> None:
    """Enable the first available local FastF1 cache directory."""
    for cache_dir in _FASTF1_CACHE_DIRS:
        path = Path(cache_dir)
        if path.exists():
            fastf1.Cache.configure(cache_dir=str(path))
            return
    Path(_FASTF1_CACHE_DIRS[0]).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=_FASTF1_CACHE_DIRS[0])


def _dnf_by_driver(fetched_rows: list[dict[str, Any]] | None) -> dict[str, bool]:
    """Map driver code -> DNF flag from a freshly fetched classification."""
    if not fetched_rows:
        return {}
    return {str(row["driver"]): bool(row.get("dnf", False)) for row in fetched_rows}


def _merge_dnf_into_block(
    block_rows: list[dict[str, Any]] | None,
    dnf_by_driver: dict[str, bool],
) -> int:
    """Add a per-driver ``dnf`` flag to existing actual rows in place.

    Positions, drivers, and teams are preserved exactly; only the ``dnf`` flag is
    attached by matching on driver. This never reorders or rewrites the recorded
    classification, so a re-fetch that happens to differ in ordering cannot change
    the ground truth the predictions were scored against.
    """
    if not isinstance(block_rows, list) or not dnf_by_driver:
        return 0
    labelled = 0
    for row in block_rows:
        if not isinstance(row, dict):
            continue
        driver = str(row.get("driver", ""))
        if driver in dnf_by_driver:
            row["dnf"] = dnf_by_driver[driver]
            if row["dnf"]:
                labelled += 1
    return labelled


def _match_session_for_block(
    block_rows: list[dict[str, Any]],
    fetched: dict[str, list[dict[str, Any]] | None],
) -> str | None:
    """Pick the fetched session whose ordering best matches an existing block.

    Used for the legacy top-level race actuals, which on a sprint weekend hold the
    sprint classification rather than the Grand Prix. Returns ``None`` when no
    session matches any position or two sessions tie, so an unidentifiable block is
    left unlabelled rather than given another session's DNF flags.
    """
    block_order = [str(row.get("driver", "")) for row in block_rows]
    best_session: str | None = None
    best_score = 0
    tied = False
    for session_name, rows in fetched.items():
        if not rows:
            continue
        order = [str(row["driver"]) for row in rows]
        score = sum(1 for left, right in zip(block_order, order, strict=False) if left == right)
        if score > best_score:
            best_score = score
            best_session = session_name
            tied = False
        elif score == best_score and best_session is not None:
            tied = True
    if tied:
        logger.warning("Legacy race block matches several sessions equally; leaving it unlabelled")
        return None
    if best_session is None:
        logger.warning("Legacy race block matches no fetched session; leaving it unlabelled")
    return best_session


def _backfill_actuals(
    prediction: dict[str, Any],
    *,
    fetched: dict[str, list[dict[str, Any]] | None],
) -> int:
    """Attach DNF flags to the prediction's existing actuals blocks. Returns DNFs labelled."""
    actuals = prediction.get("actuals")
    if not isinstance(actuals, dict):
        return 0

    labelled = 0
    target_actuals = actuals.get("targets")
    if isinstance(target_actuals, dict):
        for target_key, session_name in _RACE_LIKE_TARGETS.items():
            rows = target_actuals.get(target_key)
            labelled += _merge_dnf_into_block(rows, _dnf_by_driver(fetched.get(session_name)))

    # Legacy top-level race actuals: pick the matching session (sprint vs Grand Prix).
    legacy_rows = actuals.get("race")
    if isinstance(legacy_rows, list) and legacy_rows:
        matched_session = _match_session_for_block(legacy_rows, fetched)
        if matched_session is not None:
            _merge_dnf_into_block(legacy_rows, _dnf_by_driver(fetched.get(matched_session)))

    return labelled


_ACTUAL_BLOCKS = ("race", *_RACE_LIKE_TARGETS)


def _race_like_blocks(prediction: dict[str, Any]) -> dict[str, list[Any]]:
    """The legacy race block and each race-like target block, empty when absent."""
    actuals = prediction.get("actuals")
    actuals = actuals if isinstance(actuals, dict) else {}
    targets = actuals.get("targets")
    targets = targets if isinstance(targets, dict) else {}
    blocks = {"race": actuals.get("race"), **{key: targets.get(key) for key in _RACE_LIKE_TARGETS}}
    return {name: rows if isinstance(rows, list) else [] for name, rows in blocks.items()}


def _flag_changes(original: dict[str, Any], updated: dict[str, Any]) -> tuple[dict[str, int], int]:
    """Flags added per block, and stored flags the update would flip."""
    added = dict.fromkeys(_ACTUAL_BLOCKS, 0)
    flips = 0
    before, after = _race_like_blocks(original), _race_like_blocks(updated)
    for name in _ACTUAL_BLOCKS:
        # The merge never reorders rows, so rows pair by index.
        for old, new in zip(before[name], after[name], strict=True):
            if not isinstance(old, dict) or "dnf" not in new:
                continue
            if "dnf" not in old:
                added[name] += 1
            elif bool(old["dnf"]) != bool(new["dnf"]):
                flips += 1
    return added, flips


def _build_grid(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a starting grid for re-deriving DNF probabilities."""
    quali = (prediction.get("qualifying") or {}).get("predicted_grid") or []
    rows = quali or (prediction.get("race") or {}).get("predicted_results") or []
    grid = [
        {
            "driver": str(row["driver"]),
            "team": str(row.get("team", "")),
            "position": int(row.get("position", index + 1)),
        }
        for index, row in enumerate(rows)
        if row.get("driver")
    ]
    return grid


def _fill_missing_dnf_probabilities(
    prediction: dict[str, Any],
    *,
    predictor: Any,
    year: int,
    race_name: str,
    weather: str,
    context: Any,
) -> int:
    """Fill missing predicted dnf_probability on race-like rows. Returns rows filled."""
    grid = _build_grid(prediction)
    if len(grid) < 2:
        return 0

    def _needs_fill(rows: Any) -> bool:
        return isinstance(rows, list) and any(
            isinstance(row, dict) and row.get("dnf_probability") is None for row in rows
        )

    # Only the target predicted_order rows are scored by the report. Each is filled with
    # the probability for its own race type. The legacy race.predicted_results block is left
    # alone: it is not read by the evaluation and its race mapping is ambiguous on sprint
    # weekends. Rows that already carry a probability are preserved.
    targets = prediction.get("targets") or {}
    target_rows = {
        ("grand_prix_race", False): (targets.get("grand_prix_race") or {}).get("predicted_order"),
        ("sprint_race", True): (targets.get("sprint_race") or {}).get("predicted_order"),
    }
    if not any(_needs_fill(rows) for rows in target_rows.values()):
        return 0

    filled = 0
    for (_target_key, is_sprint), rows in target_rows.items():
        if not isinstance(rows, list) or not _needs_fill(rows):
            continue
        result = predictor.predict_race(
            qualifying_grid=grid,
            weather=weather,
            race_name=race_name,
            year=year,
            is_sprint=is_sprint,
            n_simulations=200,
            prediction_context=context,
        )
        probabilities = {
            str(row["driver"]): float(row.get("dnf_probability", 0.0))
            for row in result["finish_order"]
        }
        for row in rows:
            if isinstance(row, dict) and row.get("dnf_probability") is None:
                value = probabilities.get(str(row.get("driver")))
                if value is not None:
                    row["dnf_probability"] = round(float(value), 3)
                    filled += 1

    return filled


def _fetch_race_like(year: int, race_name: str) -> dict[str, list[dict[str, Any]] | None]:
    """Fetch the race and sprint classifications, None for a session that fails."""
    fetched: dict[str, list[dict[str, Any]] | None] = {}
    for session_name in ("R", "Sprint"):
        try:
            result = fetch_actual_session_results(year, race_name, session_name)
            fetched[session_name] = cast("list[dict[str, Any]] | None", result)
        except Exception as exc:  # noqa: BLE001 - best effort per session
            logger.warning("Could not fetch %s %s: %s", race_name, session_name, exc)
            fetched[session_name] = None
    return fetched


def backfill_store(
    year: int,
    *,
    logger_inst: Any,
    export_path: Path | None,
    dry_run: bool,
    fetch_fn: Any = _fetch_race_like,
) -> int:
    """Attach DNF flags to stored predictions. Returns how many records changed.

    Reads the ArtifactStore listing only: ``get_all_predictions`` also merges local
    files, and saving a stale local copy would overwrite the stored record. Writes
    prediction records only: operational events stay in this process, and a race
    with no stored actuals yet is never fetched. Any stored flag that would flip
    stops the write.
    """
    from src.utils.operational_observability import suspend_db_writes
    from src.utils.prediction_logger import PredictionLogger

    suspend_db_writes()
    store = logger_inst.artifact_store
    if getattr(store, "storage_mode", "file_only") == "file_only":
        raise SystemExit("--store needs USE_DB_STORAGE db_only or dual_write (--env-file)")
    rows = store.list_artifacts("prediction", key_prefix=f"{year}::", limit=4096)
    # The stored run_id column can differ from the payload's (an actuals update stamps
    # its own run); the write must change DNF flags only, so it keeps the stored one.
    run_id_by_key = {row.get("artifact_key"): row.get("run_id") for row in rows}
    payloads = [logger_inst._prediction_payload_from_listing_row(row) for row in rows]
    predictions = logger_inst._deduplicate_predictions([p for p in payloads if p])

    fetched_by_race: dict[str, dict[str, list[dict[str, Any]] | None]] = {}
    changes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    added_total = dict.fromkeys(_ACTUAL_BLOCKS, 0)
    flips_total = 0
    suffix = " (dry-run)" if dry_run else ""
    for prediction in predictions:
        metadata = prediction.get("metadata", {})
        race_name = str(metadata.get("race_name", "")).strip()
        session_name = str(metadata.get("session_name", "")).strip()
        if not race_name or not session_name:
            continue
        if not any(_race_like_blocks(prediction).values()):
            continue  # no result stored yet, so nothing to label
        if race_name not in fetched_by_race:
            fetched_by_race[race_name] = fetch_fn(year, race_name)
        updated = copy.deepcopy(prediction)
        labelled = _backfill_actuals(updated, fetched=fetched_by_race[race_name])
        if updated.get("actuals") != prediction.get("actuals"):
            changes.append((prediction, updated))
            added, flips = _flag_changes(prediction, updated)
            for name, count in added.items():
                added_total[name] += count
            flips_total += flips
            logger.info(
                "%s %s: %s flags added, %s flips, %s retirements%s",
                race_name,
                session_name,
                sum(added.values()),
                flips,
                labelled,
                suffix,
            )

    logger.info(
        "Store backfill: %s record(s) change; flags added: %s Grand Prix, %s sprint, "
        "%s legacy race; %s flips%s.",
        len(changes),
        added_total["grand_prix_race"],
        added_total["sprint_race"],
        added_total["race"],
        flips_total,
        ", nothing written" if dry_run else "",
    )
    if dry_run or not changes:
        return len(changes)
    if flips_total:
        raise SystemExit(f"{flips_total} stored flag(s) would flip; nothing written")
    if export_path is None:
        raise SystemExit("--export <file> is required before --store writes anything")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        json.dumps([original for original, _ in changes], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Exported %s original record(s) to %s", len(changes), export_path)
    stamp = datetime.now(UTC).isoformat()
    for _, prediction in changes:
        metadata = prediction.setdefault("metadata", {})
        metadata["dnf_backfilled_at"] = stamp
        artifact_key = PredictionLogger._artifact_key_for_prediction(
            year, metadata["race_name"], metadata["session_name"]
        )
        store.save_artifact(
            artifact_type="prediction",
            artifact_key=artifact_key,
            data=prediction,
            version=1,
            run_id=run_id_by_key.get(artifact_key, metadata.get("run_id")),
        )
    return len(changes)


def backfill_file(
    path: Path,
    *,
    predictor: Any,
    dry_run: bool,
) -> dict[str, int]:
    """Backfill one prediction artifact. Returns counts of what changed."""
    from src.utils.prediction_context import PredictionContext

    with open(path, encoding="utf-8") as handle:
        prediction = json.load(handle)

    metadata = prediction.get("metadata", {})
    year = int(metadata.get("year", 0) or 0)
    race_name = str(metadata.get("race_name", "")).strip()
    weather = str(metadata.get("weather", "dry")).strip().lower() or "dry"
    if year <= 0 or not race_name:
        logger.warning("Skipping %s: missing year/race_name", path.name)
        return {"actual_dnfs": 0, "probs_filled": 0}

    actual_dnfs = _backfill_actuals(prediction, fetched=_fetch_race_like(year, race_name))

    # A far-future reference time keeps all sessions "available"; DNF probability does
    # not depend on practice data, so this only affects session-availability gating.
    context = PredictionContext(
        mode="historical",
        as_of_datetime=datetime(year, 12, 31, tzinfo=UTC),
        season_year=year,
    )
    probs_filled = _fill_missing_dnf_probabilities(
        prediction,
        predictor=predictor,
        year=year,
        race_name=race_name,
        weather=weather,
        context=context,
    )

    if (actual_dnfs or probs_filled) and not dry_run:
        prediction.setdefault("metadata", {})["dnf_backfilled_at"] = datetime.now(UTC).isoformat()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(prediction, handle, indent=2)

    return {"actual_dnfs": actual_dnfs, "probs_filled": probs_filled}


def main() -> None:
    """CLI entrypoint for the DNF backfill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=datetime.now(UTC).year)
    parser.add_argument(
        "--predictions-dir",
        default="data/predictions",
        help="Root directory of stored prediction artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing files.",
    )
    parser.add_argument("--store", action="store_true", help="Label stored predictions.")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--export", type=Path, default=None, help="Original records, --store.")
    args = parser.parse_args()

    if args.env_file is not None:
        load_env_file(args.env_file)
    # File mode never touches the database unless the environment says so.
    os.environ.setdefault("USE_DB_STORAGE", "file_only")
    _ = TARGET_SESSION_BY_KEY  # documents the target->session mapping source
    _enable_cache()

    if args.store:
        from src.utils.prediction_logger import PredictionLogger

        backfill_store(
            args.year,
            logger_inst=PredictionLogger(),
            export_path=args.export,
            dry_run=args.dry_run,
        )
        return

    year_dir = Path(args.predictions_dir) / str(args.year)
    if not year_dir.exists():
        logger.error("No prediction directory for %s at %s", args.year, year_dir)
        return

    files = sorted(year_dir.rglob("*.json"))
    if not files:
        logger.warning("No prediction artifacts found under %s", year_dir)
        return

    from src.predictors import Baseline2026Predictor

    predictor = Baseline2026Predictor(season_year=args.year)

    total_actual_dnfs = 0
    total_probs_filled = 0
    changed_files = 0
    for path in files:
        counts = backfill_file(path, predictor=predictor, dry_run=args.dry_run)
        if counts["actual_dnfs"] or counts["probs_filled"]:
            changed_files += 1
            logger.info(
                "%s: actual DNFs labelled=%s, predicted probabilities filled=%s%s",
                path.name,
                counts["actual_dnfs"],
                counts["probs_filled"],
                " (dry-run)" if args.dry_run else "",
            )
        total_actual_dnfs += counts["actual_dnfs"]
        total_probs_filled += counts["probs_filled"]

    logger.info(
        "Backfill complete: %s file(s) changed, %s actual DNFs labelled, "
        "%s predicted probabilities filled%s.",
        changed_files,
        total_actual_dnfs,
        total_probs_filled,
        " (dry-run, no files written)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
