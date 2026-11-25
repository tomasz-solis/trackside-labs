"""Plan row 33: does a practice gap against the pre-weekend forecast predict qualifying?

Unit: a team at a practice checkpoint. Expectation: the stored PRE main qualifying
forecast of one replay root, as team ranks (the stored forecast holds positions only).
Evidence: the practice gap the model already measures (``fp_short_gap_pct``) over the
dry sessions visible at the checkpoint, as team ranks. A card fires when the two ranks
differ by at least ``--threshold`` places. The next test: the team's qualifying rank moves
from its expected rank in the card's direction (supported) or not (weakened); wet
qualifying or a driver missing from the classification is unresolved.

The bar (threshold, minimum laps, end-team rule, null bands, the share of weekends with a
card and the lift whose interval must exclude zero) is written in docs/MODEL_LEDGER.md
before this runs on real data. Every bar value is a required argument for that reason.

    uv run python scripts/investigation_backtest.py --root data/historical_replay_arms/C39_lineup_s42 \
        --threshold 3 --min-laps 8 --ends drop --bands 4
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.ml_challenger import practice_features_from_payloads  # noqa: E402

NORMAL_CHECKPOINTS = ("FP1", "FP2", "FP3")
SPRINT_CHECKPOINTS = ("FP1",)
WetFn = Callable[[int, str, str], bool]


@dataclass(frozen=True)
class Unit:
    """One team at one practice checkpoint.

    ``moved`` is expected minus actual qualifying rank (positive = qualified better than
    expected), ``None`` when qualifying cannot resolve the card.
    """

    race: str
    checkpoint: str
    team: str
    expected_rank: float
    practice_rank: float
    moved: float | None

    @property
    def gap(self) -> float:
        """Expected minus practice rank: positive = quicker in practice than expected."""
        return self.expected_rank - self.practice_rank

    @property
    def outcome(self) -> str:
        """Supported when qualifying moved the gap's way, else weakened or unresolved."""
        if self.moved is None:
            return "unresolved"
        return "supported" if self.moved * self.gap > 0 else "weakened"


def _ranks(values: dict[str, float]) -> dict[str, float]:
    """1 = best (lowest value), ties averaged."""
    ordered = sorted(values, key=lambda team: values[team])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        for team in ordered[i : j + 1]:
            ranks[team] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _team_mean_position(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    by_team: dict[str, list[float]] = {}
    for row in rows:
        by_team.setdefault(str(row["team"]), []).append(float(row["position"]))
    return {team: float(np.mean(positions)) for team, positions in by_team.items()}


def weekend_units(
    *,
    year: int,
    race: str,
    forecast: Sequence[dict[str, Any]],
    actual: Sequence[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
    checkpoints: Sequence[str],
    is_wet: WetFn,
    min_laps: int,
    drop_ends: bool,
) -> list[Unit]:
    """Every team unit of one weekend, gated on dry sessions and clean laps."""
    q_wet = is_wet(year, race, "Q")
    units: list[Unit] = []
    for index, checkpoint in enumerate(checkpoints):
        visible = [
            snapshots[s]
            for s in checkpoints[: index + 1]
            if s in snapshots and not is_wet(year, race, s)
        ]
        if not visible:
            continue
        laps: dict[str, int] = {}
        for payload in visible:
            for team, count in (payload.get("team_clean_lap_counts") or {}).items():
                laps[team] = laps.get(team, 0) + int(count)
        features = practice_features_from_payloads(visible)
        if "fp_short_gap_pct" not in features:
            continue
        practice = {
            team: float(gap)
            for team, gap in features["fp_short_gap_pct"].dropna().items()
            if laps.get(str(team), 0) >= min_laps
        }
        expected_pos = _team_mean_position(forecast)
        teams = sorted(set(practice) & set(expected_pos))
        if len(teams) < 6:
            continue
        expected = _ranks({t: expected_pos[t] for t in teams})
        observed = _ranks({t: practice[t] for t in teams})
        actual_rank = _ranks({t: v for t, v in _team_mean_position(actual).items() if t in teams})
        for team in teams:
            if drop_ends and expected[team] in (1.0, float(len(teams))):
                continue
            moved = None if q_wet or team not in actual_rank else expected[team] - actual_rank[team]
            units.append(Unit(race, checkpoint, team, expected[team], observed[team], moved))
    return units


def _supported_share(units: Sequence[Unit]) -> tuple[int, int]:
    resolved = [u for u in units if u.outcome != "unresolved"]
    return sum(u.outcome == "supported" for u in resolved), len(resolved)


def _shuffled(units: Sequence[Unit], bands: int, rng: np.random.Generator) -> list[Unit]:
    """Practice ranks permuted across teams within each checkpoint and expected-rank band."""
    groups: dict[tuple[str, str, int], list[Unit]] = {}
    for unit in units:
        band = int((unit.expected_rank - 1) // bands)
        groups.setdefault((unit.race, unit.checkpoint, band), []).append(unit)
    out: list[Unit] = []
    for group in groups.values():
        for unit, rank in zip(
            group, rng.permutation([u.practice_rank for u in group]), strict=True
        ):
            out.append(
                Unit(
                    unit.race,
                    unit.checkpoint,
                    unit.team,
                    unit.expected_rank,
                    float(rank),
                    unit.moved,
                )
            )
    return out


def summarise(
    units_by_weekend: dict[str, list[Unit]],
    *,
    threshold: float,
    bands: int,
    shuffles: int = 200,
    bootstrap: int = 2000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Cards, shares and both lifts with weekend-bootstrap 95% intervals."""
    rng = np.random.default_rng(seed)
    weekends = sorted(units_by_weekend)
    if not weekends:
        return {"weekends": 0}
    rows: list[dict[str, Any]] = []
    for race in weekends:
        units = units_by_weekend[race]
        cards = [u for u in units if abs(u.gap) >= threshold]
        under = [u for u in units if abs(u.gap) < threshold]
        null = []
        for _ in range(shuffles):
            shuffled = [u for u in _shuffled(units, bands, rng) if abs(u.gap) >= threshold]
            null.append(_supported_share(shuffled))
        rows.append(
            {
                "cards": len(cards),
                "card": _supported_share(cards),
                "under": _supported_share(under),
                "null": null,
                "unresolved": sum(u.outcome == "unresolved" for u in cards),
                "weakened": sum(u.outcome == "weakened" for u in cards),
            }
        )

    def lifts(sample: Sequence[dict[str, Any]]) -> tuple[float, float]:
        card_s = sum(r["card"][0] for r in sample)
        card_n = sum(r["card"][1] for r in sample)
        under_s = sum(r["under"][0] for r in sample)
        under_n = sum(r["under"][1] for r in sample)
        null_s = np.sum([[k[0] for k in r["null"]] for r in sample], axis=0)
        null_n = np.sum([[k[1] for k in r["null"]] for r in sample], axis=0)
        share = card_s / card_n if card_n else np.nan
        under_share = under_s / under_n if under_n else np.nan
        null_share = float(np.nanmean(np.where(null_n > 0, null_s / np.maximum(null_n, 1), np.nan)))
        return share - under_share, share - null_share

    observed = lifts(rows)
    draws = np.array(
        [lifts([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(bootstrap)]
    )
    total_cards = sum(r["cards"] for r in rows)
    supported = sum(r["card"][0] for r in rows)
    return {
        "weekends": len(rows),
        "weekends_without_card": sum(r["cards"] == 0 for r in rows),
        "cards": total_cards,
        "cards_per_weekend": total_cards / len(rows),
        "supported": supported,
        "weakened": sum(r["weakened"] for r in rows),
        "unresolved": sum(r["unresolved"] for r in rows),
        "lift_vs_under": observed[0],
        "lift_vs_under_ci": tuple(np.nanpercentile(draws[:, 0], [2.5, 97.5]).round(3)),
        "lift_vs_shuffle": observed[1],
        "lift_vs_shuffle_ci": tuple(np.nanpercentile(draws[:, 1], [2.5, 97.5]).round(3)),
    }


@functools.cache
def _fastf1_wet(year: int, race: str, session: str) -> bool:
    """True when the cached session reported rain; an unreadable session counts as wet."""
    import fastf1

    from src.dashboard.cache import enable_fastf1_cache

    enable_fastf1_cache()
    fastf1.Cache.offline_mode(True)
    try:
        loaded = fastf1.get_session(year, race, session)
        loaded.load(laps=False, telemetry=False, weather=True, messages=False)
        return bool(loaded.weather_data["Rainfall"].any())
    except Exception:  # noqa: BLE001 - an unknown session is not evidence
        return True


def load_root(
    root: Path, year: int, *, is_wet: WetFn, min_laps: int, drop_ends: bool
) -> tuple[dict[str, list[Unit]], dict[str, list[Unit]]]:
    """Units per weekend from one replay root, normal and sprint weekends apart."""
    normal: dict[str, list[Unit]] = {}
    sprint: dict[str, list[Unit]] = {}
    for pre in sorted((root / "predictions" / str(year)).glob("*/*_pre.json")):
        payload = json.loads(pre.read_text(encoding="utf-8"))
        forecast = (payload.get("targets") or {}).get("main_qualifying", {}).get("predicted_order")
        actual = ((payload.get("actuals") or {}).get("targets") or {}).get("main_qualifying")
        if not forecast or not actual:
            continue
        race = str(payload["metadata"]["race_name"])
        is_sprint = payload["metadata"].get("weekend_format") == "sprint"
        snapshot_dir = root / "car_characteristics_snapshot" / str(year) / pre.parent.name
        checkpoints = SPRINT_CHECKPOINTS if is_sprint else NORMAL_CHECKPOINTS
        snapshots = {
            s: json.loads((snapshot_dir / f"{s.lower()}.json").read_text(encoding="utf-8"))
            for s in checkpoints
            if (snapshot_dir / f"{s.lower()}.json").exists()
        }
        units = weekend_units(
            year=year,
            race=race,
            forecast=forecast,
            actual=actual,
            snapshots=snapshots,
            checkpoints=checkpoints,
            is_wet=is_wet,
            min_laps=min_laps,
            drop_ends=drop_ends,
        )
        (sprint if is_sprint else normal)[race] = units
    return normal, sprint


def main(argv: Sequence[str] | None = None) -> int:
    """Print the row 33 tables for one replay root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--threshold", type=float, required=True, help="Card: places apart.")
    parser.add_argument("--min-laps", type=int, required=True, help="Clean laps, visible sessions.")
    parser.add_argument(
        "--ends",
        choices=("drop", "keep"),
        required=True,
        help="Drop the expected first and last team (bounded-rank trap).",
    )
    parser.add_argument(
        "--bands", type=int, required=True, help="Expected-rank band width for the shuffle null."
    )
    args = parser.parse_args(argv)

    normal, sprint = load_root(
        args.root,
        args.year,
        is_wet=_fastf1_wet,
        min_laps=args.min_laps,
        drop_ends=args.ends == "drop",
    )
    for label, units in (("Normal weekends, FP1 to FP3", normal), ("Sprint weekends, FP1", sprint)):
        result = summarise(units, threshold=args.threshold, bands=args.bands)
        print(f"\n## {label}\n")
        print("| Measure | Value |\n|---|---|")
        for key, value in result.items():
            shown = f"{value:.3f}" if isinstance(value, float) else value
            print(f"| {key} | {shown} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
