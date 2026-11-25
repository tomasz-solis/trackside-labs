"""Scoring rules of the 2027 prospective protocol (``docs/MODEL_LEDGER.md``, 2026-11-21).

Pure functions on plain rows, with no scipy or pandas, because the cron may import them:

- ``same_kind_naive``: the forecast a target is compared with.
- ``spearman``: the primary metric, equal to ``compute_prediction_accuracy``'s.
- ``likely_range_coverage``: the share of drivers inside the shown Likely range.
- ``sequential_verdict`` and ``coverage_verdict``: the call after every compared weekend.

A weekend's value for a target is the mean over its scored checkpoints of champion
minus naive correlation (``weekend_value``), or of coverage minus the 50% claim. Each
series is judged under the null its 2026 replay follows (ledger 2026-11-22):
``MARGIN_NULL`` for margins, ``COVERAGE_NULL`` for coverage.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.models.challengers import order_by_result
from src.utils.accuracy_targets import (
    TARGET_GRAND_PRIX_RACE,
    TARGET_MAIN_QUALIFYING,
    TARGET_SPRINT_QUALIFYING,
    TARGET_SPRINT_RACE,
)

Rows = list[dict[str, Any]]

# Calibrated by scripts/calibrate_protocol_threshold.py (ledger 2026-11-22): 5% false
# verdicts per series per season when champion and naive are exchangeable, looks 4 to 24.
# independent: independent weekends. drifting: AR(1) weekends, autocorrelation 0.66.
VERDICT_P_THRESHOLD = {"independent": 0.0079, "drifting": 4.9e-6}
FIRST_VERDICT_WEEKEND = 4
MAX_WEEKENDS = 24
LIKELY_RANGE_CLAIM = 0.50

SESSION_KIND = {
    TARGET_MAIN_QUALIFYING: "one_lap",
    TARGET_SPRINT_QUALIFYING: "one_lap",
    TARGET_GRAND_PRIX_RACE: "race",
    TARGET_SPRINT_RACE: "race",
}

# 2026 replay: race margins alternate (lag-1 autocorrelation -0.69, which only makes the
# independent null stricter); one-lap margins and coverage drift (up to 0.46 detrended).
MARGIN_NULL = {
    TARGET_MAIN_QUALIFYING: "drifting",
    TARGET_SPRINT_QUALIFYING: "drifting",
    TARGET_GRAND_PRIX_RACE: "independent",
    TARGET_SPRINT_RACE: "independent",
}
COVERAGE_NULL = "drifting"

# Two-sided critical |t| at each null's threshold by compared weekends (df = weekends - 1).
# Tests check every value against scipy; the cron cannot load it.
_CRITICAL_T = {
    "independent": {
        4: 6.350240,
        5: 4.925187,
        6: 4.275138,
        7: 3.909165,
        8: 3.676127,
        9: 3.515304,
        10: 3.397866,
        11: 3.308448,
        12: 3.238143,
        13: 3.181443,
        14: 3.134764,
        15: 3.095673,
        16: 3.062465,
        17: 3.033909,
        18: 3.009094,
        19: 2.987331,
        20: 2.968091,
        21: 2.950961,
        22: 2.935612,
        23: 2.921779,
        24: 2.909251,
    },
    "drifting": {
        4: 76.618939,
        5: 33.214968,
        6: 20.675535,
        7: 15.312599,
        8: 12.474241,
        9: 10.760007,
        10: 9.628311,
        11: 8.832088,
        12: 8.244610,
        13: 7.794928,
        14: 7.440538,
        15: 7.154562,
        16: 6.919240,
        17: 6.722401,
        18: 6.555442,
        19: 6.412122,
        20: 6.287809,
        21: 6.178996,
        22: 6.082982,
        23: 5.997654,
        24: 5.921338,
    },
}


@dataclass(frozen=True)
class OfficialResult:
    """One revision of a session's official classification.

    ``issued_at`` is when this revision was known (for stored actuals, when it was
    fetched). A correction is another ``OfficialResult`` for the same session with a
    later ``issued_at``.
    """

    target: str
    event: str
    session_start: datetime
    rows: Rows
    issued_at: datetime


@dataclass(frozen=True)
class Verdict:
    """The protocol's call on one target after its compared weekends so far.

    ``status`` is ``provisional`` until a look crosses its critical value, then the
    status of that look for the rest of the season; ``noise`` when a season ends
    without one. ``called_at`` is the weekend of the crossing.
    """

    status: str
    weekends: int
    mean: float | None
    t_statistic: float | None
    critical_t: float | None
    called_at: int | None = None


def same_kind_naive(
    *,
    target: str,
    champion_rows: Rows,
    results: Iterable[OfficialResult],
    information_cutoff_at: datetime,
) -> Rows | None:
    """The champion's drivers in the latest same-kind result issued before the cutoff.

    Qualifying and sprint qualifying compare with the last one-lap session, the Grand
    Prix and the sprint with the last race. A revision counts when it was issued at or
    before the cutoff, so a correction published after it never reaches the naive; of
    those, the latest session's latest revision is used. Drivers missing from the result
    follow in the champion's order. ``None`` when no result qualifies.
    """
    kind = SESSION_KIND[target]
    if information_cutoff_at.tzinfo is None:
        raise ValueError("information_cutoff_at must be timezone-aware")
    eligible = [
        result
        for result in results
        if SESSION_KIND.get(result.target) == kind and result.issued_at <= information_cutoff_at
    ]
    if not eligible:
        return None
    latest = max(eligible, key=lambda result: (result.session_start, result.issued_at))
    team_by_driver = {str(row["driver"]): row.get("team") for row in champion_rows}
    return [
        {"driver": driver, "team": team_by_driver[driver], "position": position}
        for position, driver in enumerate(order_by_result(champion_rows, latest.rows), start=1)
    ]


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for index in order[start : end + 1]:
            ranks[index] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(predicted_rows: Rows, actual_rows: Rows) -> float | None:
    """Spearman correlation over the drivers in both, as ``compute_prediction_accuracy``.

    ``None`` when fewer than two drivers match. A constant order scores 0.0, as there.
    """
    actual_by_driver = {str(row["driver"]): float(row["position"]) for row in actual_rows}
    pairs = [
        (float(row["position"]), actual_by_driver[str(row["driver"])])
        for row in predicted_rows
        if str(row["driver"]) in actual_by_driver
    ]
    if len(pairs) < 2:
        return None
    predicted = _average_ranks([p for p, _ in pairs])
    actual = _average_ranks([a for _, a in pairs])
    mean_p, mean_a = statistics.fmean(predicted), statistics.fmean(actual)
    covariance = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual, strict=True))
    spread = math.sqrt(
        sum((p - mean_p) ** 2 for p in predicted) * sum((a - mean_a) ** 2 for a in actual)
    )
    return covariance / spread if spread > 0 else 0.0


def likely_range_coverage(forecast_rows: Rows, actual_rows: Rows) -> float | None:
    """Share of matched drivers whose result fell inside the shown Likely range, ends included.

    Rows without ``likely_lo`` and ``likely_hi`` are skipped; ``None`` when none remain.
    """
    actual_by_driver = {str(row["driver"]): int(row["position"]) for row in actual_rows}
    inside = [
        int(row["likely_lo"]) <= actual_by_driver[str(row["driver"])] <= int(row["likely_hi"])
        for row in forecast_rows
        if str(row["driver"]) in actual_by_driver
        and row.get("likely_lo") is not None
        and row.get("likely_hi") is not None
    ]
    return sum(inside) / len(inside) if inside else None


def weekend_value(checkpoint_values: Iterable[float | None]) -> float | None:
    """Mean of one weekend's scored checkpoint values; ``None`` when none were scored."""
    values = [value for value in checkpoint_values if value is not None]
    return statistics.fmean(values) if values else None


def _t_statistic(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    if sd == 0:
        return 0.0 if mean == 0 else math.copysign(math.inf, mean)
    return mean / (sd / math.sqrt(len(values)))


def sequential_verdict(
    weekend_values: Sequence[float], *, null: str, season_complete: bool
) -> Verdict:
    """Call ``better`` or ``worse`` at the first weekend, from the 4th, whose |t| crosses.

    ``weekend_values`` are one champion's compared weekends in calendar order; a new
    champion starts a new list. ``null`` is the series' null, ``MARGIN_NULL[target]`` for
    a margin. Raises ``ValueError`` on a value that is not finite or on more weekends than
    the threshold was calibrated for.
    """
    critical = _CRITICAL_T[null]
    values = [float(value) for value in weekend_values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("weekend values must be finite")
    if len(values) > MAX_WEEKENDS:
        raise ValueError(f"the threshold covers {MAX_WEEKENDS} weekends; start a new list")
    for weekends in range(FIRST_VERDICT_WEEKEND, len(values) + 1):
        t = _t_statistic(values[:weekends])
        if abs(t) >= critical[weekends]:
            return Verdict(
                status="better" if t > 0 else "worse",
                weekends=len(values),
                mean=statistics.fmean(values[:weekends]),
                t_statistic=t,
                critical_t=critical[weekends],
                called_at=weekends,
            )
    return Verdict(
        status="noise" if season_complete else "provisional",
        weekends=len(values),
        mean=statistics.fmean(values) if values else None,
        t_statistic=_t_statistic(values) if len(values) >= 2 else None,
        critical_t=critical.get(len(values)),
    )


def coverage_verdict(weekend_values: Sequence[float], *, season_complete: bool) -> Verdict:
    """The same rule on coverage minus the 50% claim, under ``COVERAGE_NULL``.

    Only ``too_narrow`` is decisive. A range that covers more than it claims is reported as ``wider_than_claimed``: whole
    places make it cover 54 to 60% on the 2026 replay, which is honest, not a failure.
    """
    verdict = sequential_verdict(
        weekend_values, null=COVERAGE_NULL, season_complete=season_complete
    )
    renamed = {"worse": "too_narrow", "better": "wider_than_claimed"}
    return Verdict(
        status=renamed.get(verdict.status, verdict.status),
        weekends=verdict.weekends,
        mean=verdict.mean,
        t_statistic=verdict.t_statistic,
        critical_t=verdict.critical_t,
        called_at=verdict.called_at,
    )
