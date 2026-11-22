"""Calibrate the 2027 protocol's verdict thresholds and print their critical t tables.

The protocol (``docs/MODEL_LEDGER.md``, 2026-11-21 and 2026-11-22) checks a target after
every compared weekend from the 4th and calls a verdict when a one-sample t-test of the
weekend values gives p at or below a threshold. Each threshold keeps false verdicts at 5%
per series per season, over up to 24 looks, under the null its series follows:

- ``independent``: independent normal weekends. Grand Prix and sprint margins: on the
  2026 replay they alternate (lag-1 autocorrelation -0.69), which only lowers the rate.
- ``drifting``: AR(1) weekends with autocorrelation 0.66. Their 16-weekend samples show
  0.48 on average, the worst of the 2026 replay's detrended qualifying margins and Likely
  range coverage (0.46). Qualifying and sprint qualifying margins, and all coverage.

Both nulls are symmetric: when champion and naive are exchangeable a weekend's margin is
as likely to be -d as +d, and coverage at its 50% claim is binomial. The 2026 skewness
(up to 0.84) belongs to the champion's advantage; the last stress line shows its cost.

``src/analysis/prospective_protocol.py`` hard-codes the critical |t| (the cron cannot
load scipy); its tests check them against scipy.

    uv run python scripts/calibrate_protocol_threshold.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.prospective_protocol import (  # noqa: E402
    FIRST_VERDICT_WEEKEND,
    MAX_WEEKENDS,
    VERDICT_P_THRESHOLD,
)

# (lag-1 autocorrelation, skewness) of each null's weekend values.
NULLS = {"independent": (0.0, 0.0), "drifting": (0.66, 0.0)}


def ar1_seasons(
    rng: np.random.Generator, seasons: int, rho: float, skewness: float = 0.0
) -> np.ndarray:
    """Zero-mean, unit-variance AR(1) seasons; gamma innovations give the skewness."""
    shape = (seasons, MAX_WEEKENDS)
    if skewness:
        # An AR(1) process carries (1 - rho^2)^1.5 / (1 - rho^3) of its innovations' skewness.
        k = (2 * (1 - rho**2) ** 1.5 / (skewness * (1 - rho**3))) ** 2
        noise = (rng.gamma(k, 1.0, shape) - k) / np.sqrt(k)
    else:
        noise = rng.standard_normal(shape)
    values = np.empty(shape)
    values[:, 0] = noise[:, 0]
    for week in range(1, MAX_WEEKENDS):
        values[:, week] = rho * values[:, week - 1] + np.sqrt(1 - rho**2) * noise[:, week]
    return values


def smallest_p_per_season(values: np.ndarray) -> np.ndarray:
    """Smallest two-sided t-test p over the verdict looks, one per simulated season."""
    n = np.arange(1, values.shape[1] + 1)
    mean = np.cumsum(values, axis=1) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = np.sqrt((np.cumsum(values * values, axis=1) - n * mean * mean) / (n - 1))
        t = mean / (sd / np.sqrt(n))
    smallest = np.ones(values.shape[0])
    for weekends in range(FIRST_VERDICT_WEEKEND, values.shape[1] + 1):
        p = 2 * stats.t.sf(np.abs(t[:, weekends - 1]), weekends - 1)
        smallest = np.minimum(smallest, p)
    return smallest


def main() -> int:
    """Print each null's calibrated threshold, false-verdict rates and critical t."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    for null, (rho, skewness) in NULLS.items():
        smallest = smallest_p_per_season(ar1_seasons(rng, args.seasons, rho, skewness))
        threshold = VERDICT_P_THRESHOLD[null]
        print(
            f"{null}: calibrated p <= {np.quantile(smallest, 0.05):.3g}; in force "
            f"p <= {threshold:.3g}, false verdicts {np.mean(smallest <= threshold):.2%}"
        )
    print("False verdicts per series per season at each threshold:")
    for label, rho, skewness in (
        ("independent", 0.0, 0.0),
        ("alternating, rho -0.7", -0.7, 0.0),
        ("drifting, rho 0.3", 0.3, 0.0),
        ("drifting, rho 0.66", 0.66, 0.0),
        ("drifting, rho 0.66, skew 1.75", 0.66, 1.75),
        ("drifting, rho 0.8", 0.8, 0.0),
    ):
        smallest = smallest_p_per_season(ar1_seasons(rng, args.seasons // 4, rho, skewness))
        rates = ", ".join(
            f"{null} {np.mean(smallest <= threshold):.2%}"
            for null, threshold in VERDICT_P_THRESHOLD.items()
        )
        print(f"  {label:30s} {rates}")
    for null, threshold in VERDICT_P_THRESHOLD.items():
        print(f"Critical |t|, {null} (degrees of freedom = weekends - 1):")
        for weekends in range(FIRST_VERDICT_WEEKEND, MAX_WEEKENDS + 1):
            print(f"  {weekends:2d}: {stats.t.isf(threshold / 2, weekends - 1):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
