"""Tests for the replay-arm comparison tool's verdict logic."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.compare_replay_arms import (
    MetricRow,
    MetricsByCheckpoint,
    _bootstrap_ci,
    analyze_arm,
    assign_verdicts,
)

TARGET = "main_qualifying"


def _by_target(values: list[float]) -> dict[str, MetricsByCheckpoint]:
    """Build a tiny target -> checkpoint -> metrics dict from a list of overall_mae values."""
    checkpoints: MetricsByCheckpoint = {
        (f"Race {i}", "PRE"): {"overall_mae": value} for i, value in enumerate(values)
    }
    return {TARGET: checkpoints}


def _overall_mae_row(rows_by_target: dict[str, list[MetricRow]]) -> MetricRow:
    """Pull out the single overall_mae row from an analyze_arm result."""
    return next(row for row in rows_by_target[TARGET] if row.metric == "overall_mae")


def test_ci_excluding_zero_verdicts_better_or_worse() -> None:
    """A constant, non-zero delta gives a CI that excludes zero and a direction verdict."""
    baseline = _by_target([1.0] * 8)
    worse_candidate = _by_target([1.05] * 8)
    better_candidate = _by_target([0.95] * 8)

    worse_rows = analyze_arm(baseline, worse_candidate, [TARGET], iterations=200)
    assign_verdicts(worse_rows, None)
    assert _overall_mae_row(worse_rows).verdict == "worse"

    better_rows = analyze_arm(baseline, better_candidate, [TARGET], iterations=200)
    assign_verdicts(better_rows, None)
    assert _overall_mae_row(better_rows).verdict == "better"


def test_ci_including_zero_is_noise() -> None:
    """Deltas that straddle zero across checkpoints verdict as noise."""
    baseline = _by_target([1.0] * 8)
    candidate = _by_target([1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9])

    rows = analyze_arm(baseline, candidate, [TARGET], iterations=200)
    assign_verdicts(rows, None)
    assert _overall_mae_row(rows).verdict == "noise"


def test_delta_below_seed_floor_is_unresolvable() -> None:
    """A delta no bigger than the seed-noise floor for the same target+metric is unresolvable."""
    baseline = _by_target([1.0] * 8)
    floor_candidate = _by_target([1.10] * 8)  # seed-noise floor delta = +0.10
    small_candidate = _by_target([1.05] * 8)  # candidate delta = +0.05, at/below the floor

    floor_rows = analyze_arm(baseline, floor_candidate, [TARGET], iterations=200)
    rows = analyze_arm(baseline, small_candidate, [TARGET], iterations=200)
    assign_verdicts(rows, floor_rows)

    assert _overall_mae_row(rows).verdict == "unresolvable (below seed floor)"


def test_without_seed_floor_never_unresolvable() -> None:
    """The same small delta verdicts by CI alone when no seed floor is supplied."""
    baseline = _by_target([1.0] * 8)
    small_candidate = _by_target([1.05] * 8)

    rows = analyze_arm(baseline, small_candidate, [TARGET], iterations=200)
    assign_verdicts(rows, None)

    assert _overall_mae_row(rows).verdict == "worse"


def test_all_checkpoints_tied_is_identical_not_unresolvable() -> None:
    """An arm whose every checkpoint matches the baseline changed nothing at all.

    This is the `never activated` case from the model ledger: a variant that a runtime
    guard made champion-identical. It must not be filed under the seed floor, because
    "we could not measure this" and "this provably did nothing" are different findings.
    """
    baseline = _by_target([1.0, 2.0, 3.0, 4.0])
    identical = _by_target([1.0, 2.0, 3.0, 4.0])
    floor = analyze_arm(baseline, _by_target([1.5, 2.5, 3.5, 4.5]), [TARGET], iterations=200)

    rows = analyze_arm(baseline, identical, [TARGET], iterations=200)
    assign_verdicts(rows, floor)
    row = _overall_mae_row(rows)

    assert row.tied == row.n_paired == 4
    assert row.verdict == "identical (never activated)"


def test_seed_floor_gates_on_interval_not_point_estimate() -> None:
    """The floor threshold is the widest bound of the seed pair's CI, not its mean.

    One seed pair's observed shift is a single draw. Gating on that point estimate lets
    a candidate whose delta sits inside the plausible seed range through as a real result.
    """
    baseline = _by_target([1.0] * 8)
    # Floor deltas straddle zero: small mean, wide interval.
    floor = analyze_arm(baseline, _by_target([1.1, 0.9] * 4), [TARGET], iterations=400)
    floor_row = _overall_mae_row(floor)
    assert abs(floor_row.delta_mean) < max(abs(floor_row.ci_lo), abs(floor_row.ci_hi))

    # Candidate delta exceeds the floor's mean but sits inside its interval.
    candidate = analyze_arm(baseline, _by_target([1.05] * 8), [TARGET], iterations=400)
    assign_verdicts(candidate, floor)
    assert _overall_mae_row(candidate).verdict == "unresolvable (below seed floor)"


def test_clustered_bootstrap_matches_per_weekend_resampling() -> None:
    """Checkpoints sharing a weekend's delta must resample as one unit, not each on its own.

    Every checkpoint within a weekend carries the same delta here, so the clustered
    bootstrap over all 12 checkpoints (4 weekends x 3 checkpoints) must produce exactly
    the same resampled-mean distribution as bootstrapping the 4 weekend deltas directly:
    the effective sample size is the weekend count, not the checkpoint count.
    """
    weekend_deltas = [0.30, -0.10, 0.20, -0.40]
    deltas = np.array([delta for delta in weekend_deltas for _ in range(3)])
    weekends = np.array([f"W{i}" for i in range(len(weekend_deltas)) for _ in range(3)])
    collapsed_deltas = np.array(weekend_deltas)
    collapsed_weekends = np.array([f"W{i}" for i in range(len(weekend_deltas))])

    clustered_ci = _bootstrap_ci(deltas, weekends, 2000, np.random.default_rng(0))
    collapsed_ci = _bootstrap_ci(
        collapsed_deltas, collapsed_weekends, 2000, np.random.default_rng(0)
    )

    assert clustered_ci == pytest.approx(collapsed_ci)


def test_clustered_bootstrap_wider_than_naive_checkpoint_bootstrap() -> None:
    """Treating within-weekend checkpoints as independent understates the true CI.

    Same perfectly-correlated-within-weekend data as above, but compared against the
    old (pre-fix) behaviour of resampling every checkpoint independently: with only 4
    independent weekends behind 12 checkpoints, the clustered CI must be wider.
    """
    weekend_deltas = [0.30, -0.10, 0.20, -0.40]
    deltas = np.array([delta for delta in weekend_deltas for _ in range(3)])
    weekends = np.array([f"W{i}" for i in range(len(weekend_deltas)) for _ in range(3)])

    clustered_lo, clustered_hi = _bootstrap_ci(deltas, weekends, 4000, np.random.default_rng(0))

    # Old behaviour: resample individual checkpoints as if independent.
    rng = np.random.default_rng(0)
    n = len(deltas)
    sample_indices = rng.integers(0, n, size=(4000, n))
    naive_means = deltas[sample_indices].mean(axis=1)
    naive_lo, naive_hi = np.percentile(naive_means, [2.5, 97.5])

    assert (clustered_hi - clustered_lo) > (naive_hi - naive_lo)


def test_bootstrap_ci_singleton_weekends_matches_old_iid_checkpoint_bootstrap() -> None:
    """One checkpoint per weekend (the existing tests' shape) is unchanged by the fix.

    When every weekend contributes exactly one checkpoint, resampling weekends and
    resampling checkpoints are the same operation, so the clustered CI must equal the
    old iid-checkpoint CI bit for bit.
    """
    deltas = np.array([0.1, -0.2, 0.3, -0.4, 0.05, -0.15, 0.25, -0.35])
    weekends = np.array([f"W{i}" for i in range(len(deltas))])

    clustered_ci = _bootstrap_ci(deltas, weekends, 500, np.random.default_rng(0))

    rng = np.random.default_rng(0)
    n = len(deltas)
    sample_indices = rng.integers(0, n, size=(500, n))
    naive_means = deltas[sample_indices].mean(axis=1)
    naive_ci = tuple(float(v) for v in np.percentile(naive_means, [2.5, 97.5]))

    assert clustered_ci == naive_ci
