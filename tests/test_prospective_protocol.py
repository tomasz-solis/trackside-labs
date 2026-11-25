"""Tests for the 2027 prospective protocol's scoring rules (plan row 45).

scipy is the independent oracle here; the module itself must not import it.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy import stats
from scripts.calibrate_protocol_threshold import NULLS, ar1_seasons

from src.analysis import prospective_protocol as protocol
from src.analysis.model_evaluation import compute_prediction_accuracy
from src.analysis.prospective_protocol import (
    COVERAGE_NULL,
    FIRST_VERDICT_WEEKEND,
    MARGIN_NULL,
    MAX_WEEKENDS,
    SESSION_KIND,
    VERDICT_P_THRESHOLD,
    OfficialResult,
    coverage_verdict,
    likely_range_coverage,
    same_kind_naive,
    sequential_verdict,
    spearman,
    weekend_value,
)


def test_every_series_has_a_calibrated_null():
    assert set(MARGIN_NULL) == set(SESSION_KIND)
    assert set(MARGIN_NULL.values()) | {COVERAGE_NULL} == set(VERDICT_P_THRESHOLD) == set(NULLS)
    assert set(protocol._CRITICAL_T) == set(VERDICT_P_THRESHOLD)


@pytest.mark.parametrize("null", sorted(VERDICT_P_THRESHOLD))
def test_critical_values_are_student_t_at_the_threshold(null):
    critical = protocol._CRITICAL_T[null]
    assert sorted(critical) == list(range(FIRST_VERDICT_WEEKEND, MAX_WEEKENDS + 1))
    for weekends, value in critical.items():
        expected = stats.t.isf(VERDICT_P_THRESHOLD[null] / 2, weekends - 1)
        assert value == pytest.approx(expected, abs=5e-7)


@pytest.mark.parametrize("null", sorted(VERDICT_P_THRESHOLD))
def test_the_threshold_keeps_false_verdicts_at_five_percent(null):
    rho, skewness = NULLS[null]
    values = ar1_seasons(np.random.default_rng(45), 200_000, rho, skewness)
    n = np.arange(1, MAX_WEEKENDS + 1)
    mean = np.cumsum(values, axis=1) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = np.sqrt((np.cumsum(values**2, axis=1) - n * mean**2) / (n - 1))
        t = mean / (sd / np.sqrt(n))
    crossed = np.zeros(len(values), dtype=bool)
    for weekends, critical in protocol._CRITICAL_T[null].items():
        crossed |= np.abs(t[:, weekends - 1]) >= critical

    assert 0.047 <= crossed.mean() <= 0.051


def _oracle(values: list[float], threshold: float) -> tuple[str, int | None]:
    for weekends in range(FIRST_VERDICT_WEEKEND, len(values) + 1):
        result = stats.ttest_1samp(values[:weekends], 0.0)
        if result.pvalue <= threshold:
            return ("better" if result.statistic > 0 else "worse"), weekends
    return "provisional", None


@pytest.mark.parametrize("null", sorted(VERDICT_P_THRESHOLD))
def test_the_verdict_matches_a_scipy_t_test_at_every_look(null):
    rng = random.Random(2027)
    checked = 0
    for _ in range(400):
        length = rng.randint(FIRST_VERDICT_WEEKEND, MAX_WEEKENDS)
        mean = rng.choice((-0.4, -0.05, 0.0, 0.05, 0.1, 0.4))
        values = [rng.gauss(mean, 0.14) for _ in range(length)]
        verdict = sequential_verdict(values, null=null, season_complete=False)
        assert (verdict.status, verdict.called_at) == _oracle(values, VERDICT_P_THRESHOLD[null])
        checked += verdict.called_at is not None
    assert checked > 50


def test_no_verdict_before_the_fourth_weekend():
    verdict = sequential_verdict([0.5, 0.6, 0.55], null="independent", season_complete=False)

    assert verdict.status == "provisional"
    assert verdict.called_at is None
    assert verdict.mean == pytest.approx(0.55)
    assert verdict.t_statistic is not None and verdict.critical_t is None


def test_a_season_that_ends_without_a_crossing_is_noise():
    values = [0.1, -0.1, 0.05, -0.02, 0.03, 0.0]
    assert sequential_verdict(values, null="independent", season_complete=True).status == "noise"
    assert (
        sequential_verdict(values, null="independent", season_complete=False).status
        == "provisional"
    )
    assert sequential_verdict([], null="independent", season_complete=False).mean is None


def test_a_called_verdict_stands_for_the_season():
    values = [0.30, 0.31, 0.29, 0.30, 0.31] + [-0.5] * 10
    verdict = sequential_verdict(values, null="independent", season_complete=True)

    assert verdict.status == "better"
    assert verdict.called_at == 4
    assert verdict.weekends == 15


def test_identical_weekends_are_decided_by_their_sign():
    assert (
        sequential_verdict([0.1] * 4, null="independent", season_complete=False).status == "better"
    )
    assert (
        sequential_verdict([-0.1] * 4, null="independent", season_complete=False).status == "worse"
    )
    zero = sequential_verdict([0.0] * 5, null="independent", season_complete=False)
    assert (zero.status, zero.t_statistic) == ("provisional", 0.0)


@pytest.mark.parametrize("values", [[0.1, math.nan, 0.2, 0.1], [0.1, math.inf, 0.2, 0.1]])
def test_values_that_are_not_finite_are_refused(values):
    with pytest.raises(ValueError, match="finite"):
        sequential_verdict(values, null="independent", season_complete=False)


def test_more_weekends_than_calibrated_are_refused():
    with pytest.raises(ValueError, match="24 weekends"):
        sequential_verdict([0.0] * (MAX_WEEKENDS + 1), null="independent", season_complete=False)


# 2026 replay (C38, seed 42) weekend margins over the same-kind naive, 3 places.
_REPLAY_QUALIFYING = [0.013, -0.035, 0.076, 0.166, 0.032, 0.078, 0.01, -0.017]
_REPLAY_QUALIFYING += [-0.005, 0.032, 0.014, 0.064, 0.19, 0.274, 0.198, 0.094]
_REPLAY_GRAND_PRIX = [-0.069, 0.471, 0.061, 0.244, 0.129, 0.404, 0.23, 0.059]
_REPLAY_GRAND_PRIX += [0.369, 0.168, 0.141, 0.29, 0.17, 0.332, 0.271, 0.079]


def test_the_2026_replay_calls_the_grand_prix_and_holds_drifting_qualifying():
    grand_prix = sequential_verdict(
        _REPLAY_GRAND_PRIX, null=MARGIN_NULL["grand_prix_race"], season_complete=False
    )
    qualifying = sequential_verdict(
        _REPLAY_QUALIFYING, null=MARGIN_NULL["main_qualifying"], season_complete=False
    )
    independent = sequential_verdict(_REPLAY_QUALIFYING, null="independent", season_complete=False)

    assert (grand_prix.status, grand_prix.called_at) == ("better", 10)
    assert (qualifying.status, qualifying.called_at) == ("provisional", None)
    assert (independent.status, independent.called_at) == ("better", 16)


def test_coverage_names_only_a_too_narrow_range_as_a_failure():
    assert coverage_verdict([-0.2] * 4, season_complete=False).status == "too_narrow"
    assert coverage_verdict([0.1] * 4, season_complete=False).status == "wider_than_claimed"
    assert coverage_verdict([0.0, 0.1], season_complete=True).status == "noise"


def test_coverage_is_judged_as_a_drifting_series():
    # Crosses the independent threshold at weekend 16; coverage drifts, so it holds.
    drifting_shortfall = [-value for value in _REPLAY_QUALIFYING]

    assert coverage_verdict(drifting_shortfall, season_complete=False).status == "provisional"


def test_a_weekend_value_is_the_mean_of_its_scored_checkpoints():
    assert weekend_value([0.1, None, 0.3]) == pytest.approx(0.2)
    assert weekend_value([None]) is None


# Same-kind naive on one sprint weekend after a normal one, times in UTC.
_T0 = datetime(2027, 3, 1, 12, tzinfo=UTC)
_CHAMPION = [
    {"driver": "AAA", "team": "Alpha", "position": 1},
    {"driver": "BBB", "team": "Beta", "position": 2},
    {"driver": "CCC", "team": "Gamma", "position": 3},
    {"driver": "NEW", "team": "Gamma", "position": 4},
]


def _result(
    target: str, hours: int, order: list[str], issued_hours: float | None = None
) -> OfficialResult:
    rows = [{"driver": driver, "position": index} for index, driver in enumerate(order, start=1)]
    start = _T0 + timedelta(hours=hours)
    issued = start if issued_hours is None else _T0 + timedelta(hours=issued_hours)
    return OfficialResult(target, "event", start, rows, issued)


_RESULTS = [
    _result("main_qualifying", 0, ["CCC", "BBB", "AAA"]),  # previous event Q
    _result("grand_prix_race", 24, ["BBB", "CCC", "AAA"]),  # previous event R
    _result("sprint_qualifying", 144, ["AAA", "CCC", "BBB"]),  # this weekend SQ
    _result("sprint_race", 162, ["CCC", "AAA", "BBB"]),  # this weekend Sprint
    _result("main_qualifying", 166, ["BBB", "AAA", "CCC"]),  # this weekend Q
    _result("grand_prix_race", 190, ["AAA", "BBB", "CCC"]),  # this weekend R
]


def _naive(target: str, cutoff_hours: float) -> list[str]:
    rows = same_kind_naive(
        target=target,
        champion_rows=_CHAMPION,
        results=_RESULTS,
        information_cutoff_at=_T0 + timedelta(hours=cutoff_hours),
    )
    assert rows is not None
    assert [row["position"] for row in rows] == [1, 2, 3, 4]
    return [row["driver"] for row in rows]


def test_the_naive_is_the_latest_same_kind_result_before_the_cutoff():
    pre_sq = 143  # before sprint qualifying: only the previous event is known
    after_sq = 162 - 1 / 3600  # the SQ checkpoint's cutoff: one second before the sprint
    after_q = 190 - 1 / 3600  # the Q checkpoint's cutoff: one second before the race

    assert _naive("sprint_qualifying", pre_sq) == ["CCC", "BBB", "AAA", "NEW"]
    assert _naive("main_qualifying", after_sq) == ["AAA", "CCC", "BBB", "NEW"]
    assert _naive("sprint_race", after_sq) == ["BBB", "CCC", "AAA", "NEW"]
    assert _naive("grand_prix_race", after_q) == ["CCC", "AAA", "BBB", "NEW"]


def test_a_result_at_the_cutoff_counts_and_one_after_it_does_not():
    assert _naive("main_qualifying", 144) == ["AAA", "CCC", "BBB", "NEW"]
    assert _naive("main_qualifying", 144 - 1 / 3600) == ["CCC", "BBB", "AAA", "NEW"]


def test_the_naive_keeps_the_champion_teams_and_reports_no_result():
    rows = same_kind_naive(
        target="grand_prix_race",
        champion_rows=_CHAMPION,
        results=_RESULTS,
        information_cutoff_at=_T0 + timedelta(hours=30),
    )
    assert rows is not None
    assert {row["driver"]: row["team"] for row in rows} == {
        "AAA": "Alpha",
        "BBB": "Beta",
        "CCC": "Gamma",
        "NEW": "Gamma",
    }
    assert (
        same_kind_naive(
            target="main_qualifying",
            champion_rows=_CHAMPION,
            results=_RESULTS,
            information_cutoff_at=_T0 - timedelta(hours=1),
        )
        is None
    )


def test_a_cutoff_without_a_timezone_is_refused():
    with pytest.raises(ValueError, match="timezone"):
        same_kind_naive(
            target="main_qualifying",
            champion_rows=_CHAMPION,
            results=_RESULTS,
            information_cutoff_at=datetime(2027, 3, 9),
        )


def test_spearman_equals_the_accuracy_metric():
    rng = random.Random(7)
    drivers = [f"D{index:02d}" for index in range(22)]
    for _ in range(200):
        predicted_order = rng.sample(drivers, len(drivers))
        actual_order = rng.sample(drivers, rng.randint(2, len(drivers)))
        predicted = [{"driver": d, "position": i} for i, d in enumerate(predicted_order, start=1)]
        actual = [{"driver": d, "position": i} for i, d in enumerate(actual_order, start=1)]
        expected = compute_prediction_accuracy(predicted, actual)["spearman_rank"]
        assert spearman(predicted, actual) == pytest.approx(expected, abs=1e-12)


def test_spearman_needs_two_matched_drivers():
    assert spearman([{"driver": "AAA", "position": 1}], [{"driver": "AAA", "position": 1}]) is None


def test_likely_range_coverage_counts_the_ends_and_skips_rows_without_a_range():
    forecast = [
        {"driver": "AAA", "position": 1, "likely_lo": 1, "likely_hi": 3},
        {"driver": "BBB", "position": 2, "likely_lo": 1, "likely_hi": 3},
        {"driver": "CCC", "position": 3, "likely_lo": 2, "likely_hi": 4},
        {"driver": "DDD", "position": 4},
    ]
    actual = [
        {"driver": "AAA", "position": 3},
        {"driver": "BBB", "position": 4},
        {"driver": "CCC", "position": 2},
        {"driver": "DDD", "position": 1},
    ]
    assert likely_range_coverage(forecast, actual) == pytest.approx(2 / 3)
    assert likely_range_coverage(forecast[3:], actual) is None


def test_the_protocol_module_loads_no_scipy_or_pandas():
    import subprocess
    import sys

    code = (
        "import sys; import src.analysis.prospective_protocol; "
        "print(any(m in sys.modules for m in ('scipy', 'pandas', 'streamlit')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_a_correction_issued_after_the_cutoff_never_reaches_the_naive():
    """Codex 2026-11-23: SQ at 11:00, cutoff 12:00, reversed correction at 12:30."""
    original = _result("sprint_qualifying", 0, ["AAA", "BBB", "CCC"], issued_hours=0.5)
    correction = _result("sprint_qualifying", 0, ["CCC", "BBB", "AAA"], issued_hours=1.5)

    def naive(cutoff_hours: float) -> list[str]:
        rows = same_kind_naive(
            target="main_qualifying",
            champion_rows=_CHAMPION,
            results=[original, correction],
            information_cutoff_at=_T0 + timedelta(hours=cutoff_hours),
        )
        assert rows is not None
        return [row["driver"] for row in rows]

    assert naive(1.0) == ["AAA", "BBB", "CCC", "NEW"]
    assert naive(2.0) == ["CCC", "BBB", "AAA", "NEW"]
