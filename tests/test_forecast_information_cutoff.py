"""Information-cutoff tests: a forecast must not see sessions or races after its checkpoint.

Two leak routes are covered. Inside a weekend, the historical replay must build each
checkpoint before any later session is applied or read. Across weekends, current-season
team form must only admit races that precede the target, including when the target name
is missing from the schedule.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import pytest

import src.utils.historical_replay as replay_module
from src.predictors.baseline.data_mixin import BaselineDataMixin
from src.utils.historical_replay import (
    ReplayCheckpointRecord,
    build_checkpoint_prediction_results,
    checkpoint_sequence_for_weekend,
    run_historical_checkpoint_replay,
)

SCHEDULE = (
    ("Australian Grand Prix", "conventional"),
    ("Chinese Grand Prix", "sprint"),
    ("Japanese Grand Prix", "conventional"),
)


def _grid(*drivers: str) -> list[dict]:
    return [
        {"driver": driver, "team": "McLaren", "position": position}
        for position, driver in enumerate(drivers, start=1)
    ]


# --- Inside a weekend: the replay orchestration -------------------------------------


def _run_replay_recording_order(
    monkeypatch, tmp_path: Path, plan: list[dict], **replay_kwargs
) -> list[str]:
    """Run the replay loop with every data step stubbed, returning the call order."""
    events: list[str] = []
    sprint_races = {"Chinese Grand Prix"}

    def _apply(*, event_name, session_name, **_kwargs):
        events.append(f"apply {event_name} {session_name}")
        return True

    def _checkpoint(*, year, race_name, checkpoint_session, **_kwargs):
        events.append(f"checkpoint {race_name} {checkpoint_session}")
        return ReplayCheckpointRecord(
            year=year,
            race_name=race_name,
            checkpoint_session=checkpoint_session,
            weekend_format="stub",
            prediction_path="",
            state_summary_path="",
        )

    monkeypatch.setattr(replay_module, "_force_file_only_storage", nullcontext)
    monkeypatch.setattr(
        replay_module, "_prepare_output_root", lambda **_kwargs: tmp_path / "processed"
    )
    monkeypatch.setattr(replay_module, "_season_snapshot_plan", lambda year: plan)
    monkeypatch.setattr(
        replay_module, "is_sprint_weekend", lambda year, race_name: race_name in sprint_races
    )
    monkeypatch.setattr(replay_module, "_apply_session_update", _apply)
    monkeypatch.setattr(
        replay_module, "_walk_forward_team_strength_mapping", lambda **_kwargs: nullcontext()
    )
    monkeypatch.setattr(replay_module, "_build_race_checkpoint_record", _checkpoint)
    monkeypatch.setattr(
        replay_module,
        "update_from_sprint_race",
        lambda year, race_name, *_args, **_kwargs: events.append(f"sprint_update {race_name}"),
    )
    monkeypatch.setattr(
        replay_module,
        "update_from_race",
        lambda year, race_name, *_args, **_kwargs: events.append(f"race_update {race_name}"),
    )

    source = tmp_path / "source"
    source.mkdir()
    run_historical_checkpoint_replay(
        year=2026, source_processed_dir=source, output_root=tmp_path / "out", **replay_kwargs
    )
    return events


def test_normal_weekend_checkpoints_are_built_before_later_sessions(monkeypatch, tmp_path):
    """PRE sees no session, FPn sees through FPn, and Q/R land only after the last forecast."""
    plan = [
        {
            "event_name": "Australian Grand Prix",
            "sessions": ["FP1", "FP2", "FP3", "Q", "R"],
            "cache_dirs": [],
        }
    ]

    events = _run_replay_recording_order(monkeypatch, tmp_path, plan)

    assert events == [
        "checkpoint Australian Grand Prix PRE",
        "apply Australian Grand Prix FP1",
        "checkpoint Australian Grand Prix FP1",
        "apply Australian Grand Prix FP2",
        "checkpoint Australian Grand Prix FP2",
        "apply Australian Grand Prix FP3",
        "checkpoint Australian Grand Prix FP3",
        "apply Australian Grand Prix Q",
        "apply Australian Grand Prix R",
        "race_update Australian Grand Prix",
    ]


def test_sprint_weekend_checkpoints_are_built_before_later_sessions(monkeypatch, tmp_path):
    """The SQ forecast is built before the Sprint, Q and R sessions are applied."""
    plan = [
        {
            "event_name": "Chinese Grand Prix",
            "sessions": ["FP1", "SQ", "Sprint", "Q", "R"],
            "cache_dirs": [],
        }
    ]

    events = _run_replay_recording_order(monkeypatch, tmp_path, plan)

    assert events == [
        "checkpoint Chinese Grand Prix PRE",
        "apply Chinese Grand Prix FP1",
        "checkpoint Chinese Grand Prix FP1",
        "apply Chinese Grand Prix SQ",
        "checkpoint Chinese Grand Prix SQ",
        "apply Chinese Grand Prix SPRINT",
        "apply Chinese Grand Prix Q",
        "apply Chinese Grand Prix R",
        "sprint_update Chinese Grand Prix",
        "race_update Chinese Grand Prix",
    ]


def test_next_weekend_starts_only_after_the_previous_race_update(monkeypatch, tmp_path):
    """Race k+1's PRE forecast sees race k's result, and nothing of race k+1."""
    plan = [
        {"event_name": "Australian Grand Prix", "sessions": ["FP1", "R"], "cache_dirs": []},
        {"event_name": "Japanese Grand Prix", "sessions": ["FP1", "R"], "cache_dirs": []},
    ]

    events = _run_replay_recording_order(monkeypatch, tmp_path, plan)

    assert events.index("race_update Australian Grand Prix") < events.index(
        "checkpoint Japanese Grand Prix PRE"
    )
    assert events.index("checkpoint Japanese Grand Prix PRE") < events.index(
        "apply Japanese Grand Prix FP1"
    )


# --- Inside a weekend: which actual results a checkpoint reads -----------------------


class _StubPredictor:
    """Returns a fixed predicted grid so only the actual-results reads are under test."""

    def predict_qualifying(self, **_kwargs):
        return {"grid": _grid("NOR", "VER", "PIA")}

    def predict_race(self, **_kwargs):
        return {"finish_order": []}

    def predict_sprint_race(self, **_kwargs):
        return {"finish_order": []}


@pytest.mark.parametrize(
    ("is_sprint", "checkpoint", "allowed_reads"),
    [
        (False, "PRE", set()),
        (False, "FP1", set()),
        (False, "FP2", set()),
        (False, "FP3", set()),
        (True, "PRE", set()),
        (True, "FP1", set()),
        (True, "SQ", {"SQ"}),
    ],
)
def test_checkpoint_reads_no_actual_result_from_a_later_session(
    monkeypatch, is_sprint, checkpoint, allowed_reads
):
    """A checkpoint may read only the actual classifications of sessions already run.

    Every actual read is recorded; the race's post-penalty starting grid counts as a
    read of R, because it is only known once qualifying has happened.
    """
    assert checkpoint in checkpoint_sequence_for_weekend(is_sprint)
    reads: list[str] = []

    def _session(*, session_name, **_kwargs):
        reads.append(str(session_name).upper())
        return _grid("VER", "NOR", "PIA")

    def _starting_grid(**_kwargs):
        reads.append("R_STARTING_GRID")
        return _grid("VER", "NOR", "PIA")

    monkeypatch.setattr(replay_module, "_fetch_actual_session_results", _session)
    monkeypatch.setattr(replay_module, "_fetch_actual_starting_grid", _starting_grid)

    build_checkpoint_prediction_results(
        predictor=_StubPredictor(),
        year=2026,
        race_name="Chinese Grand Prix" if is_sprint else "Australian Grand Prix",
        checkpoint_session=checkpoint,
        weather="dry",
        is_sprint=is_sprint,
        actual_cache={},
    )

    assert set(reads) == allowed_reads


# --- Across weekends: current-season form ---------------------------------------------


class _FormPredictor(BaselineDataMixin):
    def __init__(self, data_dir: Path, saved_records: list[dict]):
        self.data_dir = Path(data_dir)
        self.artifact_store = None
        super().__init__()
        self.season_year = 2026
        self.teams = {"Ferrari": {"current_season_performance": [0.1, 0.5, 0.9]}}
        self._saved_records = saved_records

    def _load_saved_actual_race_scores(self, target_year: int) -> list[dict]:
        return self._saved_records

    def _resolve_prediction_target_year(self) -> int:
        return 2026


def _saved_records(japan_score: float) -> list[dict]:
    """Saved actuals for all three rounds, as a retrospective run would find them."""
    return [
        {"race_name": "Australian Grand Prix", "team_scores": {"Ferrari": 0.1}},
        {"race_name": "Chinese Grand Prix", "team_scores": {"Ferrari": 0.5}},
        {"race_name": "Japanese Grand Prix", "team_scores": {"Ferrari": japan_score}},
    ]


@pytest.fixture
def schedule(monkeypatch):
    monkeypatch.setattr("src.utils.weekend.get_schedule_rows", lambda year: SCHEDULE)


def test_future_saved_actuals_do_not_change_an_earlier_race_forecast(tmp_path, schedule):
    """Changing round 3's result must leave round 2's saved-actual form untouched."""
    baseline = _FormPredictor(tmp_path, _saved_records(japan_score=0.9))
    perturbed = _FormPredictor(tmp_path, _saved_records(japan_score=0.0))

    kwargs = {"team_name": "Ferrari", "target_year": 2026, "race_name": "Chinese Grand Prix"}
    assert baseline._get_saved_actual_observations(**kwargs) == [(0.1, 1)]
    assert perturbed._get_saved_actual_observations(**kwargs) == [(0.1, 1)]


def test_live_form_is_capped_to_races_before_the_target(tmp_path, schedule):
    """Live floats carry no race labels, so the cap is what keeps later rounds out."""
    predictor = _FormPredictor(tmp_path, saved_records=[])

    observations = predictor._get_current_season_observations(
        team_name="Ferrari",
        team_data=predictor.teams["Ferrari"],
        race_name="Chinese Grand Prix",
    )

    assert observations == [0.1]


def test_a_target_missing_from_the_schedule_admits_no_saved_actuals(tmp_path, schedule):
    """An alias or unlisted venue cannot be ordered, so no saved race may count as earlier."""
    predictor = _FormPredictor(tmp_path, _saved_records(japan_score=0.9))

    observations = predictor._get_saved_actual_observations(
        team_name="Ferrari", target_year=2026, race_name="Mystery Grand Prix"
    )

    assert observations == []


def test_a_target_missing_from_the_schedule_admits_no_live_form(tmp_path, schedule):
    """Without an order for the target, the live list cannot be cut to prior races."""
    predictor = _FormPredictor(tmp_path, saved_records=[])

    observations = predictor._get_current_season_observations(
        team_name="Ferrari",
        team_data=predictor.teams["Ferrari"],
        race_name="Mystery Grand Prix",
    )

    assert observations == []
    assert predictor._get_contextual_races_completed("Mystery Grand Prix") == 0


def test_an_unloadable_schedule_admits_no_current_season_form(tmp_path, monkeypatch):
    """If the schedule cannot load, every target is unordered and must fail closed."""

    def _unavailable(year):
        raise OSError("schedule unavailable")

    monkeypatch.setattr("src.utils.weekend.get_schedule_rows", _unavailable)
    predictor = _FormPredictor(tmp_path, _saved_records(japan_score=0.9))

    saved = predictor._get_saved_actual_observations(
        team_name="Ferrari", target_year=2026, race_name="Chinese Grand Prix"
    )
    live = predictor._get_current_season_observations(
        team_name="Ferrari",
        team_data=predictor.teams["Ferrari"],
        race_name="Chinese Grand Prix",
    )

    assert saved == []
    assert live == []


def _two_race_plan() -> list[dict]:
    return [
        {"event_name": "Australian Grand Prix", "sessions": ["Q", "R"], "cache_dirs": []},
        {"event_name": "Japanese Grand Prix", "sessions": ["Q", "R"], "cache_dirs": []},
    ]


def test_through_round_stops_the_replay_at_the_pinned_window(monkeypatch, tmp_path):
    """A newly cached weekend past the pinned round is neither forecast nor applied."""
    events = _run_replay_recording_order(monkeypatch, tmp_path, _two_race_plan(), through_round=1)

    assert not any("Japanese Grand Prix" in event for event in events)
    assert events[-1] == "race_update Australian Grand Prix"


@pytest.mark.parametrize("through_round", [0, 3])
def test_through_round_outside_the_cached_season_fails(monkeypatch, tmp_path, through_round):
    """A window the cache cannot fill must fail, not silently replay a shorter season."""
    with pytest.raises(ValueError, match="through_round"):
        _run_replay_recording_order(
            monkeypatch, tmp_path, _two_race_plan(), through_round=through_round
        )


def _measured_pace(monkeypatch, tmp_path: Path, payload: dict, race_name: str | None):
    """Load measured team pace for ``race_name`` from a synthetic artifact."""
    from src.utils import lap_by_lap_simulator as simulator_module

    pace_dir = tmp_path / "data" / "processed" / "team_race_pace"
    pace_dir.mkdir(parents=True)
    (pace_dir / "2026_team_race_pace.json").write_text(json.dumps(payload))
    monkeypatch.setattr(simulator_module, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.utils.weekend.get_schedule_rows", lambda year: SCHEDULE)
    simulator_module._load_measured_team_pace_deltas.cache_clear()
    try:
        return simulator_module._load_measured_team_pace_deltas(2026, race_name)
    finally:
        simulator_module._load_measured_team_pace_deltas.cache_clear()


PACE_BY_RACE = {
    "races": {
        "Australian Grand Prix": {"Fast": 0.0, "Slow": 1.0},
        "Chinese Grand Prix": {"Fast": 0.0, "Slow": 3.0},
        "Japanese Grand Prix": {"Fast": 5.0, "Slow": 0.0},
    }
}


def test_measured_team_pace_reads_only_races_before_the_target(monkeypatch, tmp_path):
    """The Japanese forecast averages Australia and China, never Japan itself."""
    deltas = _measured_pace(monkeypatch, tmp_path, PACE_BY_RACE, "Japanese Grand Prix")

    assert deltas == pytest.approx({"Fast": 1.0, "Slow": -1.0})


@pytest.mark.parametrize("race_name", ["Australian Grand Prix", "Bahrain Grand Prix", None])
def test_measured_team_pace_without_prior_races_falls_back(monkeypatch, tmp_path, race_name):
    """Round 1, a race missing from the schedule, or no target has no admissible pace."""
    assert _measured_pace(monkeypatch, tmp_path, PACE_BY_RACE, race_name) is None


def test_season_average_pace_artifact_is_not_used(monkeypatch, tmp_path):
    """An artifact without per-race gaps cannot be cut off, so it must not be read."""
    season_average = {"teams": {"Fast": {"gap_s": 0.0}, "Slow": {"gap_s": 1.0}}}

    assert _measured_pace(monkeypatch, tmp_path, season_average, "Japanese Grand Prix") is None


def _calibration_rows(year: int, race_name: str, slope: float, kinds=("qualifying", "race")):
    """Rows whose team_target_s is exactly ``slope`` times the centred team strength."""
    return [
        {
            "year": year,
            "race_name": race_name,
            "session_kind": kind,
            "team_strength_same_session": strength,
            "team_target_s": slope * (strength - 0.5),
        }
        for kind in kinds
        for strength in (0.0, 0.5, 1.0)
    ]


def _walk_forward_mapping(
    monkeypatch, tmp_path: Path, race_name: str, previous_era_only: bool = False
) -> dict:
    """Enter the replay's mapping scope for ``race_name`` and return what it points at."""
    import pandas as pd

    from src.models.team_strength_mapping import TEAM_STRENGTH_SECONDS_MAPPING_PATH_ENV

    rows = (
        _calibration_rows(2025, "Abu Dhabi Grand Prix", 1.0)
        + _calibration_rows(2026, "Australian Grand Prix", 2.0)
        + _calibration_rows(2026, "Chinese Grand Prix", 4.0)
        + _calibration_rows(2026, "Japanese Grand Prix", 100.0)
    )
    mapping_dir = tmp_path / "processed" / "team_strength_seconds_mapping"
    mapping_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(mapping_dir / "calibration_observations.csv", index=False)
    monkeypatch.setattr(replay_module, "get_schedule_rows", lambda year: SCHEDULE)
    monkeypatch.delenv(TEAM_STRENGTH_SECONDS_MAPPING_PATH_ENV, raising=False)

    with replay_module._walk_forward_team_strength_mapping(
        year=2026,
        race_name=race_name,
        processed_dir=tmp_path / "processed",
        output_root=tmp_path / "out",
        previous_era_only=previous_era_only,
    ):
        artifact = json.loads(Path(os.environ[TEAM_STRENGTH_SECONDS_MAPPING_PATH_ENV]).read_text())
    assert TEAM_STRENGTH_SECONDS_MAPPING_PATH_ENV not in os.environ
    return artifact


def test_replay_mapping_is_fitted_only_on_earlier_races(monkeypatch, tmp_path):
    """Japan's slope comes from Australia and China; Japan's own rows would make it 100."""
    artifact = _walk_forward_mapping(monkeypatch, tmp_path, "Japanese Grand Prix")

    assert artifact["prior_races"] == ["Australian Grand Prix", "Chinese Grand Prix"]
    for kind in ("qualifying", "race"):
        assert artifact["mappings"][kind]["slope_s_per_unit"] == pytest.approx(3.0)
        assert artifact["mappings"][kind]["training_years"] == [2026]


def test_replay_mapping_for_round_one_uses_the_previous_era(monkeypatch, tmp_path):
    """No 2026 race precedes Australia, so it gets the fit the live model had then."""
    artifact = _walk_forward_mapping(monkeypatch, tmp_path, "Australian Grand Prix")

    assert artifact["prior_races"] == []
    assert artifact["mappings"]["qualifying"]["slope_s_per_unit"] == pytest.approx(1.0)
    assert artifact["mappings"]["qualifying"]["training_years"] == [2025]


def test_replay_mapping_refuses_a_race_it_cannot_order(monkeypatch, tmp_path):
    """A target missing from the schedule cannot be cut off, so the replay stops."""
    with pytest.raises(ValueError, match="not in the 2026 schedule"):
        _walk_forward_mapping(monkeypatch, tmp_path, "Bahrain Grand Prix")


def test_previous_era_arm_ignores_the_current_season(monkeypatch, tmp_path):
    """The A/B arm uses the 2022 to 2025 fit at every round, even with 2026 rows available."""
    artifact = _walk_forward_mapping(
        monkeypatch, tmp_path, "Japanese Grand Prix", previous_era_only=True
    )

    assert artifact["mappings"]["race"]["slope_s_per_unit"] == pytest.approx(1.0)
    assert artifact["mappings"]["race"]["training_years"] == [2025]
