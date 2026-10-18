"""Tests for blending the actual sprint qualifying order into the main qualifying forecast."""

from src.predictors.baseline.qualifying_preparation import blend_sprint_qualifying_result

# Three simulations. The model has AAA ahead of BBB ahead of CCC every time.
RECORDS = {"AAA": [1, 1, 1], "BBB": [2, 2, 2], "CCC": [3, 3, 3], "DDD": [4, 4, 4]}
# Sprint qualifying finished the other way round. DDD set no time.
SPRINT = [
    {"driver": "CCC", "team": "C", "position": 1},
    {"driver": "BBB", "team": "B", "position": 2},
    {"driver": "AAA", "team": "A", "position": 3},
]


def _blend(weight, stage="main", checkpoint="SQ", result=SPRINT):
    return blend_sprint_qualifying_result(
        RECORDS,
        result,
        weight=weight,
        qualifying_stage=stage,
        checkpoint_session_name=checkpoint,
    )


def test_full_weight_copies_the_sprint_qualifying_order():
    blended = _blend(1.0)

    assert blended["CCC"] == [1, 1, 1]
    assert blended["BBB"] == [2, 2, 2]
    assert blended["AAA"] == [3, 3, 3]
    # No sprint qualifying place: the simulated one is kept.
    assert blended["DDD"] == [4, 4, 4]


def test_every_simulation_stays_a_permutation():
    blended = _blend(0.7)

    for sim in range(3):
        assert sorted(places[sim] for places in blended.values()) == [1, 2, 3, 4]


def test_before_sprint_qualifying_the_result_is_never_read():
    for checkpoint in ("PRE", "FP1", None):
        assert _blend(1.0, checkpoint=checkpoint) is RECORDS


def test_only_the_main_qualifying_forecast_is_blended():
    assert _blend(1.0, stage="sprint") is RECORDS


def test_shipped_at_point_seven_and_inert_without_a_result():
    from src.utils import config_loader

    assert (
        config_loader.get("baseline_predictor.qualifying.sprint_quali_result_blend_weight") == 0.7
    )
    assert _blend(0.0) is RECORDS
    assert _blend(1.0, result=None) is RECORDS
    assert _blend(1.0, checkpoint="SPRINT")["CCC"] == [1, 1, 1]
