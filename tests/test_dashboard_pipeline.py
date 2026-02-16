"""Tests for live prediction pipeline orchestration."""

import pytest

from src.dashboard import pages


def test_execute_live_prediction_pipeline_refresh_call_order(monkeypatch):
    call_order: list[str] = []

    monkeypatch.setattr(pages, "auto_update_if_needed", lambda: call_order.append("race_update"))
    monkeypatch.setattr(
        pages,
        "is_sprint_weekend",
        lambda year, race_name: (call_order.append("sprint_check"), True)[1],
    )

    def _practice_update(year: int, race_name: str, is_sprint: bool):
        call_order.append("practice_update")
        assert year == 2026
        assert race_name == "Chinese Grand Prix"
        assert is_sprint is True
        return {"updated": False, "completed_fp_sessions": []}

    monkeypatch.setattr(pages, "auto_update_practice_characteristics_if_needed", _practice_update)
    monkeypatch.setattr(
        pages,
        "get_artifact_versions",
        lambda: (call_order.append("artifact_versions"), {"k": (1, "ts")})[1],
    )

    def _run_prediction(
        race_name: str,
        weather: str,
        versions: dict,
        is_sprint: bool,
        year: int,
    ):
        call_order.append("run_prediction")
        assert race_name == "Chinese Grand Prix"
        assert weather == "dry"
        assert versions == {"k": (1, "ts")}
        assert is_sprint is True
        assert year == 2026
        return {"sprint_quali": {"grid": []}}

    monkeypatch.setattr(pages, "run_prediction", _run_prediction)

    output = pages.execute_live_prediction_pipeline(
        race_name="Chinese Grand Prix",
        weather="dry",
        year=2026,
    )

    assert call_order == [
        "race_update",
        "sprint_check",
        "practice_update",
        "artifact_versions",
        "run_prediction",
    ]
    assert output["is_sprint"] is True


def test_execute_live_prediction_pipeline_clears_cache_before_prediction_when_practice_updated(
    monkeypatch,
):
    call_order: list[str] = []

    monkeypatch.setattr(pages, "auto_update_if_needed", lambda: call_order.append("race_update"))
    monkeypatch.setattr(
        pages,
        "is_sprint_weekend",
        lambda year, race_name: (call_order.append("sprint_check"), False)[1],
    )
    monkeypatch.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint: (
            call_order.append("practice_update"),
            {"updated": True, "completed_fp_sessions": ["FP1"], "teams_updated": 2},
        )[1],
    )
    monkeypatch.setattr(
        pages,
        "get_artifact_versions",
        lambda: (call_order.append("artifact_versions"), {"k": (4, "ts4")})[1],
    )

    monkeypatch.setattr(
        pages.st,
        "cache_resource",
        type(
            "_CacheResource",
            (),
            {"clear": staticmethod(lambda: call_order.append("clear_resource"))},
        ),
    )
    monkeypatch.setattr(
        pages.st,
        "cache_data",
        type("_CacheData", (), {"clear": staticmethod(lambda: call_order.append("clear_data"))}),
    )

    def _run_prediction(
        race_name: str,
        weather: str,
        versions: dict,
        is_sprint: bool,
        year: int,
    ):
        call_order.append("run_prediction")
        assert versions == {"k": (4, "ts4")}
        assert is_sprint is False
        return {"qualifying": {"grid": []}, "race": {"finish_order": []}}

    monkeypatch.setattr(pages, "run_prediction", _run_prediction)

    pages.execute_live_prediction_pipeline("Bahrain Grand Prix", "dry", year=2026)

    assert call_order == [
        "race_update",
        "sprint_check",
        "practice_update",
        "clear_resource",
        "clear_data",
        "artifact_versions",
        "run_prediction",
    ]


def test_execute_live_prediction_pipeline_raises_when_practice_update_fails(monkeypatch):
    monkeypatch.setattr(pages, "auto_update_if_needed", lambda: None)
    monkeypatch.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    monkeypatch.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )

    run_called = {"value": False}

    def _run_prediction(
        race_name: str,
        weather: str,
        versions: dict,
        is_sprint: bool,
        year: int,
    ):
        run_called["value"] = True
        raise AssertionError("run_prediction should not be called when refresh fails")

    monkeypatch.setattr(pages, "run_prediction", _run_prediction)

    with pytest.raises(RuntimeError, match="refresh failed"):
        pages.execute_live_prediction_pipeline("Bahrain Grand Prix", "dry", year=2026)

    assert run_called["value"] is False


def test_execute_live_prediction_pipeline_raises_when_sprint_lookup_fails(monkeypatch):
    monkeypatch.setattr(pages, "auto_update_if_needed", lambda: None)
    monkeypatch.setattr(
        pages,
        "is_sprint_weekend",
        lambda year, race_name: (_ for _ in ()).throw(ValueError("bad race")),
    )
    monkeypatch.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint: {"updated": False, "completed_fp_sessions": []},
    )
    monkeypatch.setattr(pages, "get_artifact_versions", lambda: {"k": (3, "ts3")})

    monkeypatch.setattr(
        pages,
        "run_prediction",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_prediction should not execute")),
    )

    with pytest.raises(ValueError, match="bad race"):
        pages.execute_live_prediction_pipeline("Unknown GP", "dry", year=2026)
