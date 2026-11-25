"""Tests for the append-only forecast record (plan row 21).

The database-level refusal of updates and deletes (test 2) runs in Supabase:
``migrations/queries/forecast_record_append_only_check.sql``.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.analysis.canonical_rows import content_hash
from src.dashboard import forecast_record, prediction_cascade

_FIXTURES = Path(__file__).parent / "fixtures" / "forecast_record"
_NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
_CUTOFF = "2026-11-21T00:29:59+00:00"
_VERSIONS = {"config/default.yaml": (1, "cfg"), "learning_state::learning_state": (1, "learn")}


class _Query:
    def __init__(self, table: _Table) -> None:
        self.table, self.filters, self.descending, self.limit_n = table, {}, False, None

    def select(self, _columns: str) -> _Query:
        return self

    def eq(self, key: str, value: object) -> _Query:
        self.filters[key] = value
        return self

    def order(self, _column: str, desc: bool = False) -> _Query:
        self.descending = desc
        return self

    def limit(self, n: int) -> _Query:
        self.limit_n = n
        return self

    def execute(self):
        rows = [r for r in self.table.rows if all(r[k] == v for k, v in self.filters.items())]
        rows = sorted(rows, key=lambda r: r["id"], reverse=self.descending)[: self.limit_n]
        return type("Result", (), {"data": copy.deepcopy(rows)})


class _UniqueViolation(Exception):
    code = "23505"


class _Table:
    """In-memory stand-in for the table, insert only.

    Unique per ``content_hash`` and ``replaces`` as migration 008 sets it, or per
    ``content_hash`` alone as 007 left it (``unique_on_hash_only``).
    """

    def __init__(self, unique_on_hash_only: bool = False) -> None:
        self.rows: list[dict] = []
        self.unique_on_hash_only = unique_on_hash_only

    def select(self, columns: str) -> _Query:
        return _Query(self).select(columns)

    def _key(self, row: dict) -> tuple:
        return (
            (row["content_hash"],)
            if self.unique_on_hash_only
            else (
                row["content_hash"],
                row.get("replaces"),
            )
        )

    def insert(self, row: dict):
        def execute(_self):
            if any(self._key(r) == self._key(row) for r in self.rows):
                raise _UniqueViolation("duplicate key value violates unique constraint")
            self.rows.append({**copy.deepcopy(row), "id": len(self.rows) + 1})

        return type("Insert", (), {"execute": execute})()


class _Client:
    def __init__(self, unique_on_hash_only: bool = False) -> None:
        self.forecast_record = _Table(unique_on_hash_only)

    def table(self, name: str) -> _Table:
        assert name == forecast_record.TABLE
        return self.forecast_record


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _records(payload: dict, *, checkpoint: str | None = None, **overrides) -> list[dict]:
    return forecast_record.build_forecast_records(
        year=payload["year"],
        race_name=payload["race_name"],
        checkpoint=checkpoint or payload["checkpoint"],
        weather=payload["weather"],
        is_sprint=payload["is_sprint"],
        prediction_results=payload["prediction_results"],
        information_cutoff_at=overrides.pop("information_cutoff_at", _CUTOFF),
        boundary_signature="sig",
        model_version=overrides.pop("model_version", "3.9"),
        artifact_hash="art",
        artifact_versions=_VERSIONS,
    )


def test_the_same_forecast_written_twice_stores_one_row():
    client, records = _Client(), _records(_fixture("normal_las_vegas_fp2_dry"))
    assert forecast_record.write_forecast_records(records, client=client, now=_NOW) == 2
    assert forecast_record.write_forecast_records(records, client=client, now=_NOW) == 0
    assert len(client.forecast_record.rows) == 2


def test_later_sessions_and_results_never_change_an_earlier_row():
    payload = _fixture("normal_las_vegas_fp2_dry")
    client = _Client()
    forecast_record.write_forecast_records(_records(payload), client=client, now=_NOW)
    stored = copy.deepcopy(client.forecast_record.rows)

    # Results attached to the served payload, as the old record does, and a later session.
    for section in payload["prediction_results"].values():
        for row in section.get("grid", []) + section.get("finish_order", []):
            row["actual_position"] = 1
    later = copy.deepcopy(payload)
    later["prediction_results"]["race"]["finish_order"].reverse()
    for position, row in enumerate(later["prediction_results"]["race"]["finish_order"], start=1):
        row["position"] = position
    forecast_record.write_forecast_records(
        _records(later, checkpoint="FP3"), client=client, now=_NOW
    )

    earlier = [r for r in client.forecast_record.rows if r["checkpoint"] == "FP2"]
    assert earlier == stored
    assert all(content_hash(r["core"]) == r["content_hash"] for r in earlier)
    assert all("actual_position" not in row for r in earlier for row in r["core"]["rows"])


@pytest.mark.parametrize("name", ["normal_las_vegas_fp2_dry", "sprint_singapore_fp1_dry"])
def test_the_record_holds_what_the_page_shows(name):
    payload = _fixture(name)
    shown: list[list[dict]] = []

    class _Streamlit:
        def markdown(self, *_args, **_kwargs) -> None:
            return None

    prediction_cascade.render_prediction_results_core(
        prediction_results=copy.deepcopy(payload["prediction_results"]),
        is_sprint=payload["is_sprint"],
        display_prediction_result_fn=lambda section, _title, is_race: shown.append(
            section["finish_order" if is_race else "grid"]
        ),
        st_module=_Streamlit(),
    )

    records = _records(payload)
    assert records and len(records) == len([rows for rows in shown if rows])
    assert all(record["core"]["rows"] in shown for record in records)


def test_a_changed_forecast_names_what_produced_the_change():
    payload = _fixture("normal_las_vegas_fp2_dry")
    client = _Client()
    forecast_record.write_forecast_records(_records(payload), client=client, now=_NOW)
    grid = payload["prediction_results"]["qualifying"]["grid"]
    grid[0]["driver"], grid[1]["driver"] = grid[1]["driver"], grid[0]["driver"]
    forecast_record.write_forecast_records(
        _records(payload, model_version="4.0"), client=client, now=_NOW
    )

    newest = client.forecast_record.rows[-1]
    assert newest["target"] == "main_qualifying"
    assert newest["replaces"] == next(
        r["id"] for r in client.forecast_record.rows if r["target"] == "main_qualifying"
    )
    assert newest["replace_reason"] == "changed: model_version"
    assert len(client.forecast_record.rows) == 3


@pytest.mark.parametrize("cutoff", ["2026-11-20T11:59:59+00:00", None])
def test_a_forecast_after_its_cutoff_or_without_one_is_not_written(cutoff):
    client = _Client()
    records = _records(_fixture("normal_las_vegas_fp2_dry"), information_cutoff_at=cutoff)
    assert forecast_record.write_forecast_records(records, client=client, now=_NOW) == 0
    assert client.forecast_record.rows == []


def test_a_record_failure_never_stops_serving(monkeypatch, caplog):
    payload = _fixture("normal_las_vegas_fp2_dry")
    monkeypatch.setenv("USE_DB_STORAGE", "db_only")

    def broken_client():
        raise RuntimeError("table missing")

    monkeypatch.setattr("src.persistence.db.get_supabase_client", broken_client)
    forecast_record.record_served_forecast(
        year=payload["year"],
        race_name=payload["race_name"],
        checkpoint=payload["checkpoint"],
        weather=payload["weather"],
        is_sprint=payload["is_sprint"],
        prediction_results=payload["prediction_results"],
        boundary_signature="sig",
        artifact_hash="art",
        artifact_versions=_VERSIONS,
    )

    assert "Could not record forecast" in caplog.text


def _swapped(payload: dict) -> dict:
    changed = copy.deepcopy(payload)
    grid = changed["prediction_results"]["qualifying"]["grid"]
    grid[0]["driver"], grid[1]["driver"] = grid[1]["driver"], grid[0]["driver"]
    return changed


def test_a_forecast_that_returns_to_an_earlier_content_is_a_new_issuance():
    """Codex 2026-11-23: A, B, then A left B as the latest record while A was served."""
    a = _fixture("normal_las_vegas_fp2_dry")
    client = _Client()
    for payload in (a, _swapped(a), a, a):
        forecast_record.write_forecast_records(_records(payload), client=client, now=_NOW)

    quali = [r for r in client.forecast_record.rows if r["target"] == "main_qualifying"]
    assert len(quali) == 3
    assert quali[2]["content_hash"] == quali[0]["content_hash"] != quali[1]["content_hash"]
    assert [r.get("replaces") for r in quali] == [None, quali[0]["id"], quali[1]["id"]]


def test_before_migration_008_a_returning_forecast_is_skipped_not_raised():
    a = _fixture("normal_las_vegas_fp2_dry")
    client = _Client(unique_on_hash_only=True)
    sent = [
        forecast_record.write_forecast_records(_records(payload), client=client, now=_NOW)
        for payload in (a, _swapped(a), a)
    ]

    assert sent == [2, 1, 0]
    assert len(client.forecast_record.rows) == 3


def test_an_error_that_is_not_a_duplicate_key_still_raises():
    class _Down(Exception):
        code = "57P01"

    class _BrokenTable(_Table):
        def insert(self, row: dict):
            return type("Insert", (), {"execute": lambda _self: (_ for _ in ()).throw(_Down())})()

    client = _Client()
    client.forecast_record = _BrokenTable()
    with pytest.raises(_Down):
        forecast_record.write_forecast_records(
            _records(_fixture("normal_las_vegas_fp2_dry")), client=client, now=_NOW
        )
