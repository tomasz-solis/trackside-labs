"""Tests for the read-only, local forecast lineage diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_forecast_lineage import audit_forecast_lineage, main

from src.utils.accuracy_targets import sanitize_actual_rows, sanitize_prediction_rows


def _write_json(path: Path, payload: object) -> bytes:
    """Write a deterministic JSON fixture and return its exact bytes."""
    contents = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(contents)
    return contents


def test_modern_targets_report_explicit_cutoff_and_actuals(tmp_path: Path) -> None:
    """Modern target-keyed records expose target state without publication claims."""
    path = tmp_path / "modern.json"
    source = _write_json(
        path,
        {
            "metadata": {
                "session_name": "FP1",
                "model_version": "2.2",
                "run_id": "95fce572-139d-42e2-83ea-e5f36690a261",
                "information_cutoff_at": "2026-03-06T04:59:59+00:00",
                "predicted_at": "2026-10-13T10:00:00+00:00",
            },
            "targets": {
                "grand_prix_race": {"predicted_order": [{"driver": "AAA", "team": "Team A"}]},
                "main_qualifying": {"predicted_order": [{"driver": "BBB", "team": "Team B"}]},
            },
            "actuals": {
                "targets": {"grand_prix_race": [{"driver": "AAA", "team": "Team A", "position": 1}]}
            },
            "api_key": "sk-secret-fixture-value",
        },
    )

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    assert artifact["sha256"] == hashlib.sha256(source).hexdigest()
    assert artifact["checkpoint"] == {"status": "known", "value": "FP1"}
    assert artifact["model_version"] == {"status": "known", "value": "2.2"}
    assert artifact["run_id"] == {
        "status": "known",
        "value": "95fce572-139d-42e2-83ea-e5f36690a261",
    }
    assert artifact["forecast_id"] == {"status": "unknown", "value": None}
    assert artifact["as_of"] == {
        "status": "known",
        "value": "2026-03-06T04:59:59+00:00",
    }
    assert artifact["actuals_status"] == "partial"
    assert artifact["targets"] == [
        {
            "target": "grand_prix_race",
            "forecast_status": "present",
            "actuals_status": "present",
            "displayable_state": "historical_inspection_only",
            "scoreability_status": "provenance_unknown",
        },
        {
            "target": "main_qualifying",
            "forecast_status": "present",
            "actuals_status": "missing",
            "displayable_state": "historical_inspection_only",
            "scoreability_status": "not_scoreable",
        },
    ]
    assert artifact["displayable_state"] == "historical_inspection_only"
    assert artifact["scoreability_status"] == "not_scoreable"
    assert artifact["publication_status"] == "unknown"
    assert artifact["provenance_status"] == "unknown"
    assert artifact["prospective_eligibility_status"] == "unknown"


def test_unknown_target_in_mixed_record_marks_partial_coverage(tmp_path: Path) -> None:
    """Known summaries survive, while ignored target names make coverage partial."""
    path = tmp_path / "mixed-unknown-target.json"
    source = _write_json(
        path,
        {
            "targets": {
                "main_qualifying": {"predicted_order": [{"driver": "AAA", "team": "Team A"}]},
                "future_target_secret_name": {"predicted_order": []},
            },
            "actuals": {
                "targets": {"main_qualifying": [{"driver": "AAA", "team": "Team A", "position": 1}]}
            },
        },
    )

    report = audit_forecast_lineage([path])
    artifact = report["artifacts"][0]

    assert artifact["sha256"] == hashlib.sha256(source).hexdigest()
    assert artifact["artifact_status"] == "partial_coverage"
    assert artifact["target_coverage"] == {"status": "partial", "unsupported_key_count": 1}
    assert [row["target"] for row in artifact["targets"]] == ["main_qualifying"]
    assert artifact["targets"][0]["scoreability_status"] == "provenance_unknown"
    assert artifact["scoreability_status"] == "not_scoreable"
    assert artifact["displayable_state"] == "partial"
    assert "future_target_secret_name" not in json.dumps(report)


def test_unknown_only_targets_are_unsupported_and_never_scoreable(tmp_path: Path) -> None:
    """Unknown-only target schemas are visible as unsupported without echoing keys."""
    path = tmp_path / "unknown-only.json"
    _write_json(path, {"targets": {"customer_secret_target": {"predicted_order": []}}})

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    assert artifact["artifact_status"] == "unsupported_schema"
    assert artifact["target_coverage"] == {"status": "unsupported", "unsupported_key_count": 1}
    assert artifact["targets"] == []
    assert artifact["scoreability_status"] == "not_scoreable"
    assert "customer_secret_target" not in json.dumps(artifact)


def test_actuals_only_unknown_target_keys_are_counted(tmp_path: Path) -> None:
    """Unknown actual target keys count even when there is no forecast target row."""
    path = tmp_path / "unknown-actual-target.json"
    _write_json(
        path,
        {"actuals": {"targets": {"hidden_actual_target": [{"secret": "value"}]}}},
    )

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    assert artifact["target_coverage"] == {"status": "unsupported", "unsupported_key_count": 1}
    assert artifact["artifact_status"] == "unsupported_schema"
    assert artifact["scoreability_status"] == "not_scoreable"
    assert "hidden_actual_target" not in json.dumps(artifact)
    assert '"secret"' not in json.dumps(artifact)


def test_unknown_declared_top_level_target_metadata_marks_partial_coverage(tmp_path: Path) -> None:
    """Unrecognized declared legacy targets cannot disappear behind known rows."""
    path = tmp_path / "unknown-declared-target.json"
    _write_json(
        path,
        {
            "metadata": {"top_level_qualifying_target": "private_future_target"},
            "race": {"predicted_results": [{"driver": "AAA", "team": "Team A"}]},
        },
    )

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    assert artifact["target_coverage"] == {"status": "partial", "unsupported_key_count": 1}
    assert artifact["artifact_status"] == "partial_coverage"
    assert artifact["targets"][0]["target"] == "grand_prix_race"
    assert artifact["scoreability_status"] == "not_scoreable"
    assert "private_future_target" not in json.dumps(artifact)


def test_known_only_coverage_is_compatible_and_repeated_audits_are_stable(tmp_path: Path) -> None:
    """Known target semantics and original bytes stay stable across repeated reads."""
    path = tmp_path / "known-only.json"
    source = _write_json(
        path,
        {
            "targets": {
                "main_qualifying": {"predicted_order": [{"driver": "AAA", "team": "Team A"}]}
            },
            "actuals": {"targets": {"main_qualifying": [{"driver": "AAA", "team": "Team A"}]}},
        },
    )

    first = audit_forecast_lineage([path])
    second = audit_forecast_lineage([path])

    assert first == second
    artifact = first["artifacts"][0]
    assert artifact["artifact_status"] == "ok"
    assert artifact["target_coverage"] == {"status": "complete", "unsupported_key_count": 0}
    assert artifact["targets"][0]["scoreability_status"] == "provenance_unknown"
    assert artifact["scoreability_status"] == "provenance_unknown"
    assert artifact["sha256"] == hashlib.sha256(source).hexdigest()
    assert path.read_bytes() == source


def test_legacy_qualifying_and_race_shapes_do_not_imply_prospective(tmp_path: Path) -> None:
    """Legacy top-level forecasts are found while publication and provenance stay unknown."""
    path = tmp_path / "legacy.json"
    _write_json(
        path,
        {
            "metadata": {
                "session_name": "FP2",
                "model_version": "2.0",
                "source": "historical_replay",
                "predicted_at": "2026-10-13T10:00:00+00:00",
            },
            "qualifying": {"predicted_grid": [{"driver": "AAA", "team": "Team A", "position": 1}]},
            "race": {"predicted_results": [{"driver": "BBB", "team": "Team B", "position": 1}]},
            "actuals": {
                "qualifying": [],
                "race": [{"driver": "BBB", "team": "Team B", "position": 1}],
            },
        },
    )

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    assert artifact["as_of"] == {"status": "unknown", "value": None}
    assert artifact["model_version"] == {"status": "known", "value": "2.0"}
    assert artifact["record_state"] == "historical_only"
    assert artifact["actuals_status"] == "partial"
    assert artifact["publication_status"] == "unknown"
    assert artifact["provenance_status"] == "unknown"
    assert [row["target"] for row in artifact["targets"]] == [
        "grand_prix_race",
        "main_qualifying",
    ]
    assert all(row["forecast_status"] == "present" for row in artifact["targets"])
    assert artifact["displayable_state"] == "historical_inspection_only"
    assert artifact["scoreability_status"] == "not_scoreable"
    assert artifact["targets"][0]["scoreability_status"] == "provenance_unknown"
    assert artifact["targets"][1]["scoreability_status"] == "not_scoreable"


def test_invalid_and_nonfinite_json_are_reported_without_raising(tmp_path: Path) -> None:
    """Malformed files produce fixed statuses without echoing parser details."""
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"token":"do-not-echo",', encoding="utf-8")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")

    report = audit_forecast_lineage([invalid, nonfinite])

    assert sorted(item["artifact_status"] for item in report["artifacts"]) == [
        "invalid_json",
        "invalid_json_nonfinite",
    ]
    assert "do-not-echo" not in json.dumps(report)


def test_report_is_strict_deterministic_and_does_not_mutate_inputs(tmp_path: Path, capsys) -> None:
    """File order and input bytes remain stable, and CLI output is strict JSON."""
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first_bytes = _write_json(first, {"qualifying": {"predicted_grid": []}})
    second_bytes = _write_json(second, {"race": {"predicted_results": []}})
    before = {first: first.read_bytes(), second: second.read_bytes()}

    one = audit_forecast_lineage([tmp_path])
    two = audit_forecast_lineage([second, first])
    assert one == two
    assert [item["sha256"] for item in one["artifacts"]] == sorted(
        hashlib.sha256(contents).hexdigest() for contents in (first_bytes, second_bytes)
    )
    assert {path: path.read_bytes() for path in before} == before

    assert main([str(second), str(first)]) == 0
    output = capsys.readouterr().out
    json.loads(output, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert "NaN" not in output


def test_untrusted_payload_values_are_never_echoed(tmp_path: Path) -> None:
    """Unknown source fields and invalid metadata values cannot leak into the report."""
    path = tmp_path / "redaction.json"
    secret = "Bearer super-secret-material"
    _write_json(
        path,
        {
            "metadata": {
                "session_name": secret,
                "model_version": secret,
                "information_cutoff_at": secret,
                "api_token": secret,
            },
            "targets": {"main_qualifying": {"predicted_order": []}},
            "sensitive_unrecognized_value": secret,
        },
    )

    serialized = json.dumps(audit_forecast_lineage([path]))

    assert secret not in serialized


def test_numeric_only_model_version_is_redacted_as_unknown(tmp_path: Path) -> None:
    """Long numeric strings cannot be echoed as plausible model versions."""
    path = tmp_path / "numeric-version.json"
    _write_json(
        path,
        {
            "metadata": {"model_version": "1234567890"},
            "targets": {"main_qualifying": {"predicted_order": []}},
        },
    )

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    assert artifact["model_version"] == {"status": "unknown", "value": None}


def test_driver_team_row_validation_matches_consumer_sanitizers(tmp_path: Path) -> None:
    """Diagnostic row status agrees with consumer sanitizers for valid and malformed rows."""
    cases = [
        ([{"driver": "AAA", "team": "Team A"}], "present"),
        ([{}], "unsupported"),
        ([{"driver": "  ", "team": "Team A"}], "unsupported"),
        ([{"driver": "AAA"}], "unsupported"),
    ]

    for index, (rows, expected_status) in enumerate(cases):
        path = tmp_path / f"row-case-{index}.json"
        _write_json(
            path,
            {
                "targets": {"main_qualifying": {"predicted_order": rows}},
                "actuals": {"targets": {"main_qualifying": rows}},
            },
        )
        artifact = audit_forecast_lineage([path])["artifacts"][0]
        target = artifact["targets"][0]
        prediction_rows = sanitize_prediction_rows(rows)
        actual_rows = sanitize_actual_rows(rows)
        consumer_accepts_all = (
            bool(rows) and len(prediction_rows) == len(rows) and len(actual_rows) == len(rows)
        )

        assert target["forecast_status"] == expected_status
        assert target["actuals_status"] == expected_status
        assert (expected_status == "present") == consumer_accepts_all


def test_naive_cutoff_and_malformed_row_shapes_remain_unknown(tmp_path: Path) -> None:
    """Naive timestamps and scalar rows cannot establish valid lineage evidence."""
    path = tmp_path / "bad-rows.json"
    _write_json(
        path,
        {
            "metadata": {"information_cutoff_at": "2026-03-06T04:59:59"},
            "targets": {"main_qualifying": {"predicted_order": [1]}},
            "actuals": {"targets": {"main_qualifying": [1]}},
        },
    )

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    assert artifact["as_of"] == {"status": "unknown", "value": None}
    assert artifact["targets"] == [
        {
            "target": "main_qualifying",
            "forecast_status": "unsupported",
            "actuals_status": "unsupported",
            "displayable_state": "unavailable",
            "scoreability_status": "not_scoreable",
        }
    ]
    assert artifact["actuals_status"] == "unsupported"


def test_overflowing_number_is_nonfinite_json(tmp_path: Path) -> None:
    """JSON exponents that Python converts to infinity are rejected."""
    path = tmp_path / "overflow.json"
    path.write_text('{"value":1e999}', encoding="utf-8")

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    assert artifact["artifact_status"] == "invalid_json_nonfinite"


def test_unreadable_file_has_fixed_status_without_echoing_path(tmp_path: Path, monkeypatch) -> None:
    """Read errors do not expose operating-system messages or file paths."""
    path = tmp_path / "private-bearer-super-secret.json"
    path.write_text("{}", encoding="utf-8")
    real_read_bytes = Path.read_bytes

    def fail_one_file(candidate: Path) -> bytes:
        if candidate == path:
            raise OSError("permission denied: bearer super-secret")
        return real_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", fail_one_file)
    report = audit_forecast_lineage([path])

    assert report["artifacts"][0]["artifact_status"] == "unreadable"
    assert "super-secret" not in json.dumps(report)


def test_missing_or_empty_input_fails_with_strict_redacted_cli_json(tmp_path: Path, capsys) -> None:
    """Invalid and empty inputs fail explicitly without echoing their paths."""
    secret_path = tmp_path / "secret-input-path-do-not-echo"
    assert main([str(secret_path)]) == 2
    output = capsys.readouterr().out
    assert json.loads(output) == {"error": "input_unavailable", "schema_version": 1}
    assert "secret-input-path-do-not-echo" not in output

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty)]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "input_unavailable"


def test_mixed_targets_use_explicit_forecast_modes_conservatively(tmp_path: Path) -> None:
    """Mixed records distinguish inspectable rows, unavailable targets and actuals."""
    path = tmp_path / "mixed-states.json"
    _write_json(
        path,
        {
            "targets": {
                "grand_prix_race": {"predicted_order": [{"driver": "AAA", "team": "Team A"}]},
                "main_qualifying": {"eligible_at_save": True},
                "sprint_qualifying": {
                    "eligible_at_save": False,
                    "predicted_order": [{"driver": "BBB", "team": "Team B"}],
                },
                "sprint_race": {
                    "result_mode": "ACTUAL",
                    "predicted_order": [{"driver": "CCC", "team": "Team C"}],
                },
            },
            "actuals": {
                "targets": {
                    "grand_prix_race": [{"driver": "AAA", "team": "Team A", "position": 1}],
                    "sprint_race": [{"driver": "CCC", "team": "Team C", "position": 1}],
                }
            },
        },
    )

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    states = {row["target"]: row["displayable_state"] for row in artifact["targets"]}
    assert states == {
        "grand_prix_race": "historical_inspection_only",
        "main_qualifying": "unavailable",
        "sprint_qualifying": "not_a_forecast",
        "sprint_race": "not_a_forecast",
    }
    scoreability = {row["target"]: row["scoreability_status"] for row in artifact["targets"]}
    assert scoreability == {
        "grand_prix_race": "provenance_unknown",
        "main_qualifying": "not_scoreable",
        "sprint_qualifying": "not_scoreable",
        "sprint_race": "not_scoreable",
    }
    assert artifact["displayable_state"] == "mixed"
    assert artifact["scoreability_status"] == "not_scoreable"
    assert artifact["publication_status"] == "unknown"
    assert artifact["provenance_status"] == "unknown"
    assert artifact["prospective_eligibility_status"] == "unknown"


def test_modern_target_metadata_precedes_conflicting_legacy_aliases(tmp_path: Path) -> None:
    """Explicit target records decide state even when legacy aliases disagree."""
    path = tmp_path / "conflicting-target-metadata.json"
    _write_json(
        path,
        {
            "metadata": {
                "top_level_race_target": "sprint_race",
                "top_level_race_eligible_at_save": False,
                "top_level_race_result_mode": "PREDICTED",
            },
            "targets": {
                "grand_prix_race": {
                    "eligible_at_save": True,
                    "result_mode": "ACTUAL",
                    "predicted_order": [{"driver": "AAA", "team": "Team A"}],
                },
                "sprint_race": {
                    "eligible_at_save": True,
                    "result_mode": "PREDICTED",
                    "predicted_order": [{"driver": "BBB", "team": "Team B"}],
                },
            },
            "actuals": {
                "targets": {
                    "grand_prix_race": [{"driver": "AAA", "team": "Team A", "position": 1}],
                    "sprint_race": [{"driver": "BBB", "team": "Team B", "position": 2}],
                }
            },
        },
    )

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    states = {row["target"]: row["displayable_state"] for row in artifact["targets"]}
    assert states == {
        "grand_prix_race": "not_a_forecast",
        "sprint_race": "historical_inspection_only",
    }
    scoreability = {row["target"]: row["scoreability_status"] for row in artifact["targets"]}
    assert scoreability == {
        "grand_prix_race": "not_scoreable",
        "sprint_race": "provenance_unknown",
    }


def test_malformed_actual_mappings_and_rows_are_not_scoreable(tmp_path: Path) -> None:
    """Only verified list-shaped object rows count as attached actuals."""
    path = tmp_path / "malformed-actuals.json"
    _write_json(
        path,
        {
            "targets": {
                "grand_prix_race": {"predicted_order": [{"driver": "AAA", "team": "Team A"}]},
                "main_qualifying": {"predicted_order": [{"driver": "BBB", "team": "Team B"}]},
            },
            "actuals": {
                "targets": {
                    "grand_prix_race": {"api_key": "secret-value"},
                    "main_qualifying": [
                        {"driver": "BBB", "team": "Team B"},
                        "malformed-row",
                    ],
                }
            },
        },
    )

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    assert artifact["actuals_status"] == "unsupported"
    assert [row["actuals_status"] for row in artifact["targets"]] == [
        "unsupported",
        "unsupported",
    ]
    assert [row["scoreability_status"] for row in artifact["targets"]] == [
        "not_scoreable",
        "not_scoreable",
    ]
    assert "secret-value" not in json.dumps(artifact)


def test_frozen_normal_and_sprint_specimens_match_manifest_without_mutation() -> None:
    """Frozen real-shape fixtures retain manifest hashes and normal/sprint targets."""
    repo_root = Path(__file__).resolve().parents[1]
    fixture_root = repo_root / "tests/fixtures/forecast_lineage/2026-10-13"
    manifest_path = repo_root / "docs/forecast_lineage/2026-10-13/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specimen_entries = {
        Path(entry["specimen_path"]).name: entry
        for entry in manifest["artifacts"]
        if entry["specimen_path"] is not None
    }
    expected_targets = {
        "australian_fp1.json": ["grand_prix_race", "main_qualifying"],
        "miami_sq.json": ["grand_prix_race", "main_qualifying", "sprint_race"],
    }
    assert set(specimen_entries) == set(expected_targets)

    fixture_paths = [fixture_root / name for name in sorted(expected_targets)]
    original_bytes = {path: path.read_bytes() for path in fixture_paths}
    report = audit_forecast_lineage(fixture_paths)
    by_hash = {artifact["sha256"]: artifact for artifact in report["artifacts"]}

    for name, targets in expected_targets.items():
        entry = specimen_entries[name]
        contents = original_bytes[fixture_root / name]
        digest = hashlib.sha256(contents).hexdigest()
        assert digest == entry["sha256"]
        artifact = by_hash[digest]
        forecast_targets = [
            row["target"]
            for row in artifact["targets"]
            if row["displayable_state"] == "historical_inspection_only"
        ]
        assert forecast_targets == targets
        if name == "miami_sq.json":
            sprint_qualifying = next(
                row for row in artifact["targets"] if row["target"] == "sprint_qualifying"
            )
            assert sprint_qualifying["displayable_state"] == "not_a_forecast"
    assert {path: path.read_bytes() for path in original_bytes} == original_bytes


def test_deeply_nested_json_has_fixed_depth_error(tmp_path: Path) -> None:
    """A parser recursion failure becomes a stable redacted artifact status."""
    path = tmp_path / "deep.json"
    path.write_bytes(b"[" * 100_000 + b"0" + b"]" * 100_000)

    artifact = audit_forecast_lineage([path])["artifacts"][0]

    assert artifact["artifact_status"] == "invalid_json_depth"
