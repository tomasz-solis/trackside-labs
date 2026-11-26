"""The model audit page reads the ledger, and the ledger and page text keep up with config."""

from __future__ import annotations

from src.dashboard.model_ledger import (
    champion_history,
    ledger_entries,
    read_ledger,
    rows_to_markdown,
)
from src.dashboard.page_content import MODEL_INSIGHTS_MARKDOWN
from src.utils.model_version import get_model_version

_LEDGER = """# Model ledger

## Champion history

| Date | Change | Effect | Commit |
|---|---|---|---|
| 2026-10-02 | Model 3.1: sort by median. | MAE \\|bias\\| down | `a1` |
| 2026-11-14 | Model 3.9: floor off. | corr up | `b2` |

## 2026-11-01: first entry of the day

Body one.

### A sub-heading stays in the body

## 2026-11-01: second entry of the day

Body two.

## Adding an entry

Not an entry.
"""


def test_ledger_entries_newest_first_with_their_bodies() -> None:
    entries = ledger_entries(_LEDGER)

    assert [entry.title for entry in entries] == [
        "second entry of the day",
        "first entry of the day",
    ]
    assert entries[1].body == "Body one.\n\n### A sub-heading stays in the body"
    assert "Not an entry" not in entries[0].body


def test_champion_history_rows_newest_first_with_escaped_pipes() -> None:
    rows = champion_history(_LEDGER)

    assert [row["Date"] for row in rows] == ["2026-11-14", "2026-10-02"]
    assert rows[1]["Effect"] == "MAE |bias| down"
    assert champion_history("# no table here") == []


def test_rows_to_markdown_round_trips_through_the_parser() -> None:
    rows = champion_history(_LEDGER)
    table = rows_to_markdown(rows)

    assert champion_history(f"\n## Champion history\n\n{table}\n") == rows


def test_ledger_and_page_text_name_the_model_in_use() -> None:
    """A version bump must add its champion row and its line on the Model and learning page."""
    version = get_model_version()
    rows = champion_history(read_ledger())

    assert any(row["Change"].startswith(f"Model {version}:") for row in rows), (
        f"docs/MODEL_LEDGER.md Champion history has no 'Model {version}:' row"
    )
    assert f"Since {version}," in MODEL_INSIGHTS_MARKDOWN, (
        f"page_content.MODEL_INSIGHTS_MARKDOWN does not say what changed in {version}"
    )
    assert ledger_entries(read_ledger())[0].date >= rows[0]["Date"]
