"""Read docs/MODEL_LEDGER.md for the model audit page.

The ledger ships with the code and every model change adds an entry before it lands, so a
page built from it cannot fall behind the model the way hand-written text did.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parents[2] / "docs" / "MODEL_LEDGER.md"
_ENTRY_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2}): (.+)$")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


@dataclass(frozen=True)
class LedgerEntry:
    """One dated ledger entry: its heading date, title and markdown body."""

    date: str
    title: str
    body: str


@functools.cache
def read_ledger(path: Path = LEDGER_PATH) -> str:
    """Return the ledger text; it only changes with a deploy."""
    return path.read_text(encoding="utf-8")


def ledger_entries(text: str) -> list[LedgerEntry]:
    """Return the `## YYYY-MM-DD: title` entries, newest first.

    A body runs to the next `## ` heading. Entries of one date keep their reverse file
    order, so the last one written comes first.
    """
    found: list[tuple[str, int, LedgerEntry]] = []
    heading: tuple[str, str] | None = None
    body: list[str] = []

    def close() -> None:
        if heading:
            found.append((heading[0], len(found), LedgerEntry(*heading, "\n".join(body).strip())))

    for line in text.splitlines():
        if line.startswith("## "):
            close()
            match = _ENTRY_HEADING.match(line)
            heading = (match.group(1), match.group(2).strip()) if match else None
            body = []
        elif heading:
            body.append(line)
    close()
    return [entry for *_, entry in sorted(found, key=lambda item: item[:2], reverse=True)]


def champion_history(text: str) -> list[dict[str, str]]:
    """Return the rows of the `## Champion history` table, newest first."""
    marker = "\n## Champion history\n"
    if marker not in text:
        return []
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    rows = [
        [cell.strip().replace("\\|", "|") for cell in _CELL_SPLIT.split(line.strip())[1:-1]]
        for line in section.splitlines()
        if line.startswith("|")
    ]
    if len(rows) < 3:
        return []
    header, body = rows[0], rows[2:]
    return sorted(
        (dict(zip(header, row, strict=False)) for row in body),
        key=lambda row: row.get(header[0], ""),
        reverse=True,
    )


def rows_to_markdown(rows: list[dict[str, str]]) -> str:
    """Render rows as a markdown table, so long cells wrap and code spans render."""
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += [
        "| " + " | ".join(_escape_cell(row.get(name, "")) for name in header) + " |" for row in rows
    ]
    return "\n".join(lines)


def _escape_cell(value: object) -> str:
    """Escape pipes so a cell cannot split the markdown row."""
    return str(value).replace("|", r"\|")
