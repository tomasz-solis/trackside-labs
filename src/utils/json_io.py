"""Shared JSON reading helpers for analysis modules and scripts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``.

    Raises:
        ValueError: If the file's top-level JSON value is not an object.
    """
    with path.open(encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def json_safe(value: Any) -> tuple[Any, int]:
    """Return ``value`` with NaN and infinite floats replaced by None, and how many.

    Strict JSON (Supabase, most parsers) rejects NaN and Infinity; Python's ``json``
    writes them anyway. A missing measurement belongs in JSON as null.
    """
    if isinstance(value, float):
        return (value, 0) if math.isfinite(value) else (None, 1)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        count = 0
        for key, item in value.items():
            out[key], n = json_safe(item)
            count += n
        return out, count
    if isinstance(value, list | tuple):
        items = []
        count = 0
        for item in value:
            clean, n = json_safe(item)
            items.append(clean)
            count += n
        return items, count
    return value, 0
