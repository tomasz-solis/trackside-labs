"""Exact copies and content hashes of saved forecast rows, with no scoring imports.

The forecast record (``src/dashboard/forecast_record.py``) runs in the cron, which must
not load scipy; the frozen scorer reuses the same two helpers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any


def copy_canonical_rows(rows: object, *, label: str, ranked_order: bool) -> list[dict[str, Any]]:
    """Copy stored rows exactly, rejecting repairs that would change their meaning."""
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    copied = copy.deepcopy(rows)
    seen_drivers: set[str] = set()
    seen_positions: set[int] = set()
    for index, row in enumerate(copied, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        driver = row.get("driver")
        team = row.get("team")
        position = row.get("position")
        if not isinstance(driver, str) or not driver or driver != driver.strip():
            raise ValueError(f"{label}[{index}].driver must be a nonempty canonical string")
        if not isinstance(team, str) or not team or team != team.strip():
            raise ValueError(f"{label}[{index}].team must be a nonempty canonical string")
        if type(position) is not int or position < 1:
            raise ValueError(f"{label}[{index}].position must be a positive integer")
        if driver in seen_drivers or position in seen_positions:
            raise ValueError(f"{label} has duplicate driver or position")
        if ranked_order and position != index:
            raise ValueError(f"{label} must be ordered by contiguous rank")
        if ranked_order:
            lower = row.get("p5")
            upper = row.get("p95")
            if (lower is None) != (upper is None):
                raise ValueError(f"{label}[{index}] must have both p5 and p95")
            if lower is not None and upper is not None:
                for field, value in (("p5", lower), ("p95", upper)):
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError(f"{label}[{index}].{field} must be finite numeric")
                if lower <= 0 or upper < lower:
                    raise ValueError(f"{label}[{index}] has an invalid p5/p95 interval")
            for field, value in row.items():
                if field.endswith("_probability") or field == "dnf_risk":
                    if value is not None and (
                        type(value) not in (int, float) or not math.isfinite(value)
                    ):
                        raise ValueError(f"{label}[{index}].{field} must be finite numeric")
                    if value is not None and not 0 <= value <= 100:
                        raise ValueError(f"{label}[{index}].{field} must be between 0 and 100")
        seen_drivers.add(driver)
        seen_positions.add(position)
    return copied


def content_hash(value: dict[str, Any]) -> str:
    """Hash the forecast projection without outcome or scoring state."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
