"""Which served weather scenario a session is scored against (2027 protocol, ledger 2026-11-21).

The rule is fixed before the season and needs no judgement after the race: a session is
``rain`` when any driver set a timed lap on intermediate or wet tyres, otherwise ``dry``.
Laps come from FastF1, or from OpenF1 laps and stints when FastF1 has none
(``openf1_session.get_session``). A session with no readable compounds is scored ``dry``
and flagged ``known=False``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

WET_COMPOUNDS = frozenset({"INTERMEDIATE", "WET"})
DRY_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD"})


@dataclass(frozen=True)
class SessionConditions:
    """The scored scenario and the laps that decided it."""

    scenario: str
    known: bool
    timed_laps: int
    wet_laps: int
    wet_drivers: tuple[str, ...]


UNKNOWN = SessionConditions(scenario="dry", known=False, timed_laps=0, wet_laps=0, wet_drivers=())


def conditions_from_laps(laps: Any) -> SessionConditions:
    """Apply the rule to a FastF1 laps frame (``Driver``, ``Compound``, ``LapTime``)."""
    if laps is None or len(laps) == 0 or not {"Driver", "Compound", "LapTime"} <= set(laps.columns):
        return UNKNOWN
    timed = laps[laps["LapTime"].notna()]
    compounds = timed["Compound"].astype("string").str.upper()
    wet = timed[compounds.isin(WET_COMPOUNDS).fillna(False).to_numpy()]
    if wet.empty and not compounds.isin(DRY_COMPOUNDS).fillna(False).any():
        return UNKNOWN
    return SessionConditions(
        scenario="rain" if len(wet) else "dry",
        known=True,
        timed_laps=len(timed),
        wet_laps=len(wet),
        wet_drivers=tuple(sorted({str(driver) for driver in wet["Driver"]})),
    )


def session_conditions(year: int, race_name: str, session_name: str) -> SessionConditions:
    """Load one session's laps and apply the rule; ``UNKNOWN`` when no source has laps.

    Raises ``OpenF1TransportError`` when OpenF1 cannot be reached, so the caller tries
    again later instead of scoring a guess.
    """
    from src.data.openf1 import OpenF1TransportError, OpenF1Unavailable
    from src.data.openf1_session import get_session

    try:
        session = get_session(year, race_name, session_name)
    except OpenF1TransportError:
        raise
    except OpenF1Unavailable as exc:
        logger.warning("No laps for %s %s %s: %s", year, race_name, session_name, exc)
        return UNKNOWN
    return conditions_from_laps(session.laps)
