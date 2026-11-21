"""Read finished-session results from OpenF1 when FastF1 cannot load them.

FastF1's archive has returned 401 since about 2026-10-15 (FastF1 #978). OpenF1
(api.openf1.org) serves finished sessions free. This module covers what the cron
needs to move a weekend forward: session completion, competitive results and the
race starting grid. ``openf1_session`` loads laps, weather and car data into a
FastF1 session.

Every row it returns carries ``source = "openf1"`` so stored actuals say where
they came from. Set ``TRACKSIDE_OPENF1_DISABLED=1`` to turn the fallback off; the
test suite does, so no test reaches the network.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
SOURCE = "openf1"
DISABLE_ENV = "TRACKSIDE_OPENF1_DISABLED"
_TIMEOUT_SECONDS = 15
_MIN_RESULT_ROWS = 18
# How long after the scheduled end a session may still be missing results before
# the state becomes unknown rather than incomplete.
_RESULTS_GRACE = timedelta(hours=3)

# FastF1 session codes to OpenF1 session names.
SESSION_NAMES = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "SQ": "Sprint Qualifying",
    "Sprint": "Sprint",
    "Q": "Qualifying",
    "R": "Race",
}
_RACE_LIKE = {"R", "Sprint"}
# Sessions whose classification comes from OpenF1 before FastF1. FastF1 builds a sprint
# qualifying order from lap times and misses steward deletions: 2026 Miami ALB P14
# (official P19), Singapore PER P22 (official P20). OpenF1 matched the official order at both.
CLASSIFICATION_FIRST = frozenset({"SQ"})

CompletionState = Literal["completed", "incomplete", "unknown"]


class OpenF1Unavailable(RuntimeError):
    """Raised when OpenF1 is disabled, unreachable or has no data for the request."""


class OpenF1TransportError(OpenF1Unavailable):
    """OpenF1 gave no usable answer: rate limit, server error, broken read or live lock.

    Unlike missing data, this says nothing about the session, so a replay must not
    carry on as if the session had not run.
    """


# OpenF1 free tier: 3 requests a second and 30 a minute, 429 above that. The pacing is
# per process, so two replays at once go over it. The retry waits add up to 75 s, past
# the one-minute window (MODEL_LEDGER 2026-11-13).
_MIN_INTERVAL_SECONDS = 2.1
_RETRY_WAIT_SECONDS = (5.0, 10.0, 20.0, 40.0)

# The web process lives for days: a response cached before results are published
# must not outlive the publication, so entries expire.
_CACHE_TTL_SECONDS = 600.0
# During a live session OpenF1 answers 401 to every unauthenticated request, past
# sessions included. After one such answer, fail fast instead of asking again.
_LIVE_LOCK_SECONDS = 300.0
_LIVE_LOCK_TEXT = "Live F1 session in progress"

_response_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_last_request_at = 0.0
_locked_until = 0.0


def clear_cache() -> None:
    """Drop cached responses and the live-session lock (for tests)."""
    global _locked_until
    _response_cache.clear()
    _locked_until = 0.0


def _get(endpoint: str, *, cache: bool = True, **params: Any) -> list[dict[str, Any]]:
    """GET one OpenF1 endpoint and return its rows; an empty result is ``[]``."""
    if os.environ.get(DISABLE_ENV, "").strip():
        raise OpenF1Unavailable(f"{DISABLE_ENV} is set")
    if time.monotonic() < _locked_until:
        raise OpenF1TransportError("OpenF1 is closed to free users while a live session runs")
    url = f"{BASE_URL}/{endpoint}?{urllib.parse.urlencode(sorted(params.items()))}"
    cached = _response_cache.get(url)
    if cached is not None and time.monotonic() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    payload = _request_json(url)
    if isinstance(payload, dict):
        payload = []
    if not isinstance(payload, list):
        raise OpenF1TransportError(f"{url} returned {type(payload).__name__}")
    rows = [row for row in payload if isinstance(row, dict)]
    if cache:
        _response_cache[url] = (time.monotonic(), rows)
    return rows


def _request_json(url: str) -> Any:
    """Fetch one URL with spacing between requests; wait and retry on 429, 5xx or a network failure.

    A replay runs for hours and stops on a transport failure, so a DNS blip or a cut read
    must not end it. In a full OpenF1 outage a call gives up after about two minutes.
    """
    global _last_request_at, _locked_until
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    for retry_wait in (*_RETRY_WAIT_SECONDS, None):
        wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # OpenF1 answers 404 with {"detail": "No results found."} for an empty query.
            if exc.code == 404:
                return []
            if exc.code == 401 and _LIVE_LOCK_TEXT in _error_body(exc):
                _locked_until = time.monotonic() + _LIVE_LOCK_SECONDS
                raise OpenF1TransportError(f"{url} returned 401: live session in progress") from exc
            if exc.code != 429 and exc.code < 500:
                raise OpenF1TransportError(f"{url} returned {exc.code}") from exc
            failure: Exception = exc
        # A car data body is about 3 MB; a cut read raises IncompleteRead, not OSError.
        except (OSError, ValueError, http.client.HTTPException) as exc:
            failure = exc
        if retry_wait is None:
            raise OpenF1TransportError(f"{url} failed after retries: {failure}") from failure
        time.sleep(retry_wait)
    raise AssertionError("unreachable")  # pragma: no cover - the last pass returns or raises


def _error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")
    except (AttributeError, OSError, ValueError):
        return ""


def _normalize_name(name: str) -> str:
    """Casefold and strip accents so "Sao Paulo" matches "São Paulo"."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold().strip()


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def find_session(year: int, race_name: str, session_name: str) -> dict[str, Any] | None:
    """Return the OpenF1 session row for a FastF1 race and session code, if any."""
    wanted_session = SESSION_NAMES.get(str(session_name))
    if wanted_session is None:
        return None
    wanted_race = _normalize_name(race_name)
    meetings = [
        meeting
        for meeting in _get("meetings", year=int(year))
        if _normalize_name(meeting.get("meeting_name", "")) == wanted_race
        and not meeting.get("is_cancelled")
    ]
    if not meetings:
        return None
    # A rescheduled race keeps its name; the latest uncancelled meeting is the one held.
    meeting = max(meetings, key=lambda row: str(row.get("date_start", "")))
    for session in _get("sessions", meeting_key=meeting["meeting_key"]):
        if session.get("session_name") == wanted_session and not session.get("is_cancelled"):
            return session
    return None


def session_completion_state(
    year: int, race_name: str, session_name: str, *, now: datetime | None = None
) -> CompletionState:
    """Return whether a session is finished, judged by OpenF1's published results."""
    session = find_session(year, race_name, session_name)
    if session is None:
        return "unknown"
    current = now or datetime.now(UTC)
    end = _parse_time(session.get("date_end"))
    if end is None:
        return "unknown"
    if current < end:
        return "incomplete"
    if str(session_name).startswith("FP"):
        return "completed"
    results = _get("session_result", session_key=session["session_key"])
    if len(results) >= _MIN_RESULT_ROWS:
        return "completed"
    return "incomplete" if current < end + _RESULTS_GRACE else "unknown"


def _driver_table(session_key: Any, *, map_team: bool = True) -> dict[int, tuple[str, str]]:
    """Map driver number to (three-letter code, characteristics or raw team name)."""
    table: dict[int, tuple[str, str]] = {}
    for row in _get("drivers", session_key=session_key):
        try:
            number = int(row["driver_number"])
        except (KeyError, TypeError, ValueError):
            continue
        code = str(row.get("name_acronym") or "").strip()
        team_raw = str(row.get("team_name") or "").strip()
        if code and team_raw:
            table[number] = (
                code,
                (map_team_to_characteristics(team_raw) or team_raw) if map_team else team_raw,
            )
    return table


def _classified_position(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def fetch_session_results(year: int, race_name: str, session_name: str) -> list[dict[str, Any]]:
    """Return a session's classification as grid rows, positions 1 to N.

    Classified drivers keep OpenF1's position. Unclassified ones (retired, DNS, DSQ,
    or a qualifying "RT") follow in order of laps completed, as the official
    classification orders retirements, then qualifying drivers with no time at all.
    Race-like sessions carry a ``dnf`` flag.
    Raises ``OpenF1Unavailable`` when the session or its drivers cannot be read.
    """
    session = find_session(year, race_name, session_name)
    if session is None:
        raise OpenF1Unavailable(f"no OpenF1 session for {year} {race_name} {session_name}")
    results = _get("session_result", session_key=session["session_key"])
    drivers = _driver_table(session["session_key"])
    if not results or not drivers:
        raise OpenF1Unavailable(f"OpenF1 has no results for {year} {race_name} {session_name}")

    classified = sorted(
        (row for row in results if _classified_position(row.get("position")) is not None),
        key=lambda row: int(row["position"]),
    )
    unclassified = sorted(
        (row for row in results if _classified_position(row.get("position")) is None),
        key=lambda row: -int(row.get("number_of_laps") or 0),
    )
    # OpenF1 leaves out entrants with no time or no start; the official classification
    # puts them last. Car number order keeps the result stable (it matched the stored
    # Canadian 2026 SQ order).
    listed = {int(row["driver_number"]) for row in results}
    absent = [{"driver_number": number} for number in sorted(set(drivers) - listed)]
    is_race_like = str(session_name) in _RACE_LIKE
    rows: list[dict[str, Any]] = []
    ordered = [
        *((row, True) for row in classified),
        *((row, False) for row in [*unclassified, *absent]),
    ]
    for position, (row, is_classified) in enumerate(ordered, start=1):
        number = int(row["driver_number"])
        if number not in drivers:
            raise OpenF1Unavailable(f"driver {number} missing from OpenF1 drivers")
        code, team = drivers[number]
        entry: dict[str, Any] = {
            "driver": code,
            "team": team,
            "position": position,
            "source": SOURCE,
        }
        if is_race_like:
            # Same rule as the FastF1 path: a classified driver finished, even one OpenF1
            # flags as retired after covering enough distance to be classified.
            entry["dnf"] = not is_classified or bool(row.get("dsq"))
        rows.append(entry)
    return rows


def _race_start_order(year: int, race_name: str) -> tuple[list[dict[str, Any]], Any] | None:
    """Return each driver's first race ``position`` row, or None unless it is a full 1..N order."""
    race = find_session(year, race_name, "R")
    if race is None:
        return None
    first: dict[int, int] = {}
    for row in sorted(
        _get("position", session_key=race["session_key"]), key=lambda r: str(r.get("date", ""))
    ):
        if row.get("driver_number") is not None and row.get("position") is not None:
            first.setdefault(int(row["driver_number"]), int(row["position"]))
    complete = len(first) == len(_driver_table(race["session_key"]))
    if not complete or sorted(first.values()) != list(range(1, len(first) + 1)):
        return None
    order = [{"driver_number": number, "position": place} for number, place in first.items()]
    return order, race["session_key"]


def fetch_starting_grid(year: int, race_name: str) -> list[dict[str, Any]]:
    """Return the order the race started in.

    Each driver's first race ``position`` row is the order the cars lined up in: it
    matched FastF1's grid on all 20 2026 races checked, while ``starting_grid`` kept
    Austin's pit-lane starter in his slot. Before the race has position data, the grid
    published against qualifying stands in. OpenF1 marks no pit-lane starts, so every
    row is a grid start. Raises ``OpenF1Unavailable`` when neither is published.
    """
    start_order = _race_start_order(year, race_name)
    if start_order is not None:
        grid, session_key = start_order
    else:
        qualifying = find_session(year, race_name, "Q")
        if qualifying is None:
            raise OpenF1Unavailable(f"no OpenF1 qualifying session for {year} {race_name}")
        grid = _get("starting_grid", session_key=qualifying["session_key"])
        session_key = qualifying["session_key"]
    drivers = _driver_table(session_key)
    if not grid or not drivers:
        raise OpenF1Unavailable(f"OpenF1 has no starting grid for {year} {race_name}")
    rows: list[dict[str, Any]] = []
    for row in sorted(grid, key=lambda item: int(item.get("position") or 0)):
        number = int(row["driver_number"])
        if number not in drivers:
            raise OpenF1Unavailable(f"driver {number} missing from OpenF1 drivers")
        code, team = drivers[number]
        rows.append(
            {
                "driver": code,
                "team": team,
                "position": int(row["position"]),
                "start_type": "grid",
                "source": SOURCE,
            }
        )
    return rows


def fetch_qualifying_best_laps(year: int, race_name: str) -> dict[str, float]:
    """Each driver's best qualifying lap in seconds (fastest of Q1, Q2, Q3).

    For finished events only; challengers use it for earlier events, never the current
    one. Raises ``OpenF1Unavailable`` when the session cannot be read.
    """
    session = find_session(year, race_name, "Q")
    if session is None:
        raise OpenF1Unavailable(f"no OpenF1 qualifying session for {year} {race_name}")
    drivers = _driver_table(session["session_key"])
    best: dict[str, float] = {}
    for row in _get("session_result", session_key=session["session_key"]):
        durations = row.get("duration")
        laps = [
            float(lap)
            for lap in (durations if isinstance(durations, list) else [durations])
            if isinstance(lap, int | float)
        ]
        number = row.get("driver_number")
        if laps and isinstance(number, int) and number in drivers:
            best[drivers[number][0]] = min(laps)
    return best
