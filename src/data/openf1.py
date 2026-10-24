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

CompletionState = Literal["completed", "incomplete", "unknown"]


class OpenF1Unavailable(RuntimeError):
    """Raised when OpenF1 is disabled, unreachable or returns an unreadable body."""


# OpenF1 free tier: 3 requests a second and 30 a minute, 429 above that.
_MIN_INTERVAL_SECONDS = 2.1
_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_WAIT_SECONDS = 5.0

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
        raise OpenF1Unavailable("OpenF1 is closed to free users while a live session runs")
    url = f"{BASE_URL}/{endpoint}?{urllib.parse.urlencode(sorted(params.items()))}"
    cached = _response_cache.get(url)
    if cached is not None and time.monotonic() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    payload = _request_json(url)
    if isinstance(payload, dict):
        payload = []
    if not isinstance(payload, list):
        raise OpenF1Unavailable(f"{url} returned {type(payload).__name__}")
    rows = [row for row in payload if isinstance(row, dict)]
    if cache:
        _response_cache[url] = (time.monotonic(), rows)
    return rows


def _request_json(url: str) -> Any:
    """Fetch one URL with spacing between requests and a short wait on 429."""
    global _last_request_at, _locked_until
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
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
                raise OpenF1Unavailable(f"{url} returned 401: live session in progress") from exc
            if exc.code == 429 and attempt < _RATE_LIMIT_RETRIES:
                time.sleep(_RATE_LIMIT_WAIT_SECONDS * (attempt + 1))
                continue
            raise OpenF1Unavailable(f"{url} returned {exc.code}") from exc
        except (OSError, ValueError) as exc:
            raise OpenF1Unavailable(f"{url} failed: {exc}") from exc
    raise OpenF1Unavailable(f"{url} kept returning 429")  # pragma: no cover - loop returns


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


def fetch_starting_grid(year: int, race_name: str) -> list[dict[str, Any]]:
    """Return the race starting grid published against the qualifying session.

    OpenF1 keys the race grid to qualifying. Every row is a grid start; OpenF1 does
    not mark pit-lane starts. Raises ``OpenF1Unavailable`` when it is not published.
    """
    qualifying = find_session(year, race_name, "Q")
    if qualifying is None:
        raise OpenF1Unavailable(f"no OpenF1 qualifying session for {year} {race_name}")
    grid = _get("starting_grid", session_key=qualifying["session_key"])
    drivers = _driver_table(qualifying["session_key"])
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
