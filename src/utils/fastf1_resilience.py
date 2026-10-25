"""Retry, circuit-breaker and archive-refusal helpers for FastF1 network calls."""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

from src.utils.operational_observability import record_alert, record_counter

logger = logging.getLogger(__name__)

# FastF1 archive pages: host, season, then the event and session folder.
_ARCHIVE_URL = re.compile(
    r"^https://(livetiming\.formula1\.com|livetiming-mirror\.fastf1\.dev)/static/(\d{4})/"
    r"([^/]+/[^/]+/)?"
)
_ARCHIVE_REFUSAL_TTL_SECONDS = 600.0
_F1TV_TOKEN_STATE = ("fastf1_auth", "f1tv_token")
# Archive prefix -> (monotonic time of the refusal, HTTP status).
_archive_refusals: dict[str, tuple[float, int]] = {}
_archive_guard_installed = False


class FastF1CircuitOpenError(RuntimeError):
    """Raised when circuit breaker is open for an operation."""


class FastF1RetryExhaustedError(RuntimeError):
    """Raised when retries/budget are exhausted for an operation."""


@dataclass(frozen=True)
class FastF1ResiliencePolicy:
    """Policy controls for resilient FastF1 calls."""

    max_attempts: int = 3
    timeout_budget_seconds: float = 10.0
    initial_backoff_seconds: float = 0.35
    max_backoff_seconds: float = 3.0
    backoff_multiplier: float = 2.0
    circuit_breaker_failure_threshold: int = 4
    circuit_breaker_cooldown_seconds: float = 45.0


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_until_monotonic: float = 0.0


_circuit_lock = RLock()
_circuit_state: dict[str, _CircuitState] = {}


def _state_for(operation_name: str) -> _CircuitState:
    with _circuit_lock:
        return _circuit_state.setdefault(str(operation_name), _CircuitState())


def call_with_resilience(
    operation_name: str,
    fn: Callable[[], Any],
    *,
    labels: dict[str, Any] | None = None,
    policy: FastF1ResiliencePolicy | None = None,
) -> Any:
    """
    Execute FastF1 operation with retry/backoff timeout budget and circuit breaker.

    Timeout budget is a total elapsed-time budget across all attempts.
    """
    cfg = policy or FastF1ResiliencePolicy()
    op_name = str(operation_name)
    start = time.monotonic()
    backoff = max(0.0, float(cfg.initial_backoff_seconds))
    attempt = 0

    while True:
        attempt += 1
        now = time.monotonic()
        elapsed = now - start
        budget_remaining = float(cfg.timeout_budget_seconds) - elapsed
        if budget_remaining <= 0:
            record_counter(
                "fastf1_timeout_budget_exhausted_total",
                labels={**(labels or {}), "operation": op_name},
            )
            raise FastF1RetryExhaustedError(
                f"FastF1 timeout budget exhausted for operation={op_name}"
            )

        state = _state_for(op_name)
        if state.opened_until_monotonic > now:
            record_counter(
                "fastf1_circuit_open_total",
                labels={**(labels or {}), "operation": op_name},
            )
            raise FastF1CircuitOpenError(
                f"FastF1 circuit is open for operation={op_name} "
                f"(retry after {state.opened_until_monotonic - now:.1f}s)"
            )

        try:
            result = fn()
            if state.consecutive_failures > 0:
                with _circuit_lock:
                    state.consecutive_failures = 0
                    state.opened_until_monotonic = 0.0
            if attempt > 1:
                record_counter(
                    "fastf1_retry_recovered_total",
                    labels={**(labels or {}), "operation": op_name, "attempts": attempt},
                )
            return result
        except Exception as exc:
            record_counter(
                "fastf1_call_failure_total",
                labels={**(labels or {}), "operation": op_name, "attempt": attempt},
            )

            with _circuit_lock:
                state.consecutive_failures += 1
                if state.consecutive_failures >= int(cfg.circuit_breaker_failure_threshold):
                    state.opened_until_monotonic = now + float(cfg.circuit_breaker_cooldown_seconds)
                    record_counter(
                        "fastf1_circuit_trip_total",
                        labels={**(labels or {}), "operation": op_name},
                    )
                    record_alert(
                        "fastf1_circuit_trip",
                        (
                            f"FastF1 circuit opened for operation={op_name} after "
                            f"{state.consecutive_failures} consecutive failures."
                        ),
                        labels={**(labels or {}), "operation": op_name},
                    )

            if attempt >= int(cfg.max_attempts):
                raise FastF1RetryExhaustedError(
                    f"FastF1 retries exhausted for operation={op_name}; attempts={attempt}"
                ) from exc

            budget_remaining = float(cfg.timeout_budget_seconds) - (time.monotonic() - start)
            if budget_remaining <= 0:
                raise FastF1RetryExhaustedError(
                    f"FastF1 timeout budget exhausted for operation={op_name}"
                ) from exc

            sleep_seconds = min(
                max(0.0, backoff),
                max(0.0, budget_remaining / 2.0),
                float(cfg.max_backoff_seconds),
            )
            if sleep_seconds <= 0:
                raise FastF1RetryExhaustedError(
                    f"FastF1 budget did not allow retry for operation={op_name}"
                ) from exc

            record_counter(
                "fastf1_retry_attempt_total",
                labels={**(labels or {}), "operation": op_name, "attempt": attempt},
            )
            logger.warning(
                "FastF1 call failed: operation=%s attempt=%s/%s error=%s; retrying in %.2fs",
                op_name,
                attempt,
                cfg.max_attempts,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            backoff = min(float(cfg.max_backoff_seconds), backoff * float(cfg.backoff_multiplier))


def _archive_keys(url: str) -> tuple[str, str | None] | None:
    """Return the season and session prefixes of a FastF1 archive URL, else ``None``."""
    match = _ARCHIVE_URL.match(url)
    if match is None:
        return None
    host, year, session_dir = match.groups()
    season = f"{host}/static/{year}/"
    return season, (season + session_dir if session_dir else None)


def _refused_status(keys: tuple[str, str | None]) -> int | None:
    now = time.monotonic()
    for key in keys:
        refusal = _archive_refusals.get(key) if key else None
        if refusal is not None and now - refusal[0] < _ARCHIVE_REFUSAL_TTL_SECONDS:
            return refusal[1]
    return None


def _note_refusal(url: str, keys: tuple[str, str | None], status: int) -> None:
    season, session = keys
    first_page = bool(session) and url.split("?")[0].endswith("/SessionInfo.jsonStream")
    if status == 401 or (status == 403 and not session):
        key = season
    elif status in (403, 404) and first_page and session:
        # SessionInfo is the first page FastF1 asks for; without it the session is absent.
        # With an F1 TV token the F1 host answers 403 for a session not published yet.
        key = session
    else:
        return
    _archive_refusals[key] = (time.monotonic(), status)
    logger.info(
        "FastF1 archive refused %s (HTTP %s); skipping it for %.0f minutes",
        key,
        status,
        _ARCHIVE_REFUSAL_TTL_SECONDS / 60,
    )


def install_archive_guard() -> None:
    """Answer refused FastF1 archive requests from memory instead of the network.

    FastF1's archive has refused 2026 since about 2026-10-15 (FastF1 #978): the F1
    host answers 401 and the FastF1 mirror 404. Each refused session load still spent
    up to 11 of FastF1's 500 calls an hour, and one cron run used them all up before
    the OpenF1 fallbacks ran. After a 401 for a season, or a 403 or 404 for a
    session's first page, the same requests get the same status for 10 minutes
    without a call. FastF1 then fails the load as before, and callers use OpenF1.
    With an F1 TV token (FastF1 3.9) a failed sign-in also ends in a 401, so it is
    retried once per 10 minutes, not on every request.
    """
    global _archive_guard_installed
    if _archive_guard_installed:
        return

    import requests  # type: ignore[import-untyped,unused-ignore]
    from fastf1.req import Cache

    fetch = Cache.requests_get

    def guarded_get(url: str, **kwargs: Any) -> Any:
        keys = _archive_keys(url)
        if keys is not None:
            status = _refused_status(keys)
            if status is not None:
                response = requests.Response()
                response.status_code = status
                response.url = url
                response._content = b""
                return response
        response = fetch(url, **kwargs)
        if keys is not None:
            _note_refusal(url, keys, int(response.status_code))
        return response

    def requests_get(cls: type, url: str | Sequence[str], **kwargs: Any) -> Any:
        # FastF1 3.9 passes the F1 host and the mirror together and keeps the first
        # answer below 400; guard each one the same way.
        if isinstance(url, str) or not url:
            return guarded_get(url, **kwargs)  # type: ignore[arg-type]
        response = None
        for one in url:
            response = guarded_get(one, **kwargs)
            if int(response.status_code) < 400:
                return response
        return response

    Cache.requests_get = classmethod(requests_get)
    _archive_guard_installed = True


def restore_f1tv_token() -> str | None:
    """Load the saved F1 TV token into FastF1 and return its value, else ``None``.

    Render starts each cron run with an empty disk, so FastF1 would sign in on
    every run. Only runs where ``F1_USERNAME`` is set; the token is never logged.
    """
    if not os.environ.get("F1_USERNAME"):
        return None
    from fastf1.internals.f1account import SubscriptionToken, token_manager

    from src.persistence.runtime_state_store import RuntimeStateStore

    try:
        saved = RuntimeStateStore().get_record(*_F1TV_TOKEN_STATE)
        if not saved:
            return None
        token = SubscriptionToken(value=str(saved["value"]), expires_at=float(saved["expires_at"]))
        if not token.is_valid():
            return None
        token_manager.store(token)
        return token.value
    except Exception as exc:
        logger.warning("Could not restore the saved F1 TV token: %s", type(exc).__name__)
        return None


def save_f1tv_token(restored: str | None) -> None:
    """Save FastF1's F1 TV token when this run signed in for a new one."""
    if not os.environ.get("F1_USERNAME"):
        return
    from fastf1.internals.f1account import token_manager

    from src.persistence.runtime_state_store import RuntimeStateStore

    token = token_manager.get_stored()
    if token is None or token.value == restored or not token.is_valid():
        return
    try:
        RuntimeStateStore().upsert_record(
            *_F1TV_TOKEN_STATE, {"value": token.value, "expires_at": token.expires_at}
        )
    except Exception as exc:
        logger.warning("Could not save the F1 TV token: %s", type(exc).__name__)
