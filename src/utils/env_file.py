"""Shared ``.env`` loading for command-line scripts.

Kept dependency-free and side-effect-free on purpose: entrypoints such as
``scripts/warmup_precompute.py`` import this *before* the heavy ``src`` imports,
because ``src.persistence.config`` validates Supabase credentials at import time
and therefore needs the env file already applied.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(env_file: Path) -> None:
    """Load ``KEY=VALUE`` pairs from ``env_file`` into ``os.environ``.

    Existing environment variables take precedence: the file only fills gaps, so an
    explicitly exported value is never overwritten. Blank lines, comments, and lines
    without ``=`` are skipped. One matching pair of surrounding quotes is removed.

    Raises:
        FileNotFoundError: If ``env_file`` does not exist.
    """
    if not env_file.exists():
        raise FileNotFoundError(f"Env file not found: {env_file}")
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
