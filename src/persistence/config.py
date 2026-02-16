"""
Configuration for persistence layer.

Environment variables:
- USE_DB_STORAGE: 'file_only', 'db_only', 'fallback', 'dual_write'
- SUPABASE_URL: Supabase project URL (https://xxx.supabase.co)
- SUPABASE_KEY: Supabase anon public key
"""

import os
from typing import Literal

# Storage mode configuration
USE_DB_STORAGE = os.getenv("USE_DB_STORAGE", "file_only")
StorageMode = Literal["file_only", "db_only", "fallback", "dual_write"]

# Validate storage mode
VALID_MODES: set[str] = {"file_only", "db_only", "fallback", "dual_write"}
if USE_DB_STORAGE not in VALID_MODES:
    raise ValueError(
        f"Invalid USE_DB_STORAGE value: {USE_DB_STORAGE}. Must be one of: {', '.join(VALID_MODES)}"
    )

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validate credentials if DB storage is enabled
if USE_DB_STORAGE != "file_only":
    if not SUPABASE_URL:
        raise ValueError(
            f"SUPABASE_URL environment variable is required when USE_DB_STORAGE={USE_DB_STORAGE}"
        )
    if not SUPABASE_KEY:
        raise ValueError(
            f"SUPABASE_KEY environment variable is required when USE_DB_STORAGE={USE_DB_STORAGE}"
        )


def get_storage_mode() -> str:
    """Get current storage mode."""
    return USE_DB_STORAGE


def is_db_enabled() -> bool:
    """Check if database storage is enabled."""
    return USE_DB_STORAGE != "file_only"


def is_file_enabled() -> bool:
    """Check if file storage is enabled."""
    return USE_DB_STORAGE != "db_only"


def should_write_to_db() -> bool:
    """Check if writes should go to database."""
    return USE_DB_STORAGE in ("db_only", "fallback", "dual_write")


def should_write_to_file() -> bool:
    """Check if writes should go to files."""
    return USE_DB_STORAGE in ("file_only", "dual_write")


def should_read_db_first() -> bool:
    """Check if reads should try database first."""
    return USE_DB_STORAGE in ("db_only", "fallback")
