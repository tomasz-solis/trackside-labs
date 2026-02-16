"""
Supabase client wrapper for database operations.

Provides a singleton Supabase client with automatic connection pooling.
"""

import logging

from supabase import Client, create_client

from .config import SUPABASE_KEY, SUPABASE_URL, is_db_enabled

logger = logging.getLogger(__name__)

# Singleton Supabase client instance
_supabase_client: Client | None = None


def get_supabase_client() -> Client:
    """
    Get or create Supabase client (singleton pattern).

    Returns:
        Supabase client instance

    Raises:
        RuntimeError: If DB storage not enabled or credentials missing
    """
    global _supabase_client

    if not is_db_enabled():
        raise RuntimeError(
            "Database storage is not enabled. Set USE_DB_STORAGE to 'db_only', 'fallback', or 'dual_write'"
        )

    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

        try:
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info(f"Supabase client initialized: {SUPABASE_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise RuntimeError(f"Failed to connect to Supabase: {e}") from e

    return _supabase_client


def check_connection() -> tuple[bool, str]:
    """
    Health check: Verify Supabase connection is working.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        client = get_supabase_client()
        # Quick health check: query for any artifact (limit 1)
        result = client.table("artifacts").select("id").limit(1).execute()
        return True, f"Supabase connection healthy ({len(result.data)} row(s) accessible)"
    except Exception as e:
        return False, f"Supabase connection failed: {e}"


def close_client() -> None:
    """
    Close Supabase client connection.

    Note: Supabase Python client manages connections internally,
    so this is primarily for cleanup during testing or shutdown.
    """
    global _supabase_client
    if _supabase_client is not None:
        # Supabase client doesn't have explicit close method
        # Connection pooling is handled automatically
        _supabase_client = None
        logger.info("Supabase client closed")
