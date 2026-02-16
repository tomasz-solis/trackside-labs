"""
Persistence layer for F1 prediction artifacts.

Provides abstraction over file-based and database-backed storage.
"""

from .artifact_store import ArtifactStore
from .config import SUPABASE_KEY, SUPABASE_URL, USE_DB_STORAGE

__all__ = ["ArtifactStore", "USE_DB_STORAGE", "SUPABASE_URL", "SUPABASE_KEY"]
