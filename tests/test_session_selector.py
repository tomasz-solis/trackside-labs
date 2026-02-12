"""Tests for session selector weekend-type compatibility."""

from src.utils.session_selector import get_prediction_context, get_prediction_workflow


def test_get_prediction_context_accepts_conventional_alias():
    """`conventional` should be accepted as alias for legacy `normal`."""
    context = get_prediction_context("post_fp2", weekend_type="conventional")
    assert context["next_prediction"] == "qualifying"


def test_get_prediction_workflow_accepts_conventional_alias():
    """Workflow helper should accept `conventional` weekend type."""
    workflow = get_prediction_workflow("conventional")
    assert len(workflow) > 0
