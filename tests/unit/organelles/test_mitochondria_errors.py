"""Error context tests for Mitochondria."""
import pytest
from operon_ai import Mitochondria


def test_error_contains_context():
    """Metabolic errors should contain useful context."""
    mito = Mitochondria()

    # Invalid expression
    result = mito.metabolize("undefined_function()")

    assert not result.success
    assert result.error is not None
    # Error should contain expression context
    assert "undefined_function" in result.error or "not defined" in result.error.lower()


def test_error_contains_pathway_info():
    """Errors should indicate which pathway failed."""
    mito = Mitochondria()

    result = mito.metabolize("invalid syntax here!!!")

    assert not result.success
    assert result.pathway is not None  # Should still set pathway
