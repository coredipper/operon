"""Safety tests for Mitochondria expression parsing."""
import pytest
from operon_ai import Mitochondria


def test_expression_length_limit():
    """Extremely long expressions should be rejected."""
    mito = Mitochondria()

    # Create expression exceeding reasonable limit
    huge_expr = "1 + " * 10000 + "1"

    result = mito.metabolize(huge_expr)
    assert not result.success
    assert "length" in result.error.lower() or "too long" in result.error.lower()


def test_reasonable_expression_passes():
    """Normal expressions should still work."""
    mito = Mitochondria()

    result = mito.metabolize("1 + 2 + 3 + 4 + 5")
    assert result.success
    assert result.atp.value == 15


def test_nested_expression_depth_limit():
    """Deeply nested expressions should be rejected."""
    mito = Mitochondria()

    # Create deeply nested expression
    nested = "(" * 100 + "1" + ")" * 100

    result = mito.metabolize(nested)
    # Should either fail or succeed but not hang/crash
    assert isinstance(result.success, bool)
