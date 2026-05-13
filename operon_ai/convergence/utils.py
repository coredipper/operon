"""Shared utilities for convergence modules."""

from typing import Any

_HEX_CHARS = frozenset("0123456789abcdef")
_MIN_HASH_LEN = 8


def is_well_formed_hash(value: Any) -> bool:
    """Check if value is a hex string of length >= 8, lowercase only.

    Operon's truncation convention is sha256(...).hexdigest()[:N] with
    N typically 8/12/16. Eight is the floor — anything shorter is too
    weak to count as a provenance marker.
    """
    if not isinstance(value, str):
        return False
    if len(value) < _MIN_HASH_LEN:
        return False
    return all(ch in _HEX_CHARS for ch in value)
