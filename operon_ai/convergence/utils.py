from typing import Any

_HEX_CHARS = frozenset("0123456789abcdef")
_MIN_HASH_LEN = 8

def is_well_formed_hash(value: Any) -> bool:
    """Hex string of length >= _MIN_HASH_LEN, lowercase only."""
    if not isinstance(value, str):
        return False
    if len(value) < _MIN_HASH_LEN:
        return False
    return all(ch in _HEX_CHARS for ch in value)
