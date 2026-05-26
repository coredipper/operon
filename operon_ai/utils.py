import inspect
from inspect import signature
from typing import Any, Callable


def _call_arity(fn: Callable[..., Any], *args: Any) -> Any:
    """Safely calls a function with only the arguments it can accept."""
    try:
        params = list(signature(fn).parameters.values())
    except (TypeError, ValueError):
        return fn(*args)
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return fn(*args)
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return fn(*args[: len(positional)])
