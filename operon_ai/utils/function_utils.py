from inspect import signature
from typing import Any, Callable

def call_arity(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        params = list(signature(fn).parameters.values())
    except (TypeError, ValueError):
        return fn(*args)
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        return fn(*args)
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return fn(*args[:len(positional)])
