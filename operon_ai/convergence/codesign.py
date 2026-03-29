"""Zardini co-design formalization for convergence adapter composition.

Each convergence adapter is a Design Problem (DP) — a monotone map
from a resource poset to a functionality poset. The full convergence
stack is series/parallel composition of DPs. The adaptive assembly
loop (run → record → score → select) is feedback composition, and
fixed-point iteration proves whether scoring stabilizes.

Based on: Zardini, G. (2023). "Co-Design of Complex Systems: From
Compositionality to Monotone Theory." ETH Zurich.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class AdapterDP(Protocol):
    """Design Problem protocol — a monotone map between resource and functionality posets.

    A DP takes resources (inputs/constraints) and produces functionalities
    (outputs/capabilities). Monotonicity: more resources → at least as many functionalities.
    """

    def evaluate(self, resources: dict[str, Any]) -> dict[str, Any]:
        """Map resources to functionalities."""
        ...

    def is_feasible(self, resources: dict[str, Any]) -> bool:
        """Check if the given resources are sufficient to produce any functionality."""
        ...


@dataclass(frozen=True)
class DesignProblem:
    """Concrete implementation of a design problem.

    Wraps a callable that maps resource dicts to functionality dicts,
    plus a feasibility predicate.
    """

    name: str
    evaluate_fn: Callable[[dict[str, Any]], dict[str, Any]]
    feasibility_fn: Callable[[dict[str, Any]], bool] = lambda r: True

    def evaluate(self, resources: dict[str, Any]) -> dict[str, Any]:
        return self.evaluate_fn(resources)

    def is_feasible(self, resources: dict[str, Any]) -> bool:
        return self.feasibility_fn(resources)


def compose_series(dp1: AdapterDP, dp2: AdapterDP, name: str = "") -> DesignProblem:
    """Series composition: dp1's output becomes dp2's input.

    The composite is feasible iff dp1 is feasible AND dp2 is feasible
    on dp1's output.
    """

    def evaluate_fn(resources: dict[str, Any]) -> dict[str, Any]:
        intermediate = dp1.evaluate(resources)
        return dp2.evaluate(intermediate)

    def feasibility_fn(resources: dict[str, Any]) -> bool:
        if not dp1.is_feasible(resources):
            return False
        intermediate = dp1.evaluate(resources)
        return dp2.is_feasible(intermediate)

    return DesignProblem(
        name=name or f"{getattr(dp1, 'name', 'dp1')}→{getattr(dp2, 'name', 'dp2')}",
        evaluate_fn=evaluate_fn,
        feasibility_fn=feasibility_fn,
    )


def compose_parallel(dp1: AdapterDP, dp2: AdapterDP, name: str = "") -> DesignProblem:
    """Parallel composition: both DPs receive the same resources,
    outputs are merged.

    The composite is feasible iff BOTH are feasible on the given resources.
    """

    def evaluate_fn(resources: dict[str, Any]) -> dict[str, Any]:
        out1 = dp1.evaluate(resources)
        out2 = dp2.evaluate(resources)
        return {**out1, **out2}

    def feasibility_fn(resources: dict[str, Any]) -> bool:
        return dp1.is_feasible(resources) and dp2.is_feasible(resources)

    return DesignProblem(
        name=name or f"({getattr(dp1, 'name', 'dp1')}‖{getattr(dp2, 'name', 'dp2')})",
        evaluate_fn=evaluate_fn,
        feasibility_fn=feasibility_fn,
    )


def feedback_fixed_point(
    dp: AdapterDP,
    initial: dict[str, Any],
    *,
    max_iterations: int = 100,
    convergence_key: str | None = None,
    epsilon: float = 0.01,
) -> tuple[dict[str, Any], int, bool]:
    """Iterative fixed-point computation for feedback composition.

    Repeatedly applies dp.evaluate(state) until the output stabilizes
    (the value at convergence_key changes by less than epsilon) or
    max_iterations is reached.

    Returns:
        (final_state, iterations_used, converged)
    """
    state = dict(initial)
    for i in range(1, max_iterations + 1):
        new_state = dp.evaluate(state)

        if convergence_key is not None:
            old_val = state.get(convergence_key, 0.0)
            new_val = new_state.get(convergence_key, 0.0)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if abs(new_val - old_val) < epsilon:
                    return new_state, i, True
        elif new_state == state:
            return new_state, i, True

        state = new_state

    return state, max_iterations, False


def feasibility_check(dp: AdapterDP, resources: dict[str, Any]) -> dict[str, Any]:
    """Check feasibility and return diagnostic info.

    Returns dict with: feasible (bool), resources, functionalities (if feasible),
    reason (if not feasible).
    """
    feasible = dp.is_feasible(resources)
    result: dict[str, Any] = {"feasible": feasible, "resources": resources}
    if feasible:
        result["functionalities"] = dp.evaluate(resources)
    else:
        result["reason"] = "Resources insufficient for this design problem"
    return result
