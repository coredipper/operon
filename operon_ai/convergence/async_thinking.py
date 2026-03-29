"""AsyncThink-inspired Fork/Join execution within a single stage.

Provides an organizer that decomposes tasks into sub-queries, dispatches
them concurrently (simulated via sequential execution in this reference
implementation), and joins results with concurrency metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import time


@dataclass(frozen=True)
class AsyncThinkResult:
    """Result of Fork/Join execution."""

    outputs: tuple[Any, ...]
    concurrency_ratio: float  # η = avg active workers / capacity
    critical_path_ms: float  # minimum sequential latency
    total_ms: float  # actual wall-clock time
    fork_count: int  # number of sub-queries dispatched


@dataclass
class AsyncOrganizer:
    """Manages Fork/Join sub-query dispatch within a single stage.

    In this reference implementation, sub-queries run sequentially.
    A production deployment would use async/threading for true parallelism.
    """

    capacity: int = 4

    def fork(
        self,
        task: str,
        sub_queries: list[str],
        handler: Callable[[str], Any],
    ) -> AsyncThinkResult:
        """Dispatch sub-queries to handler, collect results.

        Measures per-query latency for concurrency metrics.
        """
        outputs = []
        latencies_ms = []

        for query in sub_queries:
            start = time.time()
            result = handler(query)
            elapsed_ms = (time.time() - start) * 1000
            outputs.append(result)
            latencies_ms.append(elapsed_ms)

        total_ms = sum(latencies_ms)
        critical_path = max(latencies_ms) if latencies_ms else 0.0
        # η = simulated concurrency: in sequential mode, ratio is 1/capacity
        # In true parallel mode, this would be actual concurrent workers / capacity
        active = min(len(sub_queries), self.capacity)
        eta = active / self.capacity if self.capacity > 0 else 0.0

        return AsyncThinkResult(
            outputs=tuple(outputs),
            concurrency_ratio=eta,
            critical_path_ms=critical_path,
            total_ms=total_ms,
            fork_count=len(sub_queries),
        )

    def critical_path_latency(self, task_dag: dict[str, list[str]]) -> float:
        """Compute critical-path latency of a Fork/Join DAG.

        task_dag maps task_id -> list of dependency task_ids.
        Uses dynamic programming: CPL(t) = cost(t) + max(CPL(dep) for dep in deps).

        Returns the critical path length assuming unit cost per task.
        """
        memo: dict[str, float] = {}

        def _cpl(task_id: str) -> float:
            if task_id in memo:
                return memo[task_id]
            deps = task_dag.get(task_id, [])
            if not deps:
                memo[task_id] = 1.0
            else:
                memo[task_id] = 1.0 + max(_cpl(d) for d in deps)
            return memo[task_id]

        if not task_dag:
            return 0.0
        return max(_cpl(t) for t in task_dag)


def async_stage_handler(
    organizer: AsyncOrganizer,
    decompose: Callable[[str], list[str]],
    handler: Callable[[str], Any],
    join: Callable[[list[Any]], Any] | None = None,
) -> Callable[[str], dict[str, Any]]:
    """Build a stage handler that uses Fork/Join internally.

    Args:
        organizer: AsyncOrganizer managing dispatch
        decompose: Splits task into sub-queries
        handler: Processes each sub-query
        join: Optional combiner for outputs (default: return all)

    Returns:
        A handler compatible with SkillStage.handler
    """

    def _handler(task: str) -> dict[str, Any]:
        sub_queries = decompose(task)
        result = organizer.fork(task, sub_queries, handler)

        combined = result.outputs
        if join is not None:
            combined = join(list(result.outputs))

        return {
            "output": combined,
            "async_think": {
                "fork_count": result.fork_count,
                "concurrency_ratio": result.concurrency_ratio,
                "critical_path_ms": result.critical_path_ms,
                "total_ms": result.total_ms,
            },
        }

    return _handler
