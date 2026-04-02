"""C8 FilesystemOptimizer protocol — harness-level optimization.

Distinct from C7's EvolutionaryOptimizer (prompt-level optimization).
C7 optimizes prompts within a fixed organism structure.
C8 optimizes the organism structure itself.

The separation validates that prompt-level optimization (gene expression)
and harness-level optimization (genotype search) are genuinely different
levels of biological organization.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .meta_types import AssessmentRecord, CandidateConfig


@runtime_checkable
class FilesystemOptimizer(Protocol):
    """C8 harness-level optimization protocol.

    Operates on CandidateConfig (organism configurations), not str (prompts).
    Uses filesystem history as the exogenous signal for proposers.
    """

    def seed(self, initial_configs: list[CandidateConfig]) -> None:
        """Initialize the population with seed candidates."""
        ...

    def step(self, task_id: str) -> tuple[CandidateConfig, AssessmentRecord]:
        """Run one evolution step: propose, evaluate, record."""
        ...

    def best(self) -> CandidateConfig:
        """Return the current best candidate."""
        ...

    def history(self) -> list[AssessmentRecord]:
        """Return full evaluation history."""
        ...
