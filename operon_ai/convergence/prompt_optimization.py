"""Interface definitions for prompt optimization integration (Phase C7).

Defines the Protocol that DSPy, A-Evolve, or custom optimizers must
implement to plug into SkillStage.prompt_optimizer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

from ..patterns.types import SkillStage


@runtime_checkable
class PromptOptimizer(Protocol):
    """Core optimizer protocol — score and improve a stage prompt."""

    def optimize(
        self,
        current_prompt: str,
        task: str,
        stage_name: str,
        feedback: dict[str, Any] | None = None,
    ) -> str: ...

    def score(
        self,
        prompt: str,
        task: str,
        result: Any,
        success: bool,
    ) -> float: ...


@runtime_checkable
class EvolutionaryOptimizer(Protocol):
    """Extended protocol for population-based prompt search (A-Evolve style)."""

    def initialize_population(
        self, seed_prompt: str, population_size: int
    ) -> list[str]: ...

    def evolve(
        self, population: list[str], fitness_scores: list[float]
    ) -> list[str]: ...

    def select_best(
        self, population: list[str], fitness_scores: list[float]
    ) -> str: ...


class NoOpOptimizer:
    """Reference implementation -- returns prompts unchanged."""

    def optimize(
        self,
        current_prompt: str,
        task: str,
        stage_name: str,
        feedback: dict[str, Any] | None = None,
    ) -> str:
        return current_prompt

    def score(
        self,
        prompt: str,
        task: str,
        result: Any,
        success: bool,
    ) -> float:
        return 1.0 if success else 0.0


def attach_optimizer(
    stages: list[SkillStage], optimizer: PromptOptimizer
) -> list[SkillStage]:
    """Return new stages with *optimizer* attached to each stage's prompt_optimizer hook."""
    return [
        replace(
            s,
            prompt_optimizer=lambda prompt, t=s.name: optimizer.optimize(
                prompt, "", t
            ),
        )
        for s in stages
    ]
