"""Base scenario infrastructure for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol
import random


@dataclass
class ScenarioStep:
    """One step in a benchmark scenario."""

    message: str
    perplexity: float | None = None
    ground_truth_status: str | None = None  # e.g. "healthy", "stagnant"
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)


class Scenario(Protocol):
    """Protocol for scenario generators."""

    name: str

    def generate(self, rng: random.Random) -> Iterator[ScenarioStep]:
        """Yield scenario steps."""
        ...
