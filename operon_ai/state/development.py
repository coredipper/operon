"""Developmental staging — critical periods and capability gating.

Maps telomere lifecycle progress to developmental stages with configurable
thresholds. Critical periods close permanently as the organism matures.

Biological analogy: neurodevelopmental critical periods where specific
neural circuits are maximally plastic (e.g., language acquisition, imprinting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from .telomere import Telomere, TelomereStatus


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class DevelopmentalStage(Enum):
    """Developmental stages mapped to organism maturity."""

    EMBRYONIC = "embryonic"
    JUVENILE = "juvenile"
    ADOLESCENT = "adolescent"
    MATURE = "mature"


_STAGE_ORDER: dict[DevelopmentalStage, int] = {
    DevelopmentalStage.EMBRYONIC: 0,
    DevelopmentalStage.JUVENILE: 1,
    DevelopmentalStage.ADOLESCENT: 2,
    DevelopmentalStage.MATURE: 3,
}

_PLASTICITY: dict[DevelopmentalStage, float] = {
    DevelopmentalStage.EMBRYONIC: 1.0,
    DevelopmentalStage.JUVENILE: 0.75,
    DevelopmentalStage.ADOLESCENT: 0.5,
    DevelopmentalStage.MATURE: 0.25,
}


def stage_reached(current: DevelopmentalStage, required: DevelopmentalStage) -> bool:
    """Return True if current stage >= required stage."""
    return _STAGE_ORDER[current] >= _STAGE_ORDER[required]


@dataclass(frozen=True)
class CriticalPeriod:
    """A time-limited learning window that closes as the organism matures."""

    name: str
    opens_at: DevelopmentalStage
    closes_at: DevelopmentalStage
    capability: str
    learning_rate_multiplier: float = 2.0


@dataclass(frozen=True)
class DevelopmentConfig:
    """Configurable thresholds for stage transitions.

    Thresholds are fractions of max_operations consumed.
    """

    juvenile_threshold: float = 0.10
    adolescent_threshold: float = 0.35
    mature_threshold: float = 0.70


@dataclass(frozen=True)
class StageTransition:
    """Record of a developmental stage transition."""

    from_stage: DevelopmentalStage
    to_stage: DevelopmentalStage
    tick_count: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class DevelopmentStatus:
    """Snapshot of developmental state."""

    stage: DevelopmentalStage
    telomere_status: TelomereStatus
    learning_plasticity: float
    open_periods: tuple[str, ...]
    closed_periods: tuple[str, ...]
    tick_count: int
    transitions: tuple[StageTransition, ...]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@dataclass
class DevelopmentController:
    """Developmental staging overlay on the Telomere lifecycle.

    Wraps a Telomere (composition, not inheritance) and maps consumed
    fraction to a DevelopmentalStage. Manages critical periods and
    learning plasticity.
    """

    telomere: Telomere
    config: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    critical_periods: tuple[CriticalPeriod, ...] = ()
    on_stage_change: Callable[[DevelopmentalStage, DevelopmentalStage], None] | None = None
    silent: bool = False

    _current_stage: DevelopmentalStage = field(default=DevelopmentalStage.EMBRYONIC, init=False)
    _transitions: list[StageTransition] = field(default_factory=list, init=False)
    _tick_count: int = field(default=0, init=False)

    def tick(self, cost: int = 1) -> bool:
        """Delegate to telomere.tick() and update developmental stage."""
        can_continue = self.telomere.tick(cost)
        self._tick_count += 1
        self._update_stage()
        return can_continue

    @property
    def stage(self) -> DevelopmentalStage:
        """Current developmental stage."""
        return self._current_stage

    @property
    def learning_plasticity(self) -> float:
        """Learning plasticity (1.0=maximum, decreasing with maturity)."""
        return _PLASTICITY[self._current_stage]

    def is_critical_period_open(self, period_name: str) -> bool:
        """Check if a named critical period is currently open."""
        for p in self.critical_periods:
            if p.name == period_name:
                return (
                    stage_reached(self._current_stage, p.opens_at)
                    and not stage_reached(self._current_stage, p.closes_at)
                )
        return False

    def open_critical_periods(self) -> list[CriticalPeriod]:
        """Return all currently open critical periods."""
        return [
            p for p in self.critical_periods
            if stage_reached(self._current_stage, p.opens_at)
            and not stage_reached(self._current_stage, p.closes_at)
        ]

    def closed_critical_periods(self) -> list[CriticalPeriod]:
        """Return all critical periods that have permanently closed."""
        return [
            p for p in self.critical_periods
            if stage_reached(self._current_stage, p.closes_at)
        ]

    def can_acquire_stage(self, min_stage: DevelopmentalStage) -> bool:
        """Check if current stage meets the minimum requirement."""
        return stage_reached(self._current_stage, min_stage)

    def get_status(self) -> DevelopmentStatus:
        """Full status snapshot."""
        return DevelopmentStatus(
            stage=self._current_stage,
            telomere_status=self.telomere.get_status(),
            learning_plasticity=self.learning_plasticity,
            open_periods=tuple(p.name for p in self.open_critical_periods()),
            closed_periods=tuple(p.name for p in self.closed_critical_periods()),
            tick_count=self._tick_count,
            transitions=tuple(self._transitions),
        )

    def get_statistics(self) -> dict[str, Any]:
        """Statistics dict for monitoring."""
        return {
            "stage": self._current_stage.value,
            "tick_count": self._tick_count,
            "learning_plasticity": self.learning_plasticity,
            "transitions": len(self._transitions),
            "open_periods": len(self.open_critical_periods()),
            "closed_periods": len(self.closed_critical_periods()),
        }

    # -- Internal ------------------------------------------------------------

    def _update_stage(self) -> None:
        """Compute stage from consumed fraction and transition if needed."""
        consumed = 1.0 - (
            self.telomere._telomere_length / max(self.telomere.max_operations, 1)
        )

        if consumed >= self.config.mature_threshold:
            new_stage = DevelopmentalStage.MATURE
        elif consumed >= self.config.adolescent_threshold:
            new_stage = DevelopmentalStage.ADOLESCENT
        elif consumed >= self.config.juvenile_threshold:
            new_stage = DevelopmentalStage.JUVENILE
        else:
            new_stage = DevelopmentalStage.EMBRYONIC

        if new_stage != self._current_stage and _STAGE_ORDER[new_stage] > _STAGE_ORDER[self._current_stage]:
            old = self._current_stage
            self._current_stage = new_stage
            self._transitions.append(StageTransition(
                from_stage=old,
                to_stage=new_stage,
                tick_count=self._tick_count,
            ))
            if self.on_stage_change is not None:
                self.on_stage_change(old, new_stage)
