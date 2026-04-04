"""
Quorum Sensing: Threshold-Gated Collective Action
===================================================

Biological Analogy:
Bacteria produce autoinducers (AIs) — small signaling molecules — that
accumulate in the shared environment.  When concentration crosses a
threshold (proportional to population density), coordinated gene
expression activates.  This enables consensus without a central
coordinator.

Key properties (from KEGG map02024):
- Each agent produces signal proportional to its suspicion level
- Signals accumulate in a shared well-mixed environment
- Signals decay exponentially (AHL lactonase degradation)
- Threshold scales sublinearly with population: log(N) × base
- Two signal classes: AI-1 (intra-species) and AI-2 (inter-species)

This is distinct from the vote-counting quorum in topology/quorum.py.
That module counts discrete yes/no votes tied to BioAgent.express().
This module models continuous signal accumulation with temporal decay,
operating on raw float suspicion values without requiring LLM calls.

References:
- KEGG map02024: Quorum sensing
- V. fischeri LuxI/LuxR system (AHL-mediated)
- Article §6.5.2: Coordination Without Central Control
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class AutoinducerSignal:
    """A single signal molecule deposited by an agent.

    Concentration is proportional to the agent's suspicion level —
    the biological insight that production rate encodes information,
    not just a boolean vote.
    """

    agent_id: str
    signal_type: str  # "AI-1" or "AI-2"
    concentration: float
    timestamp: float


@dataclass
class SignalEnvironment:
    """Shared medium where autoinducer signals accumulate and decay.

    Models a well-mixed environment (no spatial structure).
    Signals decay exponentially based on age, mimicking AHL
    lactonase degradation in real bacteria.
    """

    decay_half_life: float = 5.0  # Time units; from typical AHL degradation
    noise_floor: float = 0.001  # Signals below this are pruned

    # signal_type -> list of signals
    _signals: dict[str, list[AutoinducerSignal]] = field(
        default_factory=dict, repr=False,
    )

    def deposit(self, signal: AutoinducerSignal) -> None:
        """Agent deposits a signal into the environment.

        Prunes decayed signals on write to prevent unbounded growth
        while keeping get_concentration() side-effect free.
        """
        if signal.signal_type not in self._signals:
            self._signals[signal.signal_type] = []
        # Prune stale signals for this type on write (monotonic time guaranteed)
        surviving = [
            s for s in self._signals[signal.signal_type]
            if s.concentration * (2.0 ** (-(signal.timestamp - s.timestamp) / self.decay_half_life))
            >= self.noise_floor
            or signal.timestamp < s.timestamp  # keep future signals
        ]
        surviving.append(signal)
        self._signals[signal.signal_type] = surviving

    def get_concentration(self, signal_type: str, current_time: float) -> float:
        """Sum all signals of this type, applying exponential decay.

        c = Σ s_i.concentration × 2^(-(t - s_i.timestamp) / half_life)

        Pure read — does not mutate signal state.
        """
        signals = self._signals.get(signal_type, [])
        if not signals:
            return 0.0

        total = 0.0
        for s in signals:
            age = current_time - s.timestamp
            if age < 0:
                continue  # Future signal, don't count
            decay = 2.0 ** (-age / self.decay_half_life)
            total += s.concentration * decay

        return total

    def prune(self, current_time: float) -> int:
        """Remove signals that have decayed below noise floor. Returns count removed."""
        removed = 0
        for signal_type in list(self._signals):
            surviving = []
            for s in self._signals[signal_type]:
                age = current_time - s.timestamp
                decayed = s.concentration * (2.0 ** (-age / self.decay_half_life))
                if decayed >= self.noise_floor:
                    surviving.append(s)
                else:
                    removed += 1
            self._signals[signal_type] = surviving
        return removed

    def clear(self) -> None:
        """Remove all signals."""
        self._signals.clear()

    @property
    def total_signals(self) -> int:
        return sum(len(v) for v in self._signals.values())


@dataclass
class QuorumSensingBio:
    """Biologically-faithful quorum sensing based on autoinducer accumulation.

    Each agent independently deposits signals proportional to its
    suspicion level.  Activation occurs when accumulated concentration
    (with decay) crosses a threshold that scales with population size.

    Unlike majority voting, this model:
    - Weights by signal strength (suspicious agents contribute more)
    - Naturally ages out old evidence (exponential decay)
    - Scales threshold sublinearly with population (log scaling)
    - Supports multiple independent signal channels (AI-1, AI-2)
    """

    environment: SignalEnvironment = field(default_factory=SignalEnvironment)
    population_size: int = 10
    threshold_base: float = 10.0
    signal_types: list[str] = field(default_factory=lambda: ["AI-1", "AI-2"])

    def _threshold(self) -> float:
        """Activation threshold scales as log(N) × base.

        Biological observation: QS threshold depends on cell density,
        roughly logarithmic at moderate densities.
        """
        return math.log(max(2, self.population_size)) * self.threshold_base

    def produce_signal(
        self,
        agent_id: str,
        suspicion: float,
        current_time: float,
        signal_type: str = "AI-1",
    ) -> None:
        """Agent emits autoinducer proportional to suspicion.

        Args:
            agent_id: The emitting agent
            suspicion: Suspicion level in [0, 1]
            current_time: Current simulation time
            signal_type: Which signal channel to use
        """
        signal = AutoinducerSignal(
            agent_id=agent_id,
            signal_type=signal_type,
            concentration=max(0.0, suspicion),
            timestamp=current_time,
        )
        self.environment.deposit(signal)

    def should_activate(
        self, signal_type: str = "AI-1", current_time: float = 0.0,
    ) -> bool:
        """Check if accumulated concentration exceeds threshold."""
        return self.get_activation_level(signal_type, current_time) >= 1.0

    def get_activation_level(
        self, signal_type: str = "AI-1", current_time: float = 0.0,
    ) -> float:
        """Return ratio of current concentration to threshold.

        Values >= 1.0 mean activation threshold is met.
        Supports graded responses (partial activation near threshold).
        """
        threshold = self._threshold()
        if threshold <= 0:
            return 0.0
        concentration = self.environment.get_concentration(signal_type, current_time)
        return concentration / threshold

    def reset(self) -> None:
        """Clear all signals."""
        self.environment.clear()

    def get_statistics(self, current_time: float = 0.0) -> dict:
        """Return current quorum sensing state."""
        stats: dict = {
            "population_size": self.population_size,
            "threshold": self._threshold(),
            "total_signals": self.environment.total_signals,
        }
        for st in self.signal_types:
            conc = self.environment.get_concentration(st, current_time)
            stats[f"{st}_concentration"] = conc
            stats[f"{st}_activation_level"] = conc / max(0.001, self._threshold())
        return stats
