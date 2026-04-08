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
    noise_floor: float = 0.001  # Signals below this are ignored/pruned

    # signal_type -> list of signals
    _signals: dict[str, list[AutoinducerSignal]] = field(
        default_factory=dict, repr=False,
    )
    # Track last prune time per signal type to enforce monotonic pruning
    _last_prune_time: dict[str, float] = field(
        default_factory=dict, repr=False,
    )

    def deposit(self, signal: AutoinducerSignal) -> None:
        """Agent deposits a signal into the environment.

        Prunes decayed signals on write when the timestamp advances
        past the last pruned time, preventing unbounded growth while
        keeping get_concentration() side-effect free.
        """
        st = signal.signal_type
        if st not in self._signals:
            self._signals[st] = []

        # Only prune if this deposit advances time (monotonic guard)
        last = self._last_prune_time.get(st, float("-inf"))
        if signal.timestamp >= last:
            surviving = [
                s for s in self._signals[st]
                if s.concentration * (2.0 ** (-(signal.timestamp - s.timestamp) / self.decay_half_life))
                >= self.noise_floor
            ]
            self._signals[st] = surviving
            self._last_prune_time[st] = signal.timestamp

        self._signals[st].append(signal)

    def get_concentration(self, signal_type: str, current_time: float) -> float:
        """Sum all signals of this type, applying exponential decay.

        c = Σ s_i.concentration × 2^(-(t - s_i.timestamp) / half_life)

        Pure read — does not mutate signal state. Skips contributions
        that have decayed below noise_floor.
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
            contribution = s.concentration * decay
            if contribution >= self.noise_floor:
                total += contribution

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
        self._last_prune_time.clear()

    @property
    def total_signals(self) -> int:
        return sum(len(v) for v in self._signals.values())


def _verify_no_false_activation(
    params: dict,
) -> tuple[bool, dict]:
    """Derivation replay for the QS no-false-activation guarantee."""
    N = params["N"]
    s = params["s"]
    h = params["h"]
    dt = params["dt"]
    m = params["safety_margin"]
    decay = 2.0 ** (-dt / h)
    c_ss = (N * s) / (1.0 - decay)
    threshold = c_ss * m
    ratio = c_ss / threshold  # = 1/m
    return ratio < 1.0, {"c_ss": c_ss, "threshold": threshold, "ratio": ratio}


# Register for certificate serialization
from ..core.certificate import register_verify_fn as _register
_register("no_false_activation", _verify_no_false_activation)
del _register


@dataclass
class QuorumSensingBio:
    """Biologically-faithful quorum sensing based on autoinducer accumulation.

    Each agent independently deposits signals proportional to its
    suspicion level.  Activation occurs when accumulated concentration
    (with decay) crosses a threshold that scales with population size.

    Unlike majority voting, this model:
    - Weights by signal strength (suspicious agents contribute more)
    - Naturally ages out old evidence (exponential decay)
    - Scales threshold with population (auto-calibrated or log scaling)
    - Supports multiple independent signal channels (AI-1, AI-2)

    Threshold modes:
    - **Calibrated** (call ``calibrate()``): threshold derived from
      steady-state formula so normal traffic never triggers activation.
      The guarantee preserves under population changes — a categorical
      certificate per de los Riscos et al. (2603.28906) Prop 5.1.
    - **Manual** (default): ``log(N) × threshold_base``, requires
      hand-tuning threshold_base for each population size.
    """

    environment: SignalEnvironment = field(default_factory=SignalEnvironment)
    population_size: int = 10
    threshold_base: float = 10.0  # Used only when not calibrated
    signal_types: list[str] = field(default_factory=lambda: ["AI-1", "AI-2"])

    # Calibration parameters (set via calibrate())
    expected_normal_suspicion: float = 0.15
    safety_margin: float = 2.0
    emission_interval: float = 1.0  # Time units between signal deposits
    _calibrated: bool = field(default=False, repr=False)

    def calibrate(self) -> None:
        """Auto-calibrate threshold from population and signal parameters.

        Derives the activation threshold so that normal traffic (all agents
        emitting at expected_normal_suspicion) produces an activation level
        of 1/safety_margin at steady state — guaranteeing no false activation
        under normal conditions.

        Uses the exact discrete-time steady state.  Every dt time units,
        N agents deposit signals of magnitude s.  Previous signals decay
        by 2^(-dt/h) per interval.  The geometric series gives:

            c_ss = N × s / (1 - 2^(-dt/h))

        Setting threshold = c_ss × safety_margin ensures activation_level
        = c_ss / threshold = 1/safety_margin < 1.0 for all normal traffic.

        Categorical certificate (de los Riscos et al., Def 5.3.4):
        This calibration preserves the no-false-activation guarantee under
        architecture morphisms that change population_size, because the
        threshold is defined in terms of (N, s, h), not a fixed constant.
        When a convergence compiler changes N, re-calling calibrate()
        restores the guarantee — an instance of Prop 5.1.
        """
        if self.emission_interval <= 0:
            raise ValueError(
                f"emission_interval must be positive, got {self.emission_interval}"
            )
        self._calibrated = True

    def certify(self) -> "Certificate":
        """Return a certificate for the no-false-activation guarantee.

        Requires ``calibrate()`` to have been called first.

        The certificate's ``verify()`` re-derives the steady-state
        concentration and confirms ``c_ss / threshold < 1.0``.
        """
        from ..core.certificate import Certificate

        if not self._calibrated:
            raise ValueError("Must calibrate() before certifying")
        return Certificate(
            theorem="no_false_activation",
            parameters={
                "N": self.population_size,
                "s": self.expected_normal_suspicion,
                "h": self.environment.decay_half_life,
                "dt": self.emission_interval,
                "safety_margin": self.safety_margin,
            },
            conclusion="Normal traffic activation level < 1.0",
            source="QuorumSensingBio.calibrate",
            _verify_fn=_verify_no_false_activation,
        )

    def _threshold(self) -> float:
        """Activation threshold.

        When calibrated: derived from discrete steady-state formula so
        normal traffic stays below activation (structural guarantee).
        When not calibrated: log(N) × threshold_base (manual tuning).
        """
        if self._calibrated:
            if self.emission_interval <= 0:
                raise ValueError(
                    f"emission_interval must be positive, got {self.emission_interval}"
                )
            # Exact discrete steady state: geometric series with decay
            # 2^(-dt/h) per emission interval dt
            decay_per_step = 2.0 ** (
                -self.emission_interval / self.environment.decay_half_life
            )
            steady_state = (
                self.population_size
                * self.expected_normal_suspicion
                / (1.0 - decay_per_step)
            )
            return max(0.001, steady_state * self.safety_margin)
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
