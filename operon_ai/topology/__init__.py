"""
Operon Network Topologies
=========================

Biologically-inspired network topologies for multi-agent coordination.

Modules:
- loops: Feed-forward loops and negative feedback
- quorum: Consensus-based decision making
- cascade: Signal amplification and multi-stage processing
- oscillator: Periodic tasks and rhythm management
"""

from .loops import (
    CoherentFeedForwardLoop,
    NegativeFeedbackLoop,
    GateLogic,
    CircuitState,
    LoopResult,
    CircuitBreakerStats,
)

from .quorum import (
    QuorumSensing,
    EmergencyQuorum,
    VotingStrategy,
    VoteType,
    Vote,
    QuorumResult,
    AgentProfile,
)

from .cascade import (
    Cascade,
    AgentCascade,
    MAPKCascade,
    CascadeStage,
    CascadeResult,
    StageResult,
    StageStatus,
    CascadeMode,
)

from .oscillator import (
    Oscillator,
    CircadianOscillator,
    HeartbeatOscillator,
    CellCycleOscillator,
    OscillatorPhase,
    OscillatorState,
    OscillatorStatus,
    CycleResult,
    WaveformType,
)

__all__ = [
    # Loops
    "CoherentFeedForwardLoop",
    "NegativeFeedbackLoop",
    "GateLogic",
    "CircuitState",
    "LoopResult",
    "CircuitBreakerStats",
    # Quorum
    "QuorumSensing",
    "EmergencyQuorum",
    "VotingStrategy",
    "VoteType",
    "Vote",
    "QuorumResult",
    "AgentProfile",
    # Cascade
    "Cascade",
    "AgentCascade",
    "MAPKCascade",
    "CascadeStage",
    "CascadeResult",
    "StageResult",
    "StageStatus",
    "CascadeMode",
    # Oscillator
    "Oscillator",
    "CircadianOscillator",
    "HeartbeatOscillator",
    "CellCycleOscillator",
    "OscillatorPhase",
    "OscillatorState",
    "OscillatorStatus",
    "CycleResult",
    "WaveformType",
]
