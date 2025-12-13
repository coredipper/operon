"""
Operon: Biologically Inspired Architectures for Agentic Control
===============================================================

Operon brings biological control structures to AI agents using
Applied Category Theory to define rigorous "wiring diagrams".

Core Components:
    - BioAgent: The fundamental agent unit (polynomial functor)
    - Signal: Input messages to agents
    - ActionProtein: Output actions from agents
    - ApprovalToken: Proof-carrying approval for two-key execution
    - IntegrityLabel/Capability: Minimal IFC + effect tags
    - WiringDiagram: Typed wiring diagram checker (WAgent)

State Management:
    - ATP_Store: Multi-currency metabolic budget (ATP, GTP, NADH)
    - HistoneStore: Epigenetic memory with markers and decay
    - Genome: Immutable configuration with gene expression
    - Telomere: Lifecycle and senescence management

Network Topologies:
    - CoherentFeedForwardLoop: Dual-check guardrails with circuit breaker
    - NegativeFeedbackLoop: Homeostasis and error correction
    - QuorumSensing: Multi-agent consensus voting
    - Cascade: Signal amplification pipeline
    - Oscillator: Periodic task scheduling

Organelles:
    - Membrane: Input filtering and immune defense
    - Mitochondria: Deterministic computation
    - Chaperone: Output validation
    - Ribosome: Prompt synthesis
    - Lysosome: Cleanup and recycling
"""

# =============================================================================
# Core
# =============================================================================
from .core.agent import BioAgent
from .core.types import (
    Signal,
    SignalType,
    SignalStrength,
    ActionProtein,
    ActionType,
    IntegrityLabel,
    DataType,
    Capability,
    ApprovalToken,
    FoldedProtein,
    CellState,
    Pathway,
)
from .core.wagent import (
    WiringError,
    PortType,
    ModuleSpec,
    Wire,
    WiringDiagram,
)

# =============================================================================
# State Management
# =============================================================================
from .state.metabolism import (
    ATP_Store,
    MetabolicState,
    EnergyType,
    EnergyTransaction,
    MetabolicReport,
)
from .state.histone import (
    HistoneStore,
    MarkerType,
    MarkerStrength,
    EpigeneticMarker,
    RetrievalResult,
)
from .state.genome import (
    Genome,
    Gene,
    GeneType,
    ExpressionLevel,
    Mutation,
    ExpressionState,
)
from .state.telomere import (
    Telomere,
    TelomereStatus,
    LifecyclePhase,
    SenescenceReason,
    LifecycleEvent,
)

# =============================================================================
# Topologies
# =============================================================================
from .topology.loops import (
    CoherentFeedForwardLoop,
    NegativeFeedbackLoop,
    GateLogic,
    CircuitState,
    LoopResult,
    CircuitBreakerStats,
)
from .topology.quorum import (
    QuorumSensing,
    EmergencyQuorum,
    VotingStrategy,
    VoteType,
    Vote,
    QuorumResult,
    AgentProfile,
)
from .topology.cascade import (
    Cascade,
    AgentCascade,
    MAPKCascade,
    CascadeStage,
    CascadeResult,
    StageResult,
    StageStatus,
    CascadeMode,
)
from .topology.oscillator import (
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

# =============================================================================
# Organelles
# =============================================================================
from .organelles.membrane import (
    Membrane,
    ThreatLevel,
    ThreatSignature,
    FilterResult,
)
from .organelles.mitochondria import (
    Mitochondria,
    MetabolicPathway,
    ATP,
    MetabolicResult,
    Tool,
    SimpleTool,
)
from .organelles.chaperone import (
    Chaperone,
    FoldingStrategy,
    FoldingAttempt,
    EnhancedFoldedProtein,
)
from .organelles.ribosome import (
    Ribosome,
    mRNA,
    tRNA,
    Protein,
    Codon,
    CodonType,
)
from .organelles.lysosome import (
    Lysosome,
    Waste,
    WasteType,
    DigestResult,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Core
    "BioAgent",
    "Signal",
    "SignalType",
    "SignalStrength",
    "ActionProtein",
    "ActionType",
    "IntegrityLabel",
    "DataType",
    "Capability",
    "ApprovalToken",
    "FoldedProtein",
    "CellState",
    "Pathway",
    "WiringError",
    "PortType",
    "ModuleSpec",
    "Wire",
    "WiringDiagram",

    # State - Metabolism
    "ATP_Store",
    "MetabolicState",
    "EnergyType",
    "EnergyTransaction",
    "MetabolicReport",

    # State - Histone
    "HistoneStore",
    "MarkerType",
    "MarkerStrength",
    "EpigeneticMarker",
    "RetrievalResult",

    # State - Genome
    "Genome",
    "Gene",
    "GeneType",
    "ExpressionLevel",
    "Mutation",
    "ExpressionState",

    # State - Telomere
    "Telomere",
    "TelomereStatus",
    "LifecyclePhase",
    "SenescenceReason",
    "LifecycleEvent",

    # Topology - Loops
    "CoherentFeedForwardLoop",
    "NegativeFeedbackLoop",
    "GateLogic",
    "CircuitState",
    "LoopResult",
    "CircuitBreakerStats",

    # Topology - Quorum
    "QuorumSensing",
    "EmergencyQuorum",
    "VotingStrategy",
    "VoteType",
    "Vote",
    "QuorumResult",
    "AgentProfile",

    # Topology - Cascade
    "Cascade",
    "AgentCascade",
    "MAPKCascade",
    "CascadeStage",
    "CascadeResult",
    "StageResult",
    "StageStatus",
    "CascadeMode",

    # Topology - Oscillator
    "Oscillator",
    "CircadianOscillator",
    "HeartbeatOscillator",
    "CellCycleOscillator",
    "OscillatorPhase",
    "OscillatorState",
    "OscillatorStatus",
    "CycleResult",
    "WaveformType",

    # Membrane
    "Membrane",
    "ThreatLevel",
    "ThreatSignature",
    "FilterResult",

    # Mitochondria
    "Mitochondria",
    "MetabolicPathway",
    "ATP",
    "MetabolicResult",
    "Tool",
    "SimpleTool",

    # Chaperone
    "Chaperone",
    "FoldingStrategy",
    "FoldingAttempt",
    "EnhancedFoldedProtein",

    # Ribosome
    "Ribosome",
    "mRNA",
    "tRNA",
    "Protein",
    "Codon",
    "CodonType",

    # Lysosome
    "Lysosome",
    "Waste",
    "WasteType",
    "DigestResult",
]

__version__ = "0.4.0"
