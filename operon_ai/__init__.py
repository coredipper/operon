"""
Operon: Biologically Inspired Architectures for Agentic Control
===============================================================

Operon brings biological control structures to AI agents using
Applied Category Theory to define rigorous "wiring diagrams".

Core Components:
    - BioAgent: The basic signal-processing agent abstraction
    - Signal: Input messages to agents
    - ActionProtein: Output actions from agents
    - ApprovalToken: Proof-carrying approval for two-key execution
    - IntegrityLabel/Capability: Minimal IFC + effect tags
    - WiringDiagram: Typed wiring diagram checker (WAgent)
    - reviewer_gate / specialist_swarm / advise_topology / skill_organism: pattern-first API

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
from .core.denature import (
    DenatureFilter,
    SummarizeFilter,
    StripMarkupFilter,
    NormalizeFilter,
    ChainFilter,
)
from .core.coalgebra import (
    Coalgebra,
    StateMachine,
    FunctionalCoalgebra,
    ParallelCoalgebra,
    SequentialCoalgebra,
    TransitionRecord,
    BisimulationResult,
    check_bisimulation,
)
from .core.optics import (
    Optic,
    OpticError,
    LensOptic,
    PrismOptic,
    TraversalOptic,
    ComposedOptic,
)
from .core.epistemic import (
    TopologyClass,
    EpistemicAnalysis,
    ObservationProfile,
    EpistemicPartition,
    TopologyClassification,
    TopologyRecommendation,
    analyze as epistemic_analyze,
    classify_topology,
    recommend_topology,
)
from .core.wiring_runtime import (
    TypedValue,
    ModuleExecution,
    ExecutionReport,
    DiagramExecutor,
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
from .state.development import (
    DevelopmentController,
    DevelopmentConfig,
    DevelopmentalStage,
    DevelopmentStatus,
    CriticalPeriod,
    StageTransition,
    stage_reached,
)
from .state.dna_repair import (
    DNARepair,
    StateCheckpoint,
    CorruptionType,
    DamageSeverity,
    DamageReport,
    RepairStrategy,
    RepairResult,
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
from .organelles.nucleus import (
    Nucleus,
    Transcription,
)
from .organelles.plasmid import (
    Plasmid,
    PlasmidRegistry,
    PlasmidError,
    AcquisitionResult,
)

# =============================================================================
# Memory
# =============================================================================
from .memory import (
    MemoryTier,
    MemoryEntry,
    EpisodicMemory,
    BiTemporalFact,
    BiTemporalQuery,
    BiTemporalMemory,
    FactSnapshot,
    CorrectionResult,
)

# =============================================================================
# Providers
# =============================================================================
from .providers import (
    LLMProvider,
    LLMResponse,
    ProviderConfig,
    ToolSchema,
    ToolCall,
    ToolResult,
    MockProvider,
    OpenAIProvider,
    OpenAICompatibleProvider,
    AnthropicProvider,
    GeminiProvider,
    NucleusError,
    ProviderUnavailableError,
    QuotaExhaustedError,
    TranscriptionFailedError,
)

# =============================================================================
# Healing (Self-Repair Mechanisms)
# =============================================================================
from .healing import (
    ChaperoneLoop,
    HealingResult,
    HealingOutcome,
    RefoldingAttempt,
    RegenerativeSwarm,
    SwarmResult,
    SimpleWorker,
    WorkerMemory,
    WorkerStatus,
    AutophagyDaemon,
    PruneResult,
    ContextMetrics,
    ContextHealthStatus,
)

# =============================================================================
# Health (Epistemic Monitoring)
# =============================================================================
from .health import (
    EpiplexityMonitor,
    EpiplexityState,
    EpiplexityResult,
    HealthStatus,
    EmbeddingProvider,
    MockEmbeddingProvider,
)

# =============================================================================
# Surveillance - Innate Immunity
# =============================================================================
from .surveillance import (
    InnateImmunity,
    InnateCheckResult,
    TLRPattern,
    PAMPCategory,
    InflammationLevel,
    InflammationState,
    InflammationResponse,
    JSONValidator,
    LengthValidator,
    CharacterSetValidator,
)

# =============================================================================
# Coordination - Morphogen Gradients
# =============================================================================
from .coordination import (
    MorphogenType,
    MorphogenValue,
    MorphogenGradient,
    GradientUpdate,
    GradientOrchestrator,
    PhenotypeConfig,
    MorphogenSource,
    DiffusionParams,
    DiffusionField,
    SocialLearning,
    PeerExchange,
    TrustRegistry,
    AdoptionResult,
    AdoptionOutcome,
    ScaffoldingResult,
    AutoinducerSignal,
    SignalEnvironment,
    QuorumSensingBio,
)

# =============================================================================
# Multi-cellular Organization
# =============================================================================
from .multicell import (
    ExpressionProfile,
    CellType,
    DifferentiatedCell,
    TissueBoundary,
    TissueError,
    Tissue,
)
from .state.metabolism import MetabolicAccessPolicy

# =============================================================================
# Pattern-First API
# =============================================================================
from .healing import (
    SleepConsolidation,
    ConsolidationResult,
    CounterfactualResult,
    counterfactual_replay,
)
from .patterns import (
    AdaptiveRunResult,
    AdaptiveSkillOrganism,
    CognitiveMode,
    ExperienceRecord,
    InterventionKind,
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    ReviewerGate,
    ReviewerGateConfig,
    ReviewerGateResult,
    reviewer_gate,
    SignalCategory,
    SkillOrganism,
    SkillRunResult,
    SkillRuntimeComponent,
    SkillStage,
    SkillStageResult,
    SubstrateView,
    SpecialistSwarm,
    SpecialistSwarmConfig,
    SpecialistSwarmResult,
    specialist_swarm,
    TaskFingerprint,
    TelemetryEvent,
    TelemetryProbe,
    TopologyAdvice,
    WatcherComponent,
    WatcherConfig,
    WatcherIntervention,
    WatcherSignal,
    adaptive_skill_organism,
    advise_topology,
    assemble_pattern,
    CLIResult,
    cli_handler,
    cli_organism,
    consolidate,
    managed_organism,
    ManagedOrganism,
    ManagedRunResult,
    resolve_cognitive_mode,
    skill_organism,
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
    "DenatureFilter",
    "SummarizeFilter",
    "StripMarkupFilter",
    "NormalizeFilter",
    "ChainFilter",
    # Coalgebra (Paper §3.5)
    "Coalgebra",
    "StateMachine",
    "FunctionalCoalgebra",
    "ParallelCoalgebra",
    "SequentialCoalgebra",
    "TransitionRecord",
    "BisimulationResult",
    "check_bisimulation",
    # Optics (Paper §3.4)
    "Optic",
    "OpticError",
    "LensOptic",
    "PrismOptic",
    "TraversalOptic",
    "ComposedOptic",
    # Epistemic Topology (Paper §6.5.4)
    "TopologyClass",
    "EpistemicAnalysis",
    "ObservationProfile",
    "EpistemicPartition",
    "TopologyClassification",
    "TopologyRecommendation",
    "epistemic_analyze",
    "classify_topology",
    "recommend_topology",
    "TypedValue",
    "ModuleExecution",
    "ExecutionReport",
    "DiagramExecutor",
    # Healing
    "SleepConsolidation",
    "ConsolidationResult",
    "CounterfactualResult",
    "counterfactual_replay",
    # Pattern-First API
    "AdaptiveRunResult",
    "AdaptiveSkillOrganism",
    "CognitiveMode",
    "ExperienceRecord",
    "InterventionKind",
    "PatternLibrary",
    "PatternRunRecord",
    "PatternTemplate",
    "ReviewerGate",
    "ReviewerGateConfig",
    "ReviewerGateResult",
    "reviewer_gate",
    "SignalCategory",
    "SkillOrganism",
    "SkillRunResult",
    "SkillRuntimeComponent",
    "SkillStage",
    "SkillStageResult",
    "SubstrateView",
    "SpecialistSwarm",
    "SpecialistSwarmConfig",
    "SpecialistSwarmResult",
    "specialist_swarm",
    "TaskFingerprint",
    "TelemetryEvent",
    "TelemetryProbe",
    "TopologyAdvice",
    "WatcherComponent",
    "WatcherConfig",
    "WatcherIntervention",
    "WatcherSignal",
    "adaptive_skill_organism",
    "advise_topology",
    "assemble_pattern",
    "CLIResult",
    "cli_handler",
    "cli_organism",
    "consolidate",
    "managed_organism",
    "ManagedOrganism",
    "ManagedRunResult",
    "resolve_cognitive_mode",
    "skill_organism",

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

    # State - Development
    "DevelopmentController",
    "DevelopmentConfig",
    "DevelopmentalStage",
    "DevelopmentStatus",
    "CriticalPeriod",
    "StageTransition",
    "stage_reached",

    # State - DNA Repair
    "DNARepair",
    "StateCheckpoint",
    "CorruptionType",
    "DamageSeverity",
    "DamageReport",
    "RepairStrategy",
    "RepairResult",

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

    # Nucleus
    "Nucleus",
    "Transcription",

    # Plasmid (Horizontal Gene Transfer)
    "Plasmid",
    "PlasmidRegistry",
    "PlasmidError",
    "AcquisitionResult",

    # Memory
    "MemoryTier",
    "MemoryEntry",
    "EpisodicMemory",
    "BiTemporalFact",
    "BiTemporalQuery",
    "BiTemporalMemory",
    "FactSnapshot",
    "CorrectionResult",

    # Providers
    "LLMProvider",
    "LLMResponse",
    "ProviderConfig",
    "ToolSchema",
    "ToolCall",
    "ToolResult",
    "MockProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "NucleusError",
    "ProviderUnavailableError",
    "QuotaExhaustedError",
    "TranscriptionFailedError",

    # Healing (Self-Repair Mechanisms)
    "ChaperoneLoop",
    "HealingResult",
    "HealingOutcome",
    "RefoldingAttempt",
    "RegenerativeSwarm",
    "SwarmResult",
    "SimpleWorker",
    "WorkerMemory",
    "WorkerStatus",
    "AutophagyDaemon",
    "PruneResult",
    "ContextMetrics",
    "ContextHealthStatus",

    # Health (Epistemic Monitoring)
    "EpiplexityMonitor",
    "EpiplexityState",
    "EpiplexityResult",
    "HealthStatus",
    "EmbeddingProvider",
    "MockEmbeddingProvider",

    # Surveillance - Innate Immunity
    "InnateImmunity",
    "InnateCheckResult",
    "TLRPattern",
    "PAMPCategory",
    "InflammationLevel",
    "InflammationState",
    "InflammationResponse",
    "JSONValidator",
    "LengthValidator",
    "CharacterSetValidator",

    # Coordination - Morphogen Gradients
    "MorphogenType",
    "MorphogenValue",
    "MorphogenGradient",
    "GradientUpdate",
    "GradientOrchestrator",
    "PhenotypeConfig",
    # Diffusion (Paper §6.5.2 / §6.5.3)
    "MorphogenSource",
    "DiffusionParams",
    "DiffusionField",
    "SocialLearning",
    "PeerExchange",
    "TrustRegistry",
    "AdoptionResult",
    "AdoptionOutcome",
    "ScaffoldingResult",
    "AutoinducerSignal",
    "SignalEnvironment",
    "QuorumSensingBio",

    # Multi-cellular Organization
    "ExpressionProfile",
    "CellType",
    "DifferentiatedCell",
    "TissueBoundary",
    "TissueError",
    "Tissue",
    "MetabolicAccessPolicy",
]

__version__ = "0.36.0"
