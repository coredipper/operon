"""
Operon State Management
=======================

Biologically-inspired state management systems for AI agents.

Modules:
- metabolism: Energy/resource management (ATP, GTP, NADH)
- histone: Epigenetic memory with multiple marker types
- genome: Immutable configuration and trait inheritance
- telomere: Lifecycle and senescence management
"""

from .metabolism import (
    ATP_Store,
    MetabolicState,
    EnergyType,
    EnergyTransaction,
    MetabolicReport,
)

from .histone import (
    HistoneStore,
    MarkerType,
    MarkerStrength,
    EpigeneticMarker,
    RetrievalResult,
)

from .genome import (
    Genome,
    Gene,
    GeneType,
    ExpressionLevel,
    Mutation,
    ExpressionState,
)

from .telomere import (
    Telomere,
    TelomereStatus,
    LifecyclePhase,
    SenescenceReason,
    LifecycleEvent,
)

__all__ = [
    # Metabolism
    "ATP_Store",
    "MetabolicState",
    "EnergyType",
    "EnergyTransaction",
    "MetabolicReport",
    # Histone
    "HistoneStore",
    "MarkerType",
    "MarkerStrength",
    "EpigeneticMarker",
    "RetrievalResult",
    # Genome
    "Genome",
    "Gene",
    "GeneType",
    "ExpressionLevel",
    "Mutation",
    "ExpressionState",
    # Telomere
    "Telomere",
    "TelomereStatus",
    "LifecyclePhase",
    "SenescenceReason",
    "LifecycleEvent",
]
