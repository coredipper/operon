"""Quality control system (Ubiquitin-Proteasome model)."""
from .types import (
    ChainType,
    DegronType,
    DegradationResult,
    PoolExhaustionStrategy,
    UbiquitinTag,
    TaggedData,
    UbiquitinPool,
)
from .components import (
    ProvenanceContext,
    E3Ligase,
    Deubiquitinase,
    ChaperoneRepair,
)

__all__ = [
    "ChainType",
    "DegronType",
    "DegradationResult",
    "PoolExhaustionStrategy",
    "UbiquitinTag",
    "TaggedData",
    "UbiquitinPool",
    "ProvenanceContext",
    "E3Ligase",
    "Deubiquitinase",
    "ChaperoneRepair",
]
