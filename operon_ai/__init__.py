"""
Operon: Biologically Inspired Architectures for Agentic Control
===============================================================

Operon brings biological control structures to AI agents using
Applied Category Theory to define rigorous "wiring diagrams".

Core Components:
    - BioAgent: The fundamental agent unit (polynomial functor)
    - Signal: Input messages to agents
    - ATP_Store: Metabolic budget management

Topologies:
    - CoherentFeedForwardLoop: Dual-check guardrails (executor + assessor)
    - QuorumSensing: Multi-agent consensus voting

Organelles:
    - Membrane: Input filtering and immune defense
    - Mitochondria: Deterministic computation
    - Chaperone: Output validation
    - Ribosome: Prompt synthesis
    - Lysosome: Cleanup and recycling
"""

# Core
from .core.agent import BioAgent
from .core.types import Signal, ActionProtein, FoldedProtein

# State
from .state.metabolism import ATP_Store
from .state.histone import HistoneStore

# Topology
from .topology.loops import CoherentFeedForwardLoop
from .topology.quorum import QuorumSensing

# Organelles
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

__all__ = [
    # Core
    "BioAgent",
    "Signal",
    "ActionProtein",
    "FoldedProtein",

    # State
    "ATP_Store",
    "HistoneStore",

    # Topology
    "CoherentFeedForwardLoop",
    "QuorumSensing",

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

__version__ = "0.2.0"
