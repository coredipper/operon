"""
Operon Core: Fundamental Types and Agents
=========================================

The core module provides the fundamental building blocks:
- BioAgent: The polynomial functor that processes signals
- Signal: Input messages (transcription factors)
- ActionProtein: Output actions (expressed proteins)
- FoldedProtein: Validated output structures
- CellState: Aggregate agent state
- Pathway: Signal routing
"""

from .agent import BioAgent
from .types import (
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
from .wagent import (
    WiringError,
    PortType,
    ModuleSpec,
    Wire,
    WiringDiagram,
)
from .denature import (
    DenatureFilter,
    SummarizeFilter,
    StripMarkupFilter,
    NormalizeFilter,
    ChainFilter,
)
from .coalgebra import (
    Coalgebra,
    StateMachine,
    FunctionalCoalgebra,
    ParallelCoalgebra,
    SequentialCoalgebra,
    TransitionRecord,
    BisimulationResult,
    check_bisimulation,
)
from .optics import (
    Optic,
    OpticError,
    LensOptic,
    PrismOptic,
    TraversalOptic,
    ComposedOptic,
)

__all__ = [
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
    # Coalgebra (Paper §4.2)
    "Coalgebra",
    "StateMachine",
    "FunctionalCoalgebra",
    "ParallelCoalgebra",
    "SequentialCoalgebra",
    "TransitionRecord",
    "BisimulationResult",
    "check_bisimulation",
    # Optics (Paper §3.3)
    "Optic",
    "OpticError",
    "LensOptic",
    "PrismOptic",
    "TraversalOptic",
    "ComposedOptic",
]
