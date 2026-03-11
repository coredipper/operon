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
    ResourceCost,
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
    BudgetOptic,
    ComposedOptic,
)
from .analyzer import (
    Optimization,
    dependency_graph,
    find_independent_groups,
    find_dead_wires,
    critical_path,
    total_cost,
    suggest_optimizations,
)
from .optimizer import (
    OptimizedDiagram,
    OptimizationPass,
    EliminateDeadWires,
    ParallelGrouping,
    CostOrderSchedule,
    optimize,
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
    "ResourceCost",
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
    "BudgetOptic",
    "ComposedOptic",
    # Analyzer (Paper §7)
    "Optimization",
    "dependency_graph",
    "find_independent_groups",
    "find_dead_wires",
    "critical_path",
    "total_cost",
    "suggest_optimizations",
    # Optimizer (Paper §7)
    "OptimizedDiagram",
    "OptimizationPass",
    "EliminateDeadWires",
    "ParallelGrouping",
    "CostOrderSchedule",
    "optimize",
]
