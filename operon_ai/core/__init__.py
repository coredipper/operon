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
    FoldedProtein,
    CellState,
    Pathway,
)

__all__ = [
    "BioAgent",
    "Signal",
    "SignalType",
    "SignalStrength",
    "ActionProtein",
    "ActionType",
    "FoldedProtein",
    "CellState",
    "Pathway",
]
