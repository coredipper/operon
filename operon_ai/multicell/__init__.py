"""
Multi-cellular Organization
============================

Higher-order biological abstractions for multi-agent systems:

- **CellType**: Agent role template with differential gene expression
- **ExpressionProfile**: Which genes are active/silenced per role
- **DifferentiatedCell**: Concrete agent config from CellType + Genome
- **Tissue**: Group of cells sharing a morphogen gradient and boundary
- **TissueBoundary**: Typed ports for inter-tissue communication

References:
- Article Section 6.5: Multi-Cellular Organization - From Agents to Tissues
"""

from .cell_type import (
    ExpressionProfile,
    CellType,
    DifferentiatedCell,
)
from .tissue import (
    TissueBoundary,
    TissueError,
    Tissue,
)

__all__ = [
    "ExpressionProfile",
    "CellType",
    "DifferentiatedCell",
    "TissueBoundary",
    "TissueError",
    "Tissue",
]
