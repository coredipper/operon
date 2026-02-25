"""
Tissue Architecture: Hierarchical Multi-Agent Organization
============================================================

Biological Analogy:
In multicellular organisms, cells don't exist in isolation. They form
*tissues* — groups of specialized cells that share a chemical environment
(extracellular matrix, growth factors, morphogen gradients) and cooperate
to perform a function.

Tissue boundaries enforce security isolation:
- Cells within a tissue share a MorphogenGradient (common context)
- Cells can only use capabilities the tissue boundary allows
- Inter-tissue communication happens through typed boundary ports
- A tissue can appear as a single module in a higher-level diagram

This implements Paper Section 6.5.3: Tissue Architecture.

The 4-level hierarchy:
- Cell → individual agent (DifferentiatedCell)
- Tissue → cells sharing gradient + boundary (this module)
- Organ → multiple tissues for a complex function (future)
- Organism → complete system with homeostatic regulation (future)

References:
- Article Section 6.5.3: Multi-Cellular Organization - Tissue Architecture
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from operon_ai.core.types import Capability, DataType, IntegrityLabel
from operon_ai.core.wagent import WiringDiagram, ModuleSpec, PortType
from operon_ai.coordination.morphogen import MorphogenGradient
from operon_ai.coordination.diffusion import DiffusionField
from operon_ai.state.genome import Genome
from operon_ai.multicell.cell_type import CellType, DifferentiatedCell


@dataclass
class TissueBoundary:
    """
    Typed ports for inter-tissue communication.

    Defines what data can enter/exit the tissue and what capabilities
    are allowed inside. Cells within a tissue cannot exceed these
    capability bounds — this is the security isolation layer.

    Example:
        >>> boundary = TissueBoundary(
        ...     inputs={"task": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        ...     outputs={"result": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        ...     allowed_capabilities={Capability.NET, Capability.READ_FS},
        ... )
    """
    inputs: dict[str, PortType] = field(default_factory=dict)
    outputs: dict[str, PortType] = field(default_factory=dict)
    allowed_capabilities: set[Capability] = field(default_factory=set)


class TissueError(Exception):
    """Error in tissue operations."""


class Tissue:
    """
    A group of cells sharing a morphogen gradient and security boundary.

    Tissue is the fundamental multi-cellular abstraction. It provides:

    1. Shared Environment
       All cells within a tissue read the same MorphogenGradient.
       Changes to the gradient affect all cells simultaneously.

    2. Security Boundary
       Cell capabilities must be a subset of the tissue's
       allowed_capabilities. This enforces least-privilege at
       the tissue level.

    3. Internal Wiring
       Cells within a tissue are connected via a WiringDiagram.
       The tissue manages the diagram internally.

    4. Composability
       A tissue can export itself as a single ModuleSpec via
       as_module(), enabling hierarchical composition:
       Cell → Tissue → Organ → Organism.

    Example:
        >>> tissue = Tissue(
        ...     name="ClassificationTissue",
        ...     boundary=TissueBoundary(
        ...         inputs={"task": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        ...         outputs={"label": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        ...         allowed_capabilities={Capability.NET},
        ...     ),
        ... )
        >>> tissue.register_cell_type(classifier_type)
        >>> tissue.add_cell("classifier_1", "Classifier", genome)
    """

    def __init__(
        self,
        name: str,
        boundary: TissueBoundary,
        gradient: MorphogenGradient | None = None,
        diffusion_field: DiffusionField | None = None,
    ):
        self.name = name
        self.boundary = boundary
        self.gradient = gradient or MorphogenGradient()
        self._diffusion_field = diffusion_field
        self._cell_types: dict[str, CellType] = {}
        self._cells: dict[str, DifferentiatedCell] = {}
        self._diagram = WiringDiagram()

    def register_cell_type(self, cell_type: CellType) -> None:
        """
        Register a cell type that can exist in this tissue.

        Validates that the cell type's required capabilities are a
        subset of the tissue boundary's allowed capabilities.

        Raises:
            TissueError: If cell type capabilities exceed tissue boundary.
        """
        excess = cell_type.required_capabilities - self.boundary.allowed_capabilities
        if excess:
            raise TissueError(
                f"CellType '{cell_type.name}' requires capabilities "
                f"{excess} not allowed by tissue '{self.name}' boundary"
            )
        self._cell_types[cell_type.name] = cell_type

    def add_cell(
        self,
        name: str,
        cell_type_name: str,
        genome: Genome,
        inputs: dict[str, PortType] | None = None,
        outputs: dict[str, PortType] | None = None,
    ) -> DifferentiatedCell:
        """
        Instantiate a cell from a registered type + genome.

        The cell is differentiated using the tissue's shared gradient,
        added to the internal wiring diagram, and returned.

        Args:
            name: Unique name for this cell within the tissue
            cell_type_name: Name of a registered CellType
            genome: The shared genome to differentiate from
            inputs: Port types for this cell's inputs (for wiring)
            outputs: Port types for this cell's outputs (for wiring)

        Returns:
            The DifferentiatedCell produced by differentiation.

        Raises:
            TissueError: If cell type not registered or name duplicate.
        """
        if cell_type_name not in self._cell_types:
            raise TissueError(
                f"CellType '{cell_type_name}' not registered in tissue '{self.name}'. "
                f"Available: {list(self._cell_types.keys())}"
            )
        if name in self._cells:
            raise TissueError(f"Cell '{name}' already exists in tissue '{self.name}'")

        cell_type = self._cell_types[cell_type_name]
        cell = cell_type.differentiate(genome, self.gradient)
        self._cells[name] = cell

        # Register in diffusion field if present
        if self._diffusion_field is not None:
            self._diffusion_field.add_node(name)

        # Add as module to internal wiring diagram
        module = ModuleSpec(
            name=name,
            inputs=inputs or {},
            outputs=outputs or {},
            capabilities=cell.capabilities,
        )
        self._diagram.add_module(module)

        return cell

    def connect_cells(
        self,
        src: str,
        src_port: str,
        dst: str,
        dst_port: str,
    ) -> None:
        """
        Wire two cells within this tissue.

        Args:
            src: Source cell name
            src_port: Output port on source cell
            dst: Destination cell name
            dst_port: Input port on destination cell

        Raises:
            TissueError: If either cell doesn't exist.
        """
        if src not in self._cells:
            raise TissueError(f"Source cell '{src}' not in tissue '{self.name}'")
        if dst not in self._cells:
            raise TissueError(f"Destination cell '{dst}' not in tissue '{self.name}'")
        self._diagram.connect(src, src_port, dst, dst_port)

        # Register edge in diffusion field if present
        if self._diffusion_field is not None:
            self._diffusion_field.add_edge(src, dst)

    def diffuse(self, steps: int = 1) -> None:
        """
        Run morphogen diffusion for the given number of steps.

        Requires a DiffusionField to have been provided at construction.

        Raises:
            TissueError: If no diffusion field is configured.
        """
        if self._diffusion_field is None:
            raise TissueError(
                f"No diffusion field configured for tissue '{self.name}'"
            )
        self._diffusion_field.run(steps)

    def get_cell_gradient(self, name: str) -> MorphogenGradient:
        """
        Get the local morphogen gradient for a specific cell.

        If a DiffusionField is configured, returns the spatially-local
        gradient from the diffusion field.  Otherwise, falls back to
        the shared tissue gradient.

        Args:
            name: Cell name within this tissue

        Returns:
            MorphogenGradient reflecting local concentrations.

        Raises:
            TissueError: If the cell doesn't exist.
        """
        if name not in self._cells:
            raise TissueError(f"Cell '{name}' not in tissue '{self.name}'")
        if self._diffusion_field is not None:
            return self._diffusion_field.get_local_gradient(name)
        return self.gradient

    def as_module(self) -> ModuleSpec:
        """
        Export this tissue as a single module for higher-level composition.

        The tissue's boundary ports become the module's ports,
        and the tissue's allowed capabilities become the module's
        capability set. This enables hierarchical composition:
        Tissue.as_module() can be added to another WiringDiagram.

        Returns:
            ModuleSpec representing this tissue as a composable unit.
        """
        return ModuleSpec(
            name=self.name,
            inputs=self.boundary.inputs,
            outputs=self.boundary.outputs,
            capabilities=self.boundary.allowed_capabilities,
        )

    def get_cell(self, name: str) -> DifferentiatedCell | None:
        """Get a cell by name."""
        return self._cells.get(name)

    def list_cells(self) -> list[dict[str, Any]]:
        """List all cells in this tissue."""
        return [
            {
                "name": name,
                "cell_type": cell.cell_type,
                "capabilities": [c.value for c in cell.capabilities],
                "active_genes": list(cell.config.keys()),
            }
            for name, cell in self._cells.items()
        ]

    @property
    def cell_count(self) -> int:
        """Number of cells in this tissue."""
        return len(self._cells)

    @property
    def diagram(self) -> WiringDiagram:
        """Access the internal wiring diagram (read-only intent)."""
        return self._diagram

    def stats(self) -> dict[str, Any]:
        """Get tissue statistics."""
        return {
            "name": self.name,
            "cell_count": self.cell_count,
            "registered_cell_types": list(self._cell_types.keys()),
            "boundary_inputs": list(self.boundary.inputs.keys()),
            "boundary_outputs": list(self.boundary.outputs.keys()),
            "allowed_capabilities": [c.value for c in self.boundary.allowed_capabilities],
            "gradient": self.gradient.to_dict(),
        }
