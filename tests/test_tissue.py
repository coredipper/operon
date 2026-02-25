"""Tests for Tissue Architecture (Paper Section 6.5.3).

Tissue: hierarchical grouping of cells with shared morphogen gradient,
security boundary, and composable boundary ports.
"""

import pytest
from operon_ai.state.genome import Genome, Gene, GeneType, ExpressionLevel
from operon_ai.coordination.morphogen import MorphogenGradient, MorphogenType
from operon_ai.core.types import Capability, DataType, IntegrityLabel
from operon_ai.core.wagent import PortType, WiringDiagram
from operon_ai.multicell.cell_type import ExpressionProfile, CellType
from operon_ai.multicell.tissue import Tissue, TissueBoundary, TissueError


def _make_genome() -> Genome:
    return Genome(
        genes=[
            Gene("model", "gpt-4", required=True),
            Gene("classification", "enabled", gene_type=GeneType.STRUCTURAL),
            Gene("verification", "strict", gene_type=GeneType.STRUCTURAL),
        ],
        silent=True,
    )


def _make_classifier_type() -> CellType:
    return CellType(
        name="Classifier",
        expression_profile=ExpressionProfile(overrides={
            "classification": ExpressionLevel.OVEREXPRESSED,
            "verification": ExpressionLevel.SILENCED,
        }),
        required_capabilities={Capability.NET},
    )


def _make_validator_type() -> CellType:
    return CellType(
        name="Validator",
        expression_profile=ExpressionProfile(overrides={
            "classification": ExpressionLevel.SILENCED,
            "verification": ExpressionLevel.OVEREXPRESSED,
        }),
        required_capabilities=set(),
    )


def _make_tissue() -> Tissue:
    return Tissue(
        name="TestTissue",
        boundary=TissueBoundary(
            inputs={"task": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"result": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            allowed_capabilities={Capability.NET, Capability.READ_FS},
        ),
    )


class TestTissueBoundary:
    """Tests for TissueBoundary."""

    def test_empty_boundary(self):
        """Empty boundary has no ports and no capabilities."""
        boundary = TissueBoundary()
        assert boundary.inputs == {}
        assert boundary.outputs == {}
        assert boundary.allowed_capabilities == set()

    def test_boundary_with_ports(self):
        """Boundary ports are accessible."""
        boundary = TissueBoundary(
            inputs={"in": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
            outputs={"out": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            allowed_capabilities={Capability.NET},
        )
        assert "in" in boundary.inputs
        assert "out" in boundary.outputs
        assert Capability.NET in boundary.allowed_capabilities


class TestTissue:
    """Tests for Tissue."""

    def test_register_cell_type(self):
        """Registering a cell type within capability bounds succeeds."""
        tissue = _make_tissue()
        ct = _make_classifier_type()
        tissue.register_cell_type(ct)
        assert "Classifier" in tissue._cell_types

    def test_register_cell_type_exceeds_capabilities(self):
        """Registering a cell type that exceeds boundary capabilities raises."""
        tissue = Tissue(
            name="RestrictedTissue",
            boundary=TissueBoundary(allowed_capabilities=set()),
        )
        ct = _make_classifier_type()  # Requires NET
        with pytest.raises(TissueError, match="not allowed"):
            tissue.register_cell_type(ct)

    def test_add_cell(self):
        """Adding a cell from a registered type succeeds."""
        tissue = _make_tissue()
        tissue.register_cell_type(_make_classifier_type())
        genome = _make_genome()

        cell = tissue.add_cell(
            "c1", "Classifier", genome,
            inputs={"task": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"label": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
        assert cell.cell_type == "Classifier"
        assert tissue.cell_count == 1

    def test_add_cell_unregistered_type(self):
        """Adding a cell of unregistered type raises."""
        tissue = _make_tissue()
        genome = _make_genome()
        with pytest.raises(TissueError, match="not registered"):
            tissue.add_cell("c1", "Unknown", genome)

    def test_add_cell_duplicate_name(self):
        """Adding a cell with duplicate name raises."""
        tissue = _make_tissue()
        tissue.register_cell_type(_make_classifier_type())
        genome = _make_genome()

        tissue.add_cell("c1", "Classifier", genome)
        with pytest.raises(TissueError, match="already exists"):
            tissue.add_cell("c1", "Classifier", genome)

    def test_cells_share_gradient(self):
        """All cells in a tissue share the same gradient reference."""
        gradient = MorphogenGradient()
        gradient.set(MorphogenType.COMPLEXITY, 0.9)

        tissue = Tissue(
            name="SharedGradient",
            boundary=TissueBoundary(allowed_capabilities={Capability.NET}),
            gradient=gradient,
        )
        tissue.register_cell_type(_make_classifier_type())
        genome = _make_genome()

        cell1 = tissue.add_cell("c1", "Classifier", genome)
        cell2 = tissue.add_cell("c2", "Classifier", genome)

        # Both were differentiated with high complexity → similar phenotype
        assert cell1.phenotype["temperature"] == cell2.phenotype["temperature"]

    def test_as_module(self):
        """Tissue exports itself as a ModuleSpec with boundary ports."""
        tissue = _make_tissue()
        module = tissue.as_module()

        assert module.name == "TestTissue"
        assert "task" in module.inputs
        assert "result" in module.outputs
        assert Capability.NET in module.capabilities

    def test_two_tissues_compose(self):
        """Two tissues can be composed into a higher-level wiring diagram."""
        tissue_a = Tissue(
            name="TissueA",
            boundary=TissueBoundary(
                outputs={"data": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
                allowed_capabilities={Capability.NET},
            ),
        )
        tissue_b = Tissue(
            name="TissueB",
            boundary=TissueBoundary(
                inputs={"data": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
                allowed_capabilities={Capability.READ_FS},
            ),
        )

        # Compose at organism level
        organism_diagram = WiringDiagram()
        organism_diagram.add_module(tissue_a.as_module())
        organism_diagram.add_module(tissue_b.as_module())
        organism_diagram.connect("TissueA", "data", "TissueB", "data")

        # Verify wiring
        assert len(organism_diagram.modules) == 2
        assert len(organism_diagram.wires) == 1

    def test_connect_cells(self):
        """Cells within a tissue can be wired together."""
        tissue = _make_tissue()
        tissue.register_cell_type(_make_classifier_type())
        tissue.register_cell_type(_make_validator_type())
        genome = _make_genome()

        tissue.add_cell(
            "classifier", "Classifier", genome,
            outputs={"label": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
        tissue.add_cell(
            "validator", "Validator", genome,
            inputs={"label": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
        tissue.connect_cells("classifier", "label", "validator", "label")

        assert len(tissue.diagram.wires) == 1

    def test_connect_nonexistent_cell(self):
        """Connecting a nonexistent cell raises."""
        tissue = _make_tissue()
        tissue.register_cell_type(_make_classifier_type())
        genome = _make_genome()
        tissue.add_cell("c1", "Classifier", genome)

        with pytest.raises(TissueError, match="not in tissue"):
            tissue.connect_cells("c1", "out", "nonexistent", "in")

    def test_list_cells(self):
        """list_cells() returns cell summaries."""
        tissue = _make_tissue()
        tissue.register_cell_type(_make_classifier_type())
        genome = _make_genome()
        tissue.add_cell("c1", "Classifier", genome)

        cells = tissue.list_cells()
        assert len(cells) == 1
        assert cells[0]["name"] == "c1"
        assert cells[0]["cell_type"] == "Classifier"

    def test_get_cell(self):
        """get_cell() retrieves a specific cell."""
        tissue = _make_tissue()
        tissue.register_cell_type(_make_classifier_type())
        genome = _make_genome()
        tissue.add_cell("c1", "Classifier", genome)

        cell = tissue.get_cell("c1")
        assert cell is not None
        assert cell.cell_type == "Classifier"

        assert tissue.get_cell("nonexistent") is None

    def test_stats(self):
        """stats() returns tissue summary."""
        tissue = _make_tissue()
        tissue.register_cell_type(_make_classifier_type())
        genome = _make_genome()
        tissue.add_cell("c1", "Classifier", genome)

        s = tissue.stats()
        assert s["name"] == "TestTissue"
        assert s["cell_count"] == 1
        assert "Classifier" in s["registered_cell_types"]
