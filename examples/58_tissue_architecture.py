"""
Example 58: Tissue Architecture — Hierarchical Multi-Agent Organization
=========================================================================

Demonstrates how cells are grouped into tissues with shared gradients
and security boundaries, implementing Paper Section 6.5.3.

Core idea: Cells don't exist in isolation. A Tissue groups cells that:
1. Share a MorphogenGradient (common chemical environment)
2. Are bounded by capabilities (security isolation)
3. Communicate internally via a WiringDiagram
4. Expose boundary ports for inter-tissue composition

The 4-level hierarchy:
  Cell → Tissue → Organ → Organism
  (this example covers Cell → Tissue → Organism composition)

Biological Analogy:
- Tissue = group of cells sharing extracellular environment
- Tissue boundary = basement membrane (physical security barrier)
- Inter-tissue signaling = boundary ports (typed channels)
- Organism = coordinated tissue system

References:
- Article Section 6.5.3: Multi-Cellular Organization - Tissue Architecture
"""

from operon_ai.state.genome import Genome, Gene, GeneType, ExpressionLevel
from operon_ai.coordination.morphogen import MorphogenGradient, MorphogenType
from operon_ai.core.types import Capability, DataType, IntegrityLabel
from operon_ai.core.wagent import PortType, WiringDiagram
from operon_ai.multicell.cell_type import ExpressionProfile, CellType
from operon_ai.multicell.tissue import Tissue, TissueBoundary, TissueError


def main():
    try:
        print("=" * 60)
        print("Tissue Architecture — Hierarchical Multi-Agent Organization")
        print("=" * 60)

        # =================================================================
        # SECTION 1: Shared Genome
        # =================================================================
        print("\n--- Section 1: Shared Genome ---")

        genome = Genome(
            genes=[
                Gene("model", "gpt-4", required=True),
                Gene("temperature", 0.7),
                Gene("classification", "enabled", gene_type=GeneType.STRUCTURAL),
                Gene("code_execution", "sandboxed", gene_type=GeneType.STRUCTURAL),
                Gene("verification", "strict", gene_type=GeneType.STRUCTURAL),
                Gene("research_tools", "web,rag", gene_type=GeneType.STRUCTURAL),
            ],
            silent=True,
        )
        print(f"  Genome: {len(genome.list_genes())} genes")

        # =================================================================
        # SECTION 2: Define Cell Types
        # =================================================================
        print("\n--- Section 2: Cell Types ---")

        classifier_type = CellType(
            name="Classifier",
            expression_profile=ExpressionProfile(overrides={
                "classification": ExpressionLevel.OVEREXPRESSED,
                "code_execution": ExpressionLevel.SILENCED,
                "verification": ExpressionLevel.SILENCED,
            }),
            required_capabilities={Capability.NET},
            system_prompt_template="Classify incoming tasks by category and priority.",
        )

        researcher_type = CellType(
            name="Researcher",
            expression_profile=ExpressionProfile(overrides={
                "classification": ExpressionLevel.SILENCED,
                "research_tools": ExpressionLevel.OVEREXPRESSED,
            }),
            required_capabilities={Capability.NET, Capability.READ_FS},
            system_prompt_template="Research the problem thoroughly.",
        )

        validator_type = CellType(
            name="Validator",
            expression_profile=ExpressionProfile(overrides={
                "classification": ExpressionLevel.SILENCED,
                "code_execution": ExpressionLevel.SILENCED,
                "research_tools": ExpressionLevel.SILENCED,
                "verification": ExpressionLevel.OVEREXPRESSED,
            }),
            required_capabilities=set(),
            system_prompt_template="Validate the output for correctness.",
        )

        print(f"  Defined: Classifier, Researcher, Validator")

        # =================================================================
        # SECTION 3: Create Classification Tissue
        # =================================================================
        print("\n--- Section 3: Classification Tissue ---")

        classification_tissue = Tissue(
            name="ClassificationTissue",
            boundary=TissueBoundary(
                inputs={"task": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
                outputs={"label": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
                allowed_capabilities={Capability.NET},
            ),
        )
        classification_tissue.register_cell_type(classifier_type)
        classification_tissue.add_cell(
            "primary_classifier", "Classifier", genome,
            inputs={"task": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"label": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )

        print(f"  Tissue: {classification_tissue.name}")
        print(f"  Cells: {classification_tissue.cell_count}")
        print(f"  Boundary: in={list(classification_tissue.boundary.inputs)} "
              f"out={list(classification_tissue.boundary.outputs)}")

        # =================================================================
        # SECTION 4: Create Research Tissue
        # =================================================================
        print("\n--- Section 4: Research Tissue ---")

        research_gradient = MorphogenGradient()
        research_gradient.set(MorphogenType.COMPLEXITY, 0.8)

        research_tissue = Tissue(
            name="ResearchTissue",
            boundary=TissueBoundary(
                inputs={"query": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
                outputs={"findings": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
                allowed_capabilities={Capability.NET, Capability.READ_FS},
            ),
            gradient=research_gradient,
        )
        research_tissue.register_cell_type(researcher_type)
        research_tissue.add_cell(
            "researcher_1", "Researcher", genome,
            inputs={"query": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"findings": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )
        research_tissue.add_cell(
            "researcher_2", "Researcher", genome,
            inputs={"query": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"findings": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )

        print(f"  Tissue: {research_tissue.name}")
        print(f"  Cells: {research_tissue.cell_count}")
        print(f"  Gradient complexity: {research_gradient.get(MorphogenType.COMPLEXITY)}")

        # =================================================================
        # SECTION 5: Create Validation Tissue
        # =================================================================
        print("\n--- Section 5: Validation Tissue ---")

        validation_tissue = Tissue(
            name="ValidationTissue",
            boundary=TissueBoundary(
                inputs={"draft": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
                outputs={"verdict": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
                allowed_capabilities=set(),  # No external access!
            ),
        )
        validation_tissue.register_cell_type(validator_type)
        validation_tissue.add_cell(
            "validator_1", "Validator", genome,
            inputs={"draft": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
            outputs={"verdict": PortType(DataType.JSON, IntegrityLabel.VALIDATED)},
        )

        print(f"  Tissue: {validation_tissue.name}")
        print(f"  Cells: {validation_tissue.cell_count}")
        print(f"  Capabilities: none (isolated)")

        # =================================================================
        # SECTION 6: Capability Enforcement
        # =================================================================
        print("\n--- Section 6: Capability Enforcement ---")

        try:
            validation_tissue.register_cell_type(researcher_type)
            print("  ERROR: Should have been blocked!")
        except TissueError as e:
            print(f"  Blocked: {e}")
            print("  (Researcher requires NET+READ_FS, Validation allows none)")

        # =================================================================
        # SECTION 7: Compose into Organism-level Diagram
        # =================================================================
        print("\n--- Section 7: Organism-level Composition ---")

        organism = WiringDiagram()
        organism.add_module(classification_tissue.as_module())
        organism.add_module(research_tissue.as_module())
        organism.add_module(validation_tissue.as_module())

        # Wire tissues: Classification → Research → Validation
        organism.connect(
            "ClassificationTissue", "label",
            "ResearchTissue", "query",
        )
        organism.connect(
            "ResearchTissue", "findings",
            "ValidationTissue", "draft",
        )

        print(f"  Organism modules: {[m.name for m in organism.modules.values()]}")
        print(f"  Organism wires: {len(organism.wires)}")
        print(f"  Pipeline: Classification → Research → Validation")

        # Show each tissue's capabilities are isolated
        for mod in organism.modules.values():
            caps = [c.value for c in mod.capabilities] if mod.capabilities else ["none"]
            print(f"    {mod.name}: capabilities = {caps}")

        # =================================================================
        # SECTION 8: Gradient Isolation
        # =================================================================
        print("\n--- Section 8: Gradient Isolation Between Tissues ---")

        print(f"  ClassificationTissue gradient complexity: "
              f"{classification_tissue.gradient.get(MorphogenType.COMPLEXITY)}")
        print(f"  ResearchTissue gradient complexity: "
              f"{research_tissue.gradient.get(MorphogenType.COMPLEXITY)}")
        print(f"  (Different gradients — tissues have independent environments)")

        print("\n" + "=" * 60)
        print("DONE — Tissue Architecture demonstrated successfully")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
