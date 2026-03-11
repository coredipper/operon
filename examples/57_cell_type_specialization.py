"""
Example 57: Cell Type Specialization — Differential Gene Expression
====================================================================

Demonstrates how a single shared Genome produces different agent
phenotypes through ExpressionProfiles, implementing Paper Section 6.5.1.

Core idea: Every agent in a system shares the same "DNA" (model config,
tools, capabilities). What makes a Classifier different from a Validator
is which genes are *expressed* — just like a neuron and a liver cell
share the same genome but express different gene programs.

Biological Analogy:
- Genome = DNA shared by all cells (same model, same tool access)
- ExpressionProfile = transcription factor program (role-specific)
- CellType = cell lineage template (Classifier, Researcher, Validator)
- DifferentiatedCell = mature cell ready to function

References:
- Article Section 6.5.1: Cell Types and Agent Specialization
"""

from operon_ai.state.genome import Genome, Gene, GeneType, ExpressionLevel
from operon_ai.coordination.morphogen import (
    MorphogenGradient,
    MorphogenType,
    PhenotypeConfig,
)
from operon_ai.core.types import Capability
from operon_ai.multicell.cell_type import ExpressionProfile, CellType


def main():
    try:
        print("=" * 60)
        print("Cell Type Specialization — Differential Gene Expression")
        print("=" * 60)

        # =================================================================
        # SECTION 1: Define Shared Genome
        # =================================================================
        print("\n--- Section 1: Shared Genome (same for all cell types) ---")

        genome = Genome(
            genes=[
                Gene("model", "gpt-4", required=True),
                Gene("temperature", 0.7),
                Gene("max_tokens", 4096),
                Gene("classification", "multi-label", gene_type=GeneType.STRUCTURAL),
                Gene("code_execution", "sandboxed", gene_type=GeneType.STRUCTURAL),
                Gene("verification", "strict", gene_type=GeneType.STRUCTURAL),
                Gene("reasoning_depth", "detailed", gene_type=GeneType.REGULATORY),
                Gene("research_tools", "web_search,rag", gene_type=GeneType.STRUCTURAL),
            ],
            silent=True,
        )

        print(f"  Genome: {len(genome.list_genes())} genes")
        for g in genome.list_genes():
            print(f"    {g['name']:20s} = {g['value']}")

        # =================================================================
        # SECTION 2: Define Cell Types
        # =================================================================
        print("\n--- Section 2: Define Cell Types ---")

        classifier = CellType(
            name="Classifier",
            expression_profile=ExpressionProfile(overrides={
                "classification": ExpressionLevel.OVEREXPRESSED,
                "code_execution": ExpressionLevel.SILENCED,
                "verification": ExpressionLevel.SILENCED,
                "research_tools": ExpressionLevel.SILENCED,
            }),
            required_capabilities={Capability.NET},
            phenotype_config=PhenotypeConfig(
                low_complexity_temperature=0.1,
                high_complexity_temperature=0.5,
            ),
            system_prompt_template="You are a task classifier. Categorize incoming requests.",
        )

        researcher = CellType(
            name="Researcher",
            expression_profile=ExpressionProfile(overrides={
                "classification": ExpressionLevel.SILENCED,
                "research_tools": ExpressionLevel.OVEREXPRESSED,
                "reasoning_depth": ExpressionLevel.OVEREXPRESSED,
            }),
            required_capabilities={Capability.NET, Capability.READ_FS},
            phenotype_config=PhenotypeConfig(
                low_complexity_temperature=0.3,
                high_complexity_temperature=0.9,
                normal_max_tokens=8192,
            ),
            system_prompt_template="You are a research agent. Investigate thoroughly.",
        )

        validator = CellType(
            name="Validator",
            expression_profile=ExpressionProfile(overrides={
                "classification": ExpressionLevel.SILENCED,
                "code_execution": ExpressionLevel.SILENCED,
                "verification": ExpressionLevel.OVEREXPRESSED,
                "research_tools": ExpressionLevel.SILENCED,
            }),
            required_capabilities=set(),
            phenotype_config=PhenotypeConfig(
                low_complexity_temperature=0.0,
                high_complexity_temperature=0.3,
            ),
            system_prompt_template="You are a quality validator. Check for errors and compliance.",
        )

        print(f"  Defined 3 cell types: Classifier, Researcher, Validator")

        # =================================================================
        # SECTION 3: Differentiate — Same Genome, Different Phenotypes
        # =================================================================
        print("\n--- Section 3: Differentiation ---")

        gradient = MorphogenGradient()
        gradient.set(MorphogenType.COMPLEXITY, 0.8)
        gradient.set(MorphogenType.BUDGET, 0.6)

        for ct in [classifier, researcher, validator]:
            cell = ct.differentiate(genome, gradient)
            print(f"\n  {cell.cell_type}:")
            print(f"    Active genes: {list(cell.config.keys())}")
            print(f"    Temperature:  {cell.phenotype['temperature']:.2f}")
            print(f"    Max tokens:   {cell.phenotype['max_tokens']}")
            print(f"    Capabilities: {', '.join(c.value for c in cell.capabilities) or 'none'}")
            print(f"    Prompt:       {cell.system_prompt_fragment[:50]}...")

        # =================================================================
        # SECTION 4: Genome is Unchanged
        # =================================================================
        print("\n--- Section 4: Genome remains unchanged after differentiation ---")

        full_config = genome.express()
        print(f"  Genome still has {len(full_config)} active genes: {list(full_config.keys())}")

        # =================================================================
        # SECTION 5: Gradient Changes → Phenotype Changes
        # =================================================================
        print("\n--- Section 5: Environment affects phenotype ---")

        # Low complexity, low budget
        gradient_easy = MorphogenGradient()
        gradient_easy.set(MorphogenType.COMPLEXITY, 0.1)
        gradient_easy.set(MorphogenType.BUDGET, 0.2)

        cell_easy = researcher.differentiate(genome, gradient_easy)
        cell_hard = researcher.differentiate(genome, gradient)

        print(f"  Researcher in easy environment:")
        print(f"    Temperature: {cell_easy.phenotype['temperature']:.2f} (low → fast heuristics)")
        print(f"    Max tokens:  {cell_easy.phenotype['max_tokens']} (budget-limited)")
        print(f"  Researcher in hard environment:")
        print(f"    Temperature: {cell_hard.phenotype['temperature']:.2f} (high → detailed reasoning)")
        print(f"    Max tokens:  {cell_hard.phenotype['max_tokens']} (more budget)")

        print("\n" + "=" * 60)
        print("DONE — Cell Type Specialization demonstrated successfully")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
