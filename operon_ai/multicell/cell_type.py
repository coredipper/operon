"""
Cell Type Specialization: Differential Gene Expression
========================================================

Biological Analogy:
Every cell in an organism shares the same genome (DNA), yet a neuron behaves
completely differently from a liver cell. The difference is *epigenetic*:
transcription factor programs silence some genes and overexpress others,
producing distinct phenotypes from identical genotypes.

In multi-agent systems:
- Genome = shared base model configuration (same weights, same tools)
- ExpressionProfile = which capabilities are active for this role
- CellType = agent role template (Classifier, Researcher, Validator)
- DifferentiatedCell = concrete agent config ready for instantiation

This implements Paper Section 6.5.1: Cell Types and Agent Specialization.

References:
- Article Section 6.5.1: Multi-Cellular Organization - Cell Types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from operon_ai.core.types import Capability
from operon_ai.state.genome import Genome, Gene, ExpressionLevel
from operon_ai.coordination.morphogen import MorphogenGradient, PhenotypeConfig


@dataclass
class ExpressionProfile:
    """
    Defines which genes are active/silenced for a specific cell type.

    An ExpressionProfile is the *epigenome* — it doesn't change the genome,
    it controls which genes are expressed and at what level. Applying a
    profile to a genome produces a role-specific configuration.

    Example:
        >>> profile = ExpressionProfile(overrides={
        ...     "reasoning_depth": ExpressionLevel.OVEREXPRESSED,
        ...     "code_execution": ExpressionLevel.SILENCED,
        ... })
        >>> config = profile.apply(genome)
    """
    overrides: dict[str, ExpressionLevel] = field(default_factory=dict)

    def apply(self, genome: Genome) -> dict[str, Any]:
        """
        Express a genome with this profile's overrides applied.

        Creates a temporary copy of expression state, applies overrides,
        then calls genome.express(). The original genome is unchanged.

        Returns:
            dict mapping gene names to their values (silenced genes excluded).
        """
        # Save original expression states for genes we'll override
        originals: dict[str, ExpressionLevel] = {}
        for gene_name, level in self.overrides.items():
            gene = genome.get_gene(gene_name)
            if gene is None:
                continue
            # Store original so we can restore
            original_state = genome._expression.get(gene_name)
            if original_state:
                originals[gene_name] = original_state.level
            # Apply override
            genome.set_expression(gene_name, level, modifier="expression_profile")

        # Express with overrides active
        config = genome.express()

        # Restore original expression states
        for gene_name, original_level in originals.items():
            genome.set_expression(gene_name, original_level, modifier="restored")

        return config

    def list_overrides(self) -> list[dict[str, str]]:
        """List all expression overrides."""
        return [
            {"gene": name, "level": level.name}
            for name, level in self.overrides.items()
        ]


@dataclass
class DifferentiatedCell:
    """
    A concrete agent configuration produced by differentiation.

    This is the *phenotype* — the observable behavior that results from
    applying an ExpressionProfile to a Genome in a specific environment
    (MorphogenGradient).

    Fields:
        cell_type: Name of the CellType that produced this cell
        config: Gene values after expression profile is applied
        phenotype: Runtime parameters (temperature, max_tokens, etc.)
        capabilities: What this cell is allowed to do
        system_prompt_fragment: Role-specific prompt text
    """
    cell_type: str
    config: dict[str, Any]
    phenotype: dict[str, Any]
    capabilities: set[Capability]
    system_prompt_fragment: str = ""


@dataclass
class CellType:
    """
    Template for a specialized agent role.

    A CellType combines:
    1. An ExpressionProfile (which genes are active)
    2. Required capabilities (what this cell can do)
    3. A PhenotypeConfig (how gradients map to runtime params)
    4. A system prompt fragment (role-specific instructions)

    Call differentiate() with a Genome and MorphogenGradient to produce
    a DifferentiatedCell ready for instantiation.

    Example:
        >>> classifier = CellType(
        ...     name="Classifier",
        ...     expression_profile=ExpressionProfile(overrides={
        ...         "classification": ExpressionLevel.OVEREXPRESSED,
        ...         "code_execution": ExpressionLevel.SILENCED,
        ...     }),
        ...     required_capabilities={Capability.NET},
        ...     system_prompt_template="You are a task classifier.",
        ... )
        >>> cell = classifier.differentiate(shared_genome, gradient)
    """
    name: str
    expression_profile: ExpressionProfile
    required_capabilities: set[Capability] = field(default_factory=set)
    phenotype_config: PhenotypeConfig = field(default_factory=PhenotypeConfig)
    system_prompt_template: str = ""

    def differentiate(
        self,
        genome: Genome,
        gradient: MorphogenGradient | None = None,
    ) -> DifferentiatedCell:
        """
        Produce a specialized agent config from genome + environment.

        Args:
            genome: The shared genome (genotype)
            gradient: Current morphogen gradient (environment). If None,
                      a default gradient is used.

        Returns:
            DifferentiatedCell with role-specific config and phenotype.

        Raises:
            ValueError: If genome is missing required genes.
        """
        gradient = gradient or MorphogenGradient()

        # Apply expression profile to get role-specific config
        config = self.expression_profile.apply(genome)

        # Validate required genes are present in the expressed config
        missing = []
        for gene_name in self.expression_profile.overrides:
            level = self.expression_profile.overrides[gene_name]
            if level != ExpressionLevel.SILENCED and gene_name not in config:
                gene = genome.get_gene(gene_name)
                if gene is not None:
                    # Gene exists but wasn't expressed (e.g., DORMANT type)
                    pass
                else:
                    missing.append(gene_name)
        if missing:
            raise ValueError(
                f"CellType '{self.name}' requires genes not in genome: {missing}"
            )

        # Compute phenotype from gradient
        phenotype = {
            "temperature": self.phenotype_config.get_temperature(gradient),
            "max_tokens": self.phenotype_config.get_max_tokens(gradient),
            "verification_threshold": self.phenotype_config.get_verification_threshold(gradient),
        }

        return DifferentiatedCell(
            cell_type=self.name,
            config=config,
            phenotype=phenotype,
            capabilities=self.required_capabilities,
            system_prompt_fragment=self.system_prompt_template,
        )
