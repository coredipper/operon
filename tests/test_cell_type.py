"""Tests for Cell Type Specialization (Paper Section 6.5.1).

ExpressionProfile + CellType + DifferentiatedCell: a single Genome
produces different agent phenotypes through differential gene expression.
"""

import pytest
from operon_ai.state.genome import Genome, Gene, GeneType, ExpressionLevel
from operon_ai.coordination.morphogen import MorphogenGradient, MorphogenType, PhenotypeConfig
from operon_ai.core.types import Capability
from operon_ai.multicell.cell_type import ExpressionProfile, CellType, DifferentiatedCell


def _make_shared_genome() -> Genome:
    """Create a shared genome with genes for multiple roles."""
    return Genome(
        genes=[
            Gene("model", "gpt-4", required=True),
            Gene("temperature", 0.7),
            Gene("max_tokens", 4096),
            Gene("classification", "enabled", gene_type=GeneType.STRUCTURAL),
            Gene("code_execution", "enabled", gene_type=GeneType.STRUCTURAL),
            Gene("verification", "strict", gene_type=GeneType.STRUCTURAL),
            Gene("reasoning_depth", "detailed", gene_type=GeneType.REGULATORY),
            Gene("conciseness", "normal", gene_type=GeneType.REGULATORY),
        ],
        silent=True,
    )


class TestExpressionProfile:
    """Tests for ExpressionProfile."""

    def test_apply_silences_genes(self):
        """Silenced genes are excluded from expressed config."""
        genome = _make_shared_genome()
        profile = ExpressionProfile(overrides={
            "code_execution": ExpressionLevel.SILENCED,
        })

        config = profile.apply(genome)
        assert "code_execution" not in config
        assert "classification" in config  # Unaffected

    def test_apply_overexpresses_genes(self):
        """Overexpressed genes are included (expression level changes)."""
        genome = _make_shared_genome()
        profile = ExpressionProfile(overrides={
            "classification": ExpressionLevel.OVEREXPRESSED,
        })

        config = profile.apply(genome)
        assert "classification" in config

    def test_apply_preserves_original_genome(self):
        """Applying a profile does not permanently modify the genome."""
        genome = _make_shared_genome()
        profile = ExpressionProfile(overrides={
            "code_execution": ExpressionLevel.SILENCED,
        })

        # Before apply
        config_before = genome.express()
        assert "code_execution" in config_before

        # Apply profile
        profile.apply(genome)

        # After apply — genome is restored
        config_after = genome.express()
        assert "code_execution" in config_after

    def test_apply_unknown_gene_ignored(self):
        """Overrides for genes not in genome are silently ignored."""
        genome = _make_shared_genome()
        profile = ExpressionProfile(overrides={
            "nonexistent_gene": ExpressionLevel.OVEREXPRESSED,
        })

        config = profile.apply(genome)
        assert "nonexistent_gene" not in config

    def test_empty_profile_returns_default_expression(self):
        """Empty profile returns same config as genome.express()."""
        genome = _make_shared_genome()
        profile = ExpressionProfile()

        assert profile.apply(genome) == genome.express()

    def test_multiple_overrides(self):
        """Multiple genes can be silenced/overexpressed simultaneously."""
        genome = _make_shared_genome()
        profile = ExpressionProfile(overrides={
            "code_execution": ExpressionLevel.SILENCED,
            "verification": ExpressionLevel.SILENCED,
            "classification": ExpressionLevel.OVEREXPRESSED,
        })

        config = profile.apply(genome)
        assert "code_execution" not in config
        assert "verification" not in config
        assert "classification" in config


class TestCellType:
    """Tests for CellType differentiation."""

    def test_differentiate_produces_cell(self):
        """CellType.differentiate() returns a DifferentiatedCell."""
        genome = _make_shared_genome()
        ct = CellType(
            name="Classifier",
            expression_profile=ExpressionProfile(overrides={
                "classification": ExpressionLevel.OVEREXPRESSED,
                "code_execution": ExpressionLevel.SILENCED,
            }),
            required_capabilities={Capability.NET},
            system_prompt_template="You classify tasks.",
        )

        cell = ct.differentiate(genome)
        assert isinstance(cell, DifferentiatedCell)
        assert cell.cell_type == "Classifier"
        assert Capability.NET in cell.capabilities
        assert cell.system_prompt_fragment == "You classify tasks."

    def test_differentiate_applies_profile(self):
        """Differentiated cell has the profile-specific config."""
        genome = _make_shared_genome()
        ct = CellType(
            name="Classifier",
            expression_profile=ExpressionProfile(overrides={
                "code_execution": ExpressionLevel.SILENCED,
            }),
        )

        cell = ct.differentiate(genome)
        assert "code_execution" not in cell.config
        assert "model" in cell.config

    def test_different_types_same_genome(self):
        """Two different CellTypes from the same genome produce different configs."""
        genome = _make_shared_genome()

        classifier = CellType(
            name="Classifier",
            expression_profile=ExpressionProfile(overrides={
                "code_execution": ExpressionLevel.SILENCED,
                "verification": ExpressionLevel.SILENCED,
            }),
        )

        validator = CellType(
            name="Validator",
            expression_profile=ExpressionProfile(overrides={
                "classification": ExpressionLevel.SILENCED,
                "code_execution": ExpressionLevel.SILENCED,
                "verification": ExpressionLevel.OVEREXPRESSED,
            }),
        )

        cell_c = classifier.differentiate(genome)
        cell_v = validator.differentiate(genome)

        # Classifier has classification, validator doesn't
        assert "classification" in cell_c.config
        assert "classification" not in cell_v.config

        # Validator has verification, classifier doesn't
        assert "verification" in cell_v.config
        assert "verification" not in cell_c.config

    def test_phenotype_from_gradient(self):
        """DifferentiatedCell phenotype is computed from the gradient."""
        genome = _make_shared_genome()
        gradient = MorphogenGradient()
        gradient.set(MorphogenType.COMPLEXITY, 0.9)
        gradient.set(MorphogenType.BUDGET, 0.5)

        ct = CellType(
            name="Researcher",
            expression_profile=ExpressionProfile(),
            phenotype_config=PhenotypeConfig(
                low_complexity_temperature=0.2,
                high_complexity_temperature=1.0,
            ),
        )

        cell = ct.differentiate(genome, gradient)
        # High complexity → temperature close to 1.0
        assert cell.phenotype["temperature"] > 0.8

    def test_default_gradient_when_none(self):
        """Differentiation works with no gradient (uses defaults)."""
        genome = _make_shared_genome()
        ct = CellType(
            name="Worker",
            expression_profile=ExpressionProfile(),
        )

        cell = ct.differentiate(genome)
        assert cell.phenotype["temperature"] is not None
        assert cell.phenotype["max_tokens"] > 0

    def test_genome_unchanged_after_differentiation(self):
        """Genome expression state is not permanently altered."""
        genome = _make_shared_genome()
        ct = CellType(
            name="Classifier",
            expression_profile=ExpressionProfile(overrides={
                "code_execution": ExpressionLevel.SILENCED,
            }),
        )

        config_before = genome.express()
        ct.differentiate(genome)
        config_after = genome.express()

        assert config_before == config_after
