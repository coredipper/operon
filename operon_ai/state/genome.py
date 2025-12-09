"""
Genome: Immutable Configuration and Trait System
=================================================

Biological Analogy:
- DNA: The fundamental, mostly immutable configuration
- Genes: Individual traits or capabilities
- Alleles: Variations of genes (different configurations)
- Mutations: Tracked changes from original configuration
- Transcription factors: What controls gene expression
- Phenotype: Observable behavior from genotype

The Genome provides immutable configuration management for agents,
defining their fundamental traits, capabilities, and constraints
that cannot be changed through normal operation (unlike epigenetics).
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
from datetime import datetime
import hashlib
import copy
import json


class GeneType(Enum):
    """Types of genes in the genome."""
    STRUCTURAL = "structural"       # Core capabilities
    REGULATORY = "regulatory"       # Controls other genes
    HOUSEKEEPING = "housekeeping"   # Essential functions
    CONDITIONAL = "conditional"     # Context-dependent
    DORMANT = "dormant"            # Not currently expressed


class ExpressionLevel(Enum):
    """Level of gene expression."""
    SILENCED = 0      # Not expressed
    LOW = 1           # Minimal expression
    NORMAL = 2        # Standard expression
    HIGH = 3          # Elevated expression
    OVEREXPRESSED = 4 # Maximum expression


@dataclass(frozen=True)
class Gene:
    """
    A single gene (configuration trait).

    Genes are immutable once created. Their expression can be
    modified through regulatory mechanisms but not their content.
    """
    name: str
    value: Any
    gene_type: GeneType = GeneType.STRUCTURAL
    description: str = ""
    required: bool = False
    default_expression: ExpressionLevel = ExpressionLevel.NORMAL

    def get_hash(self) -> str:
        """Get unique hash for this gene."""
        content = f"{self.name}:{self.value}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class Mutation:
    """Record of a mutation (configuration change)."""
    gene_name: str
    original_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""
    approved: bool = False


@dataclass
class ExpressionState:
    """Current expression state of a gene."""
    level: ExpressionLevel = ExpressionLevel.NORMAL
    modified_at: datetime = field(default_factory=datetime.now)
    modifier: str = ""  # What caused the modification


class Genome:
    """
    Immutable Configuration Manager.

    The Genome stores the fundamental configuration of an agent
    that should not change during normal operation. It provides:

    Features:

    1. Immutable Genes
       - Configuration values that cannot be modified
       - Protected from accidental changes
       - Audited access

    2. Gene Expression
       - Control which genes are active
       - Expression levels (silenced → overexpressed)
       - Regulatory mechanisms

    3. Mutation Tracking
       - All changes are logged
       - Requires explicit approval
       - Rollback capability

    4. Inheritance
       - Create child genomes from parent
       - Selective gene inheritance
       - Mutation during replication

    5. Phenotype Generation
       - Convert genotype to observable config
       - Expression-aware output
       - Context-dependent traits

    Example:
        >>> genome = Genome()
        >>> genome.add_gene(Gene(name="model", value="gpt-4", required=True))
        >>> genome.add_gene(Gene(name="temperature", value=0.7))
        >>> genome.add_gene(Gene(name="max_tokens", value=4096))
        >>> config = genome.express()  # Get active configuration
        >>> print(config["model"])
        'gpt-4'
    """

    def __init__(
        self,
        genes: list[Gene] | None = None,
        allow_mutations: bool = False,
        mutation_rate: float = 0.0,
        on_mutation: Callable[[Mutation], bool] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Genome.

        Args:
            genes: Initial genes to add
            allow_mutations: Whether to allow configuration changes
            mutation_rate: Probability of mutation during replication
            on_mutation: Callback to approve/reject mutations
            silent: Suppress console output
        """
        self.allow_mutations = allow_mutations
        self.mutation_rate = mutation_rate
        self.on_mutation = on_mutation
        self.silent = silent

        # Gene storage
        self._genes: dict[str, Gene] = {}
        self._expression: dict[str, ExpressionState] = {}
        self._mutations: list[Mutation] = []

        # Add initial genes
        if genes:
            for gene in genes:
                self.add_gene(gene)

        # Tracking
        self._created_at = datetime.now()
        self._generation = 0
        self._parent_hash: str | None = None

    def add_gene(self, gene: Gene) -> bool:
        """
        Add a gene to the genome.

        Can only be done during initialization or with allow_mutations=True.
        """
        if gene.name in self._genes and not self.allow_mutations:
            if not self.silent:
                print(f"🧬 [Genome] Cannot overwrite gene: {gene.name}")
            return False

        self._genes[gene.name] = gene
        self._expression[gene.name] = ExpressionState(
            level=gene.default_expression
        )

        if not self.silent:
            print(f"🧬 [Genome] Added gene: {gene.name} = {gene.value}")

        return True

    def get_gene(self, name: str) -> Gene | None:
        """Get a gene by name."""
        return self._genes.get(name)

    def get_value(self, name: str, default: Any = None) -> Any:
        """Get a gene's value, considering expression level."""
        gene = self._genes.get(name)
        if not gene:
            return default

        expression = self._expression.get(name)
        if expression and expression.level == ExpressionLevel.SILENCED:
            return default

        return gene.value

    def set_expression(
        self,
        gene_name: str,
        level: ExpressionLevel,
        modifier: str = ""
    ) -> bool:
        """
        Set the expression level of a gene.

        This doesn't change the gene, just whether/how it's expressed.
        """
        if gene_name not in self._genes:
            return False

        self._expression[gene_name] = ExpressionState(
            level=level,
            modified_at=datetime.now(),
            modifier=modifier
        )

        if not self.silent:
            print(f"🧬 [Genome] Expression: {gene_name} → {level.name}")

        return True

    def silence_gene(self, gene_name: str, reason: str = "") -> bool:
        """Silence a gene (prevent expression)."""
        return self.set_expression(gene_name, ExpressionLevel.SILENCED, reason)

    def activate_gene(self, gene_name: str, reason: str = "") -> bool:
        """Activate a previously silenced gene."""
        return self.set_expression(gene_name, ExpressionLevel.NORMAL, reason)

    def mutate(
        self,
        gene_name: str,
        new_value: Any,
        reason: str = ""
    ) -> bool:
        """
        Attempt to mutate a gene.

        Requires allow_mutations=True or approval via callback.
        """
        if gene_name not in self._genes:
            return False

        original_gene = self._genes[gene_name]
        mutation = Mutation(
            gene_name=gene_name,
            original_value=original_gene.value,
            new_value=new_value,
            reason=reason,
            approved=False
        )

        # Check if mutation is allowed
        if not self.allow_mutations:
            if self.on_mutation:
                mutation.approved = self.on_mutation(mutation)
            else:
                if not self.silent:
                    print(f"🧬 [Genome] Mutation rejected: {gene_name}")
                self._mutations.append(mutation)
                return False

        if not mutation.approved and not self.allow_mutations:
            self._mutations.append(mutation)
            return False

        # Apply mutation
        mutation.approved = True
        self._mutations.append(mutation)

        # Create new gene with mutated value
        new_gene = Gene(
            name=original_gene.name,
            value=new_value,
            gene_type=original_gene.gene_type,
            description=original_gene.description,
            required=original_gene.required,
            default_expression=original_gene.default_expression
        )
        self._genes[gene_name] = new_gene

        if not self.silent:
            print(f"🧬 [Genome] Mutation: {gene_name}: {original_gene.value} → {new_value}")

        return True

    def rollback_mutation(self, gene_name: str) -> bool:
        """Rollback the last mutation on a gene."""
        # Find the last mutation for this gene
        for mutation in reversed(self._mutations):
            if mutation.gene_name == gene_name and mutation.approved:
                return self.mutate(gene_name, mutation.original_value, "rollback")
        return False

    def express(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Express the genome to get active configuration.

        Returns all genes that are not silenced, with their values.
        Conditional genes may use context to determine expression.
        """
        config = {}
        context = context or {}

        for name, gene in self._genes.items():
            expression = self._expression.get(name)

            # Skip silenced genes
            if expression and expression.level == ExpressionLevel.SILENCED:
                continue

            # Conditional genes check context
            if gene.gene_type == GeneType.CONDITIONAL:
                # Simple pattern: gene name exists in context
                if name not in context:
                    continue

            # Dormant genes are never expressed
            if gene.gene_type == GeneType.DORMANT:
                continue

            config[name] = gene.value

        return config

    def replicate(
        self,
        mutations: dict[str, Any] | None = None,
        inherit_expression: bool = True
    ) -> 'Genome':
        """
        Create a child genome (replication).

        Args:
            mutations: Specific mutations to apply
            inherit_expression: Whether to inherit expression states

        Returns:
            A new Genome instance
        """
        import random

        # Create new genome with copied genes
        child_genes = list(self._genes.values())

        child = Genome(
            genes=child_genes,
            allow_mutations=self.allow_mutations,
            mutation_rate=self.mutation_rate,
            on_mutation=self.on_mutation,
            silent=self.silent,
        )

        child._generation = self._generation + 1
        child._parent_hash = self.get_hash()

        # Inherit expression states
        if inherit_expression:
            for name, state in self._expression.items():
                child._expression[name] = ExpressionState(
                    level=state.level,
                    modifier="inherited"
                )

        # Apply specified mutations
        if mutations:
            for gene_name, new_value in mutations.items():
                child.mutate(gene_name, new_value, "replication_mutation")

        # Random mutations based on mutation rate
        if self.mutation_rate > 0:
            for gene_name in child._genes:
                if random.random() < self.mutation_rate:
                    # Simple mutation: slightly modify numeric values
                    gene = child._genes[gene_name]
                    if isinstance(gene.value, (int, float)):
                        delta = gene.value * 0.1 * (random.random() - 0.5)
                        new_value = gene.value + delta
                        if isinstance(gene.value, int):
                            new_value = int(new_value)
                        child.mutate(gene_name, new_value, "random_mutation")

        if not self.silent:
            print(f"🧬 [Genome] Replicated: generation {child._generation}")

        return child

    def get_hash(self) -> str:
        """Get unique hash for this genome."""
        content = json.dumps(
            {name: gene.value for name, gene in sorted(self._genes.items())},
            sort_keys=True,
            default=str
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def diff(self, other: 'Genome') -> dict[str, tuple[Any, Any]]:
        """
        Compare this genome to another.

        Returns dict of gene_name -> (this_value, other_value) for differences.
        """
        differences = {}

        all_genes = set(self._genes.keys()) | set(other._genes.keys())

        for name in all_genes:
            this_gene = self._genes.get(name)
            other_gene = other._genes.get(name)

            this_val = this_gene.value if this_gene else None
            other_val = other_gene.value if other_gene else None

            if this_val != other_val:
                differences[name] = (this_val, other_val)

        return differences

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the genome.

        Checks that all required genes are present and expressed.
        Returns (valid, list of errors).
        """
        errors = []

        for name, gene in self._genes.items():
            if gene.required:
                expression = self._expression.get(name)
                if expression and expression.level == ExpressionLevel.SILENCED:
                    errors.append(f"Required gene '{name}' is silenced")

        return len(errors) == 0, errors

    def get_statistics(self) -> dict:
        """Get genome statistics."""
        by_type = {}
        by_expression = {}

        for gene in self._genes.values():
            t = gene.gene_type.value
            by_type[t] = by_type.get(t, 0) + 1

        for name, state in self._expression.items():
            e = state.level.name
            by_expression[e] = by_expression.get(e, 0) + 1

        return {
            "total_genes": len(self._genes),
            "generation": self._generation,
            "parent_hash": self._parent_hash,
            "mutations_count": len(self._mutations),
            "approved_mutations": len([m for m in self._mutations if m.approved]),
            "by_type": by_type,
            "by_expression": by_expression,
            "hash": self.get_hash(),
        }

    def export(self) -> dict:
        """Export genome for serialization."""
        return {
            "genes": [
                {
                    "name": g.name,
                    "value": g.value,
                    "gene_type": g.gene_type.value,
                    "description": g.description,
                    "required": g.required,
                    "default_expression": g.default_expression.value,
                }
                for g in self._genes.values()
            ],
            "expression": {
                name: {
                    "level": state.level.value,
                    "modifier": state.modifier,
                }
                for name, state in self._expression.items()
            },
            "generation": self._generation,
            "parent_hash": self._parent_hash,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any], **kwargs) -> 'Genome':
        """Create a genome from a simple config dictionary."""
        genes = [
            Gene(name=key, value=value)
            for key, value in config.items()
        ]
        return cls(genes=genes, **kwargs)

    def list_genes(self) -> list[dict]:
        """List all genes with their current state."""
        return [
            {
                "name": gene.name,
                "value": gene.value,
                "type": gene.gene_type.value,
                "expression": self._expression[gene.name].level.name,
                "required": gene.required,
            }
            for gene in self._genes.values()
        ]
