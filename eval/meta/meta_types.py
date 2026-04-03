"""C8 Meta-Harness types: candidate configs, assessment records, and Genome mapping.

Tests whether the Genome abstraction (genes as named typed values, expression,
replication) can represent real organism configurations without information loss.
The round-trip CandidateConfig -> Genome -> CandidateConfig must be lossless.

Scientific hypothesis: the gene abstraction covers the full configuration space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from operon_ai.state.genome import Gene, GeneType, Genome


# ---------------------------------------------------------------------------
# Stage and candidate configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageConfig:
    """Configuration for a single SkillStage in an organism."""

    role: str
    mode: str  # "fixed" | "fuzzy"
    model: str | None = None  # explicit model ID or None = nucleus default
    include_stage_outputs: bool = True
    include_shared_state: bool = True
    cognitive_mode: str | None = None


@dataclass(frozen=True)
class CandidateConfig:
    """A proposed organism configuration — the unit of evolution.

    Phase A: stage configs only (modes, models, thresholds).
    Phase B: topology field enables structural mutations.
    """

    candidate_id: str
    parent_id: str | None
    iteration: int
    stage_configs: tuple[StageConfig, ...]
    intervention_policy: dict[str, Any]
    topology: str | None = None  # "sequential" | "parallel" | "fan_out" | etc
    edges: tuple[tuple[str, str], ...] = ()  # (src_stage, dst_stage) wiring pairs
    proposer: str = "seed"  # "seed" | "tournament_mutate" | "llm_explore"
    reason: str = ""


@dataclass(frozen=True)
class AssessmentRecord:
    """Result of evaluating one candidate on one task.

    Score is separate from candidate — one candidate can be assessed
    on many tasks.
    """

    candidate_id: str
    iteration: int
    task_id: str
    score: float
    tokens: int
    latency_ms: float
    feasible: bool
    proposer: str


# ---------------------------------------------------------------------------
# Genome <-> CandidateConfig mapping
# ---------------------------------------------------------------------------

# Gene naming convention:
#   stage_{i}_role, stage_{i}_mode, stage_{i}_model, ...
#   policy_{key}
#   _n_stages  (metadata gene for reconstruction)

_STAGE_FIELDS = ("role", "mode", "model", "include_stage_outputs",
                 "include_shared_state", "cognitive_mode")

_STRUCTURAL_FIELDS = frozenset({"role", "mode", "model"})


def candidate_to_genome(config: CandidateConfig) -> Genome:
    """Map CandidateConfig fields to Gene objects in a Genome.

    Gene naming: stage_{i}_{field} for stage configs, policy_{key} for
    intervention policy.  GeneType reflects the biological role:
    STRUCTURAL for mode/model (core capability), REGULATORY for booleans
    and policies (control behavior).
    """
    genes: list[Gene] = []

    # Metadata genes (needed for reconstruction)
    genes.append(Gene(
        name="_n_stages",
        value=len(config.stage_configs),
        gene_type=GeneType.HOUSEKEEPING,
        description="Number of stages in the organism",
    ))
    genes.append(Gene(
        name="_topology",
        value=config.topology,
        gene_type=GeneType.HOUSEKEEPING,
        description="Topology hint (Phase B)",
    ))

    # Stage config genes
    for i, sc in enumerate(config.stage_configs):
        for field_name in _STAGE_FIELDS:
            value = getattr(sc, field_name)
            gene_type = (GeneType.STRUCTURAL if field_name in _STRUCTURAL_FIELDS
                         else GeneType.REGULATORY)
            genes.append(Gene(
                name=f"stage_{i}_{field_name}",
                value=value,
                gene_type=gene_type,
            ))

    # Edge genes (Phase B topology wiring)
    genes.append(Gene(
        name="_n_edges",
        value=len(config.edges),
        gene_type=GeneType.HOUSEKEEPING,
    ))
    for i, (src, dst) in enumerate(config.edges):
        genes.append(Gene(name=f"edge_{i}_src", value=src, gene_type=GeneType.STRUCTURAL))
        genes.append(Gene(name=f"edge_{i}_dst", value=dst, gene_type=GeneType.STRUCTURAL))

    # Intervention policy genes
    for key, value in sorted(config.intervention_policy.items()):
        genes.append(Gene(
            name=f"policy_{key}",
            value=value,
            gene_type=GeneType.REGULATORY,
        ))

    return Genome(genes=genes, allow_mutations=True, silent=True)


def genome_to_candidate(
    genome: Genome,
    candidate_id: str,
    parent_id: str | None,
    iteration: int,
    proposer: str = "seed",
    reason: str = "",
) -> CandidateConfig:
    """Reconstruct CandidateConfig from an expressed Genome.

    Inverse of candidate_to_genome — must be lossless round-trip.
    """
    expressed = genome.express()
    n_stages = expressed["_n_stages"]

    stage_configs = []
    for i in range(n_stages):
        kwargs = {}
        for field_name in _STAGE_FIELDS:
            gene_name = f"stage_{i}_{field_name}"
            kwargs[field_name] = expressed[gene_name]
        stage_configs.append(StageConfig(**kwargs))

    policy = {}
    for key, value in expressed.items():
        if key.startswith("policy_"):
            policy[key[len("policy_"):]] = value

    topology = expressed.get("_topology")

    # Restore edges
    n_edges = expressed.get("_n_edges", 0)
    edges = []
    for i in range(n_edges):
        src = expressed.get(f"edge_{i}_src", "")
        dst = expressed.get(f"edge_{i}_dst", "")
        if src and dst:
            edges.append((src, dst))

    return CandidateConfig(
        candidate_id=candidate_id,
        parent_id=parent_id,
        iteration=iteration,
        stage_configs=tuple(stage_configs),
        intervention_policy=policy,
        topology=topology,
        edges=tuple(edges),
        proposer=proposer,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Distance provider for EpiplexityMonitor integration
# ---------------------------------------------------------------------------


@runtime_checkable
class DistanceProvider(Protocol):
    """Non-embedding novelty measurement for EpiplexityMonitor.

    Tests whether epistemic health monitoring generalizes beyond
    embedding space to arbitrary configuration metrics.
    """

    def distance(self, a: Any, b: Any) -> float:
        """Return distance in [0, 1] between two objects."""
        ...


class ConfigHammingDistance:
    """Hamming distance over CandidateConfig fields, normalized to [0, 1].

    Compares each StageConfig field and each intervention policy key,
    counting mismatches, then divides by total field count.
    """

    def distance(self, a: CandidateConfig, b: CandidateConfig) -> float:
        if not a.stage_configs and not b.stage_configs:
            return 0.0

        mismatches = 0
        total = 0

        # Compare stage configs (zip to shorter; extra stages count as mismatches)
        max_stages = max(len(a.stage_configs), len(b.stage_configs))
        for i in range(max_stages):
            if i >= len(a.stage_configs) or i >= len(b.stage_configs):
                # Missing stage = all fields differ
                mismatches += len(_STAGE_FIELDS)
                total += len(_STAGE_FIELDS)
                continue
            sa, sb = a.stage_configs[i], b.stage_configs[i]
            for field_name in _STAGE_FIELDS:
                total += 1
                if getattr(sa, field_name) != getattr(sb, field_name):
                    mismatches += 1

        # Compare intervention policies
        all_keys = set(a.intervention_policy) | set(b.intervention_policy)
        for key in all_keys:
            total += 1
            if a.intervention_policy.get(key) != b.intervention_policy.get(key):
                mismatches += 1

        # Compare topology and edges (Phase B)
        if a.topology != b.topology:
            total += 1
            mismatches += 1
        else:
            total += 1

        a_edges = set(a.edges)
        b_edges = set(b.edges)
        all_edges = a_edges | b_edges
        if all_edges:
            total += len(all_edges)
            mismatches += len(a_edges ^ b_edges)  # symmetric difference

        return mismatches / total if total > 0 else 0.0
