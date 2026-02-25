"""
Morphogen Diffusion: Spatially Varying Concentrations on Graphs
================================================================

Paper §6.4: Agents at different positions in the wiring topology
experience different morphogen concentrations, enabling local
gradient-based coordination.

Biological Analogy:
In embryonic development, morphogens diffuse from localized sources
through tissue.  Cells near the source see high concentration; cells
far away see low concentration.  The resulting gradient drives
spatially patterned gene expression (e.g., Bicoid in Drosophila).

Since Operon agents don't have physical positions (Paper §6.5 line 135),
we use **graph adjacency** from the wiring topology as the spatial model.
Each node in the graph represents an agent/cell; edges represent
connections through which morphogens can diffuse.

Diffusion algorithm (per step):
1. Emit   — each source adds emission_rate to its node
2. Diffuse — for each node, diffusion_rate fraction flows to neighbors
             (split evenly among neighbors)
3. Decay  — all concentrations multiplied by (1 - decay_rate)
4. Clamp  — values > 1.0 clamped to 1.0, values < min_concentration set to 0.0

References:
- Article Section 6.4: Morphogen Diffusion
- Article Section 6.5: Graph-Based Spatial Model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .morphogen import MorphogenGradient, MorphogenType


@dataclass(frozen=True)
class MorphogenSource:
    """
    A localized source of morphogen emission.

    Biological Analogy:
    The signaling center (e.g., ZPA in limb bud) that secretes
    a specific morphogen at a defined rate.

    Attributes:
        node_id: Node in the diffusion graph that emits
        morphogen_type: Which morphogen is emitted
        emission_rate: Amount added per step
        max_concentration: Cap for this source's node
    """

    node_id: str
    morphogen_type: MorphogenType
    emission_rate: float
    max_concentration: float = 1.0


@dataclass(frozen=True)
class DiffusionParams:
    """
    Parameters controlling the diffusion dynamics.

    Attributes:
        diffusion_rate: Fraction of concentration that flows to
                        each neighbor per step (split evenly).
        decay_rate: Fraction of concentration lost per step
                    (models morphogen degradation).
        min_concentration: Values below this snap to zero
                          (avoids floating-point dust).
    """

    diffusion_rate: float = 0.1
    decay_rate: float = 0.05
    min_concentration: float = 0.001


class DiffusionField:
    """
    Graph-based morphogen diffusion field.

    Manages a graph of nodes (agents/cells) connected by edges
    (wiring topology).  Morphogen sources emit into specific nodes,
    and concentrations diffuse along edges according to DiffusionParams.

    Example:
        >>> field = DiffusionField()
        >>> for n in ["A", "B", "C"]:
        ...     field.add_node(n)
        >>> field.add_edge("A", "B")
        >>> field.add_edge("B", "C")
        >>> field.add_source(MorphogenSource("A", MorphogenType.COMPLEXITY, 0.5))
        >>> field.run(10)
        >>> # A has high concentration, B medium, C lower
    """

    def __init__(self, params: DiffusionParams | None = None):
        self.params = params or DiffusionParams()
        self._nodes: dict[str, dict[MorphogenType, float]] = {}
        self._adjacency: dict[str, set[str]] = {}
        self._sources: list[MorphogenSource] = []

    # ── Graph construction ──────────────────────────────────────────

    def add_node(self, node_id: str) -> None:
        """Add a node to the diffusion graph."""
        if node_id not in self._nodes:
            self._nodes[node_id] = {}
            self._adjacency[node_id] = set()

    def add_edge(self, node_a: str, node_b: str, *, bidirectional: bool = True) -> None:
        """
        Connect two nodes for morphogen diffusion.

        Raises:
            KeyError: If either node hasn't been added.
        """
        if node_a not in self._nodes:
            raise KeyError(f"Unknown node: {node_a!r}")
        if node_b not in self._nodes:
            raise KeyError(f"Unknown node: {node_b!r}")
        self._adjacency[node_a].add(node_b)
        if bidirectional:
            self._adjacency[node_b].add(node_a)

    # ── Source management ───────────────────────────────────────────

    def add_source(self, source: MorphogenSource) -> None:
        """Register a morphogen emission source."""
        if source.node_id not in self._nodes:
            raise KeyError(f"Unknown node: {source.node_id!r}")
        self._sources.append(source)

    def remove_source(self, node_id: str, morphogen_type: MorphogenType) -> None:
        """Remove sources matching node_id and morphogen_type."""
        self._sources = [
            s for s in self._sources
            if not (s.node_id == node_id and s.morphogen_type == morphogen_type)
        ]

    # ── Concentration access ────────────────────────────────────────

    def set_concentration(
        self, node_id: str, morphogen_type: MorphogenType, value: float
    ) -> None:
        """Set concentration at a specific node."""
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node: {node_id!r}")
        self._nodes[node_id][morphogen_type] = value

    def get_concentration(
        self, node_id: str, morphogen_type: MorphogenType
    ) -> float:
        """Get concentration at a specific node (default 0.0)."""
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node: {node_id!r}")
        return self._nodes[node_id].get(morphogen_type, 0.0)

    def get_local_gradient(self, node_id: str) -> MorphogenGradient:
        """
        Build a MorphogenGradient from concentrations at a specific node.

        This bridges the diffusion field to the existing MorphogenGradient
        API — each cell/agent gets its own gradient reflecting local
        concentrations rather than the shared global gradient.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node: {node_id!r}")
        gradient = MorphogenGradient()
        for mtype, value in self._nodes[node_id].items():
            gradient.set(mtype, value)
        return gradient

    # ── Simulation ──────────────────────────────────────────────────

    def step(self) -> None:
        """
        Advance the diffusion field by one time step.

        Algorithm:
        1. Emit:   each source adds emission_rate to its node
        2. Diffuse: diffusion_rate fraction flows to each neighbor
        3. Decay:  all concentrations * (1 - decay_rate)
        4. Clamp:  cap at 1.0, snap below min_concentration to 0.0
        """
        # 1. Emit
        for source in self._sources:
            current = self._nodes[source.node_id].get(source.morphogen_type, 0.0)
            self._nodes[source.node_id][source.morphogen_type] = min(
                current + source.emission_rate, source.max_concentration
            )

        # 2. Diffuse — accumulate deltas, then apply
        deltas: dict[str, dict[MorphogenType, float]] = {
            nid: {} for nid in self._nodes
        }
        for node_id, concentrations in self._nodes.items():
            neighbors = self._adjacency[node_id]
            if not neighbors:
                continue
            for mtype, conc in concentrations.items():
                outflow = conc * self.params.diffusion_rate
                per_neighbor = outflow / len(neighbors)
                # Remove from source
                deltas[node_id][mtype] = deltas[node_id].get(mtype, 0.0) - outflow
                # Add to neighbors
                for neighbor in neighbors:
                    deltas[neighbor][mtype] = deltas[neighbor].get(mtype, 0.0) + per_neighbor

        for node_id in self._nodes:
            for mtype, delta in deltas[node_id].items():
                current = self._nodes[node_id].get(mtype, 0.0)
                self._nodes[node_id][mtype] = current + delta

        # 3. Decay
        for node_id in self._nodes:
            for mtype in list(self._nodes[node_id].keys()):
                self._nodes[node_id][mtype] *= (1.0 - self.params.decay_rate)

        # 4. Clamp
        for node_id in self._nodes:
            for mtype in list(self._nodes[node_id].keys()):
                val = self._nodes[node_id][mtype]
                if val > 1.0:
                    self._nodes[node_id][mtype] = 1.0
                elif val < self.params.min_concentration:
                    del self._nodes[node_id][mtype]  # Snap to zero (absent = 0.0)

    def run(self, steps: int) -> None:
        """Advance the diffusion field by multiple steps."""
        for _ in range(steps):
            self.step()

    # ── Inspection ──────────────────────────────────────────────────

    def snapshot(self) -> dict[str, dict[str, float]]:
        """
        Return a snapshot of all concentrations.

        Returns:
            {node_id: {morphogen_type_value: concentration, ...}, ...}
        """
        return {
            node_id: {
                mtype.value: round(conc, 6)
                for mtype, conc in concentrations.items()
            }
            for node_id, concentrations in self._nodes.items()
        }

    @classmethod
    def from_adjacency(
        cls,
        adjacency: dict[str, list[str]],
        params: DiffusionParams | None = None,
    ) -> DiffusionField:
        """
        Create a DiffusionField from an adjacency list.

        Args:
            adjacency: {node_id: [neighbor_ids, ...], ...}
            params: Optional diffusion parameters

        Example:
            >>> field = DiffusionField.from_adjacency({
            ...     "A": ["B"],
            ...     "B": ["A", "C"],
            ...     "C": ["B"],
            ... })
        """
        df = cls(params=params)
        for node_id in adjacency:
            df.add_node(node_id)
        for node_id, neighbors in adjacency.items():
            for neighbor in neighbors:
                if neighbor in df._nodes:
                    df._adjacency[node_id].add(neighbor)
        return df
