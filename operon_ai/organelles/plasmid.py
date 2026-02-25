"""
Plasmid Registry: Horizontal Gene Transfer for Agents
=====================================================

Paper §6.2, Eq. 12: Agent_new = Agent_old ⊗ ToolSchema

In biology, plasmids are small DNA molecules that bacteria exchange
via horizontal gene transfer (HGT).  A bacterium can acquire new
capabilities (e.g. antibiotic resistance) without vertical
inheritance.

This module provides a PlasmidRegistry from which a Mitochondria
can dynamically acquire (engulf) or release tools at runtime,
subject to capability gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.types import Capability
from .mitochondria import SimpleTool


class PlasmidError(ValueError):
    """Raised for plasmid registry or acquisition errors."""


@dataclass(frozen=True)
class Plasmid:
    """A transferable tool definition stored in a registry.

    Like a biological plasmid carrying genes for antibiotic
    resistance, a Plasmid carries a callable plus metadata
    needed for capability-gated acquisition.
    """

    name: str
    description: str
    func: Callable[..., Any]
    tags: frozenset[str] = frozenset()
    version: str = "1.0.0"
    required_capabilities: frozenset[Capability] = frozenset()
    parameters_schema: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def to_tool(self) -> SimpleTool:
        """Convert this plasmid into a SimpleTool for engulfment."""
        return SimpleTool(
            name=self.name,
            description=self.description,
            func=self.func,
            required_capabilities=set(self.required_capabilities),
            parameters_schema=dict(self.parameters_schema),
        )


@dataclass
class AcquisitionResult:
    """Result of a plasmid acquisition attempt."""

    success: bool
    plasmid_name: str
    error: str | None = None


class PlasmidRegistry:
    """Searchable registry of plasmids available for horizontal transfer.

    Analogous to the environmental pool of plasmids that bacteria
    can uptake via conjugation or transformation.
    """

    def __init__(self) -> None:
        self._plasmids: dict[str, Plasmid] = {}

    def register(self, plasmid: Plasmid) -> None:
        """Add a plasmid to the registry."""
        if plasmid.name in self._plasmids:
            raise PlasmidError(
                f"Plasmid already registered: {plasmid.name}"
            )
        self._plasmids[plasmid.name] = plasmid

    def unregister(self, name: str) -> None:
        """Remove a plasmid from the registry."""
        if name not in self._plasmids:
            raise PlasmidError(f"Unknown plasmid: {name}")
        del self._plasmids[name]

    def get(self, name: str) -> Plasmid:
        """Retrieve a plasmid by name."""
        if name not in self._plasmids:
            raise PlasmidError(f"Unknown plasmid: {name}")
        return self._plasmids[name]

    def list_available(self) -> list[dict[str, Any]]:
        """List all available plasmids with metadata."""
        return [
            {
                "name": p.name,
                "description": p.description,
                "tags": sorted(p.tags),
                "version": p.version,
                "required_capabilities": sorted(
                    c.value for c in p.required_capabilities
                ),
            }
            for p in self._plasmids.values()
        ]

    def search(
        self, query: str, tags: set[str] | None = None
    ) -> list[Plasmid]:
        """Search plasmids by name/description text and optional tags.

        Matching is case-insensitive.  If ``tags`` is provided, only
        plasmids whose tags intersect with the query tags are returned.
        """
        query_lower = query.lower()
        results: list[Plasmid] = []
        for p in self._plasmids.values():
            text_match = (
                query_lower in p.name.lower()
                or query_lower in p.description.lower()
            )
            if not text_match:
                continue
            if tags is not None and not tags.intersection(p.tags):
                continue
            results.append(p)
        return results

    def __len__(self) -> int:
        return len(self._plasmids)

    def __contains__(self, name: str) -> bool:
        return name in self._plasmids
