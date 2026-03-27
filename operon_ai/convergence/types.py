"""Shared types for convergence adapters.

Every adapter (Swarms, DeerFlow, AnimaWorks) produces an :class:`ExternalTopology`
that :func:`analyze_external_topology` consumes.  This design keeps the analysis
pipeline source-agnostic: adding a new orchestration target means writing one
parser function, not changing the analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.epistemic import TopologyRecommendation
from ..patterns.repository import PatternTemplate, TaskFingerprint
from ..patterns.types import TopologyAdvice


@dataclass(frozen=True)
class ExternalTopology:
    """Serializable representation of an external system's agent topology.

    Adapters parse framework-specific configs into this common type.
    The ``source`` field is a tag, not a type discriminant — all downstream
    code treats every ``ExternalTopology`` identically.
    """

    source: str
    """Origin framework: ``"swarms"``, ``"deerflow"``, or ``"animaworks"``."""

    pattern_name: str
    """Name of the workflow pattern (e.g., ``"SequentialWorkflow"``, ``"HierarchicalSwarm"``)."""

    agents: tuple[dict[str, Any], ...]
    """Agent specifications as plain dicts.  Each dict should have at least
    ``"name"`` and ``"role"`` keys; additional keys are framework-specific."""

    edges: tuple[tuple[str, str], ...]
    """Directed edges ``(from_agent, to_agent)`` describing the communication or
    control-flow graph."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary framework-specific metadata (e.g., sandbox mode, recursion limit)."""


@dataclass(frozen=True)
class AdapterResult:
    """Result of analyzing an external topology with Operon's epistemic theorems.

    Combines topology advice, an optional template for the PatternLibrary,
    structural warnings, and a composite risk score.
    """

    topology_advice: TopologyAdvice
    """Operon's topology recommendation for the analyzed structure."""

    suggested_template: PatternTemplate | None
    """A ``PatternTemplate`` derived from the external topology, ready to
    register in a ``PatternLibrary``.  ``None`` if conversion failed."""

    warnings: tuple[str, ...]
    """Structural concerns identified by the epistemic analysis (e.g.,
    'error amplification exceeds tolerance', 'excessive sequential handoffs')."""

    risk_score: float
    """Composite risk score in ``[0.0, 1.0]``.  Higher means the topology
    is more likely to amplify errors or incur coordination overhead."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Optional runtime configuration hints for compiled topologies.

    Used by Phase C5 compilers to pass deployment parameters alongside
    the compiled workflow.
    """

    provider: str = "mock"
    """Model provider to use (``"openai"``, ``"anthropic"``, ``"mock"``, etc.)."""

    timeout: float = 30.0
    """Default per-stage timeout in seconds."""

    max_retries: int = 1
    """Maximum retry count per stage on failure."""

    sandbox: str = "none"
    """Execution sandbox mode: ``"none"``, ``"docker"``, ``"k8s"``."""

    metadata: dict[str, Any] = field(default_factory=dict)
