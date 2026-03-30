"""Swarms convergence adapter -- parse Swarms workflow patterns into Operon types.

This is the core convergence adapter.  It provides:

- :func:`parse_swarm_topology` -- convert Swarms-style agent specs and edges
  into an :class:`ExternalTopology`.
- :func:`analyze_external_topology` -- apply Operon's four epistemic theorems
  to *any* ``ExternalTopology`` (shared across all adapters).
- :func:`swarm_to_template` -- convert an ``ExternalTopology`` into a
  :class:`PatternTemplate` for the :class:`PatternLibrary`.

Design principles:
  - NO imports of the ``swarms`` package -- only ``operon_ai`` imports.
  - All inputs are plain dicts / tuples.
  - Pure functions, no side effects.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..core.epistemic import (
    TopologyClass,
    TopologyRecommendation,
    classify_topology,
    error_amplification_bound,
    recommend_topology,
    sequential_penalty,
    tool_density,
)
from ..core.types import Capability, DataType, IntegrityLabel
from ..core.wagent import ModuleSpec, PortType, WiringDiagram
from ..patterns.repository import PatternTemplate, TaskFingerprint
from ..patterns.types import TopologyAdvice
from .types import AdapterResult, ExternalTopology

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PORT = PortType(DataType.JSON, IntegrityLabel.VALIDATED)

# Thresholds used to convert theorem outputs into a scalar risk score.
_ERROR_BOUND_WARN_THRESHOLD = 3.0
_SEQUENTIAL_OVERHEAD_WARN_THRESHOLD = 0.3
_TOOL_DENSITY_WARN_THRESHOLD = 4.0

# Weights for the composite risk score (must sum to 1.0).
_W_ERROR = 0.40
_W_SEQUENTIAL = 0.30
_W_DENSITY = 0.20
_W_TOPOLOGY_MISMATCH = 0.10

# Map Swarms pattern names to broad task shapes.
_PATTERN_TO_SHAPE: dict[str, str] = {
    "sequentialworkflow": "sequential",
    "hierarchicalswarm": "mixed",
    "concurrentworkflow": "parallel",
    "graphworkflow": "mixed",
}

# Map task shapes to PatternTemplate topology labels.
def _shape_to_topology(shape: str, n_agents: int) -> str:
    """Pick topology label, accounting for multi-stage sequential workflows."""
    if shape == "sequential":
        return "skill_organism" if n_agents > 1 else "single_worker"
    return "specialist_swarm"


# ---------------------------------------------------------------------------
# parse_swarm_topology
# ---------------------------------------------------------------------------


def parse_swarm_topology(
    pattern_name: str,
    agent_specs: list[dict[str, Any]],
    edges: list[tuple[str, str]],
    **metadata: Any,
) -> ExternalTopology:
    """Parse Swarms-style configuration into an :class:`ExternalTopology`.

    Parameters
    ----------
    pattern_name:
        The Swarms workflow class name, e.g. ``"SequentialWorkflow"``,
        ``"HierarchicalSwarm"``, ``"ConcurrentWorkflow"``, ``"GraphWorkflow"``.
    agent_specs:
        List of dicts with at least ``"name"`` and ``"role"`` keys.
        Additional keys (``"capabilities"``, ``"model"``, etc.) are preserved.
    edges:
        Directed ``(from_agent, to_agent)`` tuples describing the
        communication or control-flow graph.
    **metadata:
        Arbitrary extra info attached to the topology.

    Returns
    -------
    ExternalTopology
        A frozen, source-agnostic representation ready for analysis.
    """
    agents = tuple(dict(spec) for spec in agent_specs)
    return ExternalTopology(
        source="swarms",
        pattern_name=pattern_name,
        agents=agents,
        edges=tuple(edges),
        metadata=dict(metadata),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_wiring_diagram(topology: ExternalTopology) -> WiringDiagram:
    """Construct a :class:`WiringDiagram` from an external topology.

    Each agent becomes a module with a single ``"out"`` output port and a
    single ``"in"`` input port (both ``JSON / VALIDATED``).  Each edge
    connects the source's ``"out"`` to the destination's ``"in"``.

    When an agent has multiple incoming edges we create numbered input ports
    (``"in_0"``, ``"in_1"``, ...) so every wire has a unique destination port.
    """
    diagram = WiringDiagram()

    # Count incoming edges per agent so we know how many input ports to create.
    in_counts: dict[str, int] = {}
    for _, dst in topology.edges:
        in_counts[dst] = in_counts.get(dst, 0) + 1

    for spec in topology.agents:
        name = spec["name"]
        n_in = in_counts.get(name, 0)

        # Build input ports.
        if n_in <= 1:
            inputs = {"in": _DEFAULT_PORT} if n_in == 1 else {}
        else:
            inputs = {f"in_{i}": _DEFAULT_PORT for i in range(n_in)}

        # Every agent gets an output port (it may or may not be wired).
        outputs = {"out": _DEFAULT_PORT}

        # Propagate capabilities/skills onto the ModuleSpec.
        # Only convert recognized Capability enum values; skip unknown tool names.
        _cap_values = {c.value for c in Capability}
        agent_caps = _get_agent_capabilities(spec)
        caps = {Capability(c) for c in agent_caps if c in _cap_values}

        diagram.add_module(ModuleSpec(name=name, inputs=inputs, outputs=outputs, capabilities=caps))

    # Wire edges.  Track how many wires have already landed on each dst.
    dst_cursor: dict[str, int] = {}
    for src, dst in topology.edges:
        n_in = in_counts.get(dst, 0)
        idx = dst_cursor.get(dst, 0)
        dst_cursor[dst] = idx + 1

        dst_port = "in" if n_in <= 1 else f"in_{idx}"
        diagram.connect(src, "out", dst, dst_port)

    return diagram


def _classify_task_shape(topology: ExternalTopology) -> str:
    """Derive a broad task shape from the topology structure.

    Uses the pattern name for Swarms sources, and structural heuristics
    for DeerFlow/AnimaWorks/unknown sources.
    """
    key = topology.pattern_name.strip().lower().replace("_", "").replace("-", "")
    # Check Swarms-specific pattern names first.
    shape = _PATTERN_TO_SHAPE.get(key, None)
    if shape is not None:
        return shape
    # Structural heuristic: no edges → parallel; linear chain → sequential.
    if len(topology.edges) == 0:
        return "parallel"
    # Detect a true linear chain: every node has indegree ≤ 1 and outdegree ≤ 1.
    in_deg: dict[str, int] = {}
    out_deg: dict[str, int] = {}
    for src, dst in topology.edges:
        out_deg[src] = out_deg.get(src, 0) + 1
        in_deg[dst] = in_deg.get(dst, 0) + 1
    agent_names = {a.get("name", "") for a in topology.agents}
    edge_nodes = {s for s, _ in topology.edges} | {d for _, d in topology.edges}
    # True linear chain: indegree/outdegree ≤ 1, exactly one head (in=0,out=1),
    # one tail (in=1,out=0), n-1 edges, all edge nodes are agents, connected.
    # Chain must cover ALL agents (no isolated nodes allowed).
    if edge_nodes != agent_names:
        return "mixed"
    max_in = max((in_deg.get(n, 0) for n in agent_names), default=0)
    max_out = max((out_deg.get(n, 0) for n in agent_names), default=0)
    if max_in > 1 or max_out > 1:
        return "mixed"
    heads = [n for n in agent_names if in_deg.get(n, 0) == 0 and out_deg.get(n, 0) == 1]
    tails = [n for n in agent_names if in_deg.get(n, 0) == 1 and out_deg.get(n, 0) == 0]
    if len(heads) == 1 and len(tails) == 1 and len(topology.edges) == len(agent_names) - 1:
        return "sequential"
    return "mixed"


def _get_agent_capabilities(spec: dict[str, Any]) -> list[str]:
    """Extract capabilities from an agent spec, normalizing skills/capabilities."""
    caps = spec.get("capabilities") or spec.get("skills") or []
    if isinstance(caps, (list, tuple, set, frozenset)):
        return [str(c) for c in caps]
    return []


def _has_capabilities(topology: ExternalTopology) -> bool:
    """Return ``True`` if any agent spec declares capabilities or skills."""
    return any(_get_agent_capabilities(spec) for spec in topology.agents)


def _count_tools(topology: ExternalTopology) -> int:
    """Count the total number of distinct capabilities across all agents."""
    tools: set[str] = set()
    for spec in topology.agents:
        tools.update(_get_agent_capabilities(spec))
    return len(tools)


def _agents_independent(topology: ExternalTopology) -> bool:
    """Heuristic: agents are independent if there are no edges."""
    return len(topology.edges) == 0


# ---------------------------------------------------------------------------
# analyze_external_topology
# ---------------------------------------------------------------------------


def analyze_external_topology(topology: ExternalTopology) -> AdapterResult:
    """Apply Operon's epistemic theorems to any :class:`ExternalTopology`.

    Steps:

    1. Build a :class:`WiringDiagram` from the topology's agents and edges.
    2. Run :func:`classify_topology` to get the topology class.
    3. Run :func:`error_amplification_bound` -- warn if bounds exceed thresholds.
    4. Run :func:`sequential_penalty` -- warn if handoff overhead is high.
    5. Run :func:`tool_density` if agents have capabilities.
    6. Use :func:`recommend_topology` to get Operon's recommendation.
    7. Build a :class:`TopologyAdvice` from the recommendation.
    8. Build a :class:`PatternTemplate` from the topology.
    9. Compute ``risk_score`` as a weighted combination of theorem outputs.
    10. Return :class:`AdapterResult`.
    """
    warnings: list[str] = []

    # 1. Build wiring diagram.
    diagram = _build_wiring_diagram(topology)

    # 2. Classify topology.
    classification = classify_topology(diagram)

    # 3. Error amplification.
    err = error_amplification_bound(diagram)
    if err.centralized_bound > _ERROR_BOUND_WARN_THRESHOLD:
        warnings.append(
            f"Error amplification: centralized bound {err.centralized_bound:.1f} "
            f"exceeds threshold {_ERROR_BOUND_WARN_THRESHOLD:.1f} "
            f"({err.n_agents} agents, detection rate {err.detection_rate})"
        )

    # 4. Sequential penalty.
    #    The epistemic ``sequential_penalty`` uses cost-weighted critical paths;
    #    external topologies rarely carry cost annotations, so we supplement it
    #    with a *structural* handoff estimate derived directly from edge count.
    seq = sequential_penalty(diagram)
    n_agents = len(topology.agents)
    n_edges = len(topology.edges)
    structural_overhead = (n_edges * 0.4 / n_agents) if n_agents > 0 else 0.0
    effective_overhead = max(seq.overhead_ratio, structural_overhead)
    if effective_overhead > _SEQUENTIAL_OVERHEAD_WARN_THRESHOLD:
        warnings.append(
            f"Sequential overhead: {effective_overhead:.2f} "
            f"exceeds threshold {_SEQUENTIAL_OVERHEAD_WARN_THRESHOLD:.2f} "
            f"({n_edges} handoffs across {n_agents} agents)"
        )

    # 5. Tool density.
    #    The WiringDiagram only carries recognized Capability enum values, so
    #    external tool names (e.g., "web_search") are invisible to tool_density().
    #    Supplement with a raw count from the topology specs.
    density = tool_density(diagram)
    raw_tool_count = _count_tools(topology)
    effective_tools = max(density.total_tools, raw_tool_count)
    modules_with_tools = sum(1 for s in topology.agents if _get_agent_capabilities(s))
    effective_modules = max(density.num_modules, modules_with_tools)
    # planning_cost_ratio = number of capability-bearing modules (per tool_density()).
    effective_planning_ratio = max(density.planning_cost_ratio, float(effective_modules))
    if effective_planning_ratio > _TOOL_DENSITY_WARN_THRESHOLD:
        warnings.append(
            f"Tool density: planning cost ratio {effective_planning_ratio:.1f} "
            f"exceeds threshold {_TOOL_DENSITY_WARN_THRESHOLD:.1f} "
            f"({effective_tools} tools across {effective_modules} modules)"
        )

    # 6. Operon's recommendation.
    task_shape = _classify_task_shape(topology)
    n_subtasks = len(topology.agents)
    n_tools = _count_tools(topology)
    independent = _agents_independent(topology)
    rec = recommend_topology(
        num_subtasks=n_subtasks,
        subtasks_independent=independent,
        num_tools=n_tools,
        error_tolerance=0.1,
    )

    # Check if the external topology's shape diverges from Operon's recommendation.
    topology_mismatch = 0.0
    if classification.topology_class != rec.recommended:
        topology_mismatch = 1.0
        warnings.append(
            f"Topology mismatch: external topology is {classification.topology_class.value} "
            f"but Operon recommends {rec.recommended.value} -- {rec.rationale}"
        )

    # 7. TopologyAdvice.
    pattern_label = _shape_to_topology(task_shape, len(topology.agents))
    advice = TopologyAdvice(
        recommended_pattern=pattern_label,
        suggested_api=f"{pattern_label}(...)",
        topology=rec.recommended,
        rationale=rec.rationale,
        raw=rec,
    )

    # 8. PatternTemplate (dispatch by source for correct naming/tags).
    template = topology_to_template(topology)

    # 9. Composite risk score in [0.0, 1.0].
    n = max(err.n_agents, 1)
    error_risk = min(err.centralized_bound / (n * 2), 1.0)
    seq_risk = min(effective_overhead / 1.0, 1.0)
    density_risk = min(effective_planning_ratio / 10.0, 1.0) if effective_planning_ratio > 0 else 0.0

    risk_score = (
        _W_ERROR * error_risk
        + _W_SEQUENTIAL * seq_risk
        + _W_DENSITY * density_risk
        + _W_TOPOLOGY_MISMATCH * topology_mismatch
    )
    risk_score = round(min(max(risk_score, 0.0), 1.0), 4)

    # 10. Return.
    return AdapterResult(
        topology_advice=advice,
        suggested_template=template,
        warnings=tuple(warnings),
        risk_score=risk_score,
    )


# ---------------------------------------------------------------------------
# topology_to_template (source-agnostic)
# ---------------------------------------------------------------------------


def topology_to_template(topology: ExternalTopology) -> PatternTemplate:
    """Convert any :class:`ExternalTopology` to a :class:`PatternTemplate`.

    Dispatches to source-specific converters when available, otherwise
    builds a generic template preserving the source metadata.
    """
    # Dispatch to source-specific converters that understand metadata semantics.
    if topology.source == "animaworks":
        from .animaworks_adapter import animaworks_to_template
        org_config = topology.metadata.get("_org_config")
        if org_config is not None:
            return animaworks_to_template(org_config)
    elif topology.source == "deerflow":
        from .deerflow_adapter import deerflow_to_template
        session_config = topology.metadata.get("_session_config")
        if session_config is not None:
            return deerflow_to_template(session_config)
    elif topology.source == "ralph":
        from .ralph_adapter import ralph_to_template
        ralph_config = topology.metadata.get("_ralph_config")
        if ralph_config is not None:
            return ralph_to_template(ralph_config)
    elif topology.source == "aevolve":
        from .aevolve_adapter import aevolve_to_template
        aevolve_manifest = topology.metadata.get("_aevolve_manifest")
        if aevolve_manifest is not None:
            return aevolve_to_template(aevolve_manifest)

    # Generic fallback for Swarms and unknown sources.
    task_shape = _classify_task_shape(topology)
    topo_label = _shape_to_topology(task_shape, len(topology.agents))

    stage_specs = tuple(
        {
            "name": spec["name"],
            "role": spec.get("role", "worker"),
            "mode": "fuzzy",
        }
        for spec in topology.agents
    )

    roles = tuple(spec.get("role", "worker") for spec in topology.agents)
    tools = _count_tools(topology)
    source = topology.source or "unknown"

    fingerprint = TaskFingerprint(
        task_shape=task_shape,
        tool_count=tools,
        subtask_count=len(topology.agents),
        required_roles=roles,
        tags=(f"{source}:{topology.pattern_name}",),
    )

    return PatternTemplate(
        template_id=uuid4().hex[:8],
        name=f"{source}_{topology.pattern_name}",
        topology=topo_label,
        stage_specs=stage_specs,
        intervention_policy={"mode": "default"},
        fingerprint=fingerprint,
        tags=(source, topology.pattern_name),
    )


# ---------------------------------------------------------------------------
# swarm_to_template (convenience alias)
# ---------------------------------------------------------------------------


def swarm_to_template(topology: ExternalTopology) -> PatternTemplate:
    """Convert a Swarms topology to a :class:`PatternTemplate`.

    The template is ready to register in a :class:`PatternLibrary`.

    Mapping:
    - ``pattern_name`` drives the ``topology`` field via ``_SHAPE_TO_TOPOLOGY``.
    - ``stage_specs`` are derived from the agent specs.
    - A :class:`TaskFingerprint` is synthesized from the topology shape.
    """
    task_shape = _classify_task_shape(topology)
    topo_label = _shape_to_topology(task_shape, len(topology.agents))

    stage_specs = tuple(
        {
            "name": spec["name"],
            "role": spec.get("role", "worker"),
            "mode": "fuzzy",
        }
        for spec in topology.agents
    )

    roles = tuple(spec.get("role", "worker") for spec in topology.agents)

    fingerprint = TaskFingerprint(
        task_shape=task_shape,
        tool_count=_count_tools(topology),
        subtask_count=len(topology.agents),
        required_roles=roles,
        tags=(f"swarms:{topology.pattern_name}",),
    )

    return PatternTemplate(
        template_id=uuid4().hex[:8],
        name=f"swarms_{topology.pattern_name}",
        topology=topo_label,
        stage_specs=stage_specs,
        intervention_policy={"mode": "default"},
        fingerprint=fingerprint,
        tags=("swarms", topology.pattern_name),
    )
