"""Mock evaluator for the C6 convergence evaluation harness.

For each task x configuration pair:

1. Builds a SkillOrganism from the task's required_roles.
2. Compiles through the config's real compiler (Swarms/DeerFlow/Ralph/Scion).
3. Parses compiled dict back through adapter -> ExternalTopology.
4. Runs analyze_external_topology() -> real AdapterResult with risk_score.
5. Derives synthetic metrics from the real analysis outputs.

The Operon-native path builds topology edges matching the task shape
(parallel=none, mixed=hub-spoke, sequential=chain) using advise_topology().

**Limitation:** Guided external configs (swarms_operon, scion_operon) change
stage modes but not the compiled topology structure, because SkillOrganism
is inherently a sequential pipeline. Genuine topology differences between
guided and unguided require live evaluation with the actual frameworks (C6b).
The mock evaluator demonstrates the *analysis pipeline* and *metric
collection*, not real guidance effects on external frameworks.

All compilers and adapters are imported from operon_ai.convergence.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.patterns.advisor import advise_topology

from operon_ai.convergence.swarms_compiler import organism_to_swarms
from operon_ai.convergence.swarms_adapter import (
    analyze_external_topology,
    parse_swarm_topology,
)
from operon_ai.convergence.deerflow_compiler import organism_to_deerflow
from operon_ai.convergence.deerflow_adapter import parse_deerflow_session
from operon_ai.convergence.ralph_compiler import organism_to_ralph
from operon_ai.convergence.ralph_adapter import parse_ralph_config
from operon_ai.convergence.scion_compiler import organism_to_scion
from operon_ai.convergence.types import ExternalTopology, RuntimeConfig

from .configurations import ConfigurationSpec
from .metrics import RunMetrics
from .structural_variation import topology_distance
from .tasks import TaskDefinition


# ---------------------------------------------------------------------------
# Compiler registry
# ---------------------------------------------------------------------------

_COMPILERS: dict[str, Any] = {
    "organism_to_swarms": organism_to_swarms,
    "organism_to_deerflow": organism_to_deerflow,
    "organism_to_ralph": organism_to_ralph,
    "organism_to_scion": organism_to_scion,
}


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def _swarms_dict_to_topology(compiled: dict) -> ExternalTopology:
    """Parse a Swarms compiler output dict into an ExternalTopology."""
    return parse_swarm_topology(
        pattern_name=compiled["workflow_type"],
        agent_specs=compiled["agents"],
        edges=compiled["edges"],
    )


def _deerflow_dict_to_topology(compiled: dict) -> ExternalTopology:
    """Parse a DeerFlow compiler output dict into an ExternalTopology."""
    return parse_deerflow_session(compiled)


def _ralph_dict_to_topology(compiled: dict) -> ExternalTopology:
    """Parse a Ralph compiler output dict into an ExternalTopology."""
    return parse_ralph_config(compiled)


def _scion_dict_to_topology(compiled: dict) -> ExternalTopology:
    """Parse a Scion compiler output dict into an ExternalTopology.

    Scion has no dedicated adapter, so we construct the ExternalTopology
    directly from the compiled grove config.
    """
    agents_raw = compiled.get("agents", [])
    messaging = compiled.get("messaging", [])

    # Filter out the watcher agent added by the compiler.
    agent_specs = []
    for a in agents_raw:
        if a.get("name") == "operon-watcher":
            continue
        template = a.get("template", {})
        agent_specs.append({
            "name": a["name"],
            "role": a["name"],
            "capabilities": template.get("skills", []),
        })

    edges = [(m["from"], m["to"]) for m in messaging]

    return ExternalTopology(
        source="scion",
        pattern_name="ScionGrove",
        agents=tuple(agent_specs),
        edges=tuple(edges),
        metadata=compiled,
    )


_ADAPTER_DISPATCH: dict[str, Any] = {
    "parse_swarm_topology": _swarms_dict_to_topology,
    "parse_deerflow_session": _deerflow_dict_to_topology,
    "parse_ralph_config": _ralph_dict_to_topology,
}

# Guidance risk reduction factor.
# Guidance is applied structurally (via advise_topology in _build_organism),
# not as a post-hoc risk multiplier.


# ---------------------------------------------------------------------------
# Organism builder
# ---------------------------------------------------------------------------


def _build_organism(task: TaskDefinition, *, guided: bool = False):
    """Build a SkillOrganism from a task's full definition.

    When guided=True, uses advise_topology() to inform the stage structure:
    - If Operon recommends a reviewer gate, the last stage gets mode="fixed"
    - The advice object is returned alongside the organism for topology alignment

    Unguided configs always use mode="fuzzy" for all stages (generic pipeline).
    """
    fast = Nucleus(provider=MockProvider(responses={}))
    deep = Nucleus(provider=MockProvider(responses={}))

    advice = None
    if guided:
        # Guided configs use stricter error_tolerance (0.01) which triggers
        # reviewer-gate recommendations for sequential tasks, producing
        # materially different organisms from unguided baselines.
        advice = advise_topology(
            task_shape=task.task_shape,
            tool_count=task.tool_count,
            subtask_count=task.subtask_count,
            error_tolerance=0.01,
        )

    stages = []
    n_roles = len(task.required_roles)
    for i, role in enumerate(task.required_roles):
        if guided and advice:
            # Guided: map Operon's recommendation to stage modes.
            rec = advice.recommended_pattern
            if "reviewer" in rec or "gate" in rec:
                # Reviewer gate: last stage is observational (fixed).
                mode = "fixed" if i == n_roles - 1 else "fuzzy"
            elif "swarm" in rec or "specialist" in rec:
                # Specialist swarm: all stages are independent (fuzzy).
                mode = "fuzzy"
            elif "single" in rec:
                # Single worker: one stage does everything.
                mode = "fuzzy"
            else:
                mode = "fuzzy"
        else:
            # Unguided: all stages are generic fuzzy (no structural insight).
            mode = "fuzzy"

        stages.append(SkillStage(
            name=f"{role}_{i}",
            role=role,
            instructions=f"Perform {role} duties for: {task.description}",
            mode=mode,
            handler=lambda t, _r=role: {_r: "done"},
        ))

    organism = skill_organism(
        stages=stages,
        fast_nucleus=fast,
        deep_nucleus=deep,
    )
    return organism, advice


# ---------------------------------------------------------------------------
# Operon-native evaluation (no external compiler)
# ---------------------------------------------------------------------------


def _evaluate_operon_native(
    task: TaskDefinition,
    rng: random.Random,
) -> RunMetrics:
    """Evaluate using Operon's native adaptive topology -- no compiler needed.

    Builds the organism and directly analyzes its structure as if it were
    an external topology (sequential chain of stages).
    """
    organism, advice = _build_organism(task, guided=True)
    stages = organism.stages

    # Build ExternalTopology with edge structure matching the task shape
    # (not always a linear chain). Operon-native uses advised topology.
    agent_specs = []
    for s in stages:
        agent_specs.append({
            "name": s.name,
            "role": s.role,
        })
    # Edge structure from task shape (guided by advise_topology).
    if task.task_shape == "parallel":
        # Parallel: no edges (independent agents).
        edges = []
    elif task.task_shape == "mixed":
        # Mixed: first stage fans out to all others (hub-and-spoke).
        if len(stages) > 1:
            edges = [(stages[0].name, stages[i].name) for i in range(1, len(stages))]
        else:
            edges = []
    else:
        # Sequential: linear chain.
        edges = [(stages[i].name, stages[i + 1].name) for i in range(len(stages) - 1)]

    # Use advised pattern name (from advise_topology) for the native path.
    pattern_name = advice.recommended_pattern if advice else "SkillOrganism"
    topology = ExternalTopology(
        source="operon",
        pattern_name=pattern_name,
        agents=tuple(agent_specs),
        edges=tuple(edges),
        metadata={},
    )

    result = analyze_external_topology(topology)

    # Operon-native config uses advise_topology to inform organism construction,
    # which naturally produces a topology-aligned organism with lower risk.
    risk_score = result.risk_score

    stage_count = len(stages)
    success = rng.random() > risk_score
    token_cost = int(1000 * stage_count * (1.0 + risk_score))
    # Sequential overhead proportional to edge count.
    seq_overhead = len(edges) * 0.1 if task.task_shape == "sequential" else 0.0
    latency_ms = 500.0 * stage_count * (1.0 + seq_overhead)
    intervention_count = len(result.warnings)

    # Structural variation: compare task shape to realized topology.
    # Compute realized shape from the actual topology structure.
    from operon_ai.convergence.swarms_adapter import _classify_task_shape
    realized_shape = _classify_task_shape(topology)
    variation = topology_distance(task.task_shape, realized_shape)

    return RunMetrics(
        task_id=task.task_id,
        config_id="operon_adaptive",
        success=success,
        token_cost=token_cost,
        latency_ms=latency_ms,
        intervention_count=intervention_count,
        convergence_rate=1.0 - risk_score,
        structural_variation=variation,
        risk_score=risk_score,
        stage_count=stage_count,
    )


# ---------------------------------------------------------------------------
# MockEvaluator
# ---------------------------------------------------------------------------


class MockEvaluator:
    """Evaluates task x configuration pairs using real Operon compilers and adapters.

    Uses deterministic RNG for reproducibility.
    """

    def __init__(self, seed: int = 1337) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def evaluate(self, task: TaskDefinition, config: ConfigurationSpec) -> RunMetrics:
        """Evaluate a single task under a single configuration.

        Steps:
        1. Build SkillOrganism from task.required_roles.
        2. Compile using the config's compiler.
        3. Parse compiled dict back through adapter -> ExternalTopology.
        4. Run analyze_external_topology() -> AdapterResult.
        5. Derive synthetic metrics from risk_score.
        6. For guided configs, reduce risk_score by 30%.
        """
        # Stable per-evaluation RNG (hashlib, not hash() which varies across processes).
        seed_str = f"{self._seed}:{task.task_id}:{config.config_id}"
        pair_seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        rng = random.Random(pair_seed)

        # Handle operon-native config (no compiler).
        if config.compiler_fn is None:
            return _evaluate_operon_native(task, rng)

        # Step 1: Build organism (guided configs get topology-informed mode assignment).
        organism, _advice = _build_organism(task, guided=config.structural_guidance)
        stage_count = len(organism.stages)

        # Step 2: Compile through the real compiler.
        compiler = _COMPILERS[config.compiler_fn]
        runtime_cfg = RuntimeConfig(provider="mock")
        compiled = compiler(organism, config=runtime_cfg)

        # Step 3: Parse back through adapter.
        if config.adapter_fn and config.adapter_fn in _ADAPTER_DISPATCH:
            topology = _ADAPTER_DISPATCH[config.adapter_fn](compiled)
        elif config.framework == "scion":
            topology = _scion_dict_to_topology(compiled)
        else:
            # Fallback: build topology from compiled agents.
            topology = _scion_dict_to_topology(compiled)

        # Step 4: Analyze through Operon's epistemic engine.
        adapter_result = analyze_external_topology(topology)

        # Step 5: Derive metrics.
        risk_score = adapter_result.risk_score

        # Step 6: Guided configs already produce different organisms (topology-
        # informed mode assignment), which naturally affects the risk score.
        # No synthetic reduction needed — the structural difference is real.

        success = rng.random() > risk_score
        token_cost = int(1000 * stage_count * (1.0 + risk_score))

        # Latency: sequential tasks have overhead from edges.
        seq_overhead = 0.0
        if len(topology.edges) > 0:
            seq_overhead = len(topology.edges) * 0.1 / max(len(topology.agents), 1)
        latency_ms = 500.0 * stage_count * (1.0 + seq_overhead)

        intervention_count = len(adapter_result.warnings)

        # Structural variation.
        from operon_ai.convergence.swarms_adapter import _classify_task_shape
        realized_shape = _classify_task_shape(topology)
        variation = topology_distance(task.task_shape, realized_shape)

        convergence_rate = 1.0 - risk_score

        return RunMetrics(
            task_id=task.task_id,
            config_id=config.config_id,
            success=success,
            token_cost=token_cost,
            latency_ms=latency_ms,
            intervention_count=intervention_count,
            convergence_rate=convergence_rate,
            structural_variation=variation,
            risk_score=risk_score,
            stage_count=stage_count,
        )
